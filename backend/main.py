"""
IBVAP — Intelligent Border Video Analytics Platform
Main FastAPI Application

Run:  python main.py
  or: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import uuid
import os
import base64
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DATABASE_PATH, FRONTEND_DIR, SNAPSHOTS_DIR, DEMO_VIDEOS_DIR, DATA_DIR
from database import Database
from services.detector import ObjectDetector
from services.virtual_fence import VirtualFence
from services.alert_engine import AlertEngine
from services.night_enhance import NightEnhancer
from services.behavior_engine import BehaviorEngine
from services.frs_engine import FRSEngine
from services.anpr_engine import ANPREngine
from services.stream_manager import StreamManager


# ═══════════════════════════════════════════════════════════════
# Globals
# ═══════════════════════════════════════════════════════════════
db: Database = None
detector: ObjectDetector = None
fence: VirtualFence = None
alert_engine: AlertEngine = None
night_enhancer: NightEnhancer = None
behavior_engine: BehaviorEngine = None
frs_engine: FRSEngine = None
anpr_engine: ANPREngine = None
stream_manager: StreamManager = None

# WebSocket connections for real-time alerts
ws_clients: set[WebSocket] = set()


# ═══════════════════════════════════════════════════════════════
# App lifecycle
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, detector, fence, alert_engine, night_enhancer, behavior_engine, frs_engine, anpr_engine, stream_manager

    print("=" * 60)
    print("  IBVAP — Intelligent Border Video Analytics Platform")
    print("  Starting up...")
    print("=" * 60)

    db = Database(DATABASE_PATH)
    detector = ObjectDetector()
    fence = VirtualFence()
    night_enhancer = NightEnhancer()
    alert_engine = AlertEngine(db)
    behavior_engine = BehaviorEngine()
    frs_engine = FRSEngine(db)
    anpr_engine = ANPREngine(db)

    stream_manager = StreamManager(
        detector, fence, alert_engine, night_enhancer, db, behavior_engine, frs_engine, anpr_engine
    )

    # Get the running event loop and wire alert push
    running_loop = asyncio.get_running_loop()
    alert_engine.on_new_alert = lambda a: asyncio.run_coroutine_threadsafe(
        broadcast_alert(a), running_loop
    )

    # Start background stats broadcaster
    stats_task = asyncio.create_task(broadcast_stats())

    # Auto-start cameras that were previously active
    for cam in db.list_cameras():
        stream_manager.add_camera(cam["id"], cam["source"])
        if cam["status"] == "active":
            stream_manager.start_camera(cam["id"])

    print("[IBVAP] System ready ✓")
    print(f"[IBVAP] Dashboard → http://localhost:8000")
    print("=" * 60)

    yield

    # Cancel stats task on shutdown
    stats_task.cancel()

    # Shutdown
    print("[IBVAP] Shutting down...")
    stream_manager.stop_all()


app = FastAPI(
    title="IBVAP",
    description="Intelligent Border Video Analytics Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve snapshots & data
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")



# ═══════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════
class AddCameraRequest(BaseModel):
    name: Optional[str] = None
    source: str
    location: Optional[str] = "Border Checkpost"
    auto_start: bool = True


class SetFenceRequest(BaseModel):
    zones: list[dict]  # [{"id": "z1", "name": "Zone A", "points": [[x,y], ...]}, ...]


class NightModeRequest(BaseModel):
    mode: Optional[str] = None
    enabled: Optional[bool] = None



class VisionModeRequest(BaseModel):
    mode: str  # "normal", "clahe", "flir", "nvg"


class WatchlistTargetRequest(BaseModel):
    name: str
    category: str
    danger_level: Optional[str] = "HIGH"
    threat_level: Optional[str] = None
    notes: Optional[str] = ""
    photo_url: Optional[str] = ""
    image_base64: Optional[str] = ""


class HotlistVehicleRequest(BaseModel):
    plate_number: str
    vehicle_model: Optional[str] = "Vehicle"
    reason: str
    danger_level: Optional[str] = "HIGH"
    threat_level: Optional[str] = None
    status: Optional[str] = "ACTIVE"


class ScanVehicleRequest(BaseModel):
    camera_id: str
    plate_number: str
    vehicle_type: str
    confidence: Optional[float] = 0.95
    is_hotlist: Optional[int] = 0


class ThreatLevelRequest(BaseModel):
    level: int  # 1, 3, 5


# ═══════════════════════════════════════════════════════════════
# WebSocket — Real-time alerts & stats
# ═══════════════════════════════════════════════════════════════
async def broadcast_alert(alert: dict):
    """Push a new alert to all connected WebSocket clients."""
    global ws_clients
    message = json.dumps({"type": "alert", "data": alert})
    dead = set()
    for ws in list(ws_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    ws_clients -= dead


async def broadcast_stats():
    """Periodically push stats to all WebSocket clients."""
    global ws_clients
    while True:
        if ws_clients and stream_manager:
            stats = stream_manager.get_stats()
            message = json.dumps({"type": "stats", "data": stats})
            dead = set()
            for ws in list(ws_clients):
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.add(ws)
            ws_clients -= dead
        await asyncio.sleep(1)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    print(f"[WS] Client connected ({len(ws_clients)} total)")
    try:
        while True:
            # Keep connection alive; handle client messages if needed
            data = await websocket.receive_text()
            msg = json.loads(data)
            # Handle client commands
            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        ws_clients.discard(websocket)
        print(f"[WS] Client disconnected ({len(ws_clients)} total)")
    except Exception:
        ws_clients.discard(websocket)


# ═══════════════════════════════════════════════════════════════
# REST API — Cameras
# ═══════════════════════════════════════════════════════════════
@app.post("/api/cameras")
async def add_camera(req: AddCameraRequest):
    source_clean = req.source.strip()
    if req.name and req.name.strip():
        cam_name = req.name.strip()
    elif "youtube.com" in source_clean or "youtu.be" in source_clean:
        cam_name = "YouTube Live CCTV"
    elif source_clean.startswith("rtsp://"):
        cam_name = "RTSP Camera Feed"
    elif source_clean.startswith("http://") or source_clean.startswith("https://"):
        cam_name = "Online Stream Feed"
    else:
        cam_name = Path(source_clean).stem.replace("_", " ").title() or "Surveillance Feed"

    cam_id = f"cam_{uuid.uuid4().hex[:8]}"
    cam = db.add_camera(cam_id, cam_name, source_clean, req.location or "Border Checkpost")
    stream_manager.add_camera(cam_id, source_clean)
    if req.auto_start:
        ok = stream_manager.start_camera(cam_id)
        if not ok:
            return JSONResponse(
                {"error": f"Cannot open video source: {source_clean}", "camera": cam},
                status_code=400,
            )
        # Update camera name if YouTube title was discovered
        cs = stream_manager.cameras.get(cam_id)
        if cs and cs.stream_title and (not req.name or not req.name.strip()):
            db.update_camera_name(cam_id, cs.stream_title)

    return {"camera": db.get_camera(cam_id)}


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file from browser, save to demo_videos, and immediately start surveillance."""
    safe_name = file.filename.replace(" ", "_")
    target_path = DEMO_VIDEOS_DIR / f"upload_{uuid.uuid4().hex[:6]}_{safe_name}"
    with open(target_path, "wb") as f:
        content = await file.read()
        f.write(content)

    cam_name = Path(safe_name).stem.replace("_", " ").title()
    cam_id = f"cam_{uuid.uuid4().hex[:8]}"
    cam = db.add_camera(cam_id, cam_name, str(target_path), "Local Video Upload")
    stream_manager.add_camera(cam_id, str(target_path))
    ok = stream_manager.start_camera(cam_id)
    if not ok:
        return JSONResponse({"error": "Failed to open uploaded video", "camera": cam}, status_code=400)
    return {"camera": db.get_camera(cam_id), "path": str(target_path)}


@app.get("/api/cameras")
async def list_cameras():
    cameras = db.list_cameras()
    # Inject live stats
    for cam in cameras:
        cs = stream_manager.cameras.get(cam["id"])
        if cs:
            cam["night_mode"] = cs.night_mode
            cam["live"] = {
                "fps": round(cs.fps, 1),
                "persons": cs.person_count,
                "vehicles": cs.vehicle_count,
                "intrusions": cs.intrusion_count,
                "running": cs.running,
                "night_mode": cs.night_mode,
            }
    return {"cameras": cameras}



@app.post("/api/cameras/{camera_id}/start")
async def start_camera(camera_id: str):
    cam = db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    if camera_id not in stream_manager.cameras:
        stream_manager.add_camera(camera_id, cam["source"])
    ok = stream_manager.start_camera(camera_id)
    if not ok:
        raise HTTPException(400, f"Cannot open source: {cam['source']}")
    return {"status": "started"}


@app.post("/api/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    stream_manager.stop_camera(camera_id)
    return {"status": "stopped"}


@app.delete("/api/cameras")
async def clear_all_cameras():
    stream_manager.stop_all()
    stream_manager.cameras.clear()
    db.clear_all_cameras()
    return {"status": "all_cleared"}


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    stream_manager.remove_camera(camera_id)
    db.delete_camera(camera_id)
    return {"status": "deleted"}


# ── Night mode & Vision mode ────────────────────────────────────
@app.post("/api/cameras/{camera_id}/night-mode")
async def set_night_mode(camera_id: str, req: NightModeRequest):
    """
    Set camera tactical night vision mode.
    Supported modes: 'clahe' (Tactical CLAHE + Adaptive Gamma),
                     'thermal' (FLIR Ironbow False-Color),
                     'nvg' (Gen 3+ Green Phosphor NVG),
                     'off' (Raw Camera Feed).
    """
    cam_info = db.get_camera(camera_id)
    if not cam_info and camera_id not in stream_manager.cameras:
        raise HTTPException(404, f"Camera '{camera_id}' not found")

    if req.mode is not None:
        mode = req.mode.lower().strip()
    elif req.enabled is not None:
        mode = "clahe" if req.enabled else "off"
    else:
        mode = "off"

    # Alias flir -> thermal
    if mode == "flir":
        mode = "thermal"

    valid_modes = {"off", "clahe", "thermal", "nvg"}
    if mode not in valid_modes:
        raise HTTPException(
            400,
            f"Invalid night vision mode '{mode}'. Allowed modes: {sorted(list(valid_modes))}",
        )

    db.update_night_mode(camera_id, mode)
    stream_manager.set_night_mode(camera_id, mode)

    return {
        "camera_id": camera_id,
        "night_mode": mode,
        "status": "active" if mode != "off" else "off",
    }


@app.post("/api/cameras/{camera_id}/vision-mode")
async def set_vision_mode(camera_id: str, req: VisionModeRequest):
    mode = req.mode.lower().strip()
    if mode == "flir":
        mode = "thermal"
    elif mode == "normal":
        mode = "off"

    valid_modes = {"off", "clahe", "thermal", "nvg"}
    if mode not in valid_modes:
        raise HTTPException(400, f"Invalid vision mode '{mode}'. Allowed modes: {sorted(list(valid_modes))}")

    db.update_night_mode(camera_id, mode)
    stream_manager.set_night_mode(camera_id, mode)
    return {"status": "ok", "vision_mode": mode}



# ── Virtual fence ───────────────────────────────────────────────
@app.post("/api/cameras/{camera_id}/fence")
async def set_fence(camera_id: str, req: SetFenceRequest):
    db.update_fence_zones(camera_id, req.zones)
    fence.set_zones(camera_id, req.zones)
    return {"zones": fence.get_zones(camera_id)}


@app.get("/api/cameras/{camera_id}/fence")
async def get_fence(camera_id: str):
    return {"zones": fence.get_zones(camera_id)}


# ═══════════════════════════════════════════════════════════════
# REST API — Video Stream (MJPEG)
# ═══════════════════════════════════════════════════════════════
@app.get("/api/cameras/{camera_id}/stream")
async def camera_stream(camera_id: str):
    """MJPEG streaming endpoint for a camera."""
    cam = stream_manager.cameras.get(camera_id)
    if not cam or not cam.running:
        raise HTTPException(404, "Camera not active")

    return StreamingResponse(
        stream_manager.generate_mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ═══════════════════════════════════════════════════════════════
# REST API — Alerts
# ═══════════════════════════════════════════════════════════════
@app.get("/api/alerts")
async def list_alerts(
    limit: int = Query(50, ge=1, le=500),
    camera_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
):
    return {
        "alerts": db.list_alerts(
            limit=limit,
            camera_id=camera_id,
            alert_type=alert_type,
            severity=severity,
        )
    }


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    db.acknowledge_alert(alert_id)
    return {"status": "acknowledged"}


@app.delete("/api/alerts")
async def clear_alerts(camera_id: Optional[str] = None):
    db.clear_alerts(camera_id)
    return {"status": "cleared"}


# ═══════════════════════════════════════════════════════════════
# REST API — Analytics
# ═══════════════════════════════════════════════════════════════
@app.get("/api/analytics")
async def get_analytics():
    return {
        "alert_counts": db.get_alert_counts(),
        "behavioral_breakdown": db.get_behavioral_breakdown(),
        "hourly_alerts": db.get_hourly_alerts(24),
        "live_stats": stream_manager.get_stats() if stream_manager else {},
    }


# ═══════════════════════════════════════════════════════════════
# REST API — Demo Videos
# ═══════════════════════════════════════════════════════════════
@app.get("/api/demo-videos")
async def list_demo_videos():
    """List available demo video files."""
    videos = []
    for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
        for f in DEMO_VIDEOS_DIR.glob(ext):
            videos.append({"name": f.stem, "path": str(f), "size_mb": round(f.stat().st_size / 1e6, 1)})
    return {"videos": videos}


# ═══════════════════════════════════════════════════════════════
# REST API — Tactical Status & Defense Threat Level
# ═══════════════════════════════════════════════════════════════
tactical_state = {
    "threat_level": 5,  # 5: LOW, 3: ELEVATED, 1: CRITICAL
    "operator_name": "INSP. V. SHARMA [SSB 42nd BN]",
    "operator_callsign": "SENTRY-ALPHA-1",
    "station_id": "C2 TACTICAL CONSOLE 01",
    "sector_name": "SECTOR-IV (INDO-NEPAL FRONTIER - BIRGUNJ AXIS)",
    "active_bops": 7,
    "total_bops": 7,
}

@app.get("/api/tactical/status")
async def get_tactical_status():
    return tactical_state

@app.post("/api/tactical/threat-level")
async def set_threat_level(req: ThreatLevelRequest):
    tactical_state["threat_level"] = req.level
    # Broadcast to all WS clients
    for ws in list(ws_clients):
        try:
            await ws.send_text(json.dumps({"type": "threat_level", "data": {"level": req.level}}))
        except Exception:
            pass
    return {"status": "ok", "threat_level": req.level}


# ═══════════════════════════════════════════════════════════════
# REST API — FRS Watchlist
# ═══════════════════════════════════════════════════════════════
@app.get("/api/frs/watchlist")
async def get_frs_watchlist():
    if frs_engine:
        frs_engine.reload_watchlist()
    return {"watchlist": db.list_watchlist()}

@app.post("/api/frs/watchlist")
async def add_frs_target(req: WatchlistTargetRequest):
    threat = req.threat_level or req.danger_level or "HIGH"
    img_bytes = None
    if req.image_base64:
        try:
            raw_b64 = req.image_base64.split(",")[-1]
            img_bytes = base64.b64decode(raw_b64)
        except Exception:
            pass

    if frs_engine:
        target = frs_engine.add_person(
            name=req.name.strip(),
            category=req.category.strip(),
            threat_level=threat,
            notes=req.notes or "",
            image_bytes=img_bytes,
            photo_url=req.photo_url or "",
        )
    else:
        target_id = f"TGT-{uuid.uuid4().hex[:4].upper()}"
        photo = req.photo_url or f"https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150&auto=format&fit=crop&q=80"
        target = db.add_watchlist_target(
            target_id=target_id,
            name=req.name.strip(),
            category=req.category,
            danger_level=threat,
            notes=req.notes or "",
            photo_url=photo,
        )
    return {"target": target}

@app.post("/api/frs/watchlist/upload")
async def upload_frs_target(
    name: str = Query(...),
    category: str = Query(...),
    threat_level: str = Query("HIGH"),
    notes: str = Query(""),
    file: UploadFile = File(...),
):
    content = await file.read()
    if frs_engine:
        target = frs_engine.add_person(
            name=name.strip(),
            category=category.strip(),
            threat_level=threat_level,
            notes=notes,
            image_bytes=content,
        )
    else:
        target_id = f"TGT-{uuid.uuid4().hex[:4].upper()}"
        target = db.add_watchlist_target(
            target_id=target_id,
            name=name.strip(),
            category=category.strip(),
            danger_level=threat_level,
            notes=notes,
        )
    return {"target": target}

@app.delete("/api/frs/watchlist/{target_id}")
async def delete_frs_target(target_id: str):
    if frs_engine:
        frs_engine.delete_person(target_id)
    else:
        db.delete_watchlist_target(target_id)
    return {"status": "deleted"}

@app.post("/api/frs/simulate-match")
async def simulate_frs_match(target_id: Optional[str] = Query(None)):
    watchlist = db.list_watchlist()
    if not watchlist:
        raise HTTPException(404, "No targets in watchlist")
    
    target = None
    if target_id:
        target = db.get_watchlist_target(target_id)
    if not target:
        target = watchlist[0]

    location = "BOP-17 Sector IV Perimeter"
    db.record_frs_match(target["id"], location)

    alert_type = "WATCHLIST_SUSPECT_DETECTED"
    severity = "critical" if target.get("danger_level") in ("CRITICAL", "HIGH") or target.get("threat_level") in ("CRITICAL", "HIGH") else "medium"
    msg = f"WATCHLIST SUSPECT DETECTED: {target['name']} ({target['category'].upper()}, 98% biometric match, Target ID: {target['id']})"
    
    alert = db.add_alert(
        camera_id="cam_bop17",
        alert_type=alert_type,
        message=msg,
        severity=severity,
        details={
            "target_id": target["id"],
            "name": target["name"],
            "category": target["category"],
            "threat_level": target.get("threat_level", "CRITICAL"),
            "confidence": 98.4,
            "photo_url": target.get("photo_url", ""),
            "location": location,
        },
    )
    await broadcast_alert(alert)
    return {"status": "matched", "target": target, "alert": alert}


# ═══════════════════════════════════════════════════════════════
# REST API — ANPR Vehicle Scanner & Hotlist
# ═══════════════════════════════════════════════════════════════
@app.get("/api/anpr/hotlist")
async def get_anpr_hotlist():
    if anpr_engine:
        anpr_engine.reload_hotlist()
    return {"hotlist": db.list_hotlist()}

@app.post("/api/anpr/hotlist")
async def add_anpr_hotlist(req: HotlistVehicleRequest):
    threat = req.threat_level or req.danger_level or "HIGH"
    model = req.vehicle_model or "Vehicle"
    if anpr_engine:
        vehicle = anpr_engine.add_hotlist_plate(
            plate_number=req.plate_number,
            vehicle_model=model,
            reason=req.reason,
            threat_level=threat,
            status=req.status or "ACTIVE",
        )
    else:
        vehicle = db.add_hotlist_vehicle(
            plate_number=req.plate_number,
            vehicle_model=model,
            reason=req.reason,
            danger_level=threat,
            status=req.status or "ACTIVE",
        )
    return {"vehicle": vehicle}

@app.delete("/api/anpr/hotlist/{plate}")
async def delete_anpr_hotlist(plate: str):
    if anpr_engine:
        anpr_engine.delete_hotlist_plate(plate)
    else:
        db.delete_hotlist_vehicle(plate)
    return {"status": "deleted"}

@app.get("/api/anpr/scans")
async def get_anpr_scans(limit: int = Query(50, ge=1, le=200)):
    if anpr_engine:
        scans = anpr_engine.list_scans(limit)
        if scans:
            return {"scans": scans}
    return {"scans": db.list_scans(limit)}

@app.post("/api/anpr/scans")
async def record_anpr_scan(req: ScanVehicleRequest):
    scan = db.record_scan(
        camera_id=req.camera_id,
        plate_number=req.plate_number,
        vehicle_type=req.vehicle_type,
        confidence=req.confidence or 0.95,
        is_hotlist=req.is_hotlist or 0,
    )
    return {"scan": scan}

@app.post("/api/anpr/simulate-scan")
async def simulate_anpr_scan(plate: Optional[str] = Query(None)):
    hotlist = db.list_hotlist()
    target_veh = None
    if plate:
        for v in hotlist:
            if v["plate_number"].upper() == plate.upper():
                target_veh = v
                break
    if not target_veh and hotlist:
        target_veh = hotlist[0]
    
    plate_no = target_veh["plate_number"] if target_veh else "UP 53 AZ 4421"
    model = target_veh["vehicle_model"] if target_veh else "White Mahindra Bolero"
    reason = target_veh["reason"] if target_veh else "Suspected Contraband Carrier"

    scan = db.record_scan(
        camera_id="cam_bop14",
        plate_number=plate_no,
        vehicle_type=model,
        confidence=0.97,
        is_hotlist=1,
    )

    alert_type = "BLACKLIST_VEHICLE_DETECTED"
    msg = f"BLACKLIST VEHICLE DETECTED: [{plate_no}] ({model}) Reason: {reason}"
    alert = db.add_alert(
        camera_id="cam_bop14",
        alert_type=alert_type,
        message=msg,
        severity="critical",
        details={
            "plate_number": plate_no,
            "vehicle_model": model,
            "reason": reason,
            "confidence": 97.0,
            "location": "BOP-14 Vehicle Checkpost",
        },
    )
    await broadcast_alert(alert)
    return {"status": "scanned", "scan": scan, "alert": alert}


# ═══════════════════════════════════════════════════════════════
# Serve Frontend
# ═══════════════════════════════════════════════════════════════
if FRONTEND_DIR.exists():
    css_dir = FRONTEND_DIR / "css"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    js_dir = FRONTEND_DIR / "js"
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")


@app.get("/")
async def serve_dashboard():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")

