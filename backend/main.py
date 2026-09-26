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

# Configure FFmpeg low-delay and auto-reconnect flags
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "reconnect;1|reconnect_streamed;1|reconnect_delay_max;5|fflags;nobuffer|flags;low_delay|analyzeduration;1000000|probesize;1000000"
)

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DATABASE_PATH, FRONTEND_DIR, SNAPSHOTS_DIR, EVENTS_DIR, DEMO_VIDEOS_DIR
from database import Database
from services.detector import ObjectDetector
from services.virtual_fence import VirtualFence
from services.alert_engine import AlertEngine
from services.night_enhance import NightEnhancer
from services.stream_manager import StreamManager


# ═══════════════════════════════════════════════════════════════
# Globals
# ═══════════════════════════════════════════════════════════════
db: Database = None
detector: ObjectDetector = None
fence: VirtualFence = None
alert_engine: AlertEngine = None
night_enhancer: NightEnhancer = None
stream_manager: StreamManager = None

# WebSocket connections for real-time alerts
ws_clients: set[WebSocket] = set()


# ═══════════════════════════════════════════════════════════════
# App lifecycle
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, detector, fence, alert_engine, night_enhancer, stream_manager

    print("=" * 60)
    print("  IBVAP — Intelligent Border Video Analytics Platform")
    print("  Starting up...")
    print("=" * 60)

    db = Database(DATABASE_PATH)
    detector = ObjectDetector()
    fence = VirtualFence()
    night_enhancer = NightEnhancer()
    alert_engine = AlertEngine(db)

    stream_manager = StreamManager(detector, fence, alert_engine, night_enhancer, db)
    alert_engine.stream_manager = stream_manager

    # Get the running event loop and wire alert, ANPR, and AI metadata push
    running_loop = asyncio.get_running_loop()
    alert_engine.on_new_alert = lambda a: asyncio.run_coroutine_threadsafe(
        broadcast_alert(a), running_loop
    )
    if stream_manager and stream_manager.anpr_engine:
        stream_manager.anpr_engine.on_plate_captured = lambda p: asyncio.run_coroutine_threadsafe(
            broadcast_plate(p), running_loop
        )
    if stream_manager:
        stream_manager.on_ai_metadata = lambda m: asyncio.run_coroutine_threadsafe(
            broadcast_ai_metadata(m), running_loop
        )

    # Start background stats broadcaster
    stats_task = asyncio.create_task(broadcast_stats())

    # Auto-start cameras that were previously active in a non-blocking thread
    import threading
    def _auto_start_cams():
        for cam in db.list_cameras():
            stream_manager.add_camera(cam["id"], cam["source"])
            if cam["status"] == "active":
                try:
                    stream_manager.start_camera(cam["id"])
                except Exception as e:
                    print(f"[IBVAP] Auto-start error on {cam['id']}: {e}")

    threading.Thread(target=_auto_start_cams, daemon=True, name="CamAutoStarter").start()

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

# Serve snapshots and structured evidence vault
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")
app.mount("/api/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="api_snapshots")
app.mount("/api/events", StaticFiles(directory=str(EVENTS_DIR)), name="api_events")
app.mount("/events", StaticFiles(directory=str(EVENTS_DIR)), name="events")



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
    enabled: bool


# ═══════════════════════════════════════════════════════════════
# WebSocket — Real-time alerts & stats
# ═══════════════════════════════════════════════════════════════
async def broadcast_alert(alert: dict):
    """Push a new alert to all connected WebSocket clients."""
    message = json.dumps({"type": "alert", "data": alert})
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    ws_clients -= dead


async def broadcast_plate(plate: dict):
    """Push newly captured license plate photo and details to all connected WebSocket clients."""
    message = json.dumps({"type": "plate_detected", "data": plate})
    dead = set()
    for ws in list(ws_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


async def broadcast_ai_metadata(meta: dict):
    """Push real-time AI detections and bounding boxes to WebSocket clients."""
    message = json.dumps({"type": "ai_metadata", "data": meta})
    dead = set()
    for ws in list(ws_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


async def broadcast_stats():
    """Periodically push stats and real-time observability telemetry to all WebSocket clients."""
    while True:
        if ws_clients and stream_manager:
            try:
                stats = stream_manager.get_stats()
                telemetry = stream_manager.get_telemetry()
                message = json.dumps({
                    "type": "stats",
                    "data": stats,
                    "telemetry": telemetry,
                })
                dead = set()
                for ws in list(ws_clients):
                    try:
                        await ws.send_text(message)
                    except Exception:
                        dead.add(ws)
                ws_clients.difference_update(dead)
            except Exception as e:
                print(f"[WS Stats Error] {e}")
        await asyncio.sleep(1)


@app.get("/api/stats")
async def get_stats():
    """Fetch current aggregate detection metrics and camera telemetry."""
    if not stream_manager:
        return {"total_persons": 0, "total_vehicles": 0, "total_threats": 0, "cameras": {}}
    return {
        "stats": stream_manager.get_stats(),
        "telemetry": stream_manager.get_telemetry(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    print(f"[WS] Client connected ({len(ws_clients)} total)")
    if stream_manager:
        try:
            stats = stream_manager.get_stats()
            telemetry = stream_manager.get_telemetry()
            await websocket.send_text(json.dumps({
                "type": "stats",
                "data": stats,
                "telemetry": telemetry,
            }))
        except Exception:
            pass
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except Exception:
                pass
    except WebSocketDisconnect:
        ws_clients.discard(websocket)
        print(f"[WS] Client disconnected ({len(ws_clients)} total)")
    except Exception:
        ws_clients.discard(websocket)


@app.websocket("/ws/stream/{camera_id}")
async def websocket_camera_stream(websocket: WebSocket, camera_id: str):
    """
    Ultra-low latency WebSocket binary frame endpoint.
    Sends raw pre-encoded JPEG bytes of newest frame directly over WebSocket,
    bypassing HTTP chunked multipart buffering in browsers.
    """
    await websocket.accept()
    cam = stream_manager.cameras.get(camera_id) if stream_manager else None
    if not cam:
        await websocket.close(code=1008, reason="Camera not found or inactive")
        return

    last_sent_id = -1
    try:
        while cam.running:
            jpeg_bytes, stream_age_ms, frame_id = stream_manager.get_latest_jpeg(camera_id)
            if jpeg_bytes is not None and frame_id != last_sent_id:
                last_sent_id = frame_id
                await websocket.send_bytes(jpeg_bytes)
            await asyncio.sleep(0.015)
    except (WebSocketDisconnect, Exception):
        pass



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
            cam["live"] = {
                "fps": round(cs.fps, 1),
                "persons": cs.person_count,
                "vehicles": cs.vehicle_count,
                "threats": getattr(cs, "threat_count", 0),
                "crowds": getattr(cs, "crowd_count", 0),
                "plates": getattr(cs, "plate_count", 0),
                "intrusions": cs.intrusion_count,
                "running": cs.running,
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


# ── Night mode ──────────────────────────────────────────────────
@app.post("/api/cameras/{camera_id}/night-mode")
async def set_night_mode(camera_id: str, req: NightModeRequest):
    db.update_night_mode(camera_id, req.enabled)
    cam = stream_manager.cameras.get(camera_id)
    if cam:
        cam.night_mode = req.enabled
    return {"night_mode": req.enabled}


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
async def camera_stream(camera_id: str, annotated: bool = Query(True)):
    """MJPEG streaming endpoint for a camera (with real-time baked detection annotations)."""
    cam = stream_manager.cameras.get(camera_id)
    if not cam or not cam.running:
        raise HTTPException(404, "Camera not active")

    return StreamingResponse(
        stream_manager.generate_mjpeg(camera_id, annotated=annotated),
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
):
    return {"alerts": db.list_alerts(limit=limit, camera_id=camera_id)}


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    db.acknowledge_alert(alert_id)
    return {"status": "acknowledged"}


@app.delete("/api/alerts")
async def clear_alerts(camera_id: Optional[str] = None):
    db.clear_alerts(camera_id)
    return {"status": "cleared"}


# ═══════════════════════════════════════════════════════════════
# REST API — ANPR License Plates
# ═══════════════════════════════════════════════════════════════
@app.get("/api/license-plates")
async def list_license_plates(
    limit: int = Query(100, ge=1, le=500),
    camera_id: Optional[str] = None,
    search: Optional[str] = None,
):
    return {"plates": db.list_license_plates(limit=limit, camera_id=camera_id, search=search)}


@app.get("/api/recent-plates")
async def get_recent_plates():
    """Retrieve recently recognized license plates with photo snapshots."""
    plates = stream_manager.anpr_engine.captured_plates if (stream_manager and stream_manager.anpr_engine) else []
    return {"plates": plates}


@app.delete("/api/license-plates")
async def clear_license_plates(camera_id: Optional[str] = None):
    db.clear_license_plates(camera_id)
    return {"status": "cleared"}


# ═══════════════════════════════════════════════════════════════
# REST API — Telemetry & Observability
# ═══════════════════════════════════════════════════════════════
@app.get("/api/telemetry")
async def get_telemetry():
    """Expose real-time system and camera pipeline observability telemetry."""
    if not stream_manager:
        raise HTTPException(status_code=503, detail="Stream manager not initialized")
    return JSONResponse(stream_manager.get_telemetry())


@app.get("/api/evidence")
async def list_evidence(limit: int = 30):
    """
    List structured multi-crop evidence folders archived under EVENTS_DIR.
    Each entry includes metadata, URLs to full_frame.jpg, vehicle.jpg/object_crop.jpg, plate.jpg, and clip.mp4.
    """
    results = []
    try:
        if EVENTS_DIR.exists():
            for cam_dir in sorted(EVENTS_DIR.iterdir(), reverse=True):
                if not cam_dir.is_dir():
                    continue
                for date_dir in sorted(cam_dir.iterdir(), reverse=True):
                    if not date_dir.is_dir():
                        continue
                    for ev_dir in sorted(date_dir.iterdir(), reverse=True):
                        if not ev_dir.is_dir():
                            continue
                        rel_path = f"{cam_dir.name}/{date_dir.name}/{ev_dir.name}"
                        meta_file = ev_dir / "metadata.json"
                        meta = {}
                        if meta_file.exists():
                            try:
                                with open(meta_file, "r") as mf:
                                    meta = json.load(mf)
                            except Exception:
                                pass

                        entry = {
                            "event_dir": ev_dir.name,
                            "camera_id": cam_dir.name,
                            "date": date_dir.name,
                            "metadata": meta,
                            "full_frame_url": f"/api/events/{rel_path}/full_frame.jpg" if (ev_dir / "full_frame.jpg").exists() else None,
                            "vehicle_url": f"/api/events/{rel_path}/vehicle.jpg" if (ev_dir / "vehicle.jpg").exists() else None,
                            "object_crop_url": f"/api/events/{rel_path}/object_crop.jpg" if (ev_dir / "object_crop.jpg").exists() else None,
                            "plate_url": f"/api/events/{rel_path}/plate.jpg" if (ev_dir / "plate.jpg").exists() else None,
                            "clip_url": f"/api/events/{rel_path}/incident_clip.mp4" if (ev_dir / "incident_clip.mp4").exists() else None,
                        }
                        results.append(entry)
                        if len(results) >= limit:
                            break
                    if len(results) >= limit:
                        break
                if len(results) >= limit:
                    break
    except Exception as e:
        print(f"[IBVAP API] Evidence listing notice: {e}")
    return {"evidence": results}


# ═══════════════════════════════════════════════════════════════
# REST API — Analytics
# ═══════════════════════════════════════════════════════════════

@app.get("/api/analytics")
async def get_analytics():
    return {
        "alert_counts": db.get_alert_counts(),
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

