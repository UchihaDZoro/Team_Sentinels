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
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DATABASE_PATH, FRONTEND_DIR, SNAPSHOTS_DIR, DEMO_VIDEOS_DIR
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

# Serve snapshots
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")



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


async def broadcast_stats():
    """Periodically push stats to all WebSocket clients."""
    while True:
        if ws_clients and stream_manager:
            stats = stream_manager.get_stats()
            message = json.dumps({"type": "stats", "data": stats})
            dead = set()
            for ws in ws_clients:
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
    cam_name = req.name.strip() if req.name and req.name.strip() else Path(req.source).stem.replace("_", " ").title() or "Surveillance Feed"
    cam_id = f"cam_{uuid.uuid4().hex[:8]}"
    cam = db.add_camera(cam_id, cam_name, req.source, req.location or "Border Checkpost")
    stream_manager.add_camera(cam_id, req.source)
    if req.auto_start:
        ok = stream_manager.start_camera(cam_id)
        if not ok:
            return JSONResponse(
                {"error": f"Cannot open video source: {req.source}", "camera": cam},
                status_code=400,
            )
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
async def camera_stream(camera_id: str):
    """MJPEG streaming endpoint for a camera."""
    cam = stream_manager.cameras.get(camera_id)
    if not cam or not cam.running:
        raise HTTPException(404, "Camera not active")

    return StreamingResponse(
        stream_manager.generate_mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
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

