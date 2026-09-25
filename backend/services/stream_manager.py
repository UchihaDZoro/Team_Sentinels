"""
IBVAP Stream Manager
Manages video capture from multiple sources and runs the AI processing pipeline.
Supports: RTSP streams, video files (looped), webcams.
"""
import cv2
import json
import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional

from config import FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS
from services.detector import ObjectDetector
from services.virtual_fence import VirtualFence
from services.alert_engine import AlertEngine
from services.night_enhance import NightEnhancer
from services.stream_resolver import resolve_stream_source, is_network_stream, is_youtube_url


class CameraStream:
    """State container for a single camera's processing loop."""

    def __init__(self, camera_id: str, source: str):
        self.camera_id = camera_id
        self.source = source
        self.resolved_source = source
        self.is_stream = False
        self.is_live = False
        self.stream_title = ""
        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.night_mode = False

        # Latest processed frame (ready for MJPEG streaming)
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # Per-frame stats
        self.fps: float = 0.0
        self.person_count: int = 0
        self.vehicle_count: int = 0
        self.intrusion_count: int = 0
        self.total_detections: int = 0


class StreamManager:
    """
    Manages all camera streams and the AI processing pipeline.
    Each camera runs in its own background thread.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        fence: VirtualFence,
        alert_engine: AlertEngine,
        night_enhancer: NightEnhancer,
        db,
    ):
        self.detector = detector
        self.fence = fence
        self.alert_engine = alert_engine
        self.night_enhancer = night_enhancer
        self.db = db

        self.cameras: dict[str, CameraStream] = {}
        self._global_stats = {
            "total_persons": 0,
            "total_vehicles": 0,
            "total_intrusions": 0,
        }

    def add_camera(self, camera_id: str, source: str) -> bool:
        """Register a camera source. Does not start processing."""
        if camera_id in self.cameras:
            self.stop_camera(camera_id)

        self.cameras[camera_id] = CameraStream(camera_id, source)
        return True

    def start_camera(self, camera_id: str) -> bool:
        """Open the video source and start the processing thread."""
        cam = self.cameras.get(camera_id)
        if not cam:
            return False

        # Try to open source
        source = cam.source

        # If source is a directory or path without file, auto-resolve to first video inside
        try:
            p = Path(source)
            if p.exists() and p.is_dir():
                vids = []
                for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
                    vids.extend(p.glob(ext))
                if vids:
                    source = str(vids[0])
                    cam.source = source
                    if hasattr(self.db, "update_camera_source"):
                        self.db.update_camera_source(camera_id, source)
                    print(f"[IBVAP] Auto-resolved folder to video file: {source}")
        except Exception:
            pass

        # Resolve online stream (YouTube, HLS m3u8, RTSP, etc.)
        resolved_source, meta = resolve_stream_source(source)
        cam.resolved_source = resolved_source
        cam.is_stream = meta.get("type") in ("youtube", "web_stream", "network_direct")
        cam.is_live = meta.get("is_live", False)
        cam.stream_title = meta.get("title", "")

        # Try parsing as integer (webcam index)
        try:
            source_val = int(resolved_source)
        except ValueError:
            source_val = resolved_source

        print(f"[IBVAP] Opening video capture for '{camera_id}' ({meta.get('type')}) -> {str(source_val)[:80]}...")
        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            print(f"[IBVAP] ✗ Cannot open source: {source} (resolved: {str(source_val)[:80]})")
            return False

        cam.cap = cap
        cam.running = True

        # Load night mode from DB
        cam_info = self.db.get_camera(camera_id)
        if cam_info:
            cam.night_mode = bool(cam_info.get("night_mode", 0))
            # Load fence zones
            zones_json = cam_info.get("fence_zones", "[]")
            zones = json.loads(zones_json) if isinstance(zones_json, str) else zones_json
            if zones:
                self.fence.set_zones(camera_id, zones)

        # Start processing thread
        cam.thread = threading.Thread(
            target=self._process_loop, args=(camera_id,), daemon=True
        )
        cam.thread.start()
        self.db.update_camera_status(camera_id, "active")
        print(f"[IBVAP] ✓ Camera '{camera_id}' started — source: {source} (live={cam.is_live})")
        return True

    def stop_camera(self, camera_id: str):
        """Stop processing and release the video source."""
        cam = self.cameras.get(camera_id)
        if not cam:
            return
        cam.running = False
        if cam.thread and cam.thread.is_alive():
            cam.thread.join(timeout=3)
        if cam.cap:
            cam.cap.release()
            cam.cap = None
        self.db.update_camera_status(camera_id, "inactive")
        print(f"[IBVAP] ■ Camera '{camera_id}' stopped")

    def remove_camera(self, camera_id: str):
        """Stop and remove a camera completely."""
        self.stop_camera(camera_id)
        self.cameras.pop(camera_id, None)

    def stop_all(self):
        for cid in list(self.cameras.keys()):
            self.stop_camera(cid)

    def get_stats(self) -> dict:
        """Aggregate stats across all cameras."""
        stats = {
            "total_persons": 0,
            "total_vehicles": 0,
            "total_intrusions": 0,
            "cameras": {},
        }
        for cid, cam in self.cameras.items():
            stats["total_persons"] += cam.person_count
            stats["total_vehicles"] += cam.vehicle_count
            stats["total_intrusions"] += cam.intrusion_count
            stats["cameras"][cid] = {
                "fps": round(cam.fps, 1),
                "persons": cam.person_count,
                "vehicles": cam.vehicle_count,
                "intrusions": cam.intrusion_count,
                "running": cam.running,
            }
        return stats

    # ── MJPEG frame generator ───────────────────────────────────
    def generate_mjpeg(self, camera_id: str):
        """Yields MJPEG frames for StreamingResponse."""
        cam = self.cameras.get(camera_id)
        if not cam:
            return

        wait_count = 0
        while cam and cam.running:
            with cam.frame_lock:
                frame = cam.latest_frame
            if frame is None:
                # Send immediate buffer frame so browser HTTP request completes and renders
                wait_count += 1
                if wait_count % 5 == 1:
                    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder,
                        "CONNECTING LIVE STREAM...",
                        (FRAME_WIDTH // 2 - 160, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 136),
                        2,
                    )
                    cv2.putText(
                        placeholder,
                        "Initializing YOLO26 Analytics Engine...",
                        (FRAME_WIDTH // 2 - 180, FRAME_HEIGHT // 2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (150, 180, 200),
                        1,
                    )
                    ok, buffer = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        data = buffer.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                            + data
                            + b"\r\n"
                        )
                time.sleep(0.1)
                continue

            ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if ok:
                data = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                    + data
                    + b"\r\n"
                )
            time.sleep(1 / TARGET_FPS)

    # ── Main processing loop (runs in a thread per camera) ──────
    def _process_loop(self, camera_id: str):
        cam = self.cameras[camera_id]
        frame_interval = 1.0 / TARGET_FPS
        consecutive_read_fails = 0

        while cam.running:
            loop_start = time.time()

            if cam.cap is None or not cam.cap.isOpened():
                time.sleep(0.5)
                continue

            ret, raw_frame = cam.cap.read()
            if not ret or raw_frame is None:
                consecutive_read_fails += 1
                if cam.is_stream:
                    # Brief pause for network stream jitter
                    time.sleep(0.1)
                    if consecutive_read_fails > 25:  # ~2.5s of failed reads
                        print(f"[IBVAP] Stream read stalled for '{camera_id}', attempting reconnect...")
                        try:
                            new_url, _ = resolve_stream_source(cam.source)
                            cam.resolved_source = new_url
                            if cam.cap:
                                cam.cap.release()
                            cam.cap = cv2.VideoCapture(new_url)
                            consecutive_read_fails = 0
                            print(f"[IBVAP] Stream '{camera_id}' reconnected successfully.")
                        except Exception as e:
                            print(f"[IBVAP] Reconnect failed for '{camera_id}': {e}")
                            time.sleep(1.0)
                    continue
                else:
                    # Loop video files
                    cam.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_read_fails = 0
                    time.sleep(0.01)
                    continue

            consecutive_read_fails = 0

            # Resize for consistent processing
            frame = cv2.resize(raw_frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # ① Night-time enhancement
            if cam.night_mode or self.night_enhancer.is_dark_frame(frame):
                frame = self.night_enhancer.enhance(frame)

            # ② Object detection + tracking
            detections = self.detector.detect_and_track(frame)

            # ③ Virtual fence intrusion check
            intrusions = self.fence.check_intrusions(camera_id, detections)

            # ④ Generate alerts
            if intrusions:
                self.alert_engine.process_intrusions(camera_id, intrusions, frame)
            self.alert_engine.process_detections(camera_id, detections, frame)

            # ⑤ Annotate frame
            fence_zones = self.fence.get_zone_polygons_for_drawing(camera_id)
            annotated = ObjectDetector.annotate_frame(
                frame, detections, intrusions, fence_zones
            )

            # Update stats
            cam.person_count = sum(1 for d in detections if d["category"] == "person")
            cam.vehicle_count = sum(1 for d in detections if d["category"] == "vehicle")
            cam.intrusion_count = len(intrusions)
            cam.total_detections = len(detections)

            # Store latest frame
            with cam.frame_lock:
                cam.latest_frame = annotated

            # FPS calculation
            elapsed = time.time() - loop_start
            cam.fps = 1.0 / max(elapsed, 0.001)

            # Throttle to target FPS
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
