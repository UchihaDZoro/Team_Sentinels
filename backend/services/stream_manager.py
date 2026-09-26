"""
IBVAP Real-Time Stream Manager
Intelligent Border Video Analytics Platform
High-Performance Zero-Lag Real-Time Video Streaming & AI Security Analytics.

Key Real-Time Architectural Features (Inspired by OpenCV Threaded Streaming):
1. Zero-Lag Threaded Capture Worker: Continuously drains the demuxer / hardware socket,
   retaining ONLY the freshest frame and immediately dropping stale backlogs.
2. Direct-Frame Annotation: Every frame served on the MJPEG stream is rendered with
   vibrant bounding boxes (Green for humans, Yellow for vehicles, Vibrant Red for threats,
   hulls for crowds, and cyan for license plates).
3. Specialized Dangerous Object & Threat Detection: Firearms, knives, grenades, explosives,
   and in-flight projectiles detected and tracked with ballistic trajectories.
4. Megvii CrowdDetection: Dual-plane head/foot clustering, Soft-NMS, and mutual occlusion metrics.
5. Behavior Analysis (Hackfest2k25-kdf): Automatic loitering & fence zone tampering alerts.
"""
import os
import cv2
import json
import time
import threading
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure FFmpeg demuxer does not buffer network packets and starts at the broadcast live edge
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|probesize;32|analyzeduration;0|reorder_queue_size;0|stimeout;5000000"
)

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    TARGET_FPS,
    JPEG_QUALITY,
    WATCHDOG_CHECK_INTERVAL,
    WATCHDOG_STALL_TIMEOUT,
    MAX_RECONNECT_BACKOFF,
)
from services.detector import ObjectDetector
from services.virtual_fence import VirtualFence
from services.alert_engine import AlertEngine
from services.night_enhance import NightEnhancer
from services.stream_resolver import resolve_stream_source, is_network_stream, is_youtube_url
from services.crowd_detector import CrowdDetector
from services.anpr_engine import ANPREngine

# Attempt importing Hackfest BehaviorAnalyzer if available
try:
    import sys
    ref_hf = Path(__file__).parent.parent.parent / "ref_repos" / "Hackfest2k25-kdf"
    if ref_hf.exists() and str(ref_hf) not in sys.path:
        sys.path.insert(0, str(ref_hf))
    import behavior_analysis
    HAVE_BEHAVIOR_ANALYZER = True
except Exception:
    HAVE_BEHAVIOR_ANALYZER = False


class CameraStream:
    """State container for a single camera's real-time capture and AI pipeline."""

    def __init__(self, camera_id: str, source: str):
        self.camera_id = camera_id
        self.source = source
        self.resolved_source = source
        self.is_stream = False
        self.is_live = False
        self.stream_title = ""
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.night_mode = False

        # Status & watchdog tracking
        self.status: str = "stopped"  # "active", "stalled", "reconnecting", "stopped"
        self.is_reconnecting: bool = False
        self.reconnect_count: int = 0
        self.started_at: float = 0.0
        self.last_successful_frame_time: float = 0.0

        # Rolling historical frame buffer for pre/post event incident clip extraction (5s @ 30fps)
        self.rolling_buffer: deque = deque(maxlen=150)

        # Raw frame synchronization (Threaded Camera pattern: latest frame only)
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_raw_timestamp: float = 0.0
        self.raw_frame_lock = threading.Lock()
        self.new_raw_frame_event = threading.Event()
        self.raw_frame_id: int = 0

        # Fully annotated JPEG stream for instant browser delivery
        self.latest_jpeg: Optional[bytes] = None
        self.latest_annotated_jpeg: Optional[bytes] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.processed_frame_id: int = 0
        self.playback_frame_id: int = 0  # Backward-compatible alias
        self.latest_ai_metadata: dict = {}

        # Dedicated worker threads
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None

        # Behavior analysis module from ref_repos
        if HAVE_BEHAVIOR_ANALYZER:
            try:
                self.behavior_analyzer = behavior_analysis.BehaviorAnalyzer({
                    "behavior_analysis": {
                        "loitering_threshold_seconds": 10,
                        "fence_zones": [],
                        "fence_proximity_threshold": 15,
                    },
                    "alerting": {
                        "alert_cooldown_seconds": 25,
                    }
                })
            except Exception:
                self.behavior_analyzer = None
        else:
            self.behavior_analyzer = None

        # Telemetry & performance metrics
        self.source_fps: float = 0.0
        self.capture_fps: float = 0.0
        self.inference_fps: float = 0.0
        self.fps: float = 0.0
        self.tracking_fps: float = 0.0
        self.processing_latency_ms: float = 0.0
        self.ai_latency_ms: float = 0.0
        self.capture_delay_ms: float = 0.0
        self.end_to_end_latency_ms: float = 0.0
        self.stream_age_ms: float = 0.0
        self.live_edge_distance_sec: float = 0.0
        self.dropped_frames: int = 0
        self.total_captured_frames: int = 0
        self.total_processed_frames: int = 0

        # Real-time detection counts
        self.person_count: int = 0
        self.vehicle_count: int = 0
        self.threat_count: int = 0
        self.crowd_count: int = 0
        self.plate_count: int = 0
        self.intrusion_count: int = 0
        self.total_detections: int = 0


class StreamManager:
    """
    Manages all camera streams and the real-time AI security pipeline.
    Implements multi-threaded capture, zero-lag frame dropping, and instant
    annotated MJPEG video streaming without any 10-second buffer delays.
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
        self.crowd_detector = CrowdDetector()
        self.anpr_engine = ANPREngine(db=self.db)
        self.on_ai_metadata = None
        self._camera_op_lock = threading.Lock()

        self.cameras: dict[str, CameraStream] = {}

        # Stream Health Watchdog
        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="IBVAP-Watchdog"
        )
        self._watchdog_thread.start()

    def add_camera(self, camera_id: str, source: str) -> bool:
        """Register a camera source. Does not start processing."""
        if camera_id in self.cameras:
            self.stop_camera(camera_id)

        self.cameras[camera_id] = CameraStream(camera_id, source)
        return True

    def start_camera(self, camera_id: str) -> bool:
        """Open the video source and launch the real-time capture and processing threads."""
        with self._camera_op_lock:
            cam = self.cameras.get(camera_id)
            if not cam:
                cam_info = self.db.get_camera(camera_id) if hasattr(self.db, "get_camera") else None
                if cam_info and cam_info.get("source"):
                    self.add_camera(camera_id, cam_info["source"])
                    cam = self.cameras.get(camera_id)
                else:
                    return False

            if cam.running:
                return True

            source = cam.source

            # Folder auto-resolution to first playable video
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

            # Resolve stream (YouTube Live, HLS m3u8, RTSP, webcam)
            resolved_source, meta = resolve_stream_source(source)
            cam.resolved_source = resolved_source
            cam.is_stream = meta.get("type") in ("youtube", "web_stream", "network_direct")
            cam.is_live = meta.get("is_live", False)
            cam.stream_title = meta.get("title", "")

            try:
                source_val = int(resolved_source)
            except ValueError:
                source_val = resolved_source

            print(f"[IBVAP] Opening real-time capture for '{camera_id}' ({meta.get('type')}, live={cam.is_live})...")
            cap = cv2.VideoCapture(source_val, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source_val)

            if not cap.isOpened():
                print(f"[IBVAP] ✗ Cannot open source: {source} (resolved: {str(source_val)[:80]})")
                return False

            # Set hardware buffer to 2 frames to avoid decoder lag
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except Exception:
                pass

            cam.cap = cap
            cam.running = True
            cam.status = "active"
            cam.started_at = time.time()
            cam.last_successful_frame_time = time.time()
            cam.is_reconnecting = False
            cam.reconnect_count = 0

            # Load night mode and fence zones from DB
            cam_info = self.db.get_camera(camera_id)
            if cam_info:
                cam.night_mode = bool(cam_info.get("night_mode", 0))
                zones_json = cam_info.get("fence_zones", "[]")
                zones = json.loads(zones_json) if isinstance(zones_json, str) else zones_json
                if zones:
                    self.fence.set_zones(camera_id, zones)
                    if cam.behavior_analyzer and hasattr(cam.behavior_analyzer, "fence_zones_poly"):
                        cam.behavior_analyzer.fence_zones_poly = [np.array(z, np.int32) for z in zones]

            # Launch Dedicated Workers (Capture + AI Processing)
            cam.capture_thread = threading.Thread(
                target=self._capture_loop, args=(camera_id,), daemon=True, name=f"Capture-{camera_id}"
            )
            cam.process_thread = threading.Thread(
                target=self._process_loop, args=(camera_id,), daemon=True, name=f"Process-{camera_id}"
            )
            cam.capture_thread.start()
            cam.process_thread.start()

            self.db.update_camera_status(camera_id, "active")
            print(f"[IBVAP] ✓ Camera '{camera_id}' active with real-time zero-lag pipeline (live={cam.is_live})")
            return True

    def stop_camera(self, camera_id: str):
        """Stop processing and release the video source cleanly."""
        with self._camera_op_lock:
            cam = self.cameras.get(camera_id)
            if not cam or not cam.running:
                return

            cam.running = False
            cam.status = "stopped"
            cam.new_raw_frame_event.set()

            if cam.capture_thread and cam.capture_thread.is_alive():
                cam.capture_thread.join(timeout=1.5)
            if cam.process_thread and cam.process_thread.is_alive():
                cam.process_thread.join(timeout=1.5)

            if cam.cap:
                try:
                    cam.cap.release()
                except Exception:
                    pass
                cam.cap = None

            self.db.update_camera_status(camera_id, "inactive")
            print(f"[IBVAP] ■ Camera '{camera_id}' stopped")

    def remove_camera(self, camera_id: str):
        """Stop and remove a camera completely."""
        self.stop_camera(camera_id)
        self.cameras.pop(camera_id, None)

    def stop_all(self):
        """Stop all active camera streams and the watchdog thread."""
        self._watchdog_running = False
        for cid in list(self.cameras.keys()):
            self.stop_camera(cid)

    def save_evidence_clip(
        self,
        camera_id: str,
        output_path: str,
        pre_seconds: float = 3.0,
        post_seconds: float = 5.0,
    ) -> bool:
        """Asynchronously writes a forensic MP4 clip of detected incident."""
        cam = self.cameras.get(camera_id)
        if not cam:
            return False

        def _record_worker():
            try:
                now = time.time()
                pre_cutoff = now - pre_seconds
                with cam.raw_frame_lock:
                    pre_frames = [f for t, f in list(cam.rolling_buffer) if t >= pre_cutoff]

                post_frames = []
                t_end = time.time() + post_seconds
                while time.time() < t_end and cam.running:
                    time.sleep(0.05)
                    if cam.latest_raw_frame is not None:
                        rh, rw = cam.latest_raw_frame.shape[:2]
                        tw, th = (640, int(rh * 640 / max(rw, 1))) if rw > 640 else (rw, rh)
                        small = cv2.resize(cam.latest_raw_frame, (tw, th), interpolation=cv2.INTER_AREA)
                        post_frames.append(small)

                all_frames = pre_frames + post_frames
                if not all_frames:
                    return

                h, w = all_frames[0].shape[:2]
                out_p = Path(output_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_p), fourcc, 15.0, (w, h))
                for f in all_frames:
                    if f.shape[:2] != (h, w):
                        f = cv2.resize(f, (w, h))
                    writer.write(f)
                writer.release()
                print(f"[IBVAP Evidence] ✓ Forensic clip saved ({len(all_frames)} frames) → {output_path}")
            except Exception as e:
                print(f"[IBVAP Evidence] ⚠ Failed to write clip: {e}")

        threading.Thread(target=_record_worker, daemon=True, name=f"Clip-{camera_id}").start()
        return True

    def get_stats(self) -> dict:
        """Aggregate detection statistics and telemetry across all cameras."""
        stats = {
            "total_persons": 0,
            "total_vehicles": 0,
            "total_threats": 0,
            "total_crowds": 0,
            "total_plates": 0,
            "total_intrusions": 0,
            "cameras": {},
        }
        for cid, cam in self.cameras.items():
            stats["total_persons"] += cam.person_count
            stats["total_vehicles"] += cam.vehicle_count
            stats["total_threats"] += cam.threat_count
            stats["total_crowds"] += cam.crowd_count
            stats["total_plates"] += cam.plate_count
            stats["total_intrusions"] += cam.intrusion_count
            stats["cameras"][cid] = {
                "fps": round(cam.fps, 1),
                "source_fps": round(cam.source_fps, 1),
                "capture_fps": round(cam.capture_fps, 1),
                "latency_ms": round(cam.processing_latency_ms, 1),
                "status": cam.status,
                "persons": cam.person_count,
                "vehicles": cam.vehicle_count,
                "threats": cam.threat_count,
                "crowds": cam.crowd_count,
                "plates": cam.plate_count,
                "intrusions": cam.intrusion_count,
                "running": cam.running,
            }
        return stats

    def get_telemetry(self) -> dict:
        """Comprehensive system and camera pipeline observability telemetry."""
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        telemetry = {
            "timestamp": time.time(),
            "system": {
                "cpu_usage_pct": round(cpu_pct, 1),
                "ram_usage_pct": round(ram.percent, 1),
                "ram_used_mb": round(ram.used / (1024 * 1024), 1),
                "ram_total_mb": round(ram.total / (1024 * 1024), 1),
                "cuda_available": False,
            },
            "cameras": {}
        }
        now = time.time()
        for cid, cam in self.cameras.items():
            idle_sec = round(now - cam.last_successful_frame_time, 2) if cam.last_successful_frame_time > 0 else 999.0
            health = "offline"
            if cam.running:
                if cam.is_reconnecting:
                    health = "reconnecting"
                elif idle_sec > WATCHDOG_STALL_TIMEOUT:
                    health = "stalled"
                else:
                    health = "healthy"

            telemetry["cameras"][cid] = {
                "camera_id": cid,
                "status": cam.status,
                "health": health,
                "fps": round(cam.fps, 1),
                "source_fps": round(cam.source_fps, 1),
                "capture_fps": round(cam.capture_fps, 1),
                "latency_ms": round(cam.processing_latency_ms, 1),
                "total_captured": cam.total_captured_frames,
                "total_processed": cam.total_processed_frames,
                "idle_seconds": idle_sec,
                "is_live": cam.is_live,
            }
        return telemetry

    # ── Real-Time MJPEG Generator ─────────────────────────────────
    def generate_mjpeg(self, camera_id: str, annotated: bool = True):
        """
        Ultra-low-latency MJPEG frame generator.
        Streams the freshly annotated frame directly to the browser with zero buffer lag.
        """
        cam = self.cameras.get(camera_id)
        if not cam:
            return

        last_sent_id = -1
        while cam and cam.running:
            jpeg_data = None
            with cam.frame_lock:
                if cam.processed_frame_id != last_sent_id:
                    jpeg_data = cam.latest_jpeg
                    last_sent_id = cam.processed_frame_id

            if jpeg_data is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg_data)).encode() + b"\r\n\r\n"
                    + jpeg_data
                    + b"\r\n"
                )
            time.sleep(0.005)

    def _get_connecting_placeholder(self) -> np.ndarray:
        """Visual placeholder displayed while connecting."""
        placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(placeholder, (20, 20), (FRAME_WIDTH - 20, FRAME_HEIGHT - 20), (20, 35, 30), 1)
        cv2.putText(placeholder, "CONNECTING REAL-TIME SURVEILLANCE FEED...", (FRAME_WIDTH // 2 - 240, FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 136), 2, cv2.LINE_AA)
        return placeholder

    # ── Stream Watchdog & Auto-Recovery ───────────────────────────
    def _watchdog_loop(self):
        """Audits all active streams and recovers stalled connections."""
        while getattr(self, "_watchdog_running", True):
            try:
                now = time.time()
                for cid, cam in list(self.cameras.items()):
                    if not cam.running or not cam.is_live or cam.is_reconnecting:
                        continue

                    idle_time = now - cam.last_successful_frame_time
                    if idle_time > WATCHDOG_STALL_TIMEOUT:
                        cam.status = "stalled"
                        print(f"[IBVAP Watchdog] Camera '{cid}' stalled ({idle_time:.1f}s idle). Auto-recovering...")
                        threading.Thread(target=self._recover_camera, args=(cid,), daemon=True).start()
            except Exception as e:
                print(f"[IBVAP Watchdog] Error in watchdog loop: {e}")
            time.sleep(WATCHDOG_CHECK_INTERVAL)

    def _recover_camera(self, camera_id: str):
        """Asynchronously recovers a stalled camera stream."""
        cam = self.cameras.get(camera_id)
        if not cam or not cam.running or cam.is_reconnecting:
            return

        cam.is_reconnecting = True
        cam.status = "reconnecting"
        cam.reconnect_count += 1
        backoff = min(MAX_RECONNECT_BACKOFF, 1.5 ** min(cam.reconnect_count, 6))
        print(f"[IBVAP Watchdog] Reconnecting '{camera_id}' (attempt #{cam.reconnect_count}, backoff={backoff:.1f}s)...")
        time.sleep(backoff)

        try:
            if cam.cap:
                try:
                    cam.cap.release()
                except Exception:
                    pass
                cam.cap = None

            resolved_source, meta = resolve_stream_source(cam.source)
            cam.resolved_source = resolved_source
            cam.is_live = meta.get("is_live", False)

            try:
                source_val = int(resolved_source)
            except ValueError:
                source_val = resolved_source

            cap = cv2.VideoCapture(source_val, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source_val)

            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                except Exception:
                    pass
                cam.cap = cap
                cam.last_successful_frame_time = time.time()
                cam.status = "active"
                print(f"[IBVAP Watchdog] ✓ Reconnection successful for '{camera_id}'")
            else:
                cam.status = "stalled"
        except Exception as e:
            print(f"[IBVAP Watchdog] Reconnection failed: {e}")
            cam.status = "stalled"
        finally:
            cam.is_reconnecting = False

    # ── Worker 1: Real-Time Capture Worker (Drops Stale Frames) ───
    def _capture_loop(self, camera_id: str):
        """
        Threaded camera capture loop (from OpenCV real-time streaming pattern).
        Drains decoder frames and keeps ONLY the freshest frame in memory.
        Guarantees zero accumulated lag over any duration.
        """
        cam = self.cameras.get(camera_id)
        if not cam:
            return

        fps = 30.0
        if cam.cap:
            try:
                cap_fps = cam.cap.get(cv2.CAP_PROP_FPS)
                if cap_fps and 1.0 < cap_fps < 120.0:
                    fps = cap_fps
            except Exception:
                pass

        frame_interval = 1.0 / max(fps, 10.0)
        cam.source_fps = fps

        while cam.running:
            if cam.cap is None or not cam.cap.isOpened():
                time.sleep(0.05)
                continue

            now = time.time()

            if cam.is_live:
                # Live stream (IP camera / RTSP / YouTube Live / Webcam)
                ret, frame = cam.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                cam.last_successful_frame_time = now
                cam.total_captured_frames += 1

                # Overwrite latest raw frame (zero lag - drops backlog)
                with cam.raw_frame_lock:
                    cam.latest_raw_frame = frame
                    cam.latest_raw_timestamp = now
                    cam.raw_frame_id += 1
                cam.new_raw_frame_event.set()

                # Save frame in rolling buffer for forensic evidence clips
                try:
                    rh, rw = frame.shape[:2]
                    tw, th = (640, int(rh * 640 / max(rw, 1))) if rw > 640 else (rw, rh)
                    small_frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
                    cam.rolling_buffer.append((now, small_frame))
                except Exception:
                    pass
            else:
                # Local recorded video file
                ret, frame = cam.cap.read()
                if not ret or frame is None:
                    # Loop video continuously
                    cam.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cam.cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue

                cam.last_successful_frame_time = now
                cam.total_captured_frames += 1

                with cam.raw_frame_lock:
                    cam.latest_raw_frame = frame
                    cam.latest_raw_timestamp = now
                    cam.raw_frame_id += 1
                cam.new_raw_frame_event.set()

                # Pace at natural presentation rate
                time.sleep(frame_interval)

    # ── Worker 2: Real-Time AI Analytics & Annotation Worker ──────
    def _process_loop(self, camera_id: str):
        """
        Picks up the freshest raw frame, runs full multi-modal security analytics,
        burns bounding boxes & HUD directly onto the frame, and generates the JPEG.
        """
        cam = self.cameras.get(camera_id)
        if not cam:
            return

        t_last = time.time()

        while cam.running:
            try:
                # Wait for fresh frame from capture thread
                if not cam.new_raw_frame_event.wait(timeout=0.08):
                    continue
                cam.new_raw_frame_event.clear()

                with cam.raw_frame_lock:
                    if cam.latest_raw_frame is None:
                        continue
                    raw_frame = cam.latest_raw_frame.copy()
                    source_timestamp = cam.latest_raw_timestamp

                t_start = time.time()
                frame = cv2.resize(raw_frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # 1. Night Enhancement
                if cam.night_mode or self.night_enhancer.is_dark_frame(frame):
                    frame = self.night_enhancer.enhance(frame)

                # 2. Object Detection & Dangerous Threat Tracking
                detections = self.detector.detect_and_track(frame, timestamp=source_timestamp)

                # 3. Megvii CrowdDetection (Soft-NMS & CrowdHuman dual-plane clustering)
                crowds = self.crowd_detector.detect_crowds(detections, camera_id=camera_id, raw_frame=raw_frame)

                # 4. ANPR (License Plates & OCR)
                plates = self.anpr_engine.detect_plates(frame, detections, camera_id)

                # 5. Virtual Fence Intrusions
                intrusions = self.fence.check_intrusions(camera_id, detections)

                # 6. Behavior Analysis from ref_repos/Hackfest2k25-kdf (Loitering & Fence zones)
                if hasattr(cam, "behavior_analyzer") and cam.behavior_analyzer:
                    try:
                        person_tracks = [
                            [d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d.get("track_id", -1), d.get("confidence", 0.5), 0]
                            for d in detections if d.get("category") == "person"
                        ]
                        b_alerts = cam.behavior_analyzer.update(person_tracks)
                        for ba in b_alerts:
                            if ba.get("type") == "loitering":
                                self.alert_engine.process_threats(camera_id, [{
                                    "bbox": [ba["position"][0]-25, ba["position"][1]-50, ba["position"][0]+25, ba["position"][1]],
                                    "class": "loitering",
                                    "confidence": 0.90,
                                    "category": "threat",
                                    "threat_label": f"SUSPICIOUS LOITERING ({int(ba.get('duration', 10))}s)",
                                    "severity": "medium",
                                }], frame)
                    except Exception:
                        pass

                # 7. Alert Engine Processing
                if intrusions:
                    self.alert_engine.process_intrusions(camera_id, intrusions, frame)
                self.alert_engine.process_threats(camera_id, detections, frame)
                self.alert_engine.process_crowds(camera_id, crowds, frame)
                self.alert_engine.process_plates(camera_id, plates, frame)

                # 8. RENDER BOUNDING BOXES & ANNOTATIONS DIRECTLY ON FRAME
                fence_zones = self.fence.get_zone_polygons_for_drawing(camera_id)
                annotated = ObjectDetector.annotate_frame(
                    frame, detections, intrusions, fence_zones, crowds=crowds, plates=plates,
                    camera_id=camera_id, anpr_engine=self.anpr_engine
                )

                # 9. Encode to JPEG once with optimal quality
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                jpeg_bytes = buf.tobytes() if ok else None

                t_end = time.time()
                latency_ms = (t_end - t_start) * 1000.0

                # 10. Update Camera State
                with cam.frame_lock:
                    cam.latest_frame = annotated
                    cam.latest_jpeg = jpeg_bytes
                    cam.latest_annotated_jpeg = jpeg_bytes
                    cam.processed_frame_id += 1
                    cam.total_processed_frames += 1
                    cam.playback_frame_id = cam.processed_frame_id
                    cam.processing_latency_ms = round(latency_ms, 1)
                    cam.person_count = sum(1 for d in detections if d["category"] == "person")
                    cam.vehicle_count = sum(1 for d in detections if d["category"] == "vehicle")
                    cam.threat_count = sum(1 for d in detections if d.get("category") == "threat" or d.get("is_threat", False))
                    cam.crowd_count = len(crowds)
                    cam.plate_count = len(plates)
                    cam.intrusion_count = len(intrusions)
                    cam.total_detections = len(detections)

                # 11. Broadcast metadata via WebSocket
                if callable(self.on_ai_metadata):
                    try:
                        self.on_ai_metadata({
                            "camera_id": camera_id,
                            "timestamp": round(source_timestamp, 3),
                            "latency_ms": round(latency_ms, 1),
                            "detections": detections,
                            "crowds": crowds,
                            "plates": plates,
                            "intrusions": intrusions,
                            "threat_count": cam.threat_count,
                            "person_count": cam.person_count,
                            "vehicle_count": cam.vehicle_count,
                        })
                    except Exception:
                        pass

                # FPS calculation
                dt = max(t_end - t_last, 0.001)
                t_last = t_end
                cam.fps = 0.8 * cam.fps + 0.2 * (1.0 / dt)
            except Exception as e:
                print(f"[IBVAP AI Error] Error in AI loop for {camera_id}: {e}")
                time.sleep(0.02)

    def get_latest_jpeg(self, camera_id: str, annotated: bool = True) -> tuple[Optional[bytes], float, int]:
        """Fetch the latest annotated JPEG frame."""
        cam = self.cameras.get(camera_id)
        if not cam:
            return None, 0.0, 0
        with cam.frame_lock:
            return cam.latest_jpeg, cam.processing_latency_ms, cam.processed_frame_id
