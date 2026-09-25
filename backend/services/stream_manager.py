"""
IBVAP Stream Manager
Intelligent Border Video Analytics Platform — Sashastra Seema Bal (SSB)

Manages video capture from multiple sources and runs the AI processing pipeline:
1. Smart Cadence & Adaptive Frame Skipping:
   - Runs full YOLO detection every N-th frame (detection_interval, e.g. 2 or 3)
   - Reuses tracked bounding boxes via ByteTrack Kalman state estimation on intermediate frames
   - Reduces inference CPU load by 50-66%
2. Multi-Mode Tactical Night Vision:
   - Modes: "off", "clahe", "thermal", "nvg"
   - Supports live mode switching per camera
3. Motion-Guided False Alarm Suppression:
   - Filters out insect flashes, rain streaks, and wind-blown foliage
   - Only triggers alerts for objects exhibiting steady directional displacement across 3+ frames
4. High-Performance Zero-Lag MJPEG Streaming:
   - Worker thread pre-encodes JPEG once per frame
   - Consumer generator delivers frames instantly on sequence change, maintaining rock-solid 20-25 FPS
"""
import cv2
import json
import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    TARGET_FPS,
    JPEG_QUALITY,
    DETECTION_INTERVAL,
    NIGHT_VISION_MODES,
    DEFAULT_NIGHT_MODE,
)
from services.detector import ObjectDetector
from services.virtual_fence import VirtualFence
from services.alert_engine import AlertEngine
from services.night_enhance import NightEnhancer, MotionFalseAlarmFilter
from services.behavior_engine import BehaviorEngine
from services.frs_engine import FRSEngine
from services.anpr_engine import ANPREngine
from services.stream_resolver import resolve_stream_source, is_network_stream, is_youtube_url


# ═════════════════════════════════════════════════════════════════════
# ByteTrack Kalman State Tracker for Intermediate Frames
# ═════════════════════════════════════════════════════════════════════
class KalmanBoxTracker:
    """
    Maintains 8-state constant velocity Kalman filter per track ID:
    State:       [cx, cy, w, h, vx, vy, vw, vh]^T
    Measurement: [cx, cy, w, h]^T
    Used on skipped intermediate frames to project smooth bounding box coordinates
    with sub-millisecond execution time, reusing ByteTrack state.
    """

    def __init__(self, track_id: int, bbox: List[float], meta: Dict):
        self.track_id = track_id
        self.meta = dict(meta)
        self.time_since_update = 0

        # cv2.KalmanFilter(dynamParams=8, measureParams=4)
        self.kf = cv2.KalmanFilter(8, 4)

        # Transition matrix F (constant velocity model)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        # Measurement matrix H
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        # Process noise covariance Q
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        for i in range(4):
            self.kf.processNoiseCov[i + 4, i + 4] = 1.0

        # Measurement noise covariance R
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0

        # Error covariance P: large initial uncertainty on velocity for rapid lock-on
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        for i in range(4):
            self.kf.errorCovPost[i + 4, i + 4] = 1000.0

        # Initialize state with initial bounding box measurement
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        init_state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(-1, 1)
        self.kf.statePost = init_state.copy()
        self.kf.statePre = init_state.copy()

    def update(self, bbox: List[float], meta: Dict):
        """Update Kalman filter with ground-truth YOLO measurement."""
        self.time_since_update = 0
        self.meta.update(meta)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        # Standard Kalman cycle: predict prior before correcting with measurement
        self.kf.predict()
        measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(-1, 1)
        self.kf.correct(measurement)


    def predict(self) -> Dict:
        """Project bounding box state forward without running neural inference."""
        self.time_since_update += 1
        state = self.kf.predict()
        cx = float(state[0, 0])
        cy = float(state[1, 0])
        w = max(4.0, float(state[2, 0]))
        h = max(4.0, float(state[3, 0]))

        x1 = max(0.0, cx - w * 0.5)
        y1 = max(0.0, cy - h * 0.5)
        x2 = min(float(FRAME_WIDTH), cx + w * 0.5)
        y2 = min(float(FRAME_HEIGHT), cy + h * 0.5)

        det = dict(self.meta)
        det["bbox"] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        det["track_id"] = self.track_id
        det["centroid"] = [round(cx, 1), round(cy, 1)]
        return det


class CameraTrackPredictor:
    """Manages active KalmanBoxTracker instances for a camera stream."""

    def __init__(self, max_age: int = 8):
        self.trackers: Dict[int, KalmanBoxTracker] = {}
        self.max_age = max_age

    def update_from_yolo(self, detections: List[Dict]):
        """Synchronize active tracks with new YOLO detections."""
        seen_ids = set()
        for det in detections:
            tid = det.get("track_id", -1)
            if tid < 0:
                continue
            seen_ids.add(tid)
            if tid in self.trackers:
                self.trackers[tid].update(det["bbox"], det)
            else:
                self.trackers[tid] = KalmanBoxTracker(tid, det["bbox"], det)

        # Remove dead tracks
        dead_ids = [
            tid
            for tid, trk in self.trackers.items()
            if trk.time_since_update > self.max_age and tid not in seen_ids
        ]
        for tid in dead_ids:
            del self.trackers[tid]

    def predict_intermediate(self) -> List[Dict]:
        """Generate predicted detections for all active tracks on intermediate frames."""
        predicted = []
        for trk in self.trackers.values():
            predicted.append(trk.predict())
        return predicted


# ═════════════════════════════════════════════════════════════════════
# Camera Stream Container
# ═════════════════════════════════════════════════════════════════════
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

        # Night vision mode: "off", "clahe", "thermal", "nvg"
        self.night_mode: str = DEFAULT_NIGHT_MODE

        # Smart cadence & Kalman tracking state
        self.detection_interval: int = DETECTION_INTERVAL
        self.frame_count: int = 0
        self.track_predictor = CameraTrackPredictor(max_age=8)

        # Frame sequence & pre-encoded JPEG for zero-lag streaming
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_jpeg: Optional[bytes] = None
        self.frame_seq: int = 0
        self.frame_lock = threading.Lock()

        # Per-frame operational stats
        self.fps: float = 0.0
        self.person_count: int = 0
        self.vehicle_count: int = 0
        self.intrusion_count: int = 0
        self.crawling_count: int = 0
        self.loitering_count: int = 0
        self.unattended_count: int = 0
        self.total_detections: int = 0


# ═════════════════════════════════════════════════════════════════════
# Stream Manager
# ═════════════════════════════════════════════════════════════════════
class StreamManager:
    """
    Manages all camera streams, smart inference cadence, night vision enhancement,
    and motion-guided false alarm suppression.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        fence: VirtualFence,
        alert_engine: AlertEngine,
        night_enhancer: NightEnhancer,
        db,
        behavior_engine: Optional[BehaviorEngine] = None,
        frs_engine: Optional[FRSEngine] = None,
        anpr_engine: Optional[ANPREngine] = None,
    ):
        self.detector = detector
        self.fence = fence
        self.alert_engine = alert_engine
        self.night_enhancer = night_enhancer
        self.db = db
        self.behavior_engine = behavior_engine or BehaviorEngine()
        self.frs_engine = frs_engine or FRSEngine(db)
        self.anpr_engine = anpr_engine or ANPREngine(db)

        # Motion-Guided False Alarm Suppression Engine
        self.motion_filter = MotionFalseAlarmFilter()

        self.cameras: Dict[str, CameraStream] = {}

    def add_camera(self, camera_id: str, source: str) -> bool:
        """Register a camera source. Does not start processing."""
        if camera_id in self.cameras:
            self.stop_camera(camera_id)

        self.cameras[camera_id] = CameraStream(camera_id, source)
        return True

    def set_night_mode(self, camera_id: str, mode: str) -> bool:
        """Update night vision mode for a camera ('off', 'clahe', 'thermal', 'nvg')."""
        mode_clean = str(mode).lower().strip()
        if mode_clean not in NIGHT_VISION_MODES:
            return False

        cam = self.cameras.get(camera_id)
        if cam:
            cam.night_mode = mode_clean

        if self.db:
            self.db.update_night_mode(camera_id, mode_clean)

        return True

    def set_detection_interval(self, camera_id: str, interval: int) -> bool:
        """Tune smart cadence interval (run YOLO every N-th frame)."""
        cam = self.cameras.get(camera_id)
        if cam:
            cam.detection_interval = max(1, interval)
            return True
        return False

    def start_camera(self, camera_id: str) -> bool:
        """Open the video source and start the processing thread."""
        cam = self.cameras.get(camera_id)
        if not cam:
            return False

        source = cam.source

        # If source is a directory, auto-resolve to first video inside
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

        print(
            f"[IBVAP] Opening video capture for '{camera_id}' ({meta.get('type')}) -> {str(source_val)[:80]}..."
        )
        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            print(f"[IBVAP] ✗ Cannot open source: {source} (resolved: {str(source_val)[:80]})")
            return False

        cam.cap = cap
        cam.running = True

        # Load persisted night mode from DB
        cam_info = self.db.get_camera(camera_id)
        if cam_info:
            raw_nm = cam_info.get("night_mode", "off")
            if isinstance(raw_nm, int):
                cam.night_mode = "clahe" if raw_nm == 1 else "off"
            elif isinstance(raw_nm, str) and raw_nm.lower() in NIGHT_VISION_MODES:
                cam.night_mode = raw_nm.lower()
            else:
                cam.night_mode = "off"

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
        print(
            f"[IBVAP] ✓ Camera '{camera_id}' started — source: {source} (live={cam.is_live}, night_mode={cam.night_mode}, cadence={cam.detection_interval})"
        )
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
        if self.behavior_engine:
            self.behavior_engine.reset_camera(camera_id)
        self.motion_filter.reset_camera(camera_id)
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
            "total_crawling": 0,
            "total_loitering": 0,
            "total_unattended": 0,
            "total_frs_matches": 0,
            "total_anpr_scans": 0,
            "cameras": {},
        }
        for cid, cam in self.cameras.items():
            frs_cnt = getattr(cam, "frs_match_count", 0)
            anpr_cnt = getattr(cam, "anpr_scan_count", 0)
            stats["total_persons"] += cam.person_count
            stats["total_vehicles"] += cam.vehicle_count
            stats["total_intrusions"] += cam.intrusion_count
            stats["total_crawling"] += cam.crawling_count
            stats["total_loitering"] += cam.loitering_count
            stats["total_unattended"] += cam.unattended_count
            stats["total_frs_matches"] += frs_cnt
            stats["total_anpr_scans"] += anpr_cnt
            stats["cameras"][cid] = {
                "fps": round(cam.fps, 1),
                "persons": cam.person_count,
                "vehicles": cam.vehicle_count,
                "intrusions": cam.intrusion_count,
                "crawling": cam.crawling_count,
                "loitering": cam.loitering_count,
                "unattended": cam.unattended_count,
                "frs_matches": frs_cnt,
                "anpr_scans": anpr_cnt,
                "running": cam.running,
                "night_mode": cam.night_mode,
                "cadence": cam.detection_interval,
            }
        return stats

    # ── MJPEG frame generator (Rock-Solid Zero-Lag) ─────────────
    def generate_mjpeg(self, camera_id: str):
        """
        Yields pre-encoded MJPEG frames for StreamingResponse.
        Delivers frames immediately as they are generated with zero buffering lag.
        """
        cam = self.cameras.get(camera_id)
        if not cam:
            return

        last_seq = -1
        stale_ticks = 0

        while cam and cam.running:
            # Poll for sequence update (ultra-low CPU sleep)
            if cam.frame_seq == last_seq:
                stale_ticks += 1
                if stale_ticks > 40:  # ~0.6s without a new frame (e.g. connecting or buffering)
                    stale_ticks = 0
                    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder,
                        "IBVAP TACTICAL STREAM CONNECTING...",
                        (FRAME_WIDTH // 2 - 210, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 136),
                        2,
                    )
                    cv2.putText(
                        placeholder,
                        "Edge Optimization & Zero-Lag Cadence Active",
                        (FRAME_WIDTH // 2 - 180, FRAME_HEIGHT // 2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (150, 180, 200),
                        1,
                    )
                    ok, ph_buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        data = ph_buf.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                            + data
                            + b"\r\n"
                        )
                time.sleep(0.015)
                continue

            stale_ticks = 0
            with cam.frame_lock:
                jpeg_data = cam.latest_jpeg
                last_seq = cam.frame_seq

            if jpeg_data is None:
                time.sleep(0.02)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg_data)).encode() + b"\r\n\r\n"
                + jpeg_data
                + b"\r\n"
            )

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
                    if consecutive_read_fails > 25:
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
                    # Loop video files cleanly
                    cam.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_read_fails = 0
                    time.sleep(0.01)
                    continue

            consecutive_read_fails = 0

            # Resize for consistent processing
            frame = cv2.resize(raw_frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # ① Night Vision Enhancement
            # Render display frame according to active mode
            if cam.night_mode and cam.night_mode != "off":
                display_frame = self.night_enhancer.enhance(frame, mode=cam.night_mode)
            elif self.night_enhancer.is_dark_frame(frame):
                # Auto-enhance low-light frames with CLAHE if dark
                display_frame = self.night_enhancer.enhance(frame, mode="clahe")
            else:
                display_frame = frame

            # AI Detection Frame:
            # When in thermal or NVG, run YOLO on high-contrast CLAHE luminance for peak object recall
            if cam.night_mode == "clahe":
                detect_frame = display_frame
            elif cam.night_mode in ("thermal", "nvg"):
                detect_frame = self.night_enhancer.enhance(frame, mode="clahe")
            elif self.night_enhancer.is_dark_frame(frame):
                detect_frame = display_frame
            else:
                detect_frame = frame

            # ② Smart Cadence & Adaptive Frame Skipping
            cam.frame_count += 1
            is_detection_frame = (cam.frame_count % cam.detection_interval == 0)

            if is_detection_frame:
                # Run full YOLO detection and tracking
                detections = self.detector.detect_and_track(detect_frame)
                # Synchronize ByteTrack Kalman state
                cam.track_predictor.update_from_yolo(detections)
            else:
                # Reuse tracked bounding boxes via ByteTrack Kalman state
                detections = cam.track_predictor.predict_intermediate()

            # ③ Update Motion-Guided False Alarm Suppression Engine (MOG2 + trajectories)
            self.motion_filter.update(camera_id, frame, detections)

            # ④ Virtual Fence & Zero-Line Intrusion Check
            raw_intrusions = self.fence.check_intrusions(camera_id, detections)

            # Suppress false alarms (insects, rain, wind-blown foliage):
            # Only allow intrusions exhibiting steady directional displacement across 3+ frames
            intrusions = self.motion_filter.filter_intrusions(camera_id, raw_intrusions)

            # ⑤ Tactical Behavioral Analytics (Loitering, Crawling, Baggage Drop, Zero-Line Ingress)
            behavioral_events = []
            if self.behavior_engine:
                behavioral_events = self.behavior_engine.process_frame(
                    camera_id, detections, self.fence, (FRAME_HEIGHT, FRAME_WIDTH)
                )

            # ⑥ FRS & ANPR Biometric Analytics
            frs_matches = []
            anpr_matches = []

            for det in detections:
                cat = det.get("category", "")
                if cat == "person" and self.frs_engine:
                    f_match = self.frs_engine.process_person_detection(display_frame, det)
                    if f_match:
                        frs_matches.append(f_match)
                elif cat == "vehicle" and self.anpr_engine:
                    v_match = self.anpr_engine.process_vehicle_detection(display_frame, det, camera_id=camera_id)
                    if v_match and (v_match.get("matched") or v_match.get("is_hotlist")):
                        anpr_matches.append(v_match)

            # ⑦ Dispatch Alerts
            if intrusions:
                self.alert_engine.process_intrusions(camera_id, intrusions, display_frame)
            if behavioral_events:
                self.alert_engine.process_behavioral_events(camera_id, behavioral_events, display_frame)
            if frs_matches:
                self.alert_engine.process_frs_matches(camera_id, frs_matches, display_frame)
            if anpr_matches:
                self.alert_engine.process_anpr_matches(camera_id, anpr_matches, display_frame)
            self.alert_engine.process_detections(camera_id, detections, display_frame)

            # ⑧ Annotate Frame with Tactical HUD, bounding boxes, FRS/ANPR badges, and night vision badge
            fence_zones = self.fence.get_zone_polygons_for_drawing(camera_id)
            annotated = ObjectDetector.annotate_frame(
                display_frame,
                detections,
                intrusions,
                fence_zones,
                behavioral_events,
                night_mode=cam.night_mode,
            )

            # Update per-camera stats
            cam.person_count = sum(1 for d in detections if d.get("category") == "person")
            cam.vehicle_count = sum(1 for d in detections if d.get("category") == "vehicle")
            cam.intrusion_count = len(intrusions)
            cam.crawling_count = sum(1 for d in detections if d.get("posture", {}).get("is_crawling"))
            cam.loitering_count = sum(1 for d in detections if d.get("dwell_time", 0.0) >= 15.0)
            cam.unattended_count = sum(1 for d in detections if d.get("unattended_telemetry", {}).get("is_unattended"))
            cam.frs_match_count = sum(1 for d in detections if d.get("frs_match", {}).get("is_threat"))
            cam.anpr_scan_count = sum(1 for d in detections if d.get("anpr"))
            cam.total_detections = len(detections)

            # ⑧ Pre-encode JPEG for zero-lag streaming
            ok, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                jpeg_bytes = jpeg_buf.tobytes()
                with cam.frame_lock:
                    cam.latest_frame = annotated
                    cam.latest_jpeg = jpeg_bytes
                    cam.frame_seq += 1

            # FPS calculation
            elapsed = time.time() - loop_start
            cam.fps = 1.0 / max(elapsed, 0.001)

            # Throttle to target FPS
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
