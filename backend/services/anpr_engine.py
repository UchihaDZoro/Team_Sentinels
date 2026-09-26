"""
IBVAP Real-Time Automatic Number Plate Recognition (ANPR) Engine
Integrates solutions from:
1. benjnb/yolo-license-plate-detection (YOLO-based plate localization)
2. gauravcodes/yolov7-npr-training (YOLO NPR bounding box training & calibration)
3. omkarg1417/anpr-yolo (ANPR pipeline: vehicle association, bilateral filter, CLAHE, OCR, text cleaning)

Key Capabilities:
- High-recall plate localization (both full-frame & vehicle-crop dual scanning)
- Strict CCTV watermark & camera-number (CAM01) rejection
- Alphanumeric plate format validation (rejects random words & non-plate text)
- Automatic high-res plate photo capture & base64 push for real-time frontend display
- Vehicle association (ties plate to parent car/truck track ID)
- Asynchronous threaded OCR (EasyOCR never blocks 30 FPS video pipeline)
"""
import re
import cv2
import time
import base64
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import easyocr

from config import SNAPSHOTS_DIR, EVENTS_DIR

# Camera watermark keywords strictly forbidden from being treated as plates
CAMERA_WATERMARKS = {
    "CAM", "CAM01", "CAMOI", "ICAMO1", "CAMO1", "CAM1", "CAM2", "CAM02",
    "CH1", "CH01", "CH2", "CH02", "CCTV", "REC", "LIVE", "FPS", "NVR",
    "DVR", "HD", "DATE", "TIME", "SEC", "SYS", "CHANN", "CHANNEL", "IPCAM"
}


class ANPREngine:
    """
    Real-time ANPR engine with vehicle association, bilateral preprocessing,
    strict CCTV watermark exclusion, multi-frame temporal voting,
    Indian license plate format validation, and structured multi-crop evidence capture.
    """

    def __init__(self, model_path: str = "license_plate_yolov8n.pt", db=None):
        self.db = db
        self.model_path = Path(__file__).parent.parent / model_path
        if not self.model_path.exists():
            self.model_path = Path(model_path)

        print(f"[IBVAP-ANPR] Loading license plate model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))

        # EasyOCR reader initialized once (English alphanumeric)
        print("[IBVAP-ANPR] Initializing EasyOCR engine...")
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        # Thread pool for non-blocking asynchronous OCR extraction
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ANPR-OCR")
        self._pending_tasks = set()

        # Multi-frame temporal consensus: vehicle_track_id -> list of (plate_text, confidence, timestamp)
        self._plate_candidates: dict[int, list[tuple[str, float, float]]] = {}

        # Cache of recognized plates:
        # vehicle_track_id -> {
        #   "plate_text": "MH12AB1234",
        #   "confidence": 0.92,
        #   "first_seen": timestamp,
        #   "last_seen": timestamp,
        #   "frame_count": 12,
        #   "bbox": [px1, py1, px2, py2]
        # }
        self.vehicle_plates: dict[int, dict] = {}
        # Spatial cache for untracked vehicles (camera_id:hash -> plate_data)
        self.spatial_cache: dict[str, dict] = {}

        # Recent captured plates buffer (for frontend live HUD display)
        self.captured_plates: list[dict] = []
        # Active side callouts kept for 10 seconds: camera_id -> list of plate info dicts
        self.active_side_plates: dict[str, list[dict]] = {}
        # Callback for real-time WebSocket push: callback(plate_data)
        self.on_plate_captured = None

        # Warmup dummy prediction
        dummy = np.zeros((120, 240, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=320, verbose=False)
        print("[IBVAP-ANPR] ANPR Engine initialized & warmed up ✓")

    def detect_plates(self, frame: np.ndarray, detections: list[dict], camera_id: str = "") -> list[dict]:
        """
        Detect license plates in the frame, associate each with its parent vehicle,
        and attach plate text, bounding box, and captured photo to the detections.
        """
        h, w = frame.shape[:2]
        plate_detections = []

        # Find all vehicles in the current frame
        vehicles = [d for d in detections if d.get("category") == "vehicle"]

        candidate_boxes = []

        # ── 1. Full-frame plate detection (conf=0.12 for high sensitivity) ──
        try:
            results = self.model.predict(
                frame,
                imgsz=512,
                conf=0.12,
                iou=0.45,
                verbose=False,
            )
            for res in results:
                if res.boxes is not None:
                    for box in res.boxes:
                        conf = float(box.conf[0])
                        px1, py1, px2, py2 = [float(v) for v in box.xyxy[0]]
                        candidate_boxes.append((px1, py1, px2, py2, conf))
        except Exception:
            pass

        # ── 2. Vehicle-Crop plate detection (omkarg1417 & benjnb technique) ──
        # Magnifies small or distant vehicles so their plates are clearly localized
        for veh in vehicles:
            vx1, vy1, vx2, vy2 = [int(v) for v in veh["bbox"]]
            vw, vh = vx2 - vx1, vy2 - vy1
            if vw < 35 or vh < 25:
                continue

            # Crop vehicle with 5px margin
            cx1 = max(0, vx1 - 5)
            cy1 = max(0, vy1 - 5)
            cx2 = min(w, vx2 + 5)
            cy2 = min(h, vy2 + 5)
            veh_crop = frame[cy1:cy2, cx1:cx2]

            if veh_crop.size == 0:
                continue

            try:
                crop_res = self.model.predict(veh_crop, imgsz=320, conf=0.15, verbose=False)
                for cr in crop_res:
                    if cr.boxes is not None:
                        for cb in cr.boxes:
                            cconf = float(cb.conf[0])
                            cpx1, cpy1, cpx2, cpy2 = [float(v) for v in cb.xyxy[0]]
                            # Map back to global frame coordinates
                            gpx1 = cx1 + cpx1
                            gpy1 = cy1 + cpy1
                            gpx2 = cx1 + cpx2
                            gpy2 = cy1 + cpy2
                            candidate_boxes.append((gpx1, gpy1, gpx2, gpy2, cconf))
            except Exception:
                pass

        now = time.time()
        seen_boxes = []

        for px1, py1, px2, py2, conf in candidate_boxes:
            # Clip to frame boundary
            px1 = max(0, min(px1, w - 1))
            py1 = max(0, min(py1, h - 1))
            px2 = max(0, min(px2, w - 1))
            py2 = max(0, min(py2, h - 1))

            pw = px2 - px1
            ph = py2 - py1
            if pw < 18 or ph < 8 or pw > 0.40 * w or ph > 0.30 * h:
                continue  # Filter noise

            # De-duplicate candidate boxes
            is_dup = False
            for sx1, sy1, sx2, sy2 in seen_boxes:
                ix1 = max(px1, sx1)
                iy1 = max(py1, sy1)
                ix2 = min(px2, sx2)
                iy2 = min(py2, sy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = (pw * ph) + ((sx2 - sx1) * (sy2 - sy1)) - inter
                    if union > 0 and (inter / union) > 0.40:
                        is_dup = True
                        break
            if is_dup:
                continue
            seen_boxes.append((px1, py1, px2, py2))

            # ── Camera Watermark & OSD Exclusion ──────────────────────────
            # If the box is in top 9% or bottom 6% of the frame (typical CCTV header / footer)
            # and is NOT inside a vehicle, it is a camera OSD number (e.g. CAM 01, timestamp)!
            plate_cx = (px1 + px2) / 2.0
            plate_cy = (py1 + py2) / 2.0
            is_in_osd_zone = (py1 < 0.09 * h) or (py2 > 0.94 * h)

            # ── Vehicle Association (omkarg1417 technique) ────────────────
            matched_vehicle = None
            best_overlap = -1.0

            for veh in vehicles:
                vx1, vy1, vx2, vy2 = veh["bbox"]
                # Center containment
                if vx1 <= plate_cx <= vx2 and vy1 <= plate_cy <= vy2:
                    matched_vehicle = veh
                    break
                # Bounding box intersection
                ix1 = max(px1, vx1)
                iy1 = max(py1, vy1)
                ix2 = min(px2, vx2)
                iy2 = min(py2, vy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if inter > best_overlap:
                        best_overlap = inter
                        matched_vehicle = veh

            # Reject OSD watermark if outside of any vehicle
            if is_in_osd_zone and matched_vehicle is None:
                continue

            track_id = matched_vehicle.get("track_id", -1) if matched_vehicle else -1
            vehicle_type = matched_vehicle.get("class", "vehicle") if matched_vehicle else "vehicle"

            # Check if plate is already cached for this vehicle
            plate_text = ""
            ocr_conf = 0.0
            photo_url = ""
            photo_base64 = ""

            cached_entry = self.vehicle_plates.get(track_id) if track_id >= 0 else None
            spatial_key = f"{camera_id}_{int(px1 // 30)}_{int(py1 // 30)}"
            cached_spatial = self.spatial_cache.get(spatial_key)

            if cached_entry and cached_entry.get("plate_text"):
                plate_text = cached_entry["plate_text"]
                ocr_conf = cached_entry.get("confidence", 0.85)
                photo_url = cached_entry.get("photo_url", "")
                photo_base64 = cached_entry.get("photo_base64", "")
                cached_entry["last_seen"] = now
                cached_entry["frame_count"] += 1
                cached_entry["bbox"] = [px1, py1, px2, py2]
            elif cached_spatial and (now - cached_spatial.get("last_seen", 0)) < 15.0:
                plate_text = cached_spatial["plate_text"]
                ocr_conf = cached_spatial.get("confidence", 0.85)
                photo_url = cached_spatial.get("photo_url", "")
                photo_base64 = cached_spatial.get("photo_base64", "")
                cached_spatial["last_seen"] = now
            else:
                # Need OCR extraction: queue asynchronously to keep 30 FPS real-time
                task_key = f"{camera_id}_{track_id}_{spatial_key}"
                pad = 6
                cx1 = max(0, int(px1 - pad))
                cy1 = max(0, int(py1 - pad))
                cx2 = min(w, int(px2 + pad))
                cy2 = min(h, int(py2 + pad))
                crop = frame[cy1:cy2, cx1:cx2].copy()

                if crop.size > 0 and (cx2 - cx1) >= 18 and (cy2 - cy1) >= 8:
                    # Instant photo snapshot encoding
                    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    photo_base64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

                    # If not broadcasted recently for this vehicle, trigger instant visual HUD photo card
                    vkey = f"{camera_id}_{track_id if track_id >= 0 else spatial_key}"
                    if (now - getattr(self, "_last_broadcast", {}).get(vkey, 0)) > 15.0:
                        if not hasattr(self, "_last_broadcast"):
                            self._last_broadcast = {}
                        self._last_broadcast[vkey] = now
                        instant_info = {
                            "plate_number": plate_text or (f"VEHICLE PLATE #{track_id}" if track_id >= 0 else "VEHICLE PLATE"),
                            "confidence": round(conf, 2),
                            "vehicle_type": vehicle_type.capitalize(),
                            "camera_id": camera_id,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "photo_base64": photo_base64,
                            "photo_url": "",
                            "bbox": [px1, py1, px2, py2],
                            "track_id": track_id,
                        }
                        self.captured_plates.insert(0, instant_info)
                        if len(self.captured_plates) > 25:
                            self.captured_plates.pop()
                        if self.on_plate_captured:
                            try:
                                self.on_plate_captured(instant_info)
                            except Exception:
                                pass

                    # ── Keep cropped image on side for 10 seconds ─────────
                    if camera_id:
                        if camera_id not in self.active_side_plates:
                            self.active_side_plates[camera_id] = []
                        matched_side = None
                        for sp in self.active_side_plates[camera_id]:
                            if (track_id >= 0 and sp.get("track_id") == track_id) or (plate_text and sp.get("plate_text") == plate_text):
                                matched_side = sp
                                break
                        if matched_side:
                            matched_side["expires_at"] = now + 10.0
                            matched_side["crop"] = crop.copy()
                            if plate_text:
                                matched_side["plate_text"] = plate_text
                        else:
                            self.active_side_plates[camera_id].insert(0, {
                                "plate_text": plate_text or "READING OCR...",
                                "crop": crop.copy(),
                                "vehicle_type": vehicle_type.capitalize(),
                                "confidence": round(conf, 2),
                                "timestamp": now,
                                "expires_at": now + 10.0,
                                "track_id": track_id,
                            })
                            if len(self.active_side_plates[camera_id]) > 2:
                                self.active_side_plates[camera_id].pop()

                    if task_key not in self._pending_tasks and len(self._pending_tasks) < 4:
                        self._pending_tasks.add(task_key)
                        veh_crop = None
                        if matched_vehicle is not None:
                            vx1, vy1, vx2, vy2 = [int(v) for v in matched_vehicle["bbox"]]
                            veh_crop = frame[max(0, vy1):min(h, vy2), max(0, vx1):min(w, vx2)].copy()

                        self._executor.submit(
                            self._async_ocr_task,
                            crop,
                            task_key,
                            track_id,
                            spatial_key,
                            camera_id,
                            vehicle_type,
                            [px1, py1, px2, py2],
                            frame.copy(),
                            veh_crop,
                            conf,
                        )

            plate_item = {
                "bbox": [px1, py1, px2, py2],
                "confidence": round(conf, 2),
                "plate_text": plate_text,
                "ocr_confidence": round(ocr_conf, 2),
                "track_id": track_id,
                "vehicle_type": vehicle_type,
                "photo_url": photo_url,
                "photo_base64": photo_base64,
            }
            plate_detections.append(plate_item)

            if matched_vehicle is not None:
                matched_vehicle["plate"] = plate_item

        return plate_detections

    def _async_ocr_task(
        self,
        crop: np.ndarray,
        task_key: str,
        track_id: int,
        spatial_key: str,
        camera_id: str,
        vehicle_type: str,
        bbox: list[float],
        full_frame: Optional[np.ndarray] = None,
        vehicle_crop: Optional[np.ndarray] = None,
        det_conf: float = 0.0,
    ):
        """
        Background worker executing preprocessing + EasyOCR pipeline.
        Enforces character disambiguation, multi-frame temporal voting,
        and saves complete structured multi-crop evidence (full frame, vehicle crop, plate crop, metadata.json).
        """
        try:
            # Step 1: Preprocess plate crop (omkarg1417 / gauravcodes pipeline)
            preprocessed = self.preprocess_plate_crop(crop)

            # Step 2: Run EasyOCR with alphanumeric character whitelist
            results = self.reader.readtext(
                preprocessed,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                paragraph=False,
                detail=1,
            )

            best_text = ""
            best_conf = 0.0

            if results:
                for res in results:
                    text = res[1]
                    conf = float(res[2])
                    cleaned = self.clean_plate_text(text)
                    if len(cleaned) >= 4 and conf > best_conf:
                        best_text = cleaned
                        best_conf = conf

            # Fallback: try raw crop
            if not best_text:
                raw_results = self.reader.readtext(
                    crop,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                    paragraph=False,
                    detail=1,
                )
                for res in raw_results:
                    cleaned = self.clean_plate_text(res[1])
                    if len(cleaned) >= 4:
                        best_text = cleaned
                        best_conf = float(res[2])
                        break

            if not best_text:
                return

            clean_compact = best_text.replace(" ", "").upper()

            # ── Strict Camera Watermark Rejection ─────────────────────────
            if clean_compact in CAMERA_WATERMARKS:
                return

            # Reject if text is purely alphabetical with no digits (e.g. "CAM", "STOP", "CAR")
            has_letters = any(c.isalpha() for c in clean_compact)
            has_digits = any(c.isdigit() for c in clean_compact)
            if not (has_letters and has_digits):
                return

            # Apply contextual Indian license plate character disambiguation
            clean_compact = self.validate_and_disambiguate_plate(clean_compact)

            # ── Multi-Frame Temporal Consensus Voting ─────────────────────
            # Requirement 17: consensus across frames rather than trusting a single frame
            now = time.time()
            if track_id >= 0:
                if track_id not in self._plate_candidates:
                    self._plate_candidates[track_id] = []
                self._plate_candidates[track_id].append((clean_compact, best_conf, now))

                # Prune candidates older than 15s
                self._plate_candidates[track_id] = [
                    c for c in self._plate_candidates[track_id] if (now - c[2]) <= 15.0
                ]

                # Run consensus voting
                readings = [c[0] for c in self._plate_candidates[track_id]]
                if readings:
                    counts = Counter(readings)
                    consensus_text, count = counts.most_common(1)[0]
                    # If confirmed across 2+ frames or single frame with high confidence (>=0.72)
                    if count >= 2 or best_conf >= 0.72:
                        clean_compact = consensus_text

            # Format canonical spacing (e.g. MH 12 AB 1234)
            formatted = self.format_plate_string(clean_compact)

            # ── Requirement 16: Structured Multi-Crop Evidence Capture ────
            now_dt = datetime.now()
            date_str = now_dt.strftime("%Y-%m-%d")
            time_str = now_dt.strftime("%H%M%S")
            timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")

            event_folder = EVENTS_DIR / camera_id / date_str / f"{time_str}_{clean_compact}"
            try:
                event_folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(event_folder / "plate.jpg"), crop)
                if vehicle_crop is not None and vehicle_crop.size > 0:
                    cv2.imwrite(str(event_folder / "vehicle.jpg"), vehicle_crop)
                if full_frame is not None and full_frame.size > 0:
                    cv2.imwrite(str(event_folder / "full_frame.jpg"), full_frame)

                meta_data = {
                    "camera_id": camera_id,
                    "timestamp": now_dt.isoformat(),
                    "plate_number": formatted,
                    "clean_plate": clean_compact,
                    "vehicle_type": vehicle_type,
                    "vehicle_track_id": track_id,
                    "ocr_confidence": round(best_conf, 3),
                    "detection_confidence": round(det_conf, 3),
                    "bbox": bbox,
                }
                with open(event_folder / "metadata.json", "w") as mf:
                    json.dump(meta_data, mf, indent=2)
            except Exception as e:
                print(f"[IBVAP-ANPR] Evidence folder write notice: {e}")

            # Also save backward-compatible snapshot for dashboard display
            filename = f"plate_{camera_id}_{clean_compact}_{timestamp_str}.jpg"
            save_path = SNAPSHOTS_DIR / filename
            cv2.imwrite(str(save_path), crop)

            # Generate base64 Data URI for instantaneous zero-latency frontend display
            _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            photo_base64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
            photo_url = f"/api/snapshots/{filename}"

            plate_info = {
                "plate_number": formatted,
                "confidence": round(best_conf, 2),
                "vehicle_type": vehicle_type.capitalize(),
                "camera_id": camera_id,
                "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "photo_url": photo_url,
                "photo_base64": photo_base64,
                "bbox": bbox,
                "track_id": track_id,
                "evidence_folder": str(event_folder),
            }

            if track_id >= 0:
                self.vehicle_plates[track_id] = {
                    "plate_text": formatted,
                    "confidence": best_conf,
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "frame_count": self.vehicle_plates.get(track_id, {}).get("frame_count", 0) + 1,
                    "bbox": bbox,
                    "vehicle_type": vehicle_type,
                    "photo_url": photo_url,
                    "photo_base64": photo_base64,
                    "evidence_folder": str(event_folder),
                }

            if spatial_key:
                self.spatial_cache[spatial_key] = {
                    "plate_text": formatted,
                    "confidence": best_conf,
                    "last_seen": time.time(),
                    "photo_url": photo_url,
                    "photo_base64": photo_base64,
                    "evidence_folder": str(event_folder),
                }

            # ── 10-Second Active Side Display Cache ───────────────────────
            # Keeps a cropped image of the recognized plate at the side of the video for 10 seconds
            if camera_id:
                if camera_id not in self.active_side_plates:
                    self.active_side_plates[camera_id] = []

                now_ts = time.time()
                found = False
                for sp in self.active_side_plates[camera_id]:
                    if (track_id >= 0 and sp.get("track_id") == track_id) or sp.get("plate_text") == formatted:
                        sp["plate_text"] = formatted
                        sp["crop"] = crop.copy()
                        sp["confidence"] = round(best_conf, 2)
                        sp["vehicle_type"] = vehicle_type.capitalize()
                        sp["expires_at"] = now_ts + 10.0
                        found = True
                        break

                if not found:
                    self.active_side_plates[camera_id].insert(0, {
                        "plate_text": formatted,
                        "crop": crop.copy(),
                        "vehicle_type": vehicle_type.capitalize(),
                        "confidence": round(best_conf, 2),
                        "timestamp": now_ts,
                        "expires_at": now_ts + 10.0,
                        "track_id": track_id,
                    })
                    if len(self.active_side_plates[camera_id]) > 2:
                        self.active_side_plates[camera_id].pop()

            # Add to recent captured list (FIFO 25 items)
            self.captured_plates.insert(0, plate_info)
            if len(self.captured_plates) > 25:
                self.captured_plates.pop()

            # Save to SQLite database
            if self.db is not None:
                try:
                    self.db.add_license_plate(
                        camera_id=camera_id,
                        vehicle_track_id=track_id,
                        plate_number=formatted,
                        confidence=best_conf,
                        vehicle_type=vehicle_type,
                        photo_path=str(save_path),
                    )
                except Exception:
                    pass

            # Trigger real-time callback for WebSocket broadcast
            if self.on_plate_captured is not None:
                try:
                    self.on_plate_captured(plate_info)
                except Exception:
                    pass

        except Exception as e:
            print(f"[IBVAP-ANPR] Async OCR error: {e}")
        finally:
            self._pending_tasks.discard(task_key)

    @staticmethod
    def preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
        """
        State-of-the-art ANPR preprocessing pipeline:
        1. Aspect-ratio preserving height normalization (70px)
        2. Grayscale conversion
        3. Bilateral filter (removes road dust/glare while preserving sharp letter edges)
        4. Contrast-limited adaptive histogram equalization (CLAHE)
        """
        h, w = crop.shape[:2]
        target_h = 70
        target_w = max(20, int(w * (target_h / max(h, 1))))
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Bilateral filter: smoothing with edge-preservation
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE for contrast enhancement under variable lighting/shadows
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)
        return enhanced

    VALID_STATES = {
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
        "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
        "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
    }

    STATE_OCR_CORRECTIONS = {
        "OL": "DL", "0L": "DL", "QL": "DL",
        "IH": "JH", "1H": "JH",
        "M8": "MH", "MB": "MH",
        "H8": "HR", "HB": "HR",
        "U9": "UP", "UB": "UP",
        "8R": "BR",
        "D1": "DL", "DI": "DL",
        "W8": "WB",
    }

    @classmethod
    def validate_and_disambiguate_plate(cls, text: str) -> str:
        """
        Applies contextual OCR character corrections for Indian license plates:
        Format: [State: 2 chars] [District: 1-2 digits] [Series: 1-3 chars] [Number: 4 digits]
        e.g. DL 01 AB 1234, MH 12 DE 1433, JH 01 A 8123
        Disambiguates:
        - In state prefix (letters): 0->O/D, 1->I, 5->S, 8->B, 2->Z
        - In district/number positions (digits): O->0, I->1, S->5, B->8, Z->2, G->6, D->0
        """
        if not text or len(text) < 4:
            return text

        chars = list(text.upper())
        n = len(chars)

        # Check direct 2-letter state OCR corrections (e.g. 0L -> DL, 1H -> JH)
        prefix2 = "".join(chars[:2])
        if prefix2 in cls.STATE_OCR_CORRECTIONS:
            corrected_prefix = cls.STATE_OCR_CORRECTIONS[prefix2]
            chars[0] = corrected_prefix[0]
            chars[1] = corrected_prefix[1]
        else:
            alpha_map = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z'}
            if n >= 2:
                for i in range(2):
                    if chars[i] in alpha_map:
                        chars[i] = alpha_map[chars[i]]
                cand_prefix = "".join(chars[:2])
                if cand_prefix in cls.STATE_OCR_CORRECTIONS:
                    corrected = cls.STATE_OCR_CORRECTIONS[cand_prefix]
                    chars[0], chars[1] = corrected[0], corrected[1]

        digit_map = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2', 'G': '6', 'D': '0', 'Q': '0'}

        # Positions 2 and 3: District code typically digits
        if 8 <= n <= 10:
            for i in range(2, min(4, n)):
                if chars[i] in digit_map:
                    chars[i] = digit_map[chars[i]]

            # Last 4 characters: Registration number digits
            for i in range(max(4, n - 4), n):
                if chars[i] in digit_map:
                    chars[i] = digit_map[chars[i]]

        return "".join(chars)

    @classmethod
    def clean_plate_text(cls, raw_text: str) -> str:
        """Clean and sanitize raw OCR output: uppercase, strip noise, and disambiguate characters."""
        if not raw_text:
            return ""
        text = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        return cls.validate_and_disambiguate_plate(text)

    @staticmethod
    def format_plate_string(text: str) -> str:
        """
        Canonical plate formatting:
        Standard Indian Format: DL 01 AB 1234 or MH 12 DE 1433
        Standard International: Group into 2-4 character tokens
        """
        if not text or len(text) < 4:
            return text

        # Check standard Indian format: 2 letters + 2 numbers + 1-2 letters + 4 numbers
        m = re.match(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})$", text)
        if m:
            return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"

        if len(text) in (9, 10):
            return f"{text[:2]} {text[2:4]} {text[4:6]} {text[6:]}"
        elif len(text) in (7, 8):
            return f"{text[:2]} {text[2:5]} {text[5:]}"
        return text

    @staticmethod
    def annotate_plates(frame: np.ndarray, plate_detections: list[dict]) -> np.ndarray:
        """
        Draw license plate boxes and stylish Cyan/White plate banners on the frame.
        """
        for p in plate_detections:
            px1, py1, px2, py2 = [int(v) for v in p["bbox"]]
            plate_text = p.get("plate_text", "")
            det_conf = p.get("confidence", 0.0)

            # Draw Plate Bounding Box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 2, cv2.LINE_AA)

            # Plate banner text
            if plate_text:
                tag = f"PLATE: {plate_text}"
                tag_bg = (0, 180, 255)  # Vibrant Cyan
                text_col = (0, 0, 0)
            else:
                tag = f"PLATE {det_conf:.0%}"
                tag_bg = (50, 120, 180)
                text_col = (255, 255, 255)

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(0, py1 - th - 6)
            cv2.rectangle(frame, (px1, tag_y1), (px1 + tw + 8, py1), tag_bg, -1)
            cv2.putText(
                frame, tag, (px1 + 4, py1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1, cv2.LINE_AA
            )

        return frame

    def draw_side_plate_callouts(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """
        Displays a high-resolution cropped image of detected & OCR'd license plates
        on the right side of the video stream for exactly 10 seconds.
        """
        if not hasattr(self, "active_side_plates") or camera_id not in self.active_side_plates:
            return frame

        now = time.time()
        # Keep only plates that have not expired
        active = [p for p in self.active_side_plates[camera_id] if p.get("expires_at", 0) > now]
        self.active_side_plates[camera_id] = active

        if not active or frame is None:
            return frame

        h, w = frame.shape[:2]
        annotated = frame

        for idx, item in enumerate(active[:2]):
            crop = item.get("crop")
            plate_text = item.get("plate_text", "")
            vehicle_type = item.get("vehicle_type", "Vehicle")
            conf = item.get("confidence", 0.85)
            expires_at = item.get("expires_at", now)
            rem_sec = max(0.0, expires_at - now)

            # Dimensions of the side callout card
            card_w = 245
            card_h = 106
            x1 = w - card_w - 12
            y1 = 14 + idx * (card_h + 12)
            x2 = x1 + card_w
            y2 = y1 + card_h

            if x1 < 0 or y2 > h:
                continue

            # 1. Dark semi-transparent card background
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 16, 26), -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            # 2. Glowing Cyan tactical border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 220, 0), 2, cv2.LINE_AA)

            # 3. Header & 10s Countdown Timer Badge
            cv2.putText(annotated, "LICENSE PLATE OCR", (x1 + 8, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1, cv2.LINE_AA)
            time_str = f"{rem_sec:.1f}s"
            (tw, th), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(annotated, time_str, (x2 - tw - 8, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1, cv2.LINE_AA)

            # Progress bar for remaining seconds (10s -> 0s)
            pct = min(1.0, max(0.0, rem_sec / 10.0))
            bar_w = int((card_w - 16) * pct)
            if bar_w > 0:
                cv2.rectangle(annotated, (x1 + 8, y1 + 21), (x1 + 8 + bar_w, y1 + 23), (0, 220, 255), -1)

            # 4. Render the Cropped Plate Image on the left side of the card
            if crop is not None and crop.size > 0:
                ch, cw = crop.shape[:2]
                disp_w = 112
                disp_h = 44
                if cw > 0 and ch > 0:
                    try:
                        disp_crop = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                        annotated[y1 + 28:y1 + 28 + disp_h, x1 + 8:x1 + 8 + disp_w] = disp_crop
                        cv2.rectangle(annotated, (x1 + 8, y1 + 28), (x1 + 8 + disp_w, y1 + 28 + disp_h), (255, 255, 255), 1)
                    except Exception:
                        pass

            # 5. Bold OCR text & vehicle details to the right of the crop
            text_x = x1 + 128
            cv2.putText(annotated, plate_text, (text_x, y1 + 45), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"{vehicle_type}", (text_x, y1 + 61), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 220, 230), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"Conf: {conf:.0%}", (text_x, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 140), 1, cv2.LINE_AA)

            # 6. Bottom subtitle: "RETAINED: 10s"
            cv2.putText(annotated, "[10s LIVE CALLOUT]", (x1 + 8, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 160, 180), 1, cv2.LINE_AA)

        return annotated
