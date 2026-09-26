"""
IBVAP Object & Threat Detector — Dual-Engine Security Analytics
Detects:
1. Humans (Vibrant Green Box) — High recall in crowds, individual person tagging
2. Vehicles: Cars, Trucks, Buses, Motorcycles (Vibrant Yellow Box)
3. Suspicious / Harmful Objects: Guns, Pistols, Grenades, Explosives/Bombs, Knives, Blades (Vibrant RED Box)
   - Dynamic Projectile & Thrown Grenade Motion Trajectory Tracking
   - Unrestricted scale for explosions & blasts
   - Negative cross-suppression against everyday items (balls, stones, phones, bottles)
   - False-alarm filtering on clothing/belts for knives
4. Integrated with ANPR Engine (License Plates) & Crowd Detection (Spatial Clustering)
"""
import os
import cv2
import time
import math
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from config import (
    YOLO_MODEL,
    THREAT_MODEL,
    POSE_MODEL,
    YOLO_IMGSZ,
    CONFIDENCE_THRESHOLD,
    THREAT_CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CLASSES_OF_INTEREST,
    PERSON_CLASSES,
    VEHICLE_CLASSES,
    BENIGN_SUPPRESSION_CLASSES,
    THREAT_CLASSES,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    ENABLE_HAND_ROI_THREATS,
    MAX_TRAJECTORY_HISTORY,
    MAX_PROJECTILE_PREDICT_FRAMES,
)
from services.model_manager import ModelManager
from services.temporal_tracker import TemporalConfirmationEngine
from services.explosion_detector import ExplosionDetector
from services.crowd_detector import CrowdDetector
from services.anpr_engine import ANPREngine

# Colour palette for drawing (BGR)
COLOUR_HUMAN = (0, 255, 0)      # Vibrant Green for Humans

COLOUR_VEHICLE = (0, 255, 255)  # Vibrant Yellow for Vehicles
COLOUR_THREAT = (0, 0, 255)     # Vibrant Alert Red for Harmful / Suspicious Objects
COLOUR_INTRUSION = (0, 0, 255)  # Red for Intrusion Alerts
COLOUR_ITEM = (255, 180, 50)    # Orange/cyan accent for neutral objects


class ProjectileTracker:
    """
    Advanced Ballistic & Threat Projectile Tracker.
    Maintains:
    - Persistent object IDs
    - Full kinematic state: timestamp, bbox, center (cx, cy), velocity vector (vx, vy), speed
    - Trajectory history up to MAX_TRAJECTORY_HISTORY
    - Dead-reckoning extrapolation for temporary occlusions (up to MAX_PROJECTILE_PREDICT_FRAMES)
    """
    def __init__(self, max_history: int = MAX_TRAJECTORY_HISTORY):
        # track_id -> {
        #   "history": [(cx, cy, timestamp), ...],
        #   "bbox": [x1, y1, x2, y2],
        #   "cls": "grenade",
        #   "conf": 0.85,
        #   "missed": 0,
        #   "vx": 0.0,
        #   "vy": 0.0,
        #   "speed": 0.0,
        # }
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        self.max_history = max_history

    def update(self, threats: list[dict], now: float) -> list[dict]:
        # Purge stale tracks older than 2.0 seconds
        stale_ids = [
            tid for tid, trk in self.tracks.items()
            if (now - trk["history"][-1][2]) > 2.0 or trk.get("missed", 0) > MAX_PROJECTILE_PREDICT_FRAMES
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        matched_tracks = set()

        for t in threats:
            raw_cls = t.get("class", "").lower()
            if raw_cls not in ("grenade", "bomb", "explosion", "knife", "gun", "threat"):
                continue

            x1, y1, x2, y2 = t["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Match to nearest active track within 180px radius
            matched_id = None
            min_dist = 180.0

            for tid, trk in self.tracks.items():
                if tid in matched_tracks:
                    continue
                hist = trk["history"]
                last_x, last_y, last_t = hist[-1]
                dt = max(0.001, now - last_t)
                dist = math.hypot(cx - last_x, cy - last_y)
                if dist < min_dist and dt < 0.6:
                    min_dist = dist
                    matched_id = tid

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
                self.tracks[matched_id] = {
                    "history": [(cx, cy, now)],
                    "bbox": [x1, y1, x2, y2],
                    "cls": raw_cls,
                    "conf": t.get("confidence", 0.5),
                    "missed": 0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "speed": 0.0,
                }
            else:
                trk = self.tracks[matched_id]
                last_x, last_y, last_t = trk["history"][-1]
                dt = max(0.005, now - last_t)
                vx = (cx - last_x) / dt
                vy = (cy - last_y) / dt
                speed = math.hypot(vx, vy)
                trk["vx"] = round(vx, 1)
                trk["vy"] = round(vy, 1)
                trk["speed"] = round(speed, 1)
                trk["bbox"] = [x1, y1, x2, y2]
                trk["conf"] = t.get("confidence", 0.5)
                trk["missed"] = 0
                trk["history"].append((cx, cy, now))
                if len(trk["history"]) > self.max_history:
                    trk["history"].pop(0)

            matched_tracks.add(matched_id)
            trk = self.tracks[matched_id]
            hist = trk["history"]

            t["projectile_track_id"] = matched_id
            t["trajectory"] = [(pt[0], pt[1]) for pt in hist]
            t["velocity"] = trk["speed"]
            t["velocity_x"] = trk["vx"]
            t["velocity_y"] = trk["vy"]

            # Flag high-speed throwing motion (speed > 110 px/s with trajectory >= 2 points)
            if trk["speed"] > 110.0 and len(hist) >= 2:
                t["is_thrown"] = True
                if raw_cls == "grenade":
                    t["threat_label"] = "GRENADE THROWN (IN FLIGHT)"
                    t["severity"] = "critical"
            else:
                t["is_thrown"] = False

        # Dead-reckoning for tracks that missed detection in this frame:
        # Keep trajectory overlay visible and extrapolate position for up to MAX_PROJECTILE_PREDICT_FRAMES
        for tid, trk in list(self.tracks.items()):
            if tid not in matched_tracks and trk["speed"] > 60.0 and len(trk["history"]) >= 3:
                trk["missed"] += 1
                if trk["missed"] <= MAX_PROJECTILE_PREDICT_FRAMES:
                    last_x, last_y, last_t = trk["history"][-1]
                    dt = max(0.01, now - last_t)
                    pred_x = last_x + trk["vx"] * dt
                    pred_y = last_y + trk["vy"] * dt
                    trk["history"].append((pred_x, pred_y, now))
                    if len(trk["history"]) > self.max_history:
                        trk["history"].pop(0)
                    bx1, by1, bx2, by2 = trk["bbox"]
                    bw, bh = bx2 - bx1, by2 - by1
                    pred_bbox = [pred_x - bw / 2, pred_y - bh / 2, pred_x + bw / 2, pred_y + bh / 2]
                    threats.append({
                        "bbox": pred_bbox,
                        "class": trk["cls"],
                        "class_id": 999,
                        "category": "threat",
                        "confidence": round(trk["conf"] * 0.85, 2),
                        "track_id": -1,
                        "projectile_track_id": tid,
                        "is_threat": True,
                        "threat_label": f"{trk['cls'].upper()} (IN FLIGHT TRAJECTORY)",
                        "severity": "critical",
                        "is_thrown": True,
                        "is_predicted": True,
                        "trajectory": [(pt[0], pt[1]) for pt in trk["history"]],
                        "velocity": trk["speed"],
                        "velocity_x": trk["vx"],
                        "velocity_y": trk["vy"],
                    })

        return threats


def resolve_model_file(model_name: str) -> Path:
    """Resolve model path checking absolute, backend, and ref_repos directories."""
    p = Path(model_name)
    if p.is_absolute() and p.exists():
        return p
    base_dir = Path(__file__).parent.parent
    b_path = base_dir / model_name
    if b_path.exists():
        return b_path
    proj_path = base_dir.parent / model_name
    if proj_path.exists():
        return proj_path
    hf_path = base_dir.parent / "ref_repos" / "Hackfest2k25-kdf" / model_name
    if hf_path.exists():
        return hf_path
    ref_dir = base_dir.parent / "ref_repos"
    if ref_dir.exists():
        matches = list(ref_dir.glob(f"**/{p.name}"))
        if matches:
            return matches[0]
    return b_path


class ObjectDetector:
    """
    Multi-Engine Real-Time Security Analytics Detector:
    Engine 1: Primary surveillance model (YOLOv11 / YOLO26 tracking humans & vehicles, high recall in crowds)
    Engine 2: Specialized Threat Model (firearms, explosions, grenades, knives from ref_repos / backend)
    Engine 3: Pose & Hand Keypoint Estimator (optional wrist keypoints for Hand ROI)
    Engine 4: Projectile Motion Tracker (ballistic trajectory history & velocity vectors)
    Engine 5: Temporal Confirmation Engine (false positive elimination)
    Engine 6: Optical Explosion & Flash Analyzer
    """

    def __init__(self):
        primary_path = resolve_model_file(YOLO_MODEL)
        print(f"[IBVAP] Loading Primary YOLO model: {primary_path} (imgsz={YOLO_IMGSZ})")
        self.model = YOLO(str(primary_path))

        # Specialized Threat Model from ref_repos / backend
        self.threat_model = None
        threat_path = resolve_model_file(THREAT_MODEL)
        if threat_path.exists():
            try:
                print(f"[IBVAP] Loading Specialized Threat model: {threat_path} (imgsz={YOLO_IMGSZ})")
                self.threat_model = YOLO(str(threat_path))
            except Exception as e:
                print(f"[IBVAP] ⚠ Could not load threat model from {threat_path}: {e}")
        else:
            print(f"[IBVAP] ⚠ Threat model weights not found at {threat_path}")

        # Pose & Hand Keypoint Model for Hand ROI Threat Detection
        self.pose_model = None
        if ENABLE_HAND_ROI_THREATS:
            pose_path = resolve_model_file(POSE_MODEL)
            if pose_path.exists():
                try:
                    print(f"[IBVAP] Loading Pose & Keypoint model: {pose_path} (imgsz=384)")
                    self.pose_model = YOLO(str(pose_path))
                except Exception as e:
                    print(f"[IBVAP] ⚠ Could not load pose model from {pose_path}: {e}")
            else:
                print(f"[IBVAP] ⚠ Pose model not found at {pose_path}")

        self.projectile_tracker = ProjectileTracker()
        self.temporal_engine = TemporalConfirmationEngine(max_idle_seconds=1.2)
        self.explosion_detector = ExplosionDetector(flash_threshold=0.28, disturbance_threshold=0.08)
        self._prev_grenade_tracks: set[int] = set()  # Track IDs of grenades last frame for disappearance detection
        self._warmup()

    def _warmup(self):
        """Run dummy inference to warm up models."""
        dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=YOLO_IMGSZ, verbose=False)
        if self.threat_model is not None:
            self.threat_model.predict(dummy, imgsz=YOLO_IMGSZ, verbose=False)
        if self.pose_model is not None:
            self.pose_model.predict(dummy, imgsz=384, verbose=False)
        print("[IBVAP] Object, Threat, and Hand-Pose Detection Engines warmed up ✓")

    def detect_and_track(self, frame: np.ndarray, timestamp: Optional[float] = None) -> list[dict]:
        """
        Run detection + tracking on a frame.
        Detects humans (green), vehicles (yellow), harmful threat objects (red),
        and tracks in-flight grenades / projectiles using true timestamp delta dt.
        """
        now = time.time() if timestamp is None else timestamp
        detections = []
        benign_items = []  # Detected everyday items for negative cross-suppression
        frame_h, frame_w = frame.shape[:2]

        # ① Primary model for humans, vehicles, and standard surveillance items
        results = self.model.track(
            frame,
            imgsz=YOLO_IMGSZ,
            persist=True,
            conf=CONFIDENCE_THRESHOLD,  # 0.18 for high recall on persons & vehicles
            iou=IOU_THRESHOLD,          # 0.65 to keep overlapping people in dense crowds
            verbose=False,
            classes=list(CLASSES_OF_INTEREST.keys()),
            tracker="bytetrack.yaml",
        )

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASSES_OF_INTEREST.get(cls_id, "unknown").lower()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                track_id = int(box.id[0]) if box.id is not None else -1
                bw = x2 - x1
                bh = y2 - y1

                # Overhead top-down vehicle correction:
                # Top-down cameras make cars look like rounded rectangles, fooling generic models into 'cell phone'
                if cls_id == 67 and (bw * bh) > 5000:
                    cls_id = 2
                    cls_name = "car"

                # Track benign items (sports balls, bottles, phones, fruits) for negative suppression
                if cls_id in BENIGN_SUPPRESSION_CLASSES:
                    benign_items.append({
                        "bbox": [x1, y1, x2, y2],
                        "class": cls_name,
                        "class_id": cls_id,
                        "conf": conf,
                    })

                # Check if COCO detected knife (class 43)
                if cls_id == 43:
                    # Prevent false knife alarms on police ties, belts, pens:
                    aspect = max(bw / max(bh, 1), bh / max(bw, 1))
                    if conf < 0.52 or aspect < 1.35 or (bw * bh) < 120:
                        continue
                    category = "threat"
                    threat_label = "BLADE / KNIFE"
                    severity = "high"
                    is_threat = True
                elif cls_name in THREAT_CLASSES:
                    category = "threat"
                    threat_meta = THREAT_CLASSES[cls_name]
                    threat_label = threat_meta.get("label", cls_name.upper())
                    severity = threat_meta.get("severity", "high")
                    is_threat = True
                elif cls_id in PERSON_CLASSES:
                    category = "person"
                    threat_label = ""
                    severity = "low"
                    is_threat = False
                elif cls_id in VEHICLE_CLASSES:
                    category = "vehicle"
                    threat_label = ""
                    severity = "low"
                    is_threat = False
                else:
                    category = "object"
                    threat_label = ""
                    severity = "low"
                    is_threat = False

                if category in ("person", "vehicle", "threat") or is_threat:
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "class": cls_name,
                        "class_id": cls_id,
                        "category": category,
                        "confidence": round(conf, 2),
                        "track_id": track_id,
                        "is_threat": is_threat,
                        "threat_label": threat_label,
                        "severity": severity,
                    })

        threat_candidates = []

        # ② Hand / Upper-body ROI Threat Scanning (Pose Keypoint Guided Pipeline)
        # Sequence: PERSON DETECTION -> PERSON TRACKING -> POSE / HAND KEYPOINTS -> HAND ROI -> THREAT DETECTOR -> PERSON ASSOCIATION
        person_detections = [d for d in detections if d.get("category") == "person"]
        if ENABLE_HAND_ROI_THREATS and self.pose_model is not None and self.threat_model is not None and person_detections:
            try:
                pose_results = self.pose_model.predict(frame, imgsz=384, conf=0.30, verbose=False)
                if pose_results and pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
                    kps_array = pose_results[0].keypoints.xy.cpu().numpy()  # [N, 17, 2]

                    for p in person_detections:
                        px1, py1, px2, py2 = p["bbox"]
                        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                        pw, ph = px2 - px1, py2 - py1
                        p_track_id = p.get("track_id", -1)

                        # Match closest pose skeleton to this detected person
                        best_kps = None
                        min_kps_dist = float("inf")
                        for kps in kps_array:
                            nx, ny = kps[0]  # Nose
                            if px1 - 25 <= nx <= px2 + 25 and py1 - 25 <= ny <= py2 + 25:
                                dist = math.hypot(pcx - nx, pcy - ny)
                                if dist < min_kps_dist:
                                    min_kps_dist = dist
                                    best_kps = kps

                        if best_kps is not None:
                            # Left wrist = 9, Right wrist = 10
                            for hand_idx, hand_name in [(9, "left"), (10, "right")]:
                                wx, wy = best_kps[hand_idx]
                                if wx > 5 and wy > 5:
                                    roi_size = max(70, min(140, int(ph * 0.35)))
                                    rx1 = max(0, int(wx - roi_size // 2))
                                    ry1 = max(0, int(wy - roi_size // 2))
                                    rx2 = min(frame_w, int(wx + roi_size // 2))
                                    ry2 = min(frame_h, int(wy + roi_size // 2))

                                    hand_crop = frame[ry1:ry2, rx1:rx2]
                                    if hand_crop.size > 0 and (rx2 - rx1) > 20 and (ry2 - ry1) > 20:
                                        h_res = self.threat_model.predict(hand_crop, imgsz=192, conf=0.20, verbose=False)
                                        for hr in h_res:
                                            if hr.boxes is not None:
                                                for hb in hr.boxes:
                                                    h_cls_id = int(hb.cls[0])
                                                    h_cls_name = self.threat_model.names.get(h_cls_id, "threat").lower()
                                                    h_conf = float(hb.conf[0])
                                                    hx1, hy1, hx2, hy2 = [float(v) for v in hb.xyxy[0]]

                                                    # Map back to full frame coordinates
                                                    gx1 = rx1 + hx1
                                                    gy1 = ry1 + hy1
                                                    gx2 = rx1 + hx2
                                                    gy2 = ry1 + hy2

                                                    threat_label = f"POSSIBLE {h_cls_name.upper()} (HELD IN {hand_name.upper()} HAND)"
                                                    threat_candidates.append({
                                                        "bbox": [gx1, gy1, gx2, gy2],
                                                        "class": h_cls_name,
                                                        "class_id": 900 + h_cls_id,
                                                        "category": "threat",
                                                        "confidence": round(h_conf, 2),
                                                        "track_id": -1,
                                                        "is_threat": True,
                                                        "threat_label": threat_label,
                                                        "severity": "critical",
                                                        "held_by_person_id": p_track_id,
                                                        "associated_hand": hand_name,
                                                        "is_hand_held": True,
                                                    })
                                                    p["has_threat"] = True
                                                    p["hand_threat"] = threat_label
            except Exception:
                pass

        # ③ Specialized Full-Frame Threat Model (Gun, Explosion, Grenade, Knife)
        if self.threat_model is not None:
            try:
                threat_results = self.threat_model.predict(
                    frame,
                    imgsz=YOLO_IMGSZ,
                    conf=0.25,  # Query candidate boxes
                    verbose=False,
                )
                for tr in threat_results:
                    if tr.boxes is None:
                        continue
                    for box in tr.boxes:
                        cls_id = int(box.cls[0])
                        raw_cls_name = self.threat_model.names.get(cls_id, "unknown").lower()
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

                        # Standardize threat label & severity
                        meta = THREAT_CLASSES.get(raw_cls_name, {})
                        threat_label = meta.get("label", raw_cls_name.upper())
                        severity = meta.get("severity", "critical")
                        min_conf = meta.get("min_conf", 0.30)

                        # Filter 1: Calibrated Confidence Threshold
                        if conf < min_conf:
                            continue

                        w = x2 - x1
                        h = y2 - y1
                        area = w * h

                        # Filter 2: Scale & Dimension Sanity
                        # EXPLOSIONS have no upper area limit! An explosion can engulf the scene!
                        if raw_cls_name == "explosion":
                            if w < 20 or h < 20:
                                continue
                            threat_label = "EXPLOSION / BLAST"
                            severity = "critical"
                        elif raw_cls_name == "grenade":
                            # Grenades can be small when airborne/thrown (min 10x10)
                            if w < 10 or h < 10 or area < 100:
                                continue
                            # Landscape line rejection
                            if w > 0.35 * frame_w or h > 0.35 * frame_h or area > 0.08 * (frame_w * frame_h):
                                continue
                        elif raw_cls_name in ("gun", "pistol", "rifle", "firearm"):
                            if w < 12 or h < 12 or area < 130:
                                continue
                            if w > 0.45 * frame_w or h > 0.50 * frame_h or area > 0.15 * (frame_w * frame_h):
                                continue
                        elif raw_cls_name == "knife":
                            if w < 12 or h < 12 or area < 130:
                                continue
                            if w > 0.35 * frame_w or h > 0.40 * frame_h or area > 0.10 * (frame_w * frame_h):
                                continue

                        # Filter 3: Negative Cross-Suppression (against everyday sports balls, bottles, phones)
                        is_false_alarm = False
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0

                        for b in benign_items:
                            bx1, by1, bx2, by2 = b["bbox"]
                            b_area = (bx2 - bx1) * (by2 - by1)
                            # Box IoU overlap
                            ix1 = max(x1, bx1)
                            iy1 = max(y1, by1)
                            ix2 = min(x2, bx2)
                            iy2 = min(y2, by2)
                            if ix2 > ix1 and iy2 > iy1:
                                inter = (ix2 - ix1) * (iy2 - iy1)
                                union = area + b_area - inter
                                # Only suppress if benign item strongly overlaps AND threat confidence is not super high
                                if union > 0 and (inter / union) >= 0.40 and conf < 0.65:
                                    is_false_alarm = True
                                    break

                        if is_false_alarm:
                            continue

                        threat_candidates.append({
                            "bbox": [x1, y1, x2, y2],
                            "class": raw_cls_name,
                            "class_id": 900 + cls_id,
                            "category": "threat",
                            "confidence": round(conf, 2),
                            "track_id": -1,
                            "is_threat": True,
                            "threat_label": threat_label,
                            "severity": severity,
                        })
            except Exception:
                pass

        # Update Projectile & Thrown Grenade Motion Tracker
        if now is None:
            now = time.time()
        threat_candidates = self.projectile_tracker.update(threat_candidates, now)

        # ③ Detect grenade disappearances → feed to ExplosionDetector
        current_grenade_track_ids: set[int] = set()
        for tc in threat_candidates:
            ptid = tc.get("projectile_track_id")
            if ptid is not None and tc.get("class", "").lower() == "grenade":
                current_grenade_track_ids.add(ptid)
        # Grenades that were tracked last frame but vanished this frame = possible detonation
        disappeared_grenades = self._prev_grenade_tracks - current_grenade_track_ids
        for ptid in disappeared_grenades:
            trk = self.projectile_tracker.tracks.get(ptid)
            if trk and isinstance(trk, dict):
                hist = trk.get("history", [])
                if hist and len(hist) >= 2:
                    last_x, last_y, _ = hist[-1]
                    self.explosion_detector.record_grenade_disappearance(last_x, last_y, now)
        self._prev_grenade_tracks = current_grenade_track_ids

        # ④ Temporal Confirmation Engine: filter threat candidates through multi-frame persistence
        # Extract person detections for spatial context
        person_dets = [d for d in detections if d.get("category") == "person"]
        verified_threats = self.temporal_engine.update(threat_candidates, now, persons=person_dets)

        # ⑤ Explosion / Blast / Smoke Event Analysis (luminance + motion independent of YOLO)
        explosion_events = self.explosion_detector.analyze_frame(frame, verified_threats, now)
        for evt in explosion_events:
            # Avoid duplicate if YOLO already found an explosion that was temporally confirmed
            already_has_explosion = any(
                t.get("class") == "explosion" for t in verified_threats
            )
            if not already_has_explosion:
                evt["track_id"] = -1
                verified_threats.append(evt)

        # ⑥ Merge verified threat candidates with primary detections (avoiding duplicates)
        for tc in verified_threats:
            x1, y1, x2, y2 = tc["bbox"]
            area = (x2 - x1) * (y2 - y1)
            is_duplicate = False

            for d in detections:
                if d.get("category") == "threat":
                    dx1, dy1, dx2, dy2 = d["bbox"]
                    ix1 = max(x1, dx1)
                    iy1 = max(y1, dy1)
                    ix2 = min(x2, dx2)
                    iy2 = min(y2, dy2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = area + ((dx2 - dx1) * (dy2 - dy1)) - inter
                        if union > 0 and (inter / union) > 0.35:
                            is_duplicate = True
                            if tc["confidence"] > d["confidence"]:
                                d["confidence"] = tc["confidence"]
                                d["threat_label"] = tc["threat_label"]
                                d["severity"] = tc["severity"]
                                if "is_thrown" in tc:
                                    d["is_thrown"] = tc["is_thrown"]
                                    d["trajectory"] = tc.get("trajectory", [])
                                    d["velocity"] = tc.get("velocity", 0.0)
                            break
            if not is_duplicate:
                detections.append(tc)

        return detections

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        detections: list[dict],
        intrusions: list[dict] | None = None,
        fence_zones: list[dict] | None = None,
        crowds: list[dict] | None = None,
        plates: list[dict] | None = None,
        camera_id: str = "default",
        anpr_engine: Any = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes, threat markers, projectile trajectories, tracking IDs,
        license plates, crowd clusters, and fence zones on the frame.
        """
        annotated = frame.copy()
        intrusion_track_ids = set()

        if intrusions:
            intrusion_track_ids = {
                intr["detection"]["track_id"] for intr in intrusions
            }

        # ── 1. Draw crowd clusters (subtle hull glow & group labels) ──
        if crowds:
            try:
                annotated = CrowdDetector.annotate_crowds(annotated, crowds)
            except Exception:
                pass

        # ── 2. Draw fence zones ────────────────────────────────────
        if fence_zones:
            for zone in fence_zones:
                pts = np.array(zone["points"], dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                label = zone.get("name", "Zone")
                cv2.putText(
                    annotated, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
                )

        # ── 3. Draw detections (Humans, Vehicles, Threats, Projectiles) ──
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class"]
            track_id = det["track_id"]
            conf = det["confidence"]
            category = det.get("category", "object")
            is_threat = (category == "threat") or det.get("is_threat", False)
            is_intrusion = track_id in intrusion_track_ids
            in_crowd = det.get("in_crowd", False)
            crowd_id = det.get("crowd_id", 0)
            is_thrown = det.get("is_thrown", False)
            trajectory = det.get("trajectory", [])

            # Dynamic Thrown Weapon / Grenade Trajectory Rendering
            if is_threat and len(trajectory) >= 2:
                # Draw ballistic flight trail
                for i in range(1, len(trajectory)):
                    pt1 = (int(trajectory[i - 1][0]), int(trajectory[i - 1][1]))
                    pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                    # Fading trail from thin to thick
                    alpha_thick = max(1, int(i * 3 / len(trajectory)))
                    cv2.line(annotated, pt1, pt2, (0, 0, 255), alpha_thick, cv2.LINE_AA)
                    cv2.circle(annotated, pt2, 3, (0, 165, 255), -1, cv2.LINE_AA)

            if is_threat:
                colour = COLOUR_THREAT
                thickness = 3
                threat_label = det.get("threat_label", cls_name.upper())
                vel_str = f" [V={det.get('velocity', 0):.0f}px/s]" if is_thrown else ""
                tag_label = f"[THREAT] {threat_label} {conf:.0%}{vel_str}"
                text_colour = (255, 255, 255)
            elif is_intrusion:
                colour = COLOUR_INTRUSION
                thickness = 3
                tag_label = f"ALERT #{track_id} {conf:.0%}" if track_id >= 0 else f"ALERT {conf:.0%}"
                text_colour = (255, 255, 255)
            elif category == "person":
                colour = COLOUR_HUMAN
                thickness = 2
                if in_crowd:
                    crowd_idx = det.get("crowd_index", 1)
                    crowd_tot = det.get("crowd_total", 0)
                    tag_label = f"HUMAN #{track_id} [CROWD #{crowd_id} {crowd_idx}/{crowd_tot}]" if track_id >= 0 else f"HUMAN [CROWD #{crowd_id}]"
                else:
                    tag_label = f"HUMAN #{track_id} {conf:.0%}" if track_id >= 0 else f"HUMAN {conf:.0%}"
                text_colour = (0, 0, 0)
            elif category == "vehicle":
                colour = COLOUR_VEHICLE
                thickness = 2
                tag_label = f"{cls_name.upper()} #{track_id} {conf:.0%}" if track_id >= 0 else f"{cls_name.upper()} {conf:.0%}"
                text_colour = (0, 0, 0)
            else:
                colour = COLOUR_ITEM
                thickness = 2
                tag_label = f"{cls_name.upper()} {conf:.0%}"
                text_colour = (0, 0, 0)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)

            # Tactical corner brackets
            corner_len = min(16, max(5, (x2 - x1) // 5), max(5, (y2 - y1) // 5))
            if corner_len > 4:
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), colour, thickness + 1)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), colour, thickness + 1)
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), colour, thickness + 1)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), colour, thickness + 1)
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), colour, thickness + 1)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), colour, thickness + 1)
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), colour, thickness + 1)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), colour, thickness + 1)

            # Label pill background
            (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            tag_y1 = max(0, y1 - th - 8)
            tag_y2 = y1
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 8, tag_y2), colour, -1)
            cv2.putText(
                annotated, tag_label, (x1 + 4, tag_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_colour, 1, cv2.LINE_AA
            )

            # Warning banner below bounding box
            if is_threat:
                banner_txt = "[!] GRENADE THROWING TRACKED" if is_thrown else "[!] HARMFUL OBJECT DETECTED"
                if cls_name == "explosion":
                    banner_txt = "[!] BLAST / EXPLOSION DETECTED"
                cv2.putText(
                    annotated, banner_txt, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOUR_THREAT, 2, cv2.LINE_AA
                )
            elif is_intrusion:
                cv2.putText(
                    annotated, "[!] INTRUSION ZONE", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOUR_INTRUSION, 2, cv2.LINE_AA
                )

        # ── 4. Draw license plates & 10s side plate callout ──────
        if plates:
            try:
                annotated = ANPREngine.annotate_plates(annotated, plates)
            except Exception:
                pass

        if anpr_engine is not None and hasattr(anpr_engine, "draw_side_plate_callouts"):
            try:
                annotated = anpr_engine.draw_side_plate_callouts(annotated, camera_id)
            except Exception:
                pass

        # ── 5. Tactical HUD overlay (top-left detection stats) ───
        person_count = sum(1 for d in detections if d["category"] == "person")
        vehicle_count = sum(1 for d in detections if d["category"] == "vehicle")
        threat_count = sum(1 for d in detections if d.get("category") == "threat" or d.get("is_threat", False))
        crowd_count = len(crowds) if crowds else 0
        plate_count = len(plates) if plates else 0

        hud_h = 70
        if intrusions:
            hud_h += 20
        if threat_count > 0:
            hud_h += 22
        if crowd_count > 0:
            hud_h += 20
        if plate_count > 0:
            hud_h += 20

        hud_w = 230
        hud_roi = annotated[8:8 + hud_h, 8:8 + hud_w]
        bg = np.zeros_like(hud_roi)
        bg[:] = (8, 12, 22)
        cv2.addWeighted(bg, 0.75, hud_roi, 0.25, 0, hud_roi)

        border_color = (0, 0, 255) if threat_count > 0 else (0, 255, 136)
        cv2.rectangle(annotated, (8, 8), (8 + hud_w, 8 + hud_h), border_color, 1 if threat_count == 0 else 2)

        cv2.putText(annotated, "SURVEILLANCE HUD", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 180), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"[+] HUMANS:   {person_count}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"[*] VEHICLES: {vehicle_count}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

        y_offset = 84
        if plate_count > 0:
            cv2.putText(annotated, f"[#] PLATES:   {plate_count}", (16, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 230, 0), 2, cv2.LINE_AA)
            y_offset += 20
        if crowd_count > 0:
            cv2.putText(annotated, f"[@] CROWDS:   {crowd_count}", (16, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 2, cv2.LINE_AA)
            y_offset += 20
        if threat_count > 0:
            cv2.putText(annotated, f"[!] THREATS:  {threat_count}", (16, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2, cv2.LINE_AA)
            y_offset += 20
        if intrusions:
            cv2.putText(annotated, f"[!] INTRUSIONS: {len(intrusions)}", (16, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)

        return annotated
