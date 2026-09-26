"""
IBVAP Alert Engine
Manages alert creation, deduplication, cooldown, and notification dispatch.
"""
import time
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Callable

import uuid
from config import ALERT_COOLDOWN_SECONDS, SNAPSHOTS_DIR, EVENTS_DIR


class AlertEngine:
    """
    Standardized Enterprise Alert Engine for IBVAP.
    - Manages alert creation, severity classification, and notification dispatch.
    - Continuous object debouncing (tracks duration without duplicate spamming).
    - Multi-crop structured evidence archiving (full_frame, object_crop, metadata.json).
    - Automatic rolling video clip export for CRITICAL and HIGH severity events.
    """

    def __init__(self, db, stream_manager=None):
        self.db = db
        self.stream_manager = stream_manager
        # Cooldown tracker: (camera_id, alert_type) → last_alert_timestamp
        self._cooldowns: dict[tuple[str, str], float] = {}
        # Active object tracker for debouncing: (camera_id, object_key) -> dict
        self._active_events: dict[tuple[str, str], dict] = {}
        # Temporal persistence tracker: (camera_id, threat_class) -> (seen_count, last_seen_time)
        self._threat_persistence: dict[tuple[str, str], int] = {}
        self._last_threat_time: dict[tuple[str, str], float] = {}
        # Callback for real-time push (set by main.py)
        self.on_new_alert: Callable | None = None

    def _is_on_cooldown(self, camera_id: str, alert_type: str) -> bool:
        key = (camera_id, alert_type)
        last = self._cooldowns.get(key, 0)
        return (time.time() - last) < ALERT_COOLDOWN_SECONDS

    def _update_cooldown(self, camera_id: str, alert_type: str):
        self._cooldowns[(camera_id, alert_type)] = time.time()

    def process_intrusions(
        self,
        camera_id: str,
        intrusions: list[dict],
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Create alerts for zone intrusions, respecting cooldown.
        Returns list of newly created alert dicts.
        """
        if not intrusions:
            return []

        new_alerts = []
        for intr in intrusions:
            det = intr["detection"]
            zone_name = intr["zone_name"]
            alert_type = f"intrusion_{det['category']}"

            if self._is_on_cooldown(camera_id, alert_type):
                continue

            # Save snapshot with structured evidence archiving
            severity = "critical" if det["category"] == "person" else "high"
            snapshot_path = ""
            if frame is not None:
                snapshot_path = self._save_snapshot(
                    camera_id=camera_id,
                    frame=frame,
                    event_type="VIRTUAL_FENCE_BREACH",
                    bbox=det.get("bbox"),
                    metadata={"zone_name": zone_name, "track_id": det.get("track_id")},
                    severity=severity,
                )

            message = (
                f"{det['class'].upper()} detected in {zone_name} "
                f"(Track #{det['track_id']}, Confidence {det['confidence']:.0%})"
            )

            alert = self.db.add_alert(
                camera_id=camera_id,
                alert_type="VIRTUAL_FENCE_BREACH",
                message=message,
                severity=severity,
                details={
                    "class": det["class"],
                    "category": det["category"],
                    "track_id": det["track_id"],
                    "confidence": det["confidence"],
                    "zone_id": intr["zone_id"],
                    "zone_name": zone_name,
                    "bbox": det["bbox"],
                },
                snapshot_path=snapshot_path,
            )

            new_alerts.append(alert)
            self._update_cooldown(camera_id, alert_type)

            # Real-time push
            if self.on_new_alert:
                self.on_new_alert(alert)

        return new_alerts

    def process_threats(
        self,
        camera_id: str,
        detections: list[dict],
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Create critical alerts for detected harmful/suspicious threat objects (weapons, guns, bombs, knives).
        Enforces temporal persistence buffer: requires >= 2 consecutive frame detections or high confidence
        to prevent single-frame false alarms from tossed stones, balls, or camera artifacts.
        """
        threats = [d for d in detections if d.get("category") == "threat" or d.get("is_threat", False)]
        if not threats:
            return []

        now = time.time()
        new_alerts = []

        for t in threats:
            raw_cls = t.get("class", "weapon").lower()
            threat_name = t.get("threat_label", raw_cls).upper()
            key = (camera_id, raw_cls)

            # Temporal persistence verification
            last_seen = self._last_threat_time.get(key, 0)
            if (now - last_seen) < 1.0:
                self._threat_persistence[key] = self._threat_persistence.get(key, 0) + 1
            else:
                self._threat_persistence[key] = 1
            self._last_threat_time[key] = now

            # Explosions, thrown grenades, and confident threats (>=0.40) trigger INSTANT ALERTS!
            # Only marginal borderline detections (<0.40) require multi-frame confirmation
            is_instant = (raw_cls == "explosion") or t.get("is_thrown", False) or (t["confidence"] >= 0.40)
            if not is_instant and self._threat_persistence[key] < 2:
                continue

            alert_type = f"threat_{raw_cls}"
            if self._is_on_cooldown(camera_id, alert_type):
                continue

            severity = t.get("severity", "critical")
            if raw_cls in ("gun", "pistol", "rifle", "firearm", "handgun"):
                std_event_type = "FIREARM_DETECTED"
            elif raw_cls in ("knife", "dagger", "blade"):
                std_event_type = "KNIFE_DETECTED"
            elif raw_cls == "grenade":
                std_event_type = "GRENADE_DETECTED"
            elif raw_cls == "explosion":
                std_event_type = "EXPLOSION_DETECTED"
            elif raw_cls == "smoke":
                std_event_type = "SMOKE_DETECTED"
            else:
                std_event_type = "DANGEROUS_OBJECT"

            snapshot_path = ""
            if frame is not None:
                snapshot_path = self._save_snapshot(
                    camera_id=camera_id,
                    frame=frame,
                    event_type=std_event_type,
                    bbox=t.get("bbox"),
                    metadata={"threat": threat_name, "is_thrown": t.get("is_thrown", False)},
                    severity=severity,
                )

            thrown_suffix = f" [THROWING TRACKED - V={t.get('velocity', 0):.0f}px/s]" if t.get("is_thrown") else ""
            message = f"CRITICAL SECURITY THREAT: {threat_name}{thrown_suffix} detected (Confidence {t['confidence']:.0%})"

            alert = self.db.add_alert(
                camera_id=camera_id,
                alert_type=std_event_type,
                message=message,
                severity=severity,
                details={
                    "threat_type": raw_cls,
                    "threat_label": threat_name,
                    "confidence": t["confidence"],
                    "bbox": t["bbox"],
                    "is_thrown": t.get("is_thrown", False),
                    "velocity": t.get("velocity", 0.0),
                },
                snapshot_path=snapshot_path,
            )
            new_alerts.append(alert)
            self._update_cooldown(camera_id, alert_type)
            if self.on_new_alert:
                self.on_new_alert(alert)

        return new_alerts

    def process_crowds(
        self,
        camera_id: str,
        crowds: list[dict],
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Create alerts for spatial crowd clusters (gatherings, dense congregations).
        """
        if not crowds:
            return []

        new_alerts = []
        for c in crowds:
            alert_type = "crowd_cluster"
            if self._is_on_cooldown(camera_id, alert_type):
                continue

            severity = c.get("severity", "high")
            count = c["count"]
            density = c.get("density_label", "CROWD")
            snapshot_path = ""
            if frame is not None:
                snapshot_path = self._save_snapshot(
                    camera_id=camera_id,
                    frame=frame,
                    event_type="CROWD_DETECTED",
                    bbox=c.get("bbox"),
                    metadata={"crowd_count": count, "density": density},
                    severity=severity,
                )

            message = f"CROWD ALERT: {count} persons gathered in close proximity ({density})"

            alert = self.db.add_alert(
                camera_id=camera_id,
                alert_type="CROWD_DETECTED",
                message=message,
                severity=severity,
                details={
                    "crowd_count": count,
                    "density_label": density,
                    "bbox": c["bbox"],
                },
                snapshot_path=snapshot_path,
            )
            new_alerts.append(alert)
            self._update_cooldown(camera_id, alert_type)
            if self.on_new_alert:
                self.on_new_alert(alert)

        return new_alerts

    def process_plates(
        self,
        camera_id: str,
        plates: list[dict],
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Process recognized license plates and emit alerts / live notifications.
        """
        if not plates:
            return []

        new_alerts = []
        for p in plates:
            plate_text = p.get("plate_text", "").strip()
            if not plate_text or len(plate_text) < 4:
                continue

            alert_type = f"plate_{plate_text.replace(' ', '')}"
            if self._is_on_cooldown(camera_id, alert_type):
                continue

            snapshot_path = self._save_snapshot(camera_id, frame) if frame is not None else ""
            message = f"VEHICLE IDENTIFIED: Plate [{plate_text}] on {p.get('vehicle_type', 'vehicle').upper()}"

            alert = self.db.add_alert(
                camera_id=camera_id,
                alert_type="plate_detected",
                message=message,
                severity="low",
                details={
                    "plate_number": plate_text,
                    "vehicle_type": p.get("vehicle_type", "vehicle"),
                    "track_id": p.get("track_id", -1),
                    "confidence": p.get("ocr_confidence", 0.8),
                    "bbox": p.get("bbox", []),
                },
                snapshot_path=snapshot_path,
            )
            new_alerts.append(alert)
            self._update_cooldown(camera_id, alert_type)
            if self.on_new_alert:
                self.on_new_alert(alert)

        return new_alerts

    def process_detections(
        self,
        camera_id: str,
        detections: list[dict],
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Legacy detection handler (crowd handling moved to process_crowds).
        """
        return []

    def _save_snapshot(
        self,
        camera_id: str,
        frame: np.ndarray,
        event_type: str = "alert",
        bbox: list | None = None,
        metadata: dict | None = None,
        severity: str = "medium",
    ) -> str:
        """
        Saves standard JPEG snapshot for immediate web dashboard display,
        and saves complete structured evidence folder with full_frame.jpg, object_crop.jpg,
        metadata.json, and triggers rolling MP4 clip recording for CRITICAL/HIGH alerts.
        """
        ts_now = datetime.now()
        ts_str = ts_now.strftime("%Y%m%d_%H%M%S_%f")
        date_str = ts_now.strftime("%Y-%m-%d")
        time_str = ts_now.strftime("%H%M%S")
        event_id = str(uuid.uuid4())[:8]

        # 1. Dashboard snapshot in SNAPSHOTS_DIR
        filename = f"{camera_id}_{ts_str}.jpg"
        filepath = SNAPSHOTS_DIR / filename
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])

        # 2. Structured evidence vault folder
        event_folder = EVENTS_DIR / camera_id / date_str / f"{time_str}_{event_type}_{event_id}"
        try:
            event_folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(event_folder / "full_frame.jpg"), frame)

            # Save object crop if bounding box available
            if bbox and len(bbox) == 4:
                bx1, by1, bx2, by2 = [int(v) for v in bbox]
                h, w = frame.shape[:2]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                if (bx2 - bx1) > 5 and (by2 - by1) > 5:
                    crop = frame[by1:by2, bx1:bx2]
                    cv2.imwrite(str(event_folder / "object_crop.jpg"), crop)

            # Metadata JSON
            meta = {
                "event_id": event_id,
                "camera_id": camera_id,
                "event_type": event_type,
                "severity": severity,
                "timestamp": ts_now.isoformat(),
                "bbox": bbox or [],
                "details": metadata or {},
            }
            with open(event_folder / "metadata.json", "w") as mf:
                json.dump(meta, mf, indent=2)

            # 3. Trigger rolling MP4 clip capture if critical or high severity
            if severity.lower() in ("critical", "high") and self.stream_manager is not None:
                clip_path = str(event_folder / "incident_clip.mp4")
                self.stream_manager.save_evidence_clip(
                    camera_id=camera_id,
                    output_path=clip_path,
                    pre_seconds=4.0,
                    post_seconds=6.0,
                )
        except Exception as e:
            print(f"[IBVAP AlertEngine] Evidence vault notice: {e}")

        return str(filepath)

