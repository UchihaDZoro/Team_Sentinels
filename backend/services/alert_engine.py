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

from config import ALERT_COOLDOWN_SECONDS, SNAPSHOTS_DIR


class AlertEngine:
    """Creates, deduplicates, and dispatches surveillance alerts."""

    def __init__(self, db):
        self.db = db
        # Cooldown tracker: (camera_id, alert_type) → last_alert_timestamp
        self._cooldowns: dict[tuple[str, str], float] = {}
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

            # Save snapshot
            snapshot_path = ""
            if frame is not None:
                snapshot_path = self._save_snapshot(camera_id, frame)

            severity = "critical" if det["category"] == "person" else "high"
            message = (
                f"{det['class'].upper()} detected in {zone_name} "
                f"(Track #{det['track_id']}, Confidence {det['confidence']:.0%})"
            )

            alert = self.db.add_alert(
                camera_id=camera_id,
                alert_type=alert_type,
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

    def process_detections(
        self,
        camera_id: str,
        detections: list[dict],
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Create alerts for notable detections (e.g. many persons detected).
        Light-touch — avoids spamming.
        """
        new_alerts = []
        person_count = sum(1 for d in detections if d["category"] == "person")
        vehicle_count = sum(1 for d in detections if d["category"] == "vehicle")

        # Alert: crowd detected (5+ persons)
        if person_count >= 5:
            alert_type = "crowd_detected"
            if not self._is_on_cooldown(camera_id, alert_type):
                snapshot_path = self._save_snapshot(camera_id, frame) if frame is not None else ""
                alert = self.db.add_alert(
                    camera_id=camera_id,
                    alert_type=alert_type,
                    message=f"Crowd detected: {person_count} persons visible",
                    severity="high",
                    details={"person_count": person_count},
                    snapshot_path=snapshot_path,
                )
                new_alerts.append(alert)
                self._update_cooldown(camera_id, alert_type)
                if self.on_new_alert:
                    self.on_new_alert(alert)

        return new_alerts

    def _save_snapshot(self, camera_id: str, frame: np.ndarray) -> str:
        """Save a JPEG snapshot and return the relative path."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{camera_id}_{ts}.jpg"
        filepath = SNAPSHOTS_DIR / filename
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return str(filepath)
