"""
IBVAP Alert Engine
Manages tactical surveillance alert creation, deduplication, cooldown,
snapshot capture, and real-time notification dispatch.
"""
import time
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any

from config import ALERT_COOLDOWN_SECONDS, SNAPSHOTS_DIR


class AlertEngine:
    """Creates, deduplicates, and dispatches surveillance and behavioral alerts."""

    def __init__(self, db):
        self.db = db
        # Cooldown tracker: (camera_id, alert_type, target_id) → last_alert_timestamp
        self._cooldowns: Dict[Tuple[str, str, Any], float] = {}
        # Callback for real-time push (wired to WebSocket broadcaster)
        self.on_new_alert: Optional[Callable] = None

    def _is_on_cooldown(
        self,
        camera_id: str,
        alert_type: str,
        target_id: Any = None,
        cooldown_seconds: float = ALERT_COOLDOWN_SECONDS,
    ) -> bool:
        key = (camera_id, alert_type, target_id)
        last = self._cooldowns.get(key, 0)
        return (time.time() - last) < cooldown_seconds

    def _update_cooldown(self, camera_id: str, alert_type: str, target_id: Any = None):
        key = (camera_id, alert_type, target_id)
        self._cooldowns[key] = time.time()

    def process_intrusions(
        self,
        camera_id: str,
        intrusions: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
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
            track_id = det.get("track_id", -1)
            alert_type = f"intrusion_{det['category']}"

            if self._is_on_cooldown(camera_id, alert_type, target_id=track_id):
                continue

            snapshot_path = ""
            if frame is not None:
                snapshot_path = self._save_snapshot(camera_id, frame)

            severity = "critical" if det["category"] == "person" else "high"
            message = (
                f"{det['class'].upper()} detected in {zone_name} "
                f"(Track #{track_id}, Confidence {det['confidence']:.0%})"
            )

            alert = self.db.add_alert(
                camera_id=camera_id,
                alert_type=alert_type,
                message=message,
                severity=severity,
                details={
                    "class": det["class"],
                    "category": det["category"],
                    "track_id": track_id,
                    "confidence": det["confidence"],
                    "zone_id": intr["zone_id"],
                    "zone_name": zone_name,
                    "bbox": det["bbox"],
                    "centroid": det.get("centroid", []),
                },
                snapshot_path=snapshot_path,
            )

            new_alerts.append(alert)
            self._update_cooldown(camera_id, alert_type, target_id=track_id)

            if self.on_new_alert:
                self.on_new_alert(alert)

        return new_alerts

    def process_behavioral_events(
        self,
        camera_id: str,
        behavioral_events: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process tactical behavioral events:
        - LOITERING_DETECTED
        - CRAWLING_INFILTRATION
        - UNATTENDED_BAGGAGE
        - DIRECTIONAL_INGRESS
        """
        if not behavioral_events:
            return []

        new_alerts = []
        for event in behavioral_events:
            event_type = event["event_type"]
            track_id = event.get("track_id", -1)

            # Behavioral cooldowns: 15-20s per target to avoid alarm fatigue
            cooldown_sec = 20.0 if event_type in ("CRAWLING_INFILTRATION", "DIRECTIONAL_INGRESS") else 25.0
            if self._is_on_cooldown(camera_id, event_type, target_id=track_id, cooldown_seconds=cooldown_sec):
                continue

            snapshot_path = ""
            if frame is not None:
                snapshot_path = self._save_snapshot(camera_id, frame)

            alert = self.db.add_alert(
                camera_id=camera_id,
                alert_type=event_type,
                message=event["message"],
                severity=event.get("severity", "high"),
                details=event.get("details", {}),
                snapshot_path=snapshot_path,
            )

            new_alerts.append(alert)
            self._update_cooldown(camera_id, event_type, target_id=track_id)

            if self.on_new_alert:
                self.on_new_alert(alert)

        return new_alerts

    def process_detections(
        self,
        camera_id: str,
        detections: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create alerts for notable crowd gatherings (5+ persons detected).
        """
        new_alerts = []
        person_count = sum(1 for d in detections if d.get("category") == "person")

        # Alert: crowd detected (5+ persons)
        if person_count >= 5:
            alert_type = "crowd_detected"
            if not self._is_on_cooldown(camera_id, alert_type, cooldown_seconds=ALERT_COOLDOWN_SECONDS * 2):
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

    def process_frs_matches(
        self,
        camera_id: str,
        frs_matches: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create WATCHLIST_SUSPECT_DETECTED alerts when FRS detects a matched suspect.
        """
        if not frs_matches:
            return []

        new_alerts = []
        for match in frs_matches:
            target_id = match.get("target_id", "UNKNOWN")
            name = match.get("name", "Unknown Suspect")
            category = match.get("category", "suspect")
            threat = match.get("threat_level") or match.get("danger_level", "HIGH")
            is_threat = match.get("is_threat", True)

            # Record match in database for surveillance stats
            if hasattr(self.db, "record_frs_match"):
                try:
                    self.db.record_frs_match(target_id, f"Camera {camera_id}")
                except Exception:
                    pass

            # Only trigger high-priority alerts for suspects / threats (not authorized personnel)
            if not is_threat:
                continue

            alert_type = "WATCHLIST_SUSPECT_DETECTED"
            # 15s cooldown per suspect per camera
            if self._is_on_cooldown(camera_id, alert_type, target_id=target_id, cooldown_seconds=15.0):
                continue

            snapshot_path = self._save_snapshot(camera_id, frame) if frame is not None else ""
            severity = "critical" if threat in ("CRITICAL", "HIGH") else "high"
            conf = match.get("confidence_pct", match.get("similarity", 0.8) * 100)

            message = (
                f"WATCHLIST SUSPECT DETECTED: {name} "
                f"({category.upper()}, {conf:.0f}% biometric match, Target ID: {target_id})"
            )

            alert = self.db.add_alert(
                camera_id=camera_id,
                alert_type=alert_type,
                message=message,
                severity=severity,
                details={
                    "target_id": target_id,
                    "name": name,
                    "category": category,
                    "threat_level": threat,
                    "confidence": conf,
                    "notes": match.get("notes", ""),
                    "photo_url": match.get("photo_url", ""),
                },
                snapshot_path=snapshot_path,
            )

            new_alerts.append(alert)
            self._update_cooldown(camera_id, alert_type, target_id=target_id)

            if self.on_new_alert:
                self.on_new_alert(alert)

        return new_alerts

    def process_anpr_matches(
        self,
        camera_id: str,
        anpr_matches: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create BLACKLIST_VEHICLE_DETECTED alerts when ANPR matches a hotlist plate.
        """
        if not anpr_matches:
            return []

        new_alerts = []
        for match in anpr_matches:
            if not match.get("matched") and not match.get("is_hotlist"):
                continue

            hotlist_plate = match.get("hotlist_plate") or match.get("plate_number", "")
            detected_plate = match.get("detected_plate") or match.get("plate_number", hotlist_plate)
            reason = match.get("reason") or match.get("hotlist_reason", "Blacklisted Vehicle Alert")
            threat = match.get("threat_level") or match.get("danger_level", "CRITICAL")
            model = match.get("vehicle_model", "Vehicle")

            alert_type = "BLACKLIST_VEHICLE_DETECTED"
            # 15s cooldown per plate per camera
            if self._is_on_cooldown(camera_id, alert_type, target_id=hotlist_plate, cooldown_seconds=15.0):
                continue

            snapshot_path = self._save_snapshot(camera_id, frame) if frame is not None else ""
            severity = "critical" if threat in ("CRITICAL", "HIGH") else "high"
            conf = match.get("confidence_pct", match.get("similarity", 0.9) * 100)

            message = (
                f"BLACKLIST VEHICLE DETECTED: [{detected_plate}] "
                f"({model} — Reason: {reason}, Match: {conf:.0f}%)"
            )

            alert = self.db.add_alert(
                camera_id=camera_id,
                alert_type=alert_type,
                message=message,
                severity=severity,
                details={
                    "plate_number": detected_plate,
                    "hotlist_plate": hotlist_plate,
                    "reason": reason,
                    "vehicle_model": model,
                    "threat_level": threat,
                    "confidence": conf,
                },
                snapshot_path=snapshot_path,
            )

            new_alerts.append(alert)
            self._update_cooldown(camera_id, alert_type, target_id=hotlist_plate)

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
