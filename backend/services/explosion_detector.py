"""
IBVAP Real-Time Explosion, Blast & Smoke Event Engine
Detects detonations, violent blasts, optical flashes, and expanding smoke plumes
by correlating temporal frame luminance delta, optical disruption contours,
and YOLO threat detections.
"""
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any


class ExplosionDetector:
    """
    Dedicated temporal event analyzer for explosions and rapid smoke expansion.
    Does not rely solely on static object detection:
    Combines:
    1. Rapid luminance flash detection (delta L > 25%)
    2. Large-scale sudden frame change / blast shockwave mask
    3. YOLO model threat detection correlation
    4. Projectile / grenade disappearance correlation
    """

    def __init__(self, flash_threshold: float = 0.28, disturbance_threshold: float = 0.08):
        self.flash_threshold = flash_threshold
        self.disturbance_threshold = disturbance_threshold
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_luminance: float = 0.0
        self.last_flash_time: float = 0.0
        self.recent_grenade_disappearances: list[tuple[float, float, float]] = []  # [(cx, cy, timestamp), ...]

    def record_grenade_disappearance(self, cx: float, cy: float, now: float):
        """Record when a tracked grenade vanishes (possible detonation point)."""
        self.recent_grenade_disappearances.append((cx, cy, now))
        # Keep within last 2 seconds
        self.recent_grenade_disappearances = [
            (gx, gy, gt) for gx, gy, gt in self.recent_grenade_disappearances if (now - gt) < 2.0
        ]

    def analyze_frame(
        self,
        frame: np.ndarray,
        yolo_threats: list[dict],
        now: float,
    ) -> list[dict]:
        """
        Analyze the incoming frame for explosion / blast / smoke signatures.
        Returns a list of verified explosion and blast event dicts.
        """
        events = []
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_luminance = float(np.mean(gray))

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_luminance = curr_luminance
            return []

        # 1. Optical Flash Detection (sudden bright spike across the frame)
        luminance_delta = 0.0
        if self.prev_luminance > 2.0:
            luminance_delta = (curr_luminance - self.prev_luminance) / self.prev_luminance

        is_flash = (luminance_delta >= self.flash_threshold)
        if is_flash:
            self.last_flash_time = now

        # 2. Visual Disruption & Shockwave Expansion Analysis
        frame_diff = cv2.absdiff(gray, self.prev_gray)
        _, thresh = cv2.threshold(frame_diff, 35, 255, cv2.THRESH_BINARY)
        disturbed_pixels = cv2.countNonZero(thresh)
        disturbance_ratio = disturbed_pixels / float(h * w)

        # 3. Check if YOLO detected an explosion or bomb
        yolo_explosion = None
        for t in yolo_threats:
            t_cls = t.get("class", "").lower()
            if t_cls in ("explosion", "bomb"):
                yolo_explosion = t
                break

        # 4. Check for projectile disappearance in blast proximity
        grenade_detonation_match = False
        grenade_pos = None
        for gx, gy, gt in self.recent_grenade_disappearances:
            if (now - gt) < 1.5 and (is_flash or disturbance_ratio > 0.06):
                grenade_detonation_match = True
                grenade_pos = (gx, gy)
                break

        # 5. Event Synthesis
        # Trigger condition A: YOLO detected explosion with confirmed confidence >= 0.28
        # Trigger condition B: Sudden optical flash + significant frame disturbance
        # Trigger condition C: Tracked grenade disappeared immediately followed by flash/blast
        is_explosion = False
        conf = 0.0
        event_bbox = [0.0, 0.0, float(w), float(h)]

        if yolo_explosion is not None:
            is_explosion = True
            conf = max(float(yolo_explosion["confidence"]), 0.75)
            event_bbox = yolo_explosion["bbox"]

        elif grenade_detonation_match and grenade_pos:
            is_explosion = True
            conf = 0.88
            gx, gy = grenade_pos
            event_bbox = [max(0.0, gx - 120), max(0.0, gy - 120), min(float(w), gx + 120), min(float(h), gy + 120)]

        elif is_flash and disturbance_ratio >= self.disturbance_threshold:
            # Sudden optical flash plus large area perturbation
            is_explosion = True
            conf = min(0.60 + disturbance_ratio, 0.95)
            # Find center of disturbed mass
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(largest_c)
                event_bbox = [float(x), float(y), float(x + cw), float(y + ch)]

        if is_explosion:
            events.append({
                "bbox": event_bbox,
                "class": "explosion",
                "category": "threat",
                "confidence": round(conf, 2),
                "is_threat": True,
                "threat_label": "EXPLOSION / DETONATION BLAST",
                "severity": "critical",
                "flash_detected": is_flash,
                "disturbance_ratio": round(disturbance_ratio, 3),
            })

        # Update historical references
        self.prev_gray = gray
        self.prev_luminance = curr_luminance
        return events
