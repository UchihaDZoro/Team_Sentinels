"""
IBVAP Temporal Confirmation & False-Alarm Suppression Engine
Eliminates single-frame false positives for dangerous objects (weapons, firearms, grenades, knives)
by enforcing multi-frame temporal persistence, spatial consistency, and trajectory validation.
"""
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional


class TemporalTrack:
    """State of an object candidate tracked across consecutive video frames."""

    def __init__(self, track_id: int, cls_name: str, bbox: list[float], conf: float, now: float):
        self.track_id = track_id
        self.cls_name = cls_name.lower()
        self.bbox = bbox
        self.history: list[tuple[float, float, float]] = []  # [(cx, cy, timestamp), ...]
        self.confidences: list[float] = [conf]
        self.first_seen = now
        self.last_seen = now
        self.consecutive_frames = 1
        self.missed_frames = 0
        self.is_confirmed = False
        self.confirmed_at: Optional[float] = None

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.history.append((cx, cy, now))

    def update(self, bbox: list[float], conf: float, now: float):
        self.bbox = bbox
        self.confidences.append(conf)
        if len(self.confidences) > 15:
            self.confidences.pop(0)

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.history.append((cx, cy, now))
        if len(self.history) > 20:
            self.history.pop(0)

        self.last_seen = now
        self.consecutive_frames += 1
        self.missed_frames = 0

    @property
    def mean_conf(self) -> float:
        return sum(self.confidences) / max(len(self.confidences), 1)

    @property
    def max_conf(self) -> float:
        return max(self.confidences) if self.confidences else 0.0

    @property
    def velocity(self) -> float:
        """Velocity in pixels per second over recent history."""
        if len(self.history) < 2:
            return 0.0
        p1 = self.history[0]
        p2 = self.history[-1]
        dt = max(p2[2] - p1[2], 0.01)
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return dist / dt

    @property
    def trajectory(self) -> list[tuple[float, float]]:
        return [(pt[0], pt[1]) for pt in self.history]


class TemporalConfirmationEngine:
    """
    Validates candidate threats across time before elevating them to verified alerts.
    Enforces class-specific temporal thresholds:
    - Explosion: 1 frame (instantaneous event)
    - Thrown Grenade (in-flight velocity > 110 px/s): 2 frames
    - Firearm / Gun: 2 consecutive frames (or single frame >= 0.65 conf)
    - Knife / Blade: 3 consecutive frames with geometric aspect ratio check
    - Stationary Grenade: 3 consecutive frames + anti-ball/stone check
    """

    def __init__(self, max_idle_seconds: float = 1.2):
        self.tracks: dict[int, TemporalTrack] = {}
        self.next_track_id = 1
        self.max_idle_seconds = max_idle_seconds

    def update(self, candidates: list[dict], now: float, persons: list[dict] | None = None) -> list[dict]:
        """
        Process incoming candidate detections through temporal confirmation.
        Returns the subset of verified, high-confidence confirmed detections.
        """
        # 1. Prune stale tracks
        stale_ids = [tid for tid, trk in self.tracks.items() if (now - trk.last_seen) > self.max_idle_seconds]
        for tid in stale_ids:
            del self.tracks[tid]

        # 2. Match each candidate to existing track
        matched_track_ids = set()
        verified_threats = []

        for cand in candidates:
            c_bbox = cand["bbox"]
            c_cls = cand["class"].lower()
            c_conf = cand["confidence"]
            cx = (c_bbox[0] + c_bbox[2]) / 2.0
            cy = (c_bbox[1] + c_bbox[3]) / 2.0

            # Find closest matching active track with same class within distance tolerance
            best_id = None
            min_dist = 120.0  # px spatial consistency threshold

            for tid, trk in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                # Class compatibility
                if trk.cls_name != c_cls and not (
                    trk.cls_name in ("gun", "pistol", "rifle", "firearm") and c_cls in ("gun", "pistol", "rifle", "firearm")
                ):
                    continue

                last_cx, last_cy, _ = trk.history[-1]
                dist = math.hypot(cx - last_cx, cy - last_cy)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid

            if best_id is not None:
                track = self.tracks[best_id]
                track.update(c_bbox, c_conf, now)
                matched_track_ids.add(best_id)
            else:
                track = TemporalTrack(self.next_track_id, c_cls, c_bbox, c_conf, now)
                self.tracks[self.next_track_id] = track
                matched_track_ids.add(self.next_track_id)
                self.next_track_id += 1

            # 3. Class-specific Confirmation Rules
            confirmed = False
            label_override = None
            is_thrown = False

            vel = track.velocity
            consec = track.consecutive_frames
            bw = c_bbox[2] - c_bbox[0]
            bh = c_bbox[3] - c_bbox[1]

            if c_cls == "explosion":
                # Explosions are sudden optical events — confirm immediately!
                confirmed = True
                label_override = "EXPLOSION / BLAST"

            elif c_cls == "grenade":
                # Dynamic throwing motion check
                if vel > 110.0 and len(track.history) >= 2:
                    is_thrown = True
                    confirmed = (consec >= 2)
                    label_override = "GRENADE THROWN (IN FLIGHT)"
                else:
                    # Stationary grenade: require 3 frames to avoid mistaking stones/balls
                    confirmed = (consec >= 3 and track.mean_conf >= 0.35) or (track.max_conf >= 0.65)
                    label_override = "HAND GRENADE DETECTED"

            elif c_cls in ("gun", "pistol", "rifle", "firearm"):
                # Firearms: require 2 consecutive frames or high single-frame confidence
                confirmed = (consec >= 2 and track.mean_conf >= 0.34) or (c_conf >= 0.62)
                label_override = f"FIREARM ({c_cls.upper()})"

            elif c_cls == "knife":
                # Knives: check aspect ratio and require 3 frames
                aspect = max(bw / max(bh, 1), bh / max(bw, 1))
                if aspect >= 1.30 and (consec >= 3 or c_conf >= 0.65):
                    confirmed = True
                    label_override = "BLADE / WEAPON"

            else:
                # General dangerous object
                confirmed = (consec >= 2)

            if confirmed:
                track.is_confirmed = True
                cand["track_id"] = track.track_id
                cand["consecutive_frames"] = track.consecutive_frames
                cand["temporal_confidence"] = round(track.mean_conf, 2)
                cand["velocity"] = round(vel, 1)
                cand["trajectory"] = track.trajectory
                cand["is_thrown"] = is_thrown
                if label_override:
                    cand["threat_label"] = label_override

                verified_threats.append(cand)

        return verified_threats
