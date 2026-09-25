"""
IBVAP Tactical Behavioral Analytics Engine
Military-grade behavioral surveillance for Sashastra Seema Bal (SSB), MHA.

Implements real-time tactical behavioral algorithms:
1. Loitering Detection (dwell-time tracking inside buffer / virtual fence zones)
2. Crawling / Prone Infiltration Detection (aspect ratio + ground-proximity heuristics)
3. Unattended Baggage / Bag Drop Detection (stationary object + personnel proximity radius)
4. Directional Zero-Line Crossing (trajectory tracking & ingress vector analysis)
"""
import time
import math
from collections import deque
from typing import Optional, Dict, List, Tuple, Any
from shapely.geometry import Point, Polygon, LineString

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


# ── Tactical Analytics Defaults ─────────────────────────────────
DEFAULT_LOITERING_THRESHOLD = 20.0        # Seconds dwelling before LOITERING_DETECTED
DEFAULT_CRAWLING_ASPECT_RATIO = 1.15       # Width / Height > 1.15 indicates prone/crawling
DEFAULT_GROUND_Y_RATIO = 0.55              # Centroid y >= 55% frame height (lower region)
DEFAULT_FENCE_PROXIMITY_PX = 80.0          # Pixels to fence perimeter
DEFAULT_UNATTENDED_TIME = 15.0             # Seconds stationary before UNATTENDED_BAGGAGE
DEFAULT_UNATTENDED_PROXIMITY_R = 120.0     # Radius in px; no personnel within R triggers alert
DEFAULT_STATIONARY_DISPLACEMENT_PX = 25.0  # Max px drift for stationary classification
DEFAULT_TRAJECTORY_HISTORY_LEN = 30        # Historical frames kept per target
DEFAULT_TRACK_LOST_TIMEOUT = 4.0           # Seconds before inactive track is pruned

BAGGAGE_CLASSES = {"backpack", "handbag", "suitcase", "umbrella"}
BAGGAGE_CLASS_IDS = {24, 25, 26, 28}


def calculate_line_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Check if line segment p1-p2 intersects line segment p3-p4."""
    try:
        line1 = LineString([p1, p2])
        line2 = LineString([p3, p4])
        return bool(line1.intersects(line2))
    except Exception:
        return False


def point_side_of_line(
    a: Tuple[float, float],
    b: Tuple[float, float],
    p: Tuple[float, float],
) -> float:
    """
    Returns signed value indicating which side of directed line a->b point p lies on.
    > 0: left side / territory side
    < 0: right side / outside
    = 0: collinear
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class TargetTrackState:
    """Maintains kinematic, posture, and dwell telemetry for a single tracked target."""

    def __init__(self, track_id: int, category: str, class_name: str, centroid: Tuple[float, float], bbox: List[float], timestamp: float):
        self.track_id = track_id
        self.category = category
        self.class_name = class_name
        self.bbox = bbox
        self.centroid = centroid

        self.first_seen = timestamp
        self.last_seen = timestamp

        # Trajectory history: deque of (x, y, timestamp)
        self.trajectory: deque[Tuple[float, float, float]] = deque(maxlen=DEFAULT_TRAJECTORY_HISTORY_LEN)
        self.trajectory.append((centroid[0], centroid[1], timestamp))

        # Kinematics
        self.velocity = (0.0, 0.0)  # (vx, vy) px/sec
        self.speed = 0.0            # px/sec

        # Posture telemetry
        self.aspect_ratio_history: deque[float] = deque(maxlen=10)
        self.posture = "standing"
        self.is_crawling = False
        self.crawling_frame_count = 0
        self.crawling_alerted = False
        self.last_crawling_alert_time = 0.0

        # Zone dwell & Loitering telemetry: zone_id -> entry_timestamp
        self.zone_entry_times: Dict[str, float] = {}
        self.current_zone_id: Optional[str] = None
        self.current_zone_name: Optional[str] = None
        self.dwell_time: float = 0.0
        self.loitering_alerted = False
        self.last_loiter_alert_time = 0.0

        # Zero-Line Ingress telemetry
        self.ingress_alerted = False
        self.last_ingress_alert_time = 0.0

        # General status tags for Tactical HUD
        self.status_tags: List[str] = []

    def update(self, centroid: Tuple[float, float], bbox: List[float], timestamp: float):
        dt = max(0.001, timestamp - self.last_seen)
        dx = centroid[0] - self.centroid[0]
        dy = centroid[1] - self.centroid[1]

        # Velocity smoothing
        vx = dx / dt
        vy = dy / dt
        speed = math.hypot(vx, vy)

        self.velocity = (round(vx, 1), round(vy, 1))
        self.speed = round(speed, 1)

        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = timestamp
        self.trajectory.append((centroid[0], centroid[1], timestamp))


class StationaryObjectTracker:
    """Tracks static unattended baggage and item drops."""

    def __init__(self, object_id: str, class_name: str, bbox: List[float], centroid: Tuple[float, float], timestamp: float):
        self.object_id = object_id
        self.class_name = class_name
        self.bbox = bbox
        self.centroid = centroid
        self.anchor_centroid = centroid

        self.first_seen = timestamp
        self.stationary_since = timestamp
        self.last_seen = timestamp

        self.stationary_duration = 0.0
        self.min_person_distance = float("inf")
        self.is_unattended = False
        self.alerted = False
        self.last_alert_time = 0.0

    def update(self, centroid: Tuple[float, float], bbox: List[float], timestamp: float):
        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = timestamp

        displacement = math.hypot(centroid[0] - self.anchor_centroid[0], centroid[1] - self.anchor_centroid[1])
        if displacement <= DEFAULT_STATIONARY_DISPLACEMENT_PX:
            self.stationary_duration = timestamp - self.stationary_since
        else:
            # Item moved; reset stationary timer
            self.anchor_centroid = centroid
            self.stationary_since = timestamp
            self.stationary_duration = 0.0
            self.alerted = False


class CameraBehaviorState:
    """Maintains state of tracked targets and unattended items for a single camera."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.tracks: Dict[int, TargetTrackState] = {}
        self.unassigned_targets: List[TargetTrackState] = []
        self.baggage_tracks: List[StationaryObjectTracker] = []
        self.last_cleanup = time.time()
        self.next_unassigned_id = 9000
        self.next_bag_id = 1

    def prune_stale_tracks(self, now: float):
        """Remove tracks not updated within timeout window."""
        stale_ids = [tid for tid, trk in self.tracks.items() if (now - trk.last_seen) > DEFAULT_TRACK_LOST_TIMEOUT]
        for tid in stale_ids:
            del self.tracks[tid]

        self.unassigned_targets = [trk for trk in self.unassigned_targets if (now - trk.last_seen) > DEFAULT_TRACK_LOST_TIMEOUT]
        self.baggage_tracks = [b for b in self.baggage_tracks if (now - b.last_seen) <= (DEFAULT_TRACK_LOST_TIMEOUT * 2)]
        self.last_cleanup = now


class BehaviorEngine:
    """
    IBVAP Tactical Behavioral Analytics Engine.
    Executes real-time behavioral heuristics and generates tactical alerts.
    """

    def __init__(
        self,
        loitering_threshold: float = DEFAULT_LOITERING_THRESHOLD,
        crawling_aspect_ratio: float = DEFAULT_CRAWLING_ASPECT_RATIO,
        ground_y_ratio: float = DEFAULT_GROUND_Y_RATIO,
        fence_proximity_px: float = DEFAULT_FENCE_PROXIMITY_PX,
        unattended_time_threshold: float = DEFAULT_UNATTENDED_TIME,
        unattended_proximity_radius: float = DEFAULT_UNATTENDED_PROXIMITY_R,
    ):
        self.loitering_threshold = loitering_threshold
        self.crawling_aspect_ratio = crawling_aspect_ratio
        self.ground_y_ratio = ground_y_ratio
        self.fence_proximity_px = fence_proximity_px
        self.unattended_time_threshold = unattended_time_threshold
        self.unattended_proximity_radius = unattended_proximity_radius

        # Camera state registry: camera_id -> CameraBehaviorState
        self.camera_states: Dict[str, CameraBehaviorState] = {}

    def get_or_create_state(self, camera_id: str) -> CameraBehaviorState:
        """Retrieve or initialize camera behavioral tracking state."""
        if camera_id not in self.camera_states:
            self.camera_states[camera_id] = CameraBehaviorState(camera_id)
        return self.camera_states[camera_id]

    def reset_camera(self, camera_id: str):
        """Reset analytics state for a given camera."""
        self.camera_states.pop(camera_id, None)

    def process_frame(
        self,
        camera_id: str,
        detections: List[Dict[str, Any]],
        fence_engine: Any,
        frame_shape: Tuple[int, int] = (FRAME_HEIGHT, FRAME_WIDTH),
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute tactical behavioral analysis on detected targets in the current frame.

        Parameters:
            camera_id: Unique identifier of the camera feed.
            detections: List of detection dictionaries from detector.
            fence_engine: VirtualFence instance managing zones.
            frame_shape: (height, width) of the surveillance frame.
            timestamp: Optional epoch timestamp (defaults to time.time()).

        Returns:
            List of generated behavioral events (LOITERING_DETECTED, CRAWLING_INFILTRATION,
            UNATTENDED_BAGGAGE, DIRECTIONAL_INGRESS).
            Also enriches detection dicts in-place with telemetry.
        """
        now = timestamp or time.time()
        state = self.get_or_create_state(camera_id)

        # Periodic cleanup of stale tracks
        if now - state.last_cleanup > 2.0:
            state.prune_stale_tracks(now)

        frame_h, frame_w = frame_shape
        events: List[Dict[str, Any]] = []

        # ── Step 1: Update Track States & Kinematics ─────────────────
        persons_in_frame: List[TargetTrackState] = []

        for det in detections:
            x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            category = det.get("category", "object")
            cls_name = det.get("class", "unknown")
            track_id = det.get("track_id", -1)

            # Assign or locate track state
            target_track: Optional[TargetTrackState] = None
            if track_id > 0:
                if track_id in state.tracks:
                    target_track = state.tracks[track_id]
                    target_track.update((cx, cy), [x1, y1, x2, y2], now)
                else:
                    target_track = TargetTrackState(track_id, category, cls_name, (cx, cy), [x1, y1, x2, y2], now)
                    state.tracks[track_id] = target_track
            else:
                # Spatial matching for targets without ByteTrack ID
                best_match = None
                best_dist = 50.0
                for unassigned in state.unassigned_targets:
                    d = math.hypot(cx - unassigned.centroid[0], cy - unassigned.centroid[1])
                    if d < best_dist and unassigned.category == category:
                        best_dist = d
                        best_match = unassigned

                if best_match:
                    target_track = best_match
                    target_track.update((cx, cy), [x1, y1, x2, y2], now)
                else:
                    state.next_unassigned_id += 1
                    target_track = TargetTrackState(
                        state.next_unassigned_id, category, cls_name, (cx, cy), [x1, y1, x2, y2], now
                    )
                    state.unassigned_targets.append(target_track)

            # Attach enriched telemetry references to detection dict
            det["centroid"] = [round(cx, 1), round(cy, 1)]
            det["velocity"] = {"vx": target_track.velocity[0], "vy": target_track.velocity[1], "speed": target_track.speed}
            det["trajectory"] = [[round(p[0], 1), round(p[1], 1)] for p in target_track.trajectory]
            det["status_tags"] = []

            if category == "person":
                persons_in_frame.append(target_track)

        # ── Step 2: Crawling / Prone Infiltration Analysis ───────────
        for det in detections:
            if det.get("category") != "person":
                continue

            x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx, cy = det["centroid"]
            aspect_ratio = round(w / h, 2)
            track_id = det.get("track_id", -1)

            target_track = state.tracks.get(track_id)
            if not target_track:
                for unassigned in state.unassigned_targets:
                    if unassigned.track_id == track_id or (unassigned.centroid[0] == cx and unassigned.centroid[1] == cy):
                        target_track = unassigned
                        break

            # Ground proximity heuristic:
            # 1. Centroid located in lower region of frame (cy >= frame_h * ground_y_ratio) or foot near base
            is_in_lower_frame = (cy >= frame_h * self.ground_y_ratio) or (y2 >= frame_h * 0.65)

            # 2. Proximity to any defined virtual fence boundary
            near_fence = False
            if fence_engine and hasattr(fence_engine, "distance_to_nearest_fence"):
                dist_to_fence = fence_engine.distance_to_nearest_fence(camera_id, Point(cx, cy))
                if dist_to_fence <= self.fence_proximity_px:
                    near_fence = True
            elif fence_engine and hasattr(fence_engine, "has_zones") and fence_engine.has_zones(camera_id):
                # Fallback: check if inside or close to any zone polygon
                foot = Point(cx, y2)
                for zone in fence_engine._zones.get(camera_id, []):
                    poly = zone.get("polygon")
                    line = zone.get("line")
                    if poly and (poly.distance(foot) <= self.fence_proximity_px or poly.contains(foot)):
                        near_fence = True
                        break
                    elif line and line.distance(foot) <= self.fence_proximity_px:
                        near_fence = True
                        break

            ground_proximity = is_in_lower_frame or near_fence

            # Crawling classification
            is_crawling = (aspect_ratio >= self.crawling_aspect_ratio) and ground_proximity
            posture_type = "prone" if is_crawling else ("crouching" if aspect_ratio > 0.85 else "standing")

            if target_track:
                target_track.aspect_ratio_history.append(aspect_ratio)
                if is_crawling:
                    target_track.crawling_frame_count += 1
                else:
                    target_track.crawling_frame_count = max(0, target_track.crawling_frame_count - 1)

                target_track.is_crawling = target_track.crawling_frame_count >= 2 or (aspect_ratio >= 1.30 and ground_proximity)
                target_track.posture = "crawling" if target_track.is_crawling else posture_type

            det["posture"] = {
                "posture": "crawling" if (target_track and target_track.is_crawling) else posture_type,
                "aspect_ratio": aspect_ratio,
                "is_crawling": target_track.is_crawling if target_track else is_crawling,
                "ground_proximity": ground_proximity,
            }

            if target_track and target_track.is_crawling:
                det["status_tags"].append("PRONE / CRAWLING")

                # Generate alert if not alerted or cooldown expired (cooldown: 20s)
                if not target_track.crawling_alerted or (now - target_track.last_crawling_alert_time) > 20.0:
                    target_track.crawling_alerted = True
                    target_track.last_crawling_alert_time = now

                    events.append({
                        "event_type": "CRAWLING_INFILTRATION",
                        "severity": "critical",
                        "message": (
                            f"CRITICAL: Crawling / Prone Infiltration detected near boundary! "
                            f"Target #{track_id} (Aspect Ratio: {aspect_ratio:.2f}, Ground Proximity: YES)"
                        ),
                        "camera_id": camera_id,
                        "track_id": track_id,
                        "detection": det,
                        "details": {
                            "track_id": track_id,
                            "aspect_ratio": aspect_ratio,
                            "ground_proximity": ground_proximity,
                            "centroid": [round(cx, 1), round(cy, 1)],
                            "bbox": det["bbox"],
                            "posture": "crawling",
                        },
                        "timestamp": now,
                    })

        # ── Step 3: Loitering Detection ──────────────────────────────
        has_fence = fence_engine and hasattr(fence_engine, "has_zones") and fence_engine.has_zones(camera_id)
        zones_data = fence_engine._zones.get(camera_id, []) if has_fence else []

        for p_track in persons_in_frame:
            cx, cy = p_track.centroid
            x1, y1, x2, y2 = p_track.bbox
            foot = Point(cx, y2)
            centre = Point(cx, cy)

            matched_zone_id: Optional[str] = None
            matched_zone_name: Optional[str] = None

            if has_fence:
                for zone in zones_data:
                    poly = zone.get("polygon")
                    line = zone.get("line")
                    if poly and (poly.contains(foot) or poly.contains(centre) or poly.distance(foot) <= 35.0):
                        matched_zone_id = zone["id"]
                        matched_zone_name = zone["name"]
                        break
                    elif line and (line.distance(foot) <= 35.0 or line.distance(centre) <= 35.0):
                        matched_zone_id = zone["id"]
                        matched_zone_name = zone["name"]
                        break
            else:
                # Default monitored buffer: lower half or border surveillance sector
                if cy >= (frame_h * 0.40):
                    matched_zone_id = "default_border_buffer"
                    matched_zone_name = "Border Monitored Buffer"

            if matched_zone_id:
                if matched_zone_id not in p_track.zone_entry_times:
                    p_track.zone_entry_times[matched_zone_id] = now
                p_track.current_zone_id = matched_zone_id
                p_track.current_zone_name = matched_zone_name

                dwell = now - p_track.zone_entry_times[matched_zone_id]
                p_track.dwell_time = round(dwell, 1)

                # Attach dwell tag to corresponding detection
                for det in detections:
                    if det.get("track_id") == p_track.track_id:
                        det["dwell_time"] = p_track.dwell_time
                        if p_track.dwell_time >= (self.loitering_threshold * 0.5):
                            det["status_tags"].append(f"DWELL: {int(p_track.dwell_time)}s")
                        break

                # Trigger alert if dwell threshold crossed
                if p_track.dwell_time >= self.loitering_threshold:
                    if not p_track.loitering_alerted or (now - p_track.last_loiter_alert_time) > 30.0:
                        p_track.loitering_alerted = True
                        p_track.last_loiter_alert_time = now

                        matching_det = next((d for d in detections if d.get("track_id") == p_track.track_id), None)
                        events.append({
                            "event_type": "LOITERING_DETECTED",
                            "severity": "high",
                            "message": (
                                f"LOITERING DETECTED: Person #{p_track.track_id} dwelled in {matched_zone_name} "
                                f"for {int(p_track.dwell_time)}s (threshold: {int(self.loitering_threshold)}s)"
                            ),
                            "camera_id": camera_id,
                            "track_id": p_track.track_id,
                            "detection": matching_det or {
                                "bbox": p_track.bbox,
                                "category": "person",
                                "class": p_track.class_name,
                                "track_id": p_track.track_id,
                                "confidence": 0.90,
                            },
                            "details": {
                                "track_id": p_track.track_id,
                                "dwell_time": p_track.dwell_time,
                                "zone_id": matched_zone_id,
                                "zone_name": matched_zone_name,
                                "bbox": p_track.bbox,
                                "centroid": [round(cx, 1), round(cy, 1)],
                            },
                            "timestamp": now,
                        })
            else:
                # Exited monitored zone: reset dwell timer
                p_track.zone_entry_times.clear()
                p_track.current_zone_id = None
                p_track.current_zone_name = None
                p_track.dwell_time = 0.0
                p_track.loitering_alerted = False

        # ── Step 4: Unattended Baggage / Bag Drop Detection ──────────
        baggage_detections = [
            d for d in detections
            if d.get("class") in BAGGAGE_CLASSES or d.get("class_id") in BAGGAGE_CLASS_IDS
        ]

        for bag_det in baggage_detections:
            cx, cy = bag_det["centroid"]
            cls_name = bag_det.get("class", "baggage")

            # Associate with existing stationary tracker or spawn new
            matched_bag: Optional[StationaryObjectTracker] = None
            min_dist = 45.0
            for bag_track in state.baggage_tracks:
                d = math.hypot(cx - bag_track.centroid[0], cy - bag_track.centroid[1])
                if d < min_dist and bag_track.class_name == cls_name:
                    min_dist = d
                    matched_bag = bag_track

            if matched_bag:
                matched_bag.update((cx, cy), bag_det["bbox"], now)
            else:
                bag_id = f"bag_{state.next_bag_id}"
                state.next_bag_id += 1
                matched_bag = StationaryObjectTracker(bag_id, cls_name, bag_det["bbox"], (cx, cy), now)
                state.baggage_tracks.append(matched_bag)

            # Measure proximity to all detected personnel
            if persons_in_frame:
                nearest_person_dist = min(
                    math.hypot(cx - p.centroid[0], cy - p.centroid[1]) for p in persons_in_frame
                )
            else:
                nearest_person_dist = 9999.0

            matched_bag.min_person_distance = round(nearest_person_dist, 1)

            # Evaluate unattended criteria
            is_unattended = (
                matched_bag.stationary_duration >= self.unattended_time_threshold
                and nearest_person_dist > self.unattended_proximity_radius
            )
            matched_bag.is_unattended = is_unattended

            bag_det["unattended_telemetry"] = {
                "stationary_duration": round(matched_bag.stationary_duration, 1),
                "nearest_person_dist": nearest_person_dist if nearest_person_dist < 9000 else None,
                "is_unattended": is_unattended,
            }

            if is_unattended:
                bag_det["status_tags"].append(f"UNATTENDED ({int(matched_bag.stationary_duration)}s)")
                # Trigger alert if not alerted or cooldown expired (cooldown: 25s)
                if not matched_bag.alerted or (now - matched_bag.last_alert_time) > 25.0:
                    matched_bag.alerted = True
                    matched_bag.last_alert_time = now

                    events.append({
                        "event_type": "UNATTENDED_BAGGAGE",
                        "severity": "high",
                        "message": (
                            f"TACTICAL ALERT: UNATTENDED {cls_name.upper()} stationary for "
                            f"{int(matched_bag.stationary_duration)}s with no personnel within "
                            f"{int(nearest_person_dist) if nearest_person_dist < 9000 else 'perimeter'}px"
                        ),
                        "camera_id": camera_id,
                        "track_id": bag_det.get("track_id", -1),
                        "detection": bag_det,
                        "details": {
                            "object_id": matched_bag.object_id,
                            "class": cls_name,
                            "stationary_duration": round(matched_bag.stationary_duration, 1),
                            "min_person_distance": nearest_person_dist if nearest_person_dist < 9000 else None,
                            "centroid": [round(cx, 1), round(cy, 1)],
                            "bbox": bag_det["bbox"],
                        },
                        "timestamp": now,
                    })
            elif matched_bag.stationary_duration > 5.0:
                bag_det["status_tags"].append(f"STATIC ({int(matched_bag.stationary_duration)}s)")

        # ── Step 5: Directional Zero-Line Crossing Analysis ──────────
        # Evaluates trajectory penetration across boundary line from outside into territory
        for det in detections:
            track_id = det.get("track_id", -1)
            target_track = state.tracks.get(track_id)
            if not target_track or len(target_track.trajectory) < 4:
                continue

            cx, cy = det["centroid"]
            # Lookback position (e.g. 5-15 frames ago)
            lookback_idx = max(0, len(target_track.trajectory) - 10)
            old_x, old_y, _ = target_track.trajectory[lookback_idx]

            movement_vector = (cx - old_x, cy - old_y)
            movement_mag = math.hypot(movement_vector[0], movement_vector[1])

            # Ingress check against defined boundary lines or fence polygon perimeter
            is_ingress = False
            crossing_zone_name = "Zero-Line"

            if fence_engine and hasattr(fence_engine, "check_directional_ingress"):
                res = fence_engine.check_directional_ingress(camera_id, (old_x, old_y), (cx, cy))
                if res.get("ingress", False):
                    is_ingress = True
                    crossing_zone_name = res.get("zone_name", "Zero-Line")
            elif zones_data:
                # Built-in boundary edge intersection check
                # Outer edge is foreign boundary; moving from outside polygon into polygon = ingress
                old_pt = Point(old_x, old_y)
                cur_pt = Point(cx, cy)
                for zone in zones_data:
                    poly = zone.get("polygon")
                    if poly and not poly.contains(old_pt) and poly.contains(cur_pt) and movement_mag >= 15.0:
                        is_ingress = True
                        crossing_zone_name = zone["name"]
                        break
            else:
                # Default Zero-Line heuristic (horizontal boundary across upper surveillance quadrant)
                # Objects moving from upper zone (foreign side, y < border_y) downward into territory (y >= border_y)
                border_y = frame_h * 0.38
                if old_y < border_y <= cy and movement_vector[1] > 18.0:
                    is_ingress = True
                    crossing_zone_name = "International Border Zero-Line"

            if is_ingress:
                det["status_tags"].append("DIRECTIONAL INGRESS")

                if not target_track.ingress_alerted or (now - target_track.last_ingress_alert_time) > 25.0:
                    target_track.ingress_alerted = True
                    target_track.last_ingress_alert_time = now

                    events.append({
                        "event_type": "DIRECTIONAL_INGRESS",
                        "severity": "critical",
                        "message": (
                            f"CRITICAL: DIRECTIONAL INGRESS across {crossing_zone_name}! "
                            f"Target #{track_id} ({det.get('class', 'target').upper()}) penetrated border into territory"
                        ),
                        "camera_id": camera_id,
                        "track_id": track_id,
                        "detection": det,
                        "details": {
                            "track_id": track_id,
                            "class": det.get("class"),
                            "vector": [round(movement_vector[0], 1), round(movement_vector[1], 1)],
                            "speed": target_track.speed,
                            "zone_name": crossing_zone_name,
                            "entry_point": [round(cx, 1), round(cy, 1)],
                            "bbox": det["bbox"],
                        },
                        "timestamp": now,
                    })

        return events
