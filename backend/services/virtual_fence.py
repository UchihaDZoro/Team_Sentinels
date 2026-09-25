"""
IBVAP Virtual Fence / Tactical Perimeter Detection
Allows operators to define polygon zones and Zero-Line boundaries on camera feeds.
Monitors perimeter breaches, intrusions, and directional border ingress.
"""
import math
from typing import Optional, List, Dict, Tuple, Any
from shapely.geometry import Point, Polygon, LineString


class VirtualFence:
    """Manages virtual fence zones, directional boundary lines, and intrusion telemetry."""

    def __init__(self):
        # camera_id → list of parsed zone dicts
        self._zones: Dict[str, List[Dict[str, Any]]] = {}

    def set_zones(self, camera_id: str, zones_data: List[Dict[str, Any]]):
        """
        Set zones or boundary lines for a camera.
        zones_data: [{"id": "z1", "name": "Restricted Area", "points": [[x,y], ...], "type": "zone"|"zero_line"}, ...]
        Points are in pixel coordinates relative to the processed frame size (960x540).
        """
        parsed = []
        for z in zones_data:
            pts = z.get("points", [])
            if len(pts) < 2:
                continue  # Need at least 2 points for a line, 3 for polygon

            zone_id = z.get("id", f"zone_{len(parsed)}")
            zone_name = z.get("name", f"Zone {len(parsed)+1}")
            zone_type = z.get("type", "zero_line" if len(pts) == 2 else "zone")

            if len(pts) == 2:
                # Directional boundary line (Zero-Line)
                line = LineString(pts)
                parsed.append({
                    "id": zone_id,
                    "name": zone_name,
                    "type": "zero_line",
                    "line": line,
                    "points": pts,
                    "polygon": None,
                })
            else:
                # Polygon zone
                poly = Polygon(pts)
                parsed.append({
                    "id": zone_id,
                    "name": zone_name,
                    "type": "zone",
                    "polygon": poly,
                    "points": pts,
                    "line": None,
                })

        self._zones[camera_id] = parsed

    def get_zones(self, camera_id: str) -> List[Dict[str, Any]]:
        """Return serialisable zone info for a camera."""
        zones = self._zones.get(camera_id, [])
        return [
            {"id": z["id"], "name": z["name"], "points": z["points"], "type": z.get("type", "zone")}
            for z in zones
        ]

    def has_zones(self, camera_id: str) -> bool:
        return bool(self._zones.get(camera_id))

    def distance_to_nearest_fence(self, camera_id: str, point: Any) -> float:
        """
        Calculate shortest distance (in pixels) from point (Point or [x,y]) to any fence boundary.
        """
        zones = self._zones.get(camera_id, [])
        if not zones:
            return float("inf")

        pt = point if isinstance(point, Point) else Point(point[0], point[1])
        min_dist = float("inf")

        for z in zones:
            if z.get("polygon"):
                # Distance to polygon boundary (or 0 if inside)
                d = z["polygon"].boundary.distance(pt) if not z["polygon"].contains(pt) else 0.0
                min_dist = min(min_dist, d)
            elif z.get("line"):
                d = z["line"].distance(pt)
                min_dist = min(min_dist, d)

        return min_dist

    def is_in_zone_or_buffer(
        self,
        camera_id: str,
        point: Any,
        buffer_px: float = 35.0,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if point is inside a zone or within a buffer margin around it.
        Returns: (is_inside, zone_id, zone_name)
        """
        zones = self._zones.get(camera_id, [])
        if not zones:
            return False, None, None

        pt = point if isinstance(point, Point) else Point(point[0], point[1])

        for z in zones:
            if z.get("polygon"):
                poly = z["polygon"]
                if poly.contains(pt) or poly.distance(pt) <= buffer_px:
                    return True, z["id"], z["name"]
            elif z.get("line"):
                if z["line"].distance(pt) <= buffer_px:
                    return True, z["id"], z["name"]

        return False, None, None

    def check_directional_ingress(
        self,
        camera_id: str,
        p_old: Tuple[float, float],
        p_new: Tuple[float, float],
    ) -> Dict[str, Any]:
        """
        Test if target trajectory vector from p_old to p_new crosses a Zero-Line
        or penetrates a polygon fence zone from outside into inside territory.
        """
        zones = self._zones.get(camera_id, [])
        if not zones:
            return {"ingress": False}

        traj_line = LineString([p_old, p_new])
        old_pt = Point(p_old)
        new_pt = Point(p_new)

        # Check Zero-Line boundary lines first (highest tactical border priority)
        for z in zones:
            if z.get("type") == "zero_line" and z.get("line"):
                boundary_line = z["line"]
                if traj_line.intersects(boundary_line):
                    coords = list(boundary_line.coords)
                    a, b = coords[0], coords[1]
                    side_old = (b[0] - a[0]) * (p_old[1] - a[1]) - (b[1] - a[1]) * (p_old[0] - a[0])
                    side_new = (b[0] - a[0]) * (p_new[1] - a[1]) - (b[1] - a[1]) * (p_new[0] - a[0])

                    if (side_old < 0 <= side_new) or (side_old <= 0 < side_new) or (side_new < 0 <= side_old) or (side_new <= 0 < side_old):
                        return {
                            "ingress": True,
                            "zone_id": z["id"],
                            "zone_name": z["name"],
                            "line_pts": z["points"],
                        }

        # Check polygon zones for perimeter ingress (outside -> inside)
        for z in zones:
            if z.get("polygon"):
                poly = z["polygon"]
                if not poly.contains(old_pt) and poly.contains(new_pt):
                    return {
                        "ingress": True,
                        "zone_id": z["id"],
                        "zone_name": z["name"],
                        "line_pts": z["points"],
                    }

        return {"ingress": False}

    def check_intrusions(self, camera_id: str, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check if any detection's bottom-centre or centroid falls inside a restricted zone.
        Returns list of intrusion events.
        """
        zones = self._zones.get(camera_id, [])
        if not zones:
            return []

        intrusions = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            foot = Point((x1 + x2) / 2, y2)
            centre = Point((x1 + x2) / 2, (y1 + y2) / 2)

            for zone in zones:
                if zone.get("polygon"):
                    poly = zone["polygon"]
                    if poly.contains(foot) or poly.contains(centre):
                        intrusions.append({
                            "detection": det,
                            "zone_id": zone["id"],
                            "zone_name": zone["name"],
                        })
                        break  # One intrusion per detection is sufficient
                elif zone.get("line"):
                    # Zero-line proximity breach (< 25px)
                    if zone["line"].distance(foot) <= 25.0 or zone["line"].distance(centre) <= 25.0:
                        intrusions.append({
                            "detection": det,
                            "zone_id": zone["id"],
                            "zone_name": zone["name"],
                        })
                        break

        return intrusions

    def get_zone_polygons_for_drawing(self, camera_id: str) -> List[Dict[str, Any]]:
        """Return polygon or line points ready for OpenCV drawing."""
        zones = self._zones.get(camera_id, [])
        return [
            {"id": z["id"], "name": z["name"], "points": z["points"], "type": z.get("type", "zone")}
            for z in zones
        ]
