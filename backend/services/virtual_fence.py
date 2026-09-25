"""
IBVAP Virtual Fence / Zone Intrusion Detection
Allows operators to define polygon zones on camera feeds.
Any detected person or vehicle entering the zone triggers an alert.
"""
import json
from shapely.geometry import Point, Polygon
from typing import Optional


class VirtualFence:
    """Manages virtual fence zones and checks for intrusions."""

    def __init__(self):
        # camera_id → list of {"id": str, "name": str, "polygon": Polygon, "points": [...]}
        self._zones: dict[str, list[dict]] = {}

    def set_zones(self, camera_id: str, zones_data: list[dict]):
        """
        Set zones for a camera.
        zones_data: [{"id": "z1", "name": "Restricted Area", "points": [[x,y], ...]}, ...]
        Points are in pixel coordinates relative to the processed frame size.
        """
        parsed = []
        for z in zones_data:
            pts = z.get("points", [])
            if len(pts) < 3:
                continue  # need at least a triangle
            parsed.append({
                "id": z.get("id", f"zone_{len(parsed)}"),
                "name": z.get("name", f"Zone {len(parsed)+1}"),
                "polygon": Polygon(pts),
                "points": pts,
            })
        self._zones[camera_id] = parsed

    def get_zones(self, camera_id: str) -> list[dict]:
        """Return serialisable zone info for a camera."""
        zones = self._zones.get(camera_id, [])
        return [
            {"id": z["id"], "name": z["name"], "points": z["points"]}
            for z in zones
        ]

    def has_zones(self, camera_id: str) -> bool:
        return bool(self._zones.get(camera_id))

    def check_intrusions(self, camera_id: str, detections: list[dict]) -> list[dict]:
        """
        Check if any detection's bottom-centre falls inside a zone.
        Returns list of intrusion events.
        """
        zones = self._zones.get(camera_id, [])
        if not zones:
            return []

        intrusions = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Use bottom-centre (foot position) for person, centre for vehicles
            foot = Point((x1 + x2) / 2, y2)
            centre = Point((x1 + x2) / 2, (y1 + y2) / 2)

            for zone in zones:
                poly = zone["polygon"]
                if poly.contains(foot) or poly.contains(centre):
                    intrusions.append({
                        "detection": det,
                        "zone_id": zone["id"],
                        "zone_name": zone["name"],
                    })
                    break  # one intrusion per detection is enough

        return intrusions

    def get_zone_polygons_for_drawing(self, camera_id: str) -> list[dict]:
        """Return polygon points ready for OpenCV drawing."""
        zones = self._zones.get(camera_id, [])
        return [
            {"id": z["id"], "name": z["name"], "points": z["points"]}
            for z in zones
        ]
