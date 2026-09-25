"""
IBVAP Object Detector — YOLOv8 with built-in tracking
Detects persons, vehicles, and objects of interest in surveillance frames.
"""
import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_MODEL,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CLASSES_OF_INTEREST,
    PERSON_CLASSES,
    VEHICLE_CLASSES,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


# Colour palette for drawing (BGR)
CATEGORY_COLOURS = {
    "person":     (0, 255, 0),     # Vibrant Green for Humans
    "vehicle":    (0, 255, 255),   # Vibrant Yellow for Vehicles (car, bus, truck, motorcycle, bicycle)
    "object":     (255, 180, 50),  # Accent cyan/orange for bags/items
    "unknown":    (180, 180, 180),
}

INTRUSION_COLOUR = (0, 0, 255)  # Red for intrusion alerts


class ObjectDetector:
    """YOLOv8-based real-time object detector with built-in tracking."""

    def __init__(self):
        print(f"[IBVAP] Loading YOLO model: {YOLO_MODEL}")
        self.model = YOLO(YOLO_MODEL)
        self._warmup()

    def _warmup(self):
        """Run a dummy inference to warm up the model."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        print("[IBVAP] YOLO model warmed up ✓")

    def detect_and_track(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection + tracking on a frame.
        Returns list of detection dicts.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            classes=list(CLASSES_OF_INTEREST.keys()),
            tracker="bytetrack.yaml",
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASSES_OF_INTEREST.get(cls_id, "unknown")
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                track_id = int(box.id[0]) if box.id is not None else -1

                category = "person" if cls_id in PERSON_CLASSES else \
                           "vehicle" if cls_id in VEHICLE_CLASSES else "object"

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": cls_name,
                    "class_id": cls_id,
                    "category": category,
                    "confidence": round(conf, 2),
                    "track_id": track_id,
                })

        return detections

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        detections: list[dict],
        intrusions: list[dict] | None = None,
        fence_zones: list[dict] | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes, labels, tracking IDs, and fence zones on the frame."""
        annotated = frame.copy()
        intrusion_track_ids = set()

        if intrusions:
            intrusion_track_ids = {
                intr["detection"]["track_id"] for intr in intrusions
            }

        # ── Draw fence zones ────────────────────────────────────
        if fence_zones:
            for zone in fence_zones:
                pts = np.array(zone["points"], dtype=np.int32)
                # Semi-transparent fill
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 180))
                cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
                # Border
                cv2.polylines(annotated, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
                # Zone label
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                label = zone.get("name", "Zone")
                cv2.putText(
                    annotated, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
                )

        # ── Draw detections ─────────────────────────────────────
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class"]
            track_id = det["track_id"]
            conf = det["confidence"]
            category = det.get("category", "object")

            is_intrusion = track_id in intrusion_track_ids

            # Choose colour based on user rules:
            # - Humans: Green (0, 255, 0)
            # - Vehicles: Yellow (0, 255, 255)
            # - Intrusions: Red (0, 0, 255)
            if is_intrusion:
                colour = INTRUSION_COLOUR
                thickness = 3
                tag_label = f"ALERT #{track_id} {conf:.0%}" if track_id >= 0 else f"ALERT {conf:.0%}"
            elif category == "person":
                colour = (0, 255, 0)       # Vibrant Green for Humans
                thickness = 2
                tag_label = f"HUMAN #{track_id} {conf:.0%}" if track_id >= 0 else f"HUMAN {conf:.0%}"
            elif category == "vehicle":
                colour = (0, 255, 255)     # Vibrant Yellow for Vehicles
                thickness = 2
                tag_label = f"{cls_name.upper()} #{track_id} {conf:.0%}" if track_id >= 0 else f"{cls_name.upper()} {conf:.0%}"
            else:
                colour = (255, 180, 50)
                thickness = 2
                tag_label = f"{cls_name.upper()} {conf:.0%}"

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)

            # Tactical corner markers for premium HUD aesthetic
            corner_len = min(14, max(4, (x2 - x1) // 5), max(4, (y2 - y1) // 5))
            if corner_len > 4:
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), colour, thickness + 1)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), colour, thickness + 1)
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), colour, thickness + 1)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), colour, thickness + 1)
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), colour, thickness + 1)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), colour, thickness + 1)
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), colour, thickness + 1)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), colour, thickness + 1)

            # Label pill background with crisp black text
            (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(0, y1 - th - 8)
            tag_y2 = y1
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 8, tag_y2), colour, -1)
            cv2.putText(
                annotated, tag_label, (x1 + 4, tag_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
            )

            # Intrusion warning label below box
            if is_intrusion:
                cv2.putText(
                    annotated, "⚠ INTRUSION ZONE", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, INTRUSION_COLOUR, 2, cv2.LINE_AA
                )

        # ── Tactical HUD overlay (top-left detection stats) ─────
        person_count = sum(1 for d in detections if d["category"] == "person")
        vehicle_count = sum(1 for d in detections if d["category"] == "vehicle")

        # Semi-transparent HUD card
        hud_h = 70 if not intrusions else 92
        overlay = annotated.copy()
        cv2.rectangle(overlay, (8, 8), (210, 8 + hud_h), (8, 12, 22), -1)
        cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)
        cv2.rectangle(annotated, (8, 8), (210, 8 + hud_h), (0, 255, 136), 1)

        cv2.putText(annotated, "SURVEILLANCE HUD", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 180), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"● HUMANS:  {person_count}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"● VEHICLES: {vehicle_count}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)
        if intrusions:
            cv2.putText(annotated, f"⚠ INTRUSIONS: {len(intrusions)}", (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)

        return annotated
