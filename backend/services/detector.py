"""
IBVAP Object Detector — YOLOv8 with built-in tracking & Tactical HUD
Detects persons, vehicles, baggage, and border surveillance objects of interest.
Carries bounding box coordinates, velocities, trajectories, and posture metadata.
"""
import math
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


# Colour palette for tactical drawing (BGR)
CATEGORY_COLOURS = {
    "person":     (0, 255, 0),     # Vibrant Green for Humans
    "vehicle":    (0, 255, 255),   # Vibrant Yellow for Vehicles
    "object":     (255, 180, 50),  # Accent cyan/orange for items
    "crawling":   (0, 0, 255),     # Crimson Red for Crawling/Prone
    "loitering":  (0, 215, 255),   # Tactical Amber/Gold for Loitering
    "unattended": (0, 140, 255),   # Orange for Unattended Baggage
    "ingress":    (0, 0, 255),     # Red for Directional Ingress
}

INTRUSION_COLOUR = (0, 0, 255)      # Red for intrusion alerts
ZERO_LINE_COLOUR = (0, 230, 255)    # Yellow/Cyan boundary marker


class ObjectDetector:
    """YOLOv8-based real-time object detector with built-in tracking and tactical telemetry."""

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
        Returns list of detection dicts with coordinates, velocities, trajectories, and posture metadata.
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

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                aspect_ratio = round(w / h, 2)

                # Preliminary posture classification
                posture_type = "crouching" if aspect_ratio > 0.85 else "standing"

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": cls_name,
                    "class_id": cls_id,
                    "category": category,
                    "confidence": round(conf, 2),
                    "track_id": track_id,
                    "centroid": [round(cx, 1), round(cy, 1)],
                    "velocity": {"vx": 0.0, "vy": 0.0, "speed": 0.0},
                    "trajectory": [[round(cx, 1), round(cy, 1)]],
                    "posture": {
                        "posture": posture_type,
                        "aspect_ratio": aspect_ratio,
                        "is_crawling": False,
                        "ground_proximity": False,
                    },
                    "dwell_time": 0.0,
                    "status_tags": [],
                })

        return detections

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        detections: list[dict],
        intrusions: list[dict] | None = None,
        fence_zones: list[dict] | None = None,
        behavioral_events: list[dict] | None = None,
        night_mode: str = "off",
    ) -> np.ndarray:
        """
        Draw military tactical bounding boxes, labels, motion trajectory trails,
        posture classifications, velocity vectors, zero-line fences, and Tactical HUD.
        """
        annotated = frame.copy()
        intrusion_track_ids = set()

        if intrusions:
            intrusion_track_ids = {
                intr["detection"]["track_id"] for intr in intrusions if "detection" in intr
            }

        # ── Draw fence zones & Zero-Lines ───────────────────────
        if fence_zones:
            for zone in fence_zones:
                pts = np.array(zone["points"], dtype=np.int32)
                zone_type = zone.get("type", "zone")
                label = zone.get("name", "Zone")

                if zone_type == "zero_line" or len(pts) == 2:
                    # Directional Zero-Line border barrier
                    p1 = tuple(pts[0])
                    p2 = tuple(pts[1])
                    cv2.line(annotated, p1, p2, ZERO_LINE_COLOUR, 3, cv2.LINE_AA)
                    # Border dashes
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    cv2.putText(
                        annotated, f"BORDER ZERO-LINE: {label.upper()}", (mid_x - 70, max(20, mid_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ZERO_LINE_COLOUR, 2, cv2.LINE_AA
                    )
                else:
                    # Semi-transparent polygon fill
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [pts], (0, 0, 180))
                    cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
                    # Tactical dashed / solid border
                    cv2.polylines(annotated, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
                    # Zone label
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))
                    cv2.putText(
                        annotated, label.upper(), (cx - 35, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA
                    )

        # ── Draw Target Trajectories (Motion Trails) ────────────
        for det in detections:
            traj = det.get("trajectory", [])
            if len(traj) >= 2:
                # Determine trail colour
                posture = det.get("posture", {})
                if posture.get("is_crawling"):
                    trail_colour = (0, 0, 255)
                elif det.get("category") == "person":
                    trail_colour = (0, 255, 120)
                elif det.get("category") == "vehicle":
                    trail_colour = (0, 220, 255)
                else:
                    trail_colour = (255, 180, 50)

                # Draw motion trail connected points with fading thickness
                num_pts = len(traj)
                for i in range(1, num_pts):
                    p_start = (int(traj[i - 1][0]), int(traj[i - 1][1]))
                    p_end = (int(traj[i][0]), int(traj[i][1]))
                    alpha = max(1, int(1 + (i / num_pts) * 2))
                    cv2.line(annotated, p_start, p_end, trail_colour, alpha, cv2.LINE_AA)

                # Draw tiny circle at current centroid
                cur_pt = (int(traj[-1][0]), int(traj[-1][1]))
                cv2.circle(annotated, cur_pt, 3, trail_colour, -1, cv2.LINE_AA)

        # ── Draw Detections & Tactical Bounding Boxes ───────────
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class"]
            track_id = det["track_id"]
            conf = det["confidence"]
            category = det.get("category", "object")
            posture = det.get("posture", {})
            is_crawling = posture.get("is_crawling", False)
            dwell_time = det.get("dwell_time", 0.0)
            status_tags = det.get("status_tags", [])
            unattended_info = det.get("unattended_telemetry", {})
            is_unattended = unattended_info.get("is_unattended", False)

            is_intrusion = track_id in intrusion_track_ids
            has_ingress = any("INGRESS" in tag for tag in status_tags)
            is_loitering = any("LOITERING" in tag or "DWELL" in tag for tag in status_tags) and (dwell_time >= 10.0)
            frs_info = det.get("frs_match")
            anpr_info = det.get("anpr")

            # Determine tactical colour and label tag:
            if frs_info and frs_info.get("is_threat"):
                colour = (0, 0, 255)       # Red for Watchlist Suspect
                thickness = 3
                tag_label = f"⚠ WATCHLIST: {frs_info['name'][:18]} {frs_info['confidence_pct']:.0f}%"
            elif frs_info and not frs_info.get("is_threat"):
                colour = (0, 230, 115)     # Emerald Green for Authorized Personnel
                thickness = 2
                tag_label = f"✓ AUTH: {frs_info['name'][:18]} {frs_info['confidence_pct']:.0f}%"
            elif anpr_info and anpr_info.get("is_hotlist"):
                colour = (0, 0, 255)       # Red for Blacklisted Vehicle
                thickness = 3
                tag_label = f"🚨 BLACKLIST: {anpr_info.get('plate_number', 'FLAGGED')}"
            elif anpr_info and anpr_info.get("plate_number"):
                colour = (0, 255, 255)     # Yellow for Vehicle with Plate
                thickness = 2
                tag_label = f"{cls_name.upper()} [{anpr_info['plate_number']}]"
            elif is_intrusion or has_ingress:
                colour = INTRUSION_COLOUR
                thickness = 3
                tag_label = f"BREACH #{track_id} {conf:.0%}" if track_id >= 0 else f"BREACH {conf:.0%}"
            elif is_crawling:
                colour = (0, 0, 255)       # Alert Red for Crawling/Prone Infiltration
                thickness = 3
                ar = posture.get("aspect_ratio", 1.0)
                tag_label = f"CRAWLING #{track_id} AR:{ar:.2f}"
            elif is_unattended:
                colour = (0, 140, 255)     # Deep Amber for Unattended Baggage
                thickness = 3
                tag_label = f"UNATTENDED {cls_name.upper()} {int(unattended_info.get('stationary_duration', 0))}s"
            elif is_loitering:
                colour = (0, 215, 255)     # Tactical Gold/Amber for Loitering
                thickness = 2
                tag_label = f"LOITERING #{track_id} {int(dwell_time)}s"
            elif category == "person":
                colour = (0, 255, 0)       # Vibrant Green for Humans
                thickness = 2
                posture_tag = f" [{posture.get('posture', 'standing').upper()}]" if posture.get('posture') != 'standing' else ""
                tag_label = f"HUMAN #{track_id} {conf:.0%}{posture_tag}" if track_id >= 0 else f"HUMAN {conf:.0%}"
            elif category == "vehicle":
                colour = (0, 255, 255)     # Vibrant Yellow for Vehicles
                thickness = 2
                tag_label = f"{cls_name.upper()} #{track_id} {conf:.0%}" if track_id >= 0 else f"{cls_name.upper()} {conf:.0%}"
            else:
                colour = (255, 180, 50)    # Cyan/Orange for objects
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
            (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            tag_y1 = max(0, y1 - th - 8)
            tag_y2 = y1
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 8, tag_y2), colour, -1)
            cv2.putText(
                annotated, tag_label, (x1 + 4, tag_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA
            )

            # Velocity vector arrow if moving
            vel = det.get("velocity", {})
            speed = vel.get("speed", 0.0)
            if speed > 10.0:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                vx = vel.get("vx", 0.0)
                vy = vel.get("vy", 0.0)
                # Normalize and scale
                mag = math.hypot(vx, vy)
                if mag > 0:
                    scale = min(30.0, speed * 0.4)
                    arr_end = (int(cx + (vx / mag) * scale), int(cy + (vy / mag) * scale))
                    cv2.arrowedLine(annotated, (cx, cy), arr_end, colour, 2, tipLength=0.35)

            # Face & License Plate Sub-Reticles
            if det.get("face_bbox"):
                fx1, fy1, fx2, fy2 = [int(v) for v in det["face_bbox"]]
                f_col = (0, 0, 255) if (frs_info and frs_info.get("is_threat")) else ((0, 230, 115) if frs_info else (0, 255, 200))
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), f_col, 1, cv2.LINE_AA)
                cv2.putText(annotated, "FRS", (fx1, max(12, fy1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, f_col, 1)

            if det.get("plate_bbox"):
                px1, py1, px2, py2 = [int(v) for v in det["plate_bbox"]]
                p_col = (0, 0, 255) if (anpr_info and anpr_info.get("is_hotlist")) else (0, 255, 255)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), p_col, 2, cv2.LINE_AA)
                cv2.putText(annotated, "ANPR", (px1, max(12, py1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, p_col, 1)

            # Secondary Tactical Warnings below bounding box
            warning_offset = y2 + 18
            if frs_info and frs_info.get("is_threat"):
                cv2.putText(
                    annotated, f"🚨 WATCHLIST SUSPECT: {frs_info['name']} [{frs_info.get('threat_level','HIGH')}]", (x1, warning_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA
                )
                warning_offset += 16
            elif frs_info and not frs_info.get("is_threat"):
                cv2.putText(
                    annotated, f"✓ AUTHORIZED: {frs_info['name']}", (x1, warning_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 115), 2, cv2.LINE_AA
                )
                warning_offset += 16

            if anpr_info and anpr_info.get("is_hotlist"):
                hm = anpr_info.get("hotlist_match", {})
                reason_txt = hm.get("reason", "Flagged Vehicle") if hm else "Flagged Vehicle"
                cv2.putText(
                    annotated, f"🚨 BLACKLIST VEHICLE: {reason_txt}", (x1, warning_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA
                )
                warning_offset += 16

            if is_intrusion:
                cv2.putText(
                    annotated, "⚠ RESTRICTED ZONE BREACH", (x1, warning_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, INTRUSION_COLOUR, 2, cv2.LINE_AA
                )
                warning_offset += 16
            if is_crawling:
                cv2.putText(
                    annotated, "⚠ PRONE INFILTRATION DETECTED", (x1, warning_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA
                )
                warning_offset += 16
            if has_ingress:
                cv2.putText(
                    annotated, "⚠ DIRECTIONAL INGRESS (ZERO-LINE)", (x1, warning_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA
                )
                warning_offset += 16
            if is_unattended:
                cv2.putText(
                    annotated, f"⚠ UNATTENDED ITEM ({int(unattended_info.get('stationary_duration', 0))}s)", (x1, warning_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 140, 255), 2, cv2.LINE_AA
                )

        # ── Tactical HUD Overlay (SSB Tactical Analytics Card) ─
        person_count = sum(1 for d in detections if d["category"] == "person")
        vehicle_count = sum(1 for d in detections if d["category"] == "vehicle")
        crawling_count = sum(1 for d in detections if d.get("posture", {}).get("is_crawling"))
        loitering_count = sum(1 for d in detections if d.get("dwell_time", 0.0) >= 15.0)
        unattended_count = sum(1 for d in detections if d.get("unattended_telemetry", {}).get("is_unattended"))
        watchlist_count = sum(1 for d in detections if d.get("frs_match", {}).get("is_threat"))
        hotlist_count = sum(1 for d in detections if d.get("anpr", {}).get("is_hotlist"))

        hud_w = 230
        hud_h = 100
        if crawling_count > 0:
            hud_h += 18
        if loitering_count > 0:
            hud_h += 18
        if unattended_count > 0:
            hud_h += 18
        if watchlist_count > 0:
            hud_h += 18
        if hotlist_count > 0:
            hud_h += 18
        if intrusions:
            hud_h += 18

        overlay = annotated.copy()
        cv2.rectangle(overlay, (8, 8), (8 + hud_w, 8 + hud_h), (8, 12, 22), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (8, 8), (8 + hud_w, 8 + hud_h), (0, 255, 136), 1)

        y_pos = 24
        cv2.putText(annotated, "SSB IBVAP TACTICAL HUD", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)

        y_pos += 20
        cv2.putText(annotated, f"● PERSONNEL:  {person_count}", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)

        y_pos += 20
        cv2.putText(annotated, f"● VEHICLES:   {vehicle_count}", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA)

        if intrusions:
            y_pos += 20
            cv2.putText(annotated, f"⚠ INTRUSIONS: {len(intrusions)}", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)

        if watchlist_count > 0:
            y_pos += 20
            cv2.putText(annotated, f"🚨 FRS MATCH:  {watchlist_count}", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)

        if hotlist_count > 0:
            y_pos += 20
            cv2.putText(annotated, f"🚨 HOTLIST HIT: {hotlist_count}", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)

        if crawling_count > 0:
            y_pos += 20
            cv2.putText(annotated, f"⚠ CRAWLING:   {crawling_count}", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)

        if loitering_count > 0:
            y_pos += 20
            cv2.putText(annotated, f"⏳ LOITERING:  {loitering_count}", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 2, cv2.LINE_AA)

        if unattended_count > 0:
            y_pos += 20
            cv2.putText(annotated, f"📦 UNATTENDED: {unattended_count}", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 140, 255), 2, cv2.LINE_AA)

        # Border status footer in HUD
        y_pos += 18
        if intrusions or crawling_count > 0:
            cv2.putText(annotated, "STATUS: PERIMETER BREACH", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, "STATUS: BORDER SECURE", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 136), 1, cv2.LINE_AA)

        # ── Tactical Night Vision Badge (Top-Right) ────────────
        if night_mode and str(night_mode).lower() != "off":
            mode_str = str(night_mode).lower().strip()
            labels = {
                "clahe": ("TACTICAL CLAHE+GAMMA", (0, 255, 180)),
                "thermal": ("FLIR THERMAL IRONBOW", (0, 140, 255)),
                "nvg": ("GEN-3+ GREEN NVG", (0, 255, 0)),
            }
            nv_text, nv_col = labels.get(mode_str, (f"NV MODE: {mode_str.upper()}", (0, 255, 136)))
            badge_text = f"● {nv_text}"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            f_w = annotated.shape[1]
            bx2 = f_w - 12
            bx1 = bx2 - tw - 16
            by1 = 8
            by2 = by1 + th + 14

            overlay_badge = annotated.copy()
            cv2.rectangle(overlay_badge, (bx1, by1), (bx2, by2), (8, 12, 22), -1)
            cv2.addWeighted(overlay_badge, 0.80, annotated, 0.20, 0, annotated)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), nv_col, 1)
            cv2.putText(
                annotated,
                badge_text,
                (bx1 + 8, by2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                nv_col,
                1,
                cv2.LINE_AA,
            )

        return annotated

