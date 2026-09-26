"""
IBVAP Real-Time Crowd Detection & Spatial Clustering Engine
Incorporates Megvii CrowdDetection & CrowdHuman Principles (CVPR 2020):
1. Soft-NMS / Occlusion-Aware Filtering (prevents greedy NMS from dropping occluded individuals in crowds)
2. Dual-Plane Spatial Clustering: Footpoint ground-plane + Head-box upper-plane association
3. Mutual Occlusion Matrix: Measures inter-pedestrian overlap ratio
4. Adaptive Density Scoring & Sliding-Window Unique Person Tracking
"""
import time
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class CrowdDetector:
    """
    Megvii CrowdDetection-inspired spatial clustering & density analysis engine.
    - Preserves occluded individuals in dense scenes via Soft-NMS re-weighting.
    - Employs CrowdHuman head-to-body dual-plane proximity clustering.
    - Quantifies crowd density, mutual occlusion ratio, and stampede risk levels.
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        distance_threshold: float = 135.0,
        soft_nms_sigma: float = 0.5,
    ):
        self.min_cluster_size = min_cluster_size
        self.distance_threshold = distance_threshold
        self.soft_nms_sigma = soft_nms_sigma
        # camera_id -> dict of track_id -> last_seen_timestamp (sliding window 10s)
        self._active_person_tracks: dict[str, dict[int, float]] = {}

    def apply_megvii_soft_nms(
        self,
        persons: list[dict],
        iou_thresh: float = 0.40,
        min_conf: float = 0.12,
    ) -> list[dict]:
        """
        Megvii CrowdDetection Soft-NMS:
        In dense crowds, standard greedy NMS discards valid pedestrians because IoU > 0.45.
        Soft-NMS decays confidence scores exponentially based on overlap instead of hard-suppressing:
        s_i = s_i * exp(-IoU(M, b_i)^2 / sigma)
        """
        if len(persons) <= 1:
            return persons

        boxes = np.array([p["bbox"] for p in persons], dtype=np.float32)
        scores = np.array([p.get("confidence", 0.5) for p in persons], dtype=np.float32)
        indices = list(range(len(persons)))

        kept = []
        while indices:
            # Pick index with max score
            max_idx = int(np.argmax(scores[indices]))
            best = indices.pop(max_idx)
            kept.append(best)

            if not indices:
                break

            b1 = boxes[best]
            for other in indices:
                b2 = boxes[other]
                # Calculate IoU
                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])

                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih

                if inter > 0:
                    area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
                    area2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
                    iou = inter / (area1 + area2 - inter)

                    if iou > iou_thresh:
                        # Soft-NMS decay
                        decay = np.exp(-(iou ** 2) / self.soft_nms_sigma)
                        scores[other] *= decay

        # Retain detections above minimum confidence
        result = []
        for idx in kept:
            if scores[idx] >= min_conf:
                p = persons[idx].copy()
                p["confidence"] = round(float(scores[idx]), 2)
                result.append(p)
        return result

    def get_tracked_person_count(self, camera_id: str, current_persons: list[dict], now: float) -> int:
        """
        Maintains unique persistent person IDs over a 10-second sliding window,
        preventing double-counting while capturing passing crowd volume.
        """
        if camera_id not in self._active_person_tracks:
            self._active_person_tracks[camera_id] = {}

        tracks = self._active_person_tracks[camera_id]

        # Register current frame tracks
        for p in current_persons:
            tid = p.get("track_id", -1)
            if tid > 0:
                tracks[tid] = now

        # Prune tracks inactive for > 10.0 seconds
        cutoff = now - 10.0
        stale = [tid for tid, last_t in tracks.items() if last_t < cutoff]
        for tid in stale:
            del tracks[tid]

        return len(tracks)

    def detect_crowds(
        self,
        detections: list[dict],
        camera_id: str = "default",
        raw_frame: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """
        Analyze all person detections and group them into spatial clusters
        using Megvii CrowdHuman dual-plane foot + head proximity analysis.
        """
        now = time.time()
        raw_persons = [d for d in detections if d.get("category") == "person"]

        # 1. Apply Megvii Soft-NMS to avoid crowd occlusion dropouts
        persons = self.apply_megvii_soft_nms(raw_persons)
        current_count = len(persons)
        tracked_count = self.get_tracked_person_count(camera_id, persons, now)

        if current_count < self.min_cluster_size:
            return []

        # 2. Extract CrowdHuman Dual-Plane Features: Foot Coordinates & Head Boxes
        feet = []
        heads = []
        heights = []
        bboxes = []

        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            bw = max(x2 - x1, 10.0)
            bh = max(y2 - y1, 10.0)

            foot_x = (x1 + x2) / 2.0
            foot_y = y2
            feet.append([foot_x, foot_y])

            # CrowdHuman Head estimation: top 28% of height, centered
            hx1 = x1 + 0.15 * bw
            hy1 = y1
            hx2 = x2 - 0.15 * bw
            hy2 = y1 + 0.28 * bh
            head_cx = (hx1 + hx2) / 2.0
            head_cy = (hy1 + hy2) / 2.0
            heads.append([head_cx, head_cy])

            heights.append(bh)
            bboxes.append([x1, y1, x2, y2])

        feet_arr = np.array(feet, dtype=np.float32)    # (N, 2)
        heads_arr = np.array(heads, dtype=np.float32)  # (N, 2)
        hts = np.array(heights, dtype=np.float32)       # (N,)
        n = len(feet_arr)

        # 3. Dual-Plane Distance Matrix (Foot Ground-Plane + Head Proximity)
        feet_diff = feet_arr[:, np.newaxis, :] - feet_arr[np.newaxis, :, :]
        feet_dist = np.sqrt(np.sum(feet_diff ** 2, axis=-1))

        heads_diff = heads_arr[:, np.newaxis, :] - heads_arr[np.newaxis, :, :]
        heads_dist = np.sqrt(np.sum(heads_diff ** 2, axis=-1))

        # Height-normalized distance (perspective correction)
        mean_h = np.sqrt(hts[:, np.newaxis] * hts[np.newaxis, :])
        norm_feet_dist = feet_dist / np.maximum(mean_h, 1.0)
        norm_head_dist = heads_dist / np.maximum(mean_h, 1.0)

        # Megvii Dual Proximity: Adjacency if either footpoints are close OR heads are tightly clustered
        adjacency = (
            (feet_dist < self.distance_threshold) |
            (norm_feet_dist < 1.45) |
            (norm_head_dist < 0.95)
        )
        np.fill_diagonal(adjacency, False)

        # 4. Connected Components (BFS Clustering)
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            queue = [i]
            visited.add(i)
            cluster_members = [i]

            while queue:
                curr = queue.pop(0)
                neighbors = np.where(adjacency[curr])[0]
                for nbr in neighbors:
                    if nbr not in visited:
                        visited.add(nbr)
                        cluster_members.append(nbr)
                        queue.append(nbr)

            if len(cluster_members) >= self.min_cluster_size:
                # Mark individual person detections
                for idx, m in enumerate(cluster_members):
                    persons[m]["in_crowd"] = True
                    persons[m]["crowd_id"] = len(clusters) + 1
                    persons[m]["crowd_index"] = idx + 1
                    persons[m]["crowd_total"] = len(cluster_members)

                # Cluster Bounding Box
                m_bboxes = [bboxes[m] for m in cluster_members]
                bx1 = min(b[0] for b in m_bboxes)
                by1 = min(b[1] for b in m_bboxes)
                bx2 = max(b[2] for b in m_bboxes)
                by2 = max(b[3] for b in m_bboxes)

                count = len(cluster_members)
                cluster_w = max(bx2 - bx1, 20.0)
                cluster_h = max(by2 - by1, 20.0)
                cluster_area = cluster_w * cluster_h
                density_score = round(count / (cluster_area / 10000.0), 2)  # persons per 100x100 px

                # 5. Mutual Occlusion Ratio (Megvii Crowd Index)
                total_inter = 0.0
                pair_count = 0
                for mi in range(count):
                    for mj in range(mi + 1, count):
                        b1 = m_bboxes[mi]
                        b2 = m_bboxes[mj]
                        ix1 = max(b1[0], b2[0])
                        iy1 = max(b1[1], b2[1])
                        ix2 = min(b1[2], b2[2])
                        iy2 = min(b1[3], b2[3])
                        iw = max(0.0, ix2 - ix1)
                        ih = max(0.0, iy2 - iy1)
                        inter = iw * ih
                        if inter > 0:
                            a1 = max(1.0, (b1[2]-b1[0]) * (b1[3]-b1[1]))
                            a2 = max(1.0, (b2[2]-b2[0]) * (b2[3]-b2[1]))
                            total_inter += inter / min(a1, a2)
                        pair_count += 1

                mutual_occlusion = round(total_inter / max(pair_count, 1), 2)

                # Density compensation: if heavy mutual occlusion, account for hidden persons
                estimated_count = count
                if mutual_occlusion > 0.35 and count >= 4:
                    occlusion_bonus = int(count * mutual_occlusion * 0.5)
                    estimated_count = count + min(occlusion_bonus, 10)

                # Severity classification
                effective_count = max(count, estimated_count)
                if effective_count >= 20 or (density_score > 3.5 and effective_count >= 12):
                    level = "CRITICAL"
                    severity = "critical"
                    density_label = "MASSIVE GATHERING / STAMPEDE RISK"
                elif effective_count >= 10 or density_score > 2.5:
                    level = "HIGH"
                    severity = "high"
                    density_label = "DENSE CROWD CONGREGATION"
                elif effective_count >= 6:
                    level = "MEDIUM"
                    severity = "medium"
                    density_label = "MODERATE GROUP CONGREGATION"
                else:
                    level = "LOW"
                    severity = "low"
                    density_label = "LOCALIZED GROUP"

                # Convex Hull Footprint
                cluster_feet = feet_arr[cluster_members]
                hull_pts = []
                if len(cluster_feet) >= 3:
                    try:
                        hull = cv2.convexHull(cluster_feet.astype(np.int32))
                        hull_pts = hull.reshape(-1, 2).tolist()
                    except Exception:
                        hull_pts = []

                clusters.append({
                    "cluster_id": len(clusters) + 1,
                    "camera_id": camera_id,
                    "count": count,
                    "estimated_count": effective_count,
                    "current_person_count": current_count,
                    "tracked_person_count": tracked_count,
                    "crowd_level": level,
                    "crowd_threshold": self.min_cluster_size,
                    "severity": severity,
                    "density_label": density_label,
                    "density_score": density_score,
                    "mutual_occlusion": mutual_occlusion,
                    "member_track_ids": [persons[m].get("track_id", -1) for m in cluster_members],
                    "bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                    "center": [float((bx1 + bx2) / 2.0), float((by1 + by2) / 2.0)],
                    "hull_pts": hull_pts,
                    "member_bboxes": m_bboxes,
                    "timestamp": now,
                })

        return clusters

    def get_crowd_telemetry(self, camera_id: str, detections: list[dict]) -> dict:
        """
        Standardized crowd telemetry for metrics dashboard and alerts.
        """
        now = time.time()
        persons = [d for d in detections if d.get("category") == "person"]
        current_count = len(persons)
        tracked_count = self.get_tracked_person_count(camera_id, persons, now)

        if current_count >= 20:
            level = "CRITICAL"
        elif current_count >= 10:
            level = "HIGH"
        elif current_count >= 5:
            level = "MEDIUM"
        elif current_count >= 2:
            level = "LOW"
        else:
            level = "NORMAL"

        return {
            "current_person_count": current_count,
            "tracked_person_count": tracked_count,
            "crowd_level": level,
            "crowd_threshold": self.min_cluster_size,
        }

    @staticmethod
    def annotate_crowds(frame: np.ndarray, crowds: list[dict]) -> np.ndarray:
        """
        Visual annotation of crowd clusters:
        - Semi-transparent amber polygon hull overlay
        - Cluster bounding box and density label
        """
        if not crowds or frame is None:
            return frame

        annotated = frame.copy()
        overlay = annotated.copy()
        has_hull = False

        for c in crowds:
            hull_pts = c.get("hull_pts", [])
            if len(hull_pts) >= 3:
                try:
                    pts = np.array(hull_pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], (0, 140, 255))
                    cv2.polylines(annotated, [pts], True, (0, 200, 255), 2, cv2.LINE_AA)
                    has_hull = True
                except Exception:
                    pass

            bbox = c.get("bbox")
            if bbox and len(bbox) == 4:
                bx1, by1, bx2, by2 = [int(v) for v in bbox]
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 165, 255), 2, cv2.LINE_AA)
                count = c.get("count", c.get("estimated_count", 0))
                density_label = c.get("density_label", "CROWD")
                label = f"[CROWD: {count} PAX | {density_label}]"
                cv2.putText(
                    annotated, label, (bx1, max(22, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2, cv2.LINE_AA
                )

        if has_hull:
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        return annotated
