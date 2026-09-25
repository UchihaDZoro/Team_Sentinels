"""
IBVAP Night Vision & Edge Optimization Service
Sashastra Seema Bal (SSB) — Ministry of Home Affairs

Features:
1. Tactical CLAHE + Adaptive Gamma correction (I_out = 255 * (I_in/255)^gamma)
   where gamma dynamically adapts to the Dark Channel Prior mean.
2. Thermal / FLIR False-Color Simulation (COLORMAP_INFERNO Ironbow) highlighting
   human and vehicle heat signatures against cool ambient terrain.
3. Green Phosphor Night Vision (NVG) with high-frequency edge sharpening.
4. Motion-Guided False Alarm Suppression (MOG2 + Directional Displacement across 3+ frames)
   filtering out insect flashes, rain streaks, and wind-blown foliage.
"""
import cv2
import numpy as np
import time
from collections import deque
from typing import Optional, Tuple, List, Dict

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE


class NightEnhancer:
    """
    Multi-mode military night vision and low-light enhancement engine.
    Supports:
      - 'clahe': Tactical CLAHE + Adaptive Gamma (dynamic luminance normalization)
      - 'thermal': Thermal / FLIR Ironbow False-Color Simulation
      - 'nvg': Gen 3+ Green Phosphor Night Vision with Edge Sharpening
      - 'off': Pass-through unmodified frame
    """

    def __init__(
        self,
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_size: Tuple[int, int] = CLAHE_TILE_SIZE,
    ):
        # Tactical CLAHE for low-light luminance boost
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size,
        )

        # High-contrast CLAHE for FLIR thermal simulation
        self.thermal_clahe = cv2.createCLAHE(
            clipLimit=4.5,
            tileGridSize=(8, 8),
        )

        # NVG light amplification CLAHE
        self.nvg_clahe = cv2.createCLAHE(
            clipLimit=3.5,
            tileGridSize=(8, 8),
        )

        # High-frequency edge sharpening kernel for NVG intensifier optics
        self.nvg_sharpen_kernel = np.array(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 5.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Precompute lookup tables for common gamma values to accelerate inference
        self._gamma_luts: Dict[int, np.ndarray] = {}

    # ── Dark Channel Prior & Adaptive Gamma ──────────────────────────
    def compute_dark_channel_mean(self, frame: np.ndarray, patch_size: int = 7) -> float:
        """
        Computes the Dark Channel Prior mean of the frame.
        In computer vision (He et al.), J^dark(x) = min_{c in {B,G,R}} I^c(x).
        The mean dark channel reflects how severely underexposed shadows are.
        """
        # Pixel-wise minimum across color channels
        dark_channel = np.min(frame, axis=2)

        if patch_size > 1:
            # Min-filter over local patch
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
            dark_channel = cv2.erode(dark_channel, kernel)

        return float(np.mean(dark_channel))

    def compute_adaptive_gamma(self, dark_mean: float) -> float:
        """
        Dynamically adapts gamma to the frame's dark channel mean.
        Formula: gamma = ln(0.5) / ln(max(D_mean, 4.0) / 255.0) clamped to [0.35, 1.0].
        - Extremely dark scenes (D_mean < 15): gamma -> 0.35 - 0.50 (aggressive tone expansion)
        - Moderate twilight (D_mean ~ 50): gamma -> 0.65 - 0.75
        - Normal/daylight (D_mean >= 120): gamma -> 1.0 (no distortion)
        """
        norm_val = max(dark_mean, 4.0) / 255.0
        # Target midpoint mapping: I_in = norm_val -> I_out = 0.5
        gamma = np.log(0.5) / np.log(norm_val)
        return float(np.clip(gamma, 0.35, 1.0))

    def _get_gamma_lut(self, gamma: float) -> np.ndarray:
        """Fetch or create uint8 lookup table for I_out = 255 * (I_in / 255)^gamma."""
        key = int(round(gamma * 100))
        lut = self._gamma_luts.get(key)
        if lut is None:
            inv_255 = 1.0 / 255.0
            lut = np.array(
                [np.clip(255.0 * ((i * inv_255) ** gamma), 0, 255) for i in range(256)],
                dtype=np.uint8,
            )
            self._gamma_luts[key] = lut
        return lut

    # ── Mode 1: Tactical CLAHE + Adaptive Gamma ─────────────────────
    def tactical_clahe_adaptive_gamma(self, frame: np.ndarray) -> np.ndarray:
        """
        Mode 1: Dynamic luminance normalization with gamma correction:
        I_out = 255 * (I_in / 255)^gamma where gamma adapts dynamically to
        the frame dark channel mean. Operates in LAB space to preserve chromaticity.
        """
        dark_mean = self.compute_dark_channel_mean(frame, patch_size=7)
        gamma = self.compute_adaptive_gamma(dark_mean)

        # Convert to LAB colour space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Step 1: Tactical CLAHE on luminance channel
        l_clahe = self.clahe.apply(l_channel)

        # Step 2: Adaptive Gamma correction via precomputed LUT
        lut = self._get_gamma_lut(gamma)
        l_gamma = cv2.LUT(l_clahe, lut)

        # Step 3: Dynamic range stretch to full 0..255 range if low dynamic range
        l_norm = cv2.normalize(l_gamma, None, 0, 255, cv2.NORM_MINMAX)

        # Recombine with original chromaticity channels
        merged = cv2.merge([l_norm, a_channel, b_channel])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # ── Mode 2: Thermal / FLIR False-Color Simulation ───────────────
    def thermal_flir_simulation(
        self, frame: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO
    ) -> np.ndarray:
        """
        Mode 2: Military infrared / thermal color mapping (FLIR Ironbow / Inferno).
        Highlights human and vehicle heat signatures against cooler backgrounds.
        - Cold ambient terrain (ground, sky, shadows) -> deep purple / black
        - Warm bodies and vehicles -> fiery crimson, amber, and glowing yellow/white
        """
        # Convert to single-channel luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Expand dynamic range with thermal CLAHE
        gray_eq = self.thermal_clahe.apply(gray)

        # High-pass unsharp mask to accentuate warm silhouettes and contours
        blurred = cv2.GaussianBlur(gray_eq, (0, 0), sigmaX=3.0)
        heat_boost = cv2.addWeighted(gray_eq, 1.6, blurred, -0.6, 0)

        # Emissivity tone curve: depress cold background floor, boost hot signatures
        # Normalise to 0..255
        norm = cv2.normalize(heat_boost, None, 0, 255, cv2.NORM_MINMAX)

        # Power curve (gamma ~ 1.2) to push cool background to deep purple/black
        # and keep hot signatures bright
        lut_thermal = self._get_gamma_lut(1.20)
        thermal_scaled = cv2.LUT(norm, lut_thermal)

        # Apply military FLIR colormap (Inferno / Ironbow)
        flir_frame = cv2.applyColorMap(thermal_scaled, colormap)
        return flir_frame

    # ── Mode 3: Green Phosphor Night Vision (NVG) ───────────────────
    def green_phosphor_nvg(
        self, frame: np.ndarray, blend_summer: bool = False
    ) -> np.ndarray:
        """
        Mode 3: Gen 3+ Military Green Phosphor Night Vision (NVG).
        Applies P43 phosphor gain matrix (545 nm green peak) with high-frequency
        edge sharpening for crisp fence and intruder silhouette identification.
        """
        # Convert to grayscale luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Photomultiplier gain amplification
        amplified = self.nvg_clahe.apply(gray)

        # High-frequency edge sharpening filter (microchannel plate halo effect)
        sharpened = cv2.filter2D(amplified, -1, self.nvg_sharpen_kernel)

        # P43 Green Phosphor Gain Matrix:
        # Peak 545nm green emission with subtle blue/amber phosphor persistence
        b_channel = (sharpened * 0.08).astype(np.uint8)
        g_channel = np.clip(sharpened.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
        r_channel = (sharpened * 0.16).astype(np.uint8)
        nvg_bgr = cv2.merge([b_channel, g_channel, r_channel])

        if blend_summer:
            summer = cv2.applyColorMap(sharpened, cv2.COLORMAP_SUMMER)
            nvg_bgr = cv2.addWeighted(nvg_bgr, 0.70, summer, 0.30, 0)

        return nvg_bgr

    # ── Unified Enhancement Dispatcher ──────────────────────────────
    def enhance(self, frame: np.ndarray, mode: str = "clahe") -> np.ndarray:
        """
        Apply requested enhancement mode:
          - 'clahe': Tactical CLAHE + Adaptive Gamma
          - 'thermal': Thermal / FLIR False-Color Simulation (Inferno)
          - 'nvg': Military Green Phosphor NVG with Edge Sharpening
          - 'off': Raw video pass-through
        """
        mode_clean = str(mode).lower().strip()
        if mode_clean == "off":
            return frame
        elif mode_clean == "thermal":
            return self.thermal_flir_simulation(frame)
        elif mode_clean == "nvg":
            return self.green_phosphor_nvg(frame)
        elif mode_clean == "clahe":
            return self.tactical_clahe_adaptive_gamma(frame)
        else:
            # Default fallback: Tactical CLAHE + Adaptive Gamma
            return self.tactical_clahe_adaptive_gamma(frame)

    def is_dark_frame(self, frame: np.ndarray, threshold: float = 60.0) -> bool:
        """Heuristic: returns True if average frame brightness is below threshold."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < threshold


# ═════════════════════════════════════════════════════════════════════
# Motion-Guided False Alarm Suppression Engine
# ═════════════════════════════════════════════════════════════════════
class MotionFalseAlarmFilter:
    """
    Suppresses border surveillance false alarms caused by:
      - Insects buzzing near infrared illuminators (erratic flutter, ephemeral 1-2 frames)
      - Rain streaks and particulate flashes (ephemeral 1-2 frames)
      - Wind-blown trees, leaves, and foliage (cyclic oscillation, near-zero net displacement)

    Validation Criteria:
      Only allows intrusion alerts for objects that exhibit steady directional
      displacement across 3+ consecutive frames and possess active foreground motion energy.
    """

    def __init__(
        self,
        history: int = 150,
        var_threshold: float = 25.0,
        min_steady_frames: int = 3,
        min_displacement: float = 6.0,
        min_linearity_ratio: float = 0.30,
        min_foreground_ratio: float = 0.02,
    ):
        self.history = history
        self.var_threshold = var_threshold
        self.min_steady_frames = min_steady_frames
        self.min_displacement = min_displacement
        self.min_linearity_ratio = min_linearity_ratio
        self.min_foreground_ratio = min_foreground_ratio

        # Per-camera MOG2 subtractors: camera_id -> cv2.BackgroundSubtractorMOG2
        self._subtractors: Dict[str, cv2.BackgroundSubtractorMOG2] = {}

        # Per-camera latest foreground masks: camera_id -> np.ndarray
        self._fg_masks: Dict[str, np.ndarray] = {}

        # Per-camera track histories:
        # camera_id -> track_id -> deque of {"centroid": (cx, cy), "frame_idx": int, "bbox": bbox, "time": float}
        self._track_histories: Dict[str, Dict[int, deque]] = {}

        # Per-camera frame index counters
        self._frame_indices: Dict[str, int] = {}

    def _get_subtractor(self, camera_id: str) -> cv2.BackgroundSubtractorMOG2:
        """Retrieve or create MOG2 background subtractor for camera."""
        if camera_id not in self._subtractors:
            self._subtractors[camera_id] = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=False,
            )
        return self._subtractors[camera_id]

    def update(
        self,
        camera_id: str,
        frame: np.ndarray,
        detections: List[Dict],
    ) -> np.ndarray:
        """
        Process incoming frame:
          1. Updates MOG2 background model and extracts foreground mask.
          2. Updates trajectory histories for all active detection track IDs.
          3. Prunes expired tracks.
        Returns the binary foreground mask.
        """
        subtractor = self._get_subtractor(camera_id)

        # Downscale slightly for high-speed MOG2 background model update
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (w // 2, h // 2)) if (w > 640) else frame
        small_mask = subtractor.apply(small_frame)

        if small_frame is not frame:
            fg_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            fg_mask = small_mask

        # Clean noise with morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        self._fg_masks[camera_id] = fg_mask

        # Advance frame counter
        curr_frame_idx = self._frame_indices.get(camera_id, 0) + 1
        self._frame_indices[camera_id] = curr_frame_idx

        # Update track trajectories
        cam_tracks = self._track_histories.setdefault(camera_id, {})
        active_ids = set()

        for det in detections:
            track_id = det.get("track_id", -1)
            if track_id < 0:
                continue

            active_ids.add(track_id)
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) * 0.5
            cy = (bbox[1] + bbox[3]) * 0.5

            if track_id not in cam_tracks:
                cam_tracks[track_id] = deque(maxlen=20)

            cam_tracks[track_id].append(
                {
                    "centroid": (cx, cy),
                    "bbox": bbox,
                    "frame_idx": curr_frame_idx,
                    "time": time.time(),
                }
            )

        # Prune dead tracks (not seen for > 30 frames)
        dead_ids = [
            tid
            for tid, hist in cam_tracks.items()
            if hist and (curr_frame_idx - hist[-1]["frame_idx"] > 30)
        ]
        for tid in dead_ids:
            del cam_tracks[tid]

        return fg_mask

    def is_steady_motion(
        self,
        camera_id: str,
        detection: Dict,
        min_frames: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluates whether a detection represents a genuine moving intruder
        or an ephemeral noise / insect / foliage false alarm.

        Returns (is_valid: bool, reason: str).
        """
        required_frames = min_frames if min_frames is not None else self.min_steady_frames
        track_id = detection.get("track_id", -1)

        # Untracked or transient single-frame detection
        if track_id < 0:
            return False, "untracked_transient_object"

        cam_tracks = self._track_histories.get(camera_id, {})
        history = cam_tracks.get(track_id)

        # ── Test 1: Temporal Persistence across 3+ Frames ─────────────
        if not history or len(history) < required_frames:
            return (
                False,
                f"ephemeral_noise_insufficient_frames ({len(history) if history else 0}/{required_frames})",
            )

        # Extract recent centroids
        centroids = [entry["centroid"] for entry in history]
        x0, y0 = centroids[0]
        xn, yn = centroids[-1]

        # ── Test 2: Net Directional Displacement ──────────────────────
        net_dx = xn - x0
        net_dy = yn - y0
        net_displacement = float(np.hypot(net_dx, net_dy))

        if net_displacement < self.min_displacement:
            return (
                False,
                f"stationary_or_jitter (net_displacement={net_displacement:.1f}px < {self.min_displacement}px)",
            )

        # ── Test 3: Directional Linearity vs Oscillating Foliage ────────
        # Calculate cumulative trajectory path length
        path_length = 0.0
        for i in range(1, len(centroids)):
            step_dx = centroids[i][0] - centroids[i - 1][0]
            step_dy = centroids[i][1] - centroids[i - 1][1]
            path_length += float(np.hypot(step_dx, step_dy))

        linearity_ratio = net_displacement / max(path_length, 1e-3)
        if linearity_ratio < self.min_linearity_ratio:
            # Wind-blown branches oscillate wildly: large path length, tiny net displacement
            return (
                False,
                f"foliage_or_insect_flutter (linearity={linearity_ratio:.2f} < {self.min_linearity_ratio:.2f})",
            )

        # ── Test 4: Foreground Motion Energy (MOG2) ───────────────────
        fg_mask = self._fg_masks.get(camera_id)
        if fg_mask is not None:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = fg_mask.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))

            if x2 > x1 and y2 > y1:
                roi = fg_mask[y1:y2, x1:x2]
                fg_ratio = np.count_nonzero(roi) / float(roi.size)
                if fg_ratio < self.min_foreground_ratio:
                    return (
                        False,
                        f"no_foreground_motion (fg_ratio={fg_ratio:.3f} < {self.min_foreground_ratio})",
                    )

        return True, "steady_directional_displacement_verified"

    def filter_intrusions(
        self,
        camera_id: str,
        intrusions: List[Dict],
    ) -> List[Dict]:
        """
        Filters intrusion list, stripping false alarms (insects, rain, foliage).
        Only retains intrusions exhibiting steady directional displacement across 3+ frames.
        """
        if not intrusions:
            return []

        validated = []
        for intr in intrusions:
            det = intr.get("detection", {})
            is_valid, reason = self.is_steady_motion(camera_id, det)
            if is_valid:
                validated.append(intr)
            else:
                track_id = det.get("track_id", -1)
                # Suppressed false alarm
                # (Optional debug log for operational visibility)
                pass

        return validated

    def reset_camera(self, camera_id: str):
        """Clean up memory when a camera is stopped or deleted."""
        self._subtractors.pop(camera_id, None)
        self._fg_masks.pop(camera_id, None)
        self._track_histories.pop(camera_id, None)
        self._frame_indices.pop(camera_id, None)
