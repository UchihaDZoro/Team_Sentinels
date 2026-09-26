"""
IBVAP Night-time Enhancement Service
Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
low-light / night-time surveillance footage without requiring IR cameras.
"""
import cv2
import numpy as np

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE


class NightEnhancer:
    """Enhances dark / night-time frames for better AI detection."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_SIZE,
        )

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement to a BGR frame."""
        # Convert to LAB colour space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to luminance channel only
        l_enhanced = self.clahe.apply(l_channel)

        # Merge channels and convert back
        merged = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        return enhanced

    def is_dark_frame(self, frame: np.ndarray, threshold: float = 60.0) -> bool:
        """Fast heuristic: returns True if average brightness is below threshold."""
        # Fast subsampling on every 8th pixel across channels (<0.05ms)
        sample = frame[::8, ::8, :]
        return float(np.mean(sample)) < threshold
