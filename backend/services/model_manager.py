"""
IBVAP Model Manager
Centralized, hardware-aware model loading and inference profile management.
Supports CUDA GPU and CPU fallback, FP16 half-precision, and profiles:
- FAST (highest FPS)
- BALANCED (optimal balance of accuracy and latency)
- ACCURATE (maximum detection recall)
"""
import os
import torch
from pathlib import Path
from typing import Optional
from ultralytics import YOLO

from config import (
    YOLO_MODEL,
    THREAT_MODEL,
    ANPR_MODEL,
    YOLO_IMGSZ,
    CONFIDENCE_THRESHOLD,
    THREAT_CONFIDENCE_THRESHOLD,
)

BASE_DIR = Path(__file__).parent.parent


class ModelManager:
    """Singleton model registry and inference profile controller."""
    _instance: Optional["ModelManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, profile: str = "balanced"):
        if getattr(self, "_initialized", False):
            return

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_half = (self.device != "cpu")
        self.profile = profile.lower()

        # Profiles configuration
        self.profiles = {
            "fast": {
                "imgsz": 416,
                "conf": 0.22,
                "threat_conf": 0.36,
                "iou": 0.60,
                "person_nms_free": True,
            },
            "balanced": {
                "imgsz": YOLO_IMGSZ,  # 512
                "conf": CONFIDENCE_THRESHOLD,  # 0.18
                "threat_conf": THREAT_CONFIDENCE_THRESHOLD,  # 0.30
                "iou": 0.68,
                "person_nms_free": True,
            },
            "accurate": {
                "imgsz": 640,
                "conf": 0.14,
                "threat_conf": 0.26,
                "iou": 0.72,
                "person_nms_free": True,
            },
        }

        # Model handles
        self.general_model: Optional[YOLO] = None
        self.threat_model: Optional[YOLO] = None
        self.plate_model: Optional[YOLO] = None

        self._load_models()
        self._initialized = True

    def _resolve_model_path(self, model_name: str) -> Path:
        """Resolve model path checking absolute, backend, and ref_repos directories."""
        p = Path(model_name)
        if p.is_absolute() and p.exists():
            return p
        # Check relative to BASE_DIR (backend/)
        b_path = BASE_DIR / model_name
        if b_path.exists():
            return b_path
        # Check relative to PROJECT_DIR (CCTron root)
        proj_path = BASE_DIR.parent / model_name
        if proj_path.exists():
            return proj_path
        # Check Hackfest ref repo
        hf_path = BASE_DIR.parent / "ref_repos" / "Hackfest2k25-kdf" / model_name
        if hf_path.exists():
            return hf_path
        # Search anywhere in ref_repos
        ref_dir = BASE_DIR.parent / "ref_repos"
        if ref_dir.exists():
            matches = list(ref_dir.glob(f"**/{p.name}"))
            if matches:
                return matches[0]
        return b_path

    def _load_models(self):
        """Load and warm up models on the target hardware."""
        print(f"[IBVAP ModelManager] Target Device: {self.device.upper()} (FP16={self.use_half}) | Profile: {self.profile.upper()}")

        # 1. Primary Surveillance Model (Humans, Vehicles, Negative suppression items)
        gen_path = self._resolve_model_path(YOLO_MODEL)
        if gen_path.exists():
            print(f"[IBVAP ModelManager] Loading General Detector: {gen_path.name}")
            self.general_model = YOLO(str(gen_path))
            self.general_model.to(self.device)
        else:
            print(f"[IBVAP ModelManager] ⚠ General detector {gen_path} not found")

        # 2. Specialized Threat Model (Firearms, Explosives, Grenades, Knives)
        threat_path = self._resolve_model_path(THREAT_MODEL)
        if threat_path.exists():
            print(f"[IBVAP ModelManager] Loading Threat Detector: {threat_path.name}")
            self.threat_model = YOLO(str(threat_path))
            self.threat_model.to(self.device)
        else:
            print(f"[IBVAP ModelManager] ⚠ Threat detector {threat_path} not found")

        # 3. License Plate Model
        plate_path = self._resolve_model_path(ANPR_MODEL)
        if plate_path.exists():
            print(f"[IBVAP ModelManager] Loading License Plate Detector: {plate_path.name}")
            self.plate_model = YOLO(str(plate_path))
            self.plate_model.to(self.device)
        else:
            print(f"[IBVAP ModelManager] ⚠ License plate detector {plate_path} not found")

    def get_profile_params(self) -> dict:
        """Retrieve active inference profile parameters."""
        return self.profiles.get(self.profile, self.profiles["balanced"])

    def set_profile(self, profile: str):
        """Switch active profile at runtime."""
        if profile.lower() in self.profiles:
            self.profile = profile.lower()
            print(f"[IBVAP ModelManager] Inference Profile switched to: {self.profile.upper()}")
