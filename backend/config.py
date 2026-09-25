"""
IBVAP Configuration
Intelligent Border Video Analytics Platform
"""
import os
from pathlib import Path

# ── Base Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
DEMO_VIDEOS_DIR = BASE_DIR / "demo_videos"
DATA_DIR = BASE_DIR / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"

# Create directories
for d in [DATA_DIR, SNAPSHOTS_DIR, DEMO_VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Database ────────────────────────────────────────────────────
DATABASE_PATH = str(DATA_DIR / "ibvap.db")

# ── Detection ───────────────────────────────────────────────────
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))

# COCO classes of interest for border surveillance
CLASSES_OF_INTEREST = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",        # drone-like object
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
}

PERSON_CLASSES = {0}
VEHICLE_CLASSES = {1, 2, 3, 5, 7}

# ── Streaming ───────────────────────────────────────────────────
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
JPEG_QUALITY = 70
TARGET_FPS = 20

# ── Alerts ──────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 8   # min gap between same alert type per camera
MAX_ALERTS_STORED = 500

# ── Night Enhancement (CLAHE) ──────────────────────────────────
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)
