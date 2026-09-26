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
EVENTS_DIR = DATA_DIR / "events"

# Create directories
for d in [DATA_DIR, SNAPSHOTS_DIR, EVENTS_DIR, DEMO_VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Database ────────────────────────────────────────────────────
DATABASE_PATH = str(DATA_DIR / "ibvap.db")

# ── Detection ───────────────────────────────────────────────────
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26n.pt")
THREAT_MODEL = os.getenv("THREAT_MODEL", "threat_yolov8n.pt")
ANPR_MODEL = os.getenv("ANPR_MODEL", "license_plate_yolov8n.pt")
POSE_MODEL = os.getenv("POSE_MODEL", "yolov8n-pose.pt")
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "512"))
# High recall for persons & vehicles in crowds (0.18 so all people are detected)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.18"))
THREAT_CONFIDENCE_THRESHOLD = float(os.getenv("THREAT_CONFIDENCE_THRESHOLD", "0.28"))
# Higher NMS IoU threshold so overlapping people in dense crowds are not deleted
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.65"))

# ── Real-Time Zero-Lag Streaming (Buffer System Completely Removed) ─
REALTIME_CAPTURE_BUFFER_SIZE = int(os.getenv("REALTIME_CAPTURE_BUFFER_SIZE", "2"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "30"))
TARGET_BUFFER_SECONDS = 0.0
MAX_LATENCY_SECONDS = 1.0

MAX_ALLOWED_FRAME_AGE_MS = 200.0
TARGET_LATENCY_MS = 50.0
ENABLE_HAND_ROI_THREATS = False
MAX_TRAJECTORY_HISTORY = 30
MAX_PROJECTILE_PREDICT_FRAMES = 3
OCR_INTERVAL_FRAMES = int(os.getenv("OCR_INTERVAL_FRAMES", "3"))
OCR_VOTING_WINDOW = int(os.getenv("OCR_VOTING_WINDOW", "5"))

# COCO classes of interest for surveillance + negative suppression
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
    29: "frisbee",     # suppress round disc false alarms
    32: "sports ball", # suppress ball/stone false alarms for grenades
    34: "baseball bat",# potential blunt weapon
    39: "bottle",      # suppress bottle false alarms for explosives
    41: "cup",         # suppress cup false alarms
    43: "knife",       # bladed weapon
    47: "apple",       # suppress round fruit false alarms
    49: "orange",      # suppress round fruit false alarms
    65: "remote",      # suppress remote false alarms
    67: "cell phone",  # suppress phone false alarms for pistol/detonator
    76: "scissors",    # sharp weapon
}

PERSON_CLASSES = {0}
VEHICLE_CLASSES = {1, 2, 3, 5, 7}

# Classes used for negative false-alarm suppression against weapons
BENIGN_SUPPRESSION_CLASSES = {
    29: "frisbee",
    32: "sports ball",
    39: "bottle",
    41: "cup",
    47: "apple",
    49: "orange",
    65: "remote",
    67: "cell phone",
}

# ── Suspicious & Harmful Objects Catalog ────────────────────────
# Master threat catalog covering firearms, explosives, grenades, bladed weapons
# Calibrated thresholds: fast reaction for weapons & explosions, no artificial area cap on explosions
THREAT_CLASSES = {
    # Firearms & Guns (accessible at conf >= 0.32, sensitive to drawn/aimed weapons)
    "gun": {"label": "FIREARM / GUN", "category": "threat", "severity": "critical", "min_conf": 0.32},
    "pistol": {"label": "PISTOL", "category": "threat", "severity": "critical", "min_conf": 0.32},
    "handgun": {"label": "HANDGUN", "category": "threat", "severity": "critical", "min_conf": 0.32},
    "rifle": {"label": "RIFLE", "category": "threat", "severity": "critical", "min_conf": 0.32},
    "firearm": {"label": "FIREARM", "category": "threat", "severity": "critical", "min_conf": 0.32},
    # Explosives & Hand Grenades (conf >= 0.30, supports small airborne grenades & huge explosions)
    "grenade": {"label": "HAND GRENADE", "category": "threat", "severity": "critical", "min_conf": 0.30},
    "explosion": {"label": "EXPLOSION / BLAST", "category": "threat", "severity": "critical", "min_conf": 0.28},
    "bomb": {"label": "BOMB / EXPLOSIVE", "category": "threat", "severity": "critical", "min_conf": 0.32},
    # Bladed & Sharp Weapons
    "knife": {"label": "BLADE / KNIFE", "category": "threat", "severity": "high", "min_conf": 0.38},
    "dagger": {"label": "DAGGER", "category": "threat", "severity": "high", "min_conf": 0.38},
    "scissors": {"label": "SHARP OBJECT", "category": "threat", "severity": "medium", "min_conf": 0.45},
    # Blunt Weapons
    "baseball bat": {"label": "BLUNT WEAPON", "category": "threat", "severity": "medium", "min_conf": 0.45},
}

# ── Crowd Detection Parameters ──────────────────────────────────
# 3 or more persons close together triggers spatial gathering analysis
CROWD_MIN_PERSONS = 3
CROWD_DISTANCE_THRESHOLD = 140.0


# ── Streaming ───────────────────────────────────────────────────
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
JPEG_QUALITY = 70
TARGET_FPS = 30
BOUNDED_BUFFER_SLOT_SIZE = 1      # strict 1-slot freshest frame buffer (zero latency accumulation)

WATCHDOG_CHECK_INTERVAL = 2.0     # seconds between stream health audits
WATCHDOG_STALL_TIMEOUT = 12.0     # max seconds without frame before auto-recovery (must exceed HLS segment duration)
MAX_RECONNECT_BACKOFF = 16.0      # max backoff delay in seconds

# ── Alerts ──────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 8   # min gap between same alert type per camera
MAX_ALERTS_STORED = 500

# ── Night Enhancement (CLAHE) ──────────────────────────────────
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)

