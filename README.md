# 🛡️ IBVAP — Intelligent Border Video Analytics Platform

**SIH 2026 | Problem Statement ID: 26187**  
**Organization:** Ministry of Home Affairs — Sashastra Seema Bal (SSB)

---

## Overview

IBVAP is an AI-driven software platform that transforms existing CCTV infrastructure into an intelligent surveillance network — **without requiring dedicated FRS, ANPR, or smart-camera hardware**.

The platform ingests live video streams from standard IP-based CCTV cameras and performs real-time video analytics using state-of-the-art **YOLO26** (NMS-free end-to-end Computer Vision), delivering:

- 👥 **Human Detection & Tracking** — Real-time person detection with multi-object tracking (Green bounding box)
- 🚗 **Vehicle Detection & Classification** — Cars, trucks, bikes, buses with tracking IDs (Yellow bounding box)
- 🔲 **Virtual Fence / Intrusion Detection** — Draw restricted zones, get instant alerts
- 🌙 **Night-time Enhancement** — CLAHE-based low-light processing without IR cameras
- 🚨 **Real-time Alert Generation** — WebSocket push alerts with sound notifications
- 📊 **Event Logging & Analytics** — Historical alert data with trend analysis

## Architecture

```
CCTV Camera (RTSP/File/Webcam)
        │
        ▼
┌─────────────────────────────┐
│   Stream Ingestor (OpenCV)  │
│         │                   │
│    Night Enhancement ◄──────┤ (auto-detect dark frames)
│         │                   │
│   YOLO26 Detection          │
│   + ByteTrack Tracking      │
│         │                   │
│   Virtual Fence Check ──────┤──► Alert Engine ──► WebSocket Push
│         │                   │         │
│   Frame Annotation          │    SQLite Storage
│         │                   │
│   MJPEG Stream Output       │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   FastAPI Backend           │
│   - REST API                │
│   - WebSocket (alerts)      │
│   - MJPEG Streaming         │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Web Dashboard             │
│   - Multi-camera grid       │
│   - Real-time alerts        │
│   - Virtual fence editor    │
│   - Analytics               │
└─────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| AI Detection | YOLOv8 (ultralytics) — Pre-trained COCO model |
| Tracking | BoT-SORT (built-in with ultralytics) |
| Video Processing | OpenCV + FFmpeg |
| Backend | FastAPI + WebSocket + MJPEG Streaming |
| Database | SQLite (WAL mode) |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Night Enhancement | CLAHE (Contrast Limited Adaptive Histogram Equalization) |

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone & Setup

```bash
cd CCTron
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Demo Videos (Optional)

Place any `.mp4` / `.avi` video files in `backend/demo_videos/` folder.
They'll appear in the "Quick Add" section of the Add Camera dialog.

Suggested sources:
- [VIRAT Dataset](https://viratdata.org/) — Real surveillance footage
- [MOT Challenge](https://motchallenge.net/) — Multi-object tracking videos
- Any road/corridor recording from a phone camera

### 3. Run the Server

```bash
python main.py
```

### 4. Open Dashboard

Visit **http://localhost:8000** in your browser.

### 5. Add Cameras

1. Click **+ ADD CAMERA**
2. Enter a name (e.g., "BOP-17 Main Gate")
3. Enter the video source:
   - Video file: `D:\videos\border_footage.mp4`
   - Webcam: `0`
   - RTSP stream: `rtsp://192.168.1.100:554/stream1`
4. Click **ADD & START**

### 6. Set Virtual Fence

1. Click the 🔲 button on any camera card
2. Click to draw polygon points on the frame
3. Right-click or press Enter to close the polygon
4. Click **SAVE FENCE**
5. Any person/vehicle entering the zone will trigger an alert! 🚨

## Features in Detail

### Human & Vehicle Detection
Uses YOLOv8n (nano) model pre-trained on COCO dataset. Detects 80+ object classes with focus on surveillance-relevant categories: persons, cars, trucks, buses, motorcycles, bicycles, backpacks, suitcases.

### Multi-Object Tracking
BoT-SORT tracking assigns persistent IDs to detected objects across frames, enabling trajectory analysis and re-identification.

### Virtual Fence
Operators define polygon zones on any camera feed. The system uses Shapely computational geometry to check if detected objects' foot-points fall within restricted zones. Intrusions trigger instant alerts.

### Night-time Enhancement
CLAHE (Contrast Limited Adaptive Histogram Equalization) applied to the luminance channel in LAB colour space. Auto-detects dark frames based on average brightness threshold.

### Alert System
- **Cooldown deduplication** — Prevents alert spam (8-second cooldown per alert type per camera)
- **Severity levels** — Critical, High, Medium, Low
- **Snapshot capture** — Saves JPEG snapshot on alert
- **Real-time push** — WebSocket push to all connected dashboards
- **Audio notification** — Web Audio API beep on new alerts

## Project Structure

```
CCTron/
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── config.py                   # Configuration
│   ├── database.py                 # SQLite database layer
│   ├── requirements.txt            # Python dependencies
│   ├── services/
│   │   ├── detector.py             # YOLOv8 detection + tracking
│   │   ├── stream_manager.py       # Camera stream processing pipeline
│   │   ├── virtual_fence.py        # Zone intrusion detection
│   │   ├── alert_engine.py         # Alert management
│   │   └── night_enhance.py        # Night-time enhancement
│   ├── data/                       # SQLite DB + snapshots
│   └── demo_videos/                # Sample surveillance footage
│
├── frontend/
│   ├── index.html                  # Dashboard
│   ├── css/styles.css              # Military-themed styling
│   └── js/app.js                   # Frontend logic
│
└── README.md
```

## Cost Comparison

| Solution | Cost per Camera | Hardware Required |
|----------|----------------|-------------------|
| Commercial FRS/ANPR | ₹5,00,000+ | Dedicated smart cameras |
| IBVAP | ~₹5,000 | Existing CCTV + mini-PC |
| **Savings** | **~100x** | **Zero additional hardware** |

## Scalability

- Edge deployment: Works on ₹30K mini-PCs (Intel NUC, Raspberry Pi 5)
- Cloud deployment: Scale horizontally with Docker containers
- Supports 4-8 cameras per instance on a standard laptop
- GPU acceleration optional (CUDA) for 15-30+ cameras

## Team

Built for **Smart India Hackathon 2026** — Problem Statement 26187

---

*IBVAP — Transforming existing CCTV infrastructure into intelligent surveillance through AI*
