# THIRD-PARTY SOFTWARE NOTICES AND LICENSES

This document contains licensing information and acknowledgments for third-party software, libraries, and open-source research models integrated into or referenced by **IBVAP (Intelligent Border Video Analytics Platform)**.

---

## 1. Primary Reference: Hackfest2k25-kdf
- **Repository:** `https://github.com/UchihaDZoro/Hackfest2k25-kdf`
- **License:** MIT License
- **Copyright (c) 2025 Hackfest Team**
- **Inspected Files & Architecture:**
  - `input_manager.py`: Multi-source input abstraction (RTSP, video files, webcams).
  - `detection.py` / `detection_video.py`: Core detection and tracking patterns.
  - `behavior_analysis.py` / `thermalbehaviour.ipynb`: Behavioral modeling and dwell time analysis.
  - `combined.py` / `live_h_map.py`: Multi-stream spatial heatmap accumulation.
  - `segmentation.py`: Polygon-based spatial containment.
  - `config.yaml` / `requirements.txt`: Configuration schemas and dependencies.
- **Integration in IBVAP:** Adapted input management into a decoupled, non-blocking capture-inference architecture (`stream_manager.py`), enhanced detection into a multi-stage pose and threat localization pipeline (`detector.py`), and unified alerts and telemetry.

---

## 2. Kaggle Open-Source ANPR References
### A. YOLO License Plate Detection
- **Source:** `https://www.kaggle.com/code/benjnb/yolo-license-plate-detection`
- **Author:** Benjamin B.
- **License:** Apache 2.0 / Open Kaggle Notebook License
- **Component Reused / Adapted:** Plate bounding box localization architecture and aspect-ratio normalization for vehicular license plates.

### B. ANPR YOLO & Preprocessing Pipeline
- **Source:** `https://www.kaggle.com/code/omkarg1417/anpr-yolo`
- **Author:** Omkar G.
- **License:** MIT License / Open Kaggle Notebook License
- **Component Reused / Adapted:** Vehicle-crop dual scanning technique (magnifying small distant vehicles before plate detection), bilateral edge-preserving smoothing, and CLAHE adaptive histogram equalization.

### C. YOLOv7 NPR Training & Indian License Plate Grammar
- **Source:** `https://www.kaggle.com/code/gauravcodes/yolov7-npr-training`
- **Author:** Gaurav Codes
- **License:** MIT License / Open Kaggle Notebook License
- **Component Reused / Adapted:** Contextual Indian license plate alphanumeric grammar rules (`[State: 2 chars][District: 2 digits][Series: 1-3 chars][Number: 4 digits]`) and state-code OCR disambiguation mapping (e.g., `0L->DL`, `1H->JH`, `M8->MH`).

---

## 3. Computer Vision & ML Libraries
### A. Ultralytics YOLO & ByteTrack
- **Repository:** `https://github.com/ultralytics/ultralytics`
- **License:** AGPL-3.0 License
- **Copyright (c) Ultralytics Inc.**
- **Usage:** Real-time general object detection (`yolo26n.pt`), pose estimation (`yolov8n-pose.pt`), and multi-object tracking (`ByteTrack`).

### B. EasyOCR
- **Repository:** `https://github.com/JaidedAI/EasyOCR`
- **License:** Apache License 2.0
- **Copyright (c) 2020 Jaided AI**
- **Usage:** Asynchronous optical character recognition on localized, preprocessed license plate crops.

### C. OpenCV (Open Source Computer Vision Library)
- **Repository:** `https://github.com/opencv/opencv` / `opencv-python`
- **License:** Apache License 2.0
- **Usage:** Video frame grabbing, hardware/software decoding, image transformations, bilateral filtering, CLAHE, and visualization overlays.

### D. PyTorch
- **Repository:** `https://github.com/pytorch/pytorch`
- **License:** Modified BSD License
- **Copyright (c) 2016-present, Facebook, Inc / Meta Platforms, Inc. and its affiliates.**
- **Usage:** Deep learning tensor execution and neural network inference.

---

## 4. Backend & Web Frameworks
### A. FastAPI & Starlette
- **Repository:** `https://github.com/tiangolo/fastapi`
- **License:** MIT License
- **Copyright (c) 2018 Sebastián Ramírez**
- **Usage:** High-performance asynchronous REST API, WebSockets, and static asset serving.

### B. Uvicorn
- **Repository:** `https://github.com/encode/uvicorn`
- **License:** BSD-3-Clause License
- **Copyright (c) 2017-present, Encode OSS Ltd.**
- **Usage:** ASGI web server implementation.
