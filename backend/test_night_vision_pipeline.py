"""
Comprehensive Verification Test Suite for IBVAP Night Vision & Edge Optimization
Sashastra Seema Bal (SSB) — Ministry of Home Affairs

Tests:
1. Tactical CLAHE + Adaptive Gamma (Dark Channel Mean adaptation)
2. Thermal / FLIR False-Color Simulation (COLORMAP_INFERNO)
3. Green Phosphor Night Vision (NVG) with Edge Sharpening
4. Motion-Guided False Alarm Suppression (MOG2 + Directional Displacement)
5. Smart Cadence & ByteTrack Kalman State intermediate box prediction
6. FastAPI POST /api/cameras/{camera_id}/night-mode endpoint validation
"""
import sys
import cv2
import numpy as np
import time
from fastapi.testclient import TestClient

from config import FRAME_WIDTH, FRAME_HEIGHT, DETECTION_INTERVAL
from services.night_enhance import NightEnhancer, MotionFalseAlarmFilter
from services.stream_manager import KalmanBoxTracker, CameraTrackPredictor, CameraStream, StreamManager
from database import Database
from main import app, db, stream_manager


def test_night_enhancer_modes():
    print("\n--- Testing NightEnhancer Multi-Mode Enhancement ---")
    enhancer = NightEnhancer()

    # Create synthetic dark border surveillance frame (dark channel mean ~ 10-15)
    dark_frame = np.random.randint(5, 30, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    # Add simulated warm object (person / engine) in center
    dark_frame[200:300, 400:500] = [80, 90, 85]

    # Test Dark Channel Prior Mean
    d_mean = enhancer.compute_dark_channel_mean(dark_frame)
    gamma = enhancer.compute_adaptive_gamma(d_mean)
    print(f"[CLAHE+Gamma] Dark channel mean: {d_mean:.2f}, Computed Adaptive Gamma: {gamma:.3f}")
    assert 0.35 <= gamma <= 0.65, f"Expected low gamma for dark scene, got {gamma}"

    # Test Mode 1: Tactical CLAHE + Adaptive Gamma
    clahe_out = enhancer.enhance(dark_frame, mode="clahe")
    assert clahe_out.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert np.mean(clahe_out) > np.mean(dark_frame), "CLAHE should brighten low-light scene"
    print(f"[CLAHE+Gamma] Brightness before: {np.mean(dark_frame):.1f} -> after: {np.mean(clahe_out):.1f} (OK)")

    # Test Mode 2: Thermal / FLIR False-Color Simulation (Inferno)
    thermal_out = enhancer.enhance(dark_frame, mode="thermal")
    assert thermal_out.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    # The warm target should show high red/yellow intensity
    warm_roi = thermal_out[220:280, 420:480]
    cold_roi = thermal_out[50:100, 50:100]
    assert np.mean(warm_roi) > np.mean(cold_roi), "Warm object must have higher thermal response than cold background"
    print(f"[FLIR] Warm target mean: {np.mean(warm_roi):.1f} vs Cold ambient: {np.mean(cold_roi):.1f} (OK)")

    # Test Mode 3: Green Phosphor Night Vision (NVG) with Edge Sharpening
    nvg_out = enhancer.enhance(dark_frame, mode="nvg")
    assert nvg_out.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    b, g, r = cv2.split(nvg_out)
    assert np.mean(g) > np.mean(b) * 2.0, "Green channel must dominate in NVG mode"
    print(f"[NVG] Green channel mean: {np.mean(g):.1f} vs Blue channel: {np.mean(b):.1f} (Dominant Green OK)")

    # Test Mode 'off'
    off_out = enhancer.enhance(dark_frame, mode="off")
    assert np.array_equal(off_out, dark_frame), "Mode 'off' must pass through unmodified frame"
    print("[OFF] Mode 'off' pass-through verified (OK)")


def test_motion_false_alarm_suppression():
    print("\n--- Testing Motion-Guided False Alarm Suppression ---")
    motion_filter = MotionFalseAlarmFilter(min_steady_frames=3, min_displacement=6.0)
    cam_id = "test_cam_01"

    base_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 30, dtype=np.uint8)

    # 1. Ephemeral Noise Test (lasts only 1 frame e.g. rain streak or single-frame insect flash)
    det_flash = {"track_id": 101, "bbox": [100, 100, 120, 120], "class": "person", "category": "person"}
    motion_filter.update(cam_id, base_frame, [det_flash])
    valid, reason = motion_filter.is_steady_motion(cam_id, det_flash)
    assert not valid, "Ephemeral noise (1 frame) must be rejected"
    print(f"[Motion Filter] Frame 1 Ephemeral Noise rejected: '{reason}' (OK)")

    # 2. Ephemeral Noise Test (2 frames)
    det_flash_2 = {"track_id": 101, "bbox": [102, 101, 122, 121], "class": "person", "category": "person"}
    motion_filter.update(cam_id, base_frame, [det_flash_2])
    valid, reason = motion_filter.is_steady_motion(cam_id, det_flash_2)
    assert not valid, "Ephemeral noise (2 frames) must still be rejected"
    print(f"[Motion Filter] Frame 2 Ephemeral Noise rejected: '{reason}' (OK)")

    # 3. Oscillating Foliage Test (tree branch shaking back and forth)
    cam_id_foliage = "cam_foliage"
    track_foliage = 202
    positions = [
        [300, 300, 340, 340],
        [308, 300, 348, 340],  # move right 8px
        [300, 300, 340, 340],  # swing back left 8px
        [307, 300, 347, 340],  # swing right again
        [301, 300, 341, 340],  # swing back to near origin (net disp ~ 1px, path ~ 25px)
    ]
    for bbox in positions:
        det = {"track_id": track_foliage, "bbox": bbox, "class": "person", "category": "person"}
        motion_filter.update(cam_id_foliage, base_frame, [det])

    valid, reason = motion_filter.is_steady_motion(cam_id_foliage, det)
    assert not valid, "Cyclic foliage oscillation must be rejected as false alarm"
    print(f"[Motion Filter] Cyclic Foliage Oscillation rejected: '{reason}' (OK)")

    # 4. Genuine Moving Intruder (steady directional displacement across 3+ frames)
    cam_id_intruder = "cam_intruder"
    track_intruder = 303
    intruder_positions = [
        [100, 200, 140, 280],
        [110, 202, 150, 282],  # moving right ~10px
        [122, 204, 162, 284],  # moving right ~12px
        [135, 205, 175, 285],  # moving right ~13px (net disp ~ 35px, linear path)
    ]
    for bbox in intruder_positions:
        # Create active foreground motion mask in MOG2
        f = base_frame.copy()
        f[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 180  # bright moving intruder
        det = {"track_id": track_intruder, "bbox": bbox, "class": "person", "category": "person"}
        motion_filter.update(cam_id_intruder, f, [det])

    valid, reason = motion_filter.is_steady_motion(cam_id_intruder, det)
    assert valid, f"Genuine intruder must be accepted, got {reason}"
    print(f"[Motion Filter] Steady Directional Intruder accepted: '{reason}' (OK)")


def test_kalman_state_cadence():
    print("\n--- Testing ByteTrack Kalman State Intermediate Cadence ---")
    predictor = CameraTrackPredictor()

    # Frame 0 (YOLO Detection Frame): target at (100, 100, 160, 200) -> center (130, 150)
    det0 = [{"track_id": 1, "bbox": [100, 100, 160, 200], "class": "person", "category": "person", "confidence": 0.90}]
    predictor.update_from_yolo(det0)

    # Frame 1 (YOLO Detection Frame): target moves to (110, 105, 170, 205) -> velocity vx=+10, vy=+5
    det1 = [{"track_id": 1, "bbox": [110, 105, 170, 205], "class": "person", "category": "person", "confidence": 0.91}]
    predictor.update_from_yolo(det1)

    # Frame 2 (Skipped Intermediate Frame): Kalman prediction without YOLO
    t0 = time.perf_counter()
    pred_dets = predictor.predict_intermediate()
    t_elapsed = (time.perf_counter() - t0) * 1000.0

    assert len(pred_dets) == 1
    pred_box = pred_dets[0]["bbox"]
    # Box should have moved right and down according to velocity
    assert pred_box[0] > 110, f"Kalman prediction should advance x, got {pred_box[0]}"
    print(f"[Kalman Cadence] Predicted Bounding Box: {pred_box} in {t_elapsed:.3f} ms (OK)")


def test_api_night_mode_endpoints():
    print("\n--- Testing FastAPI POST /api/cameras/{camera_id}/night-mode ---")
    with TestClient(app) as client:
        import main
        cam_id = "test_ssb_cam_01"
        # Ensure camera exists in db
        main.db.add_camera(cam_id, "SSB BOP-12 Checkpost", "0", "Sector 4")

        # 1. Test mode = 'clahe'
        r1 = client.post(f"/api/cameras/{cam_id}/night-mode", json={"mode": "clahe"})
        assert r1.status_code == 200, r1.text
        assert r1.json()["night_mode"] == "clahe"
        print(f"[API] Set mode 'clahe': {r1.json()} (OK)")

        # 2. Test mode = 'thermal'
        r2 = client.post(f"/api/cameras/{cam_id}/night-mode", json={"mode": "thermal"})
        assert r2.status_code == 200, r2.text
        assert r2.json()["night_mode"] == "thermal"
        print(f"[API] Set mode 'thermal': {r2.json()} (OK)")

        # 3. Test mode = 'nvg'
        r3 = client.post(f"/api/cameras/{cam_id}/night-mode", json={"mode": "nvg"})
        assert r3.status_code == 200, r3.text
        assert r3.json()["night_mode"] == "nvg"
        print(f"[API] Set mode 'nvg': {r3.json()} (OK)")

        # 4. Test mode = 'off'
        r4 = client.post(f"/api/cameras/{cam_id}/night-mode", json={"mode": "off"})
        assert r4.status_code == 200, r4.text
        assert r4.json()["night_mode"] == "off"
        print(f"[API] Set mode 'off': {r4.json()} (OK)")

        # 5. Test legacy enabled=True -> 'clahe'
        r5 = client.post(f"/api/cameras/{cam_id}/night-mode", json={"enabled": True})
        assert r5.status_code == 200, r5.text
        assert r5.json()["night_mode"] == "clahe"
        print(f"[API] Legacy enabled=True mapped to 'clahe': {r5.json()} (OK)")

        # 6. Test invalid mode rejection
        r6 = client.post(f"/api/cameras/{cam_id}/night-mode", json={"mode": "invalid_mode_xyz"})
        assert r6.status_code == 400, "Invalid mode must return HTTP 400"
        print(f"[API] Invalid mode correctly rejected with 400: {r6.json()} (OK)")



if __name__ == "__main__":
    print("=================================================================")
    print("  IBVAP Night Vision & Edge Optimization Test Suite")
    print("=================================================================")
    test_night_enhancer_modes()
    test_motion_false_alarm_suppression()
    test_kalman_state_cadence()
    test_api_night_mode_endpoints()
    print("\n=================================================================")
    print("  ALL TESTS PASSED SUCCESSFULLY! ✓")
    print("=================================================================")
