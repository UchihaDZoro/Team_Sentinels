"""
IBVAP FRS & ANPR Comprehensive Verification Suite
Tests:
1. FRS Engine: Face extraction, 216-D feature vector, watchlist seeding, cosine matcher.
2. ANPR Engine: Plate localization, character OCR, Indian syntax validator, fuzzy hotlist matcher, rolling scans.
3. Database Layer: frs_watchlist, anpr_hotlist, anpr_scans tables and CRUD.
4. Alert Engine: WATCHLIST_SUSPECT_DETECTED and BLACKLIST_VEHICLE_DETECTED alerts.
5. Stream Manager & Annotation: FRS & ANPR HUD badges.
6. REST API Endpoints: FastAPI TestClient for all FRS and ANPR endpoints.
"""
import os
import cv2
import json
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from config import DATA_DIR, DATABASE_PATH
from database import Database
from services.frs_engine import FRSEngine
from services.anpr_engine import ANPREngine, levenshtein_distance, normalized_similarity
from services.alert_engine import AlertEngine
from services.detector import ObjectDetector


def test_frs_engine():
    print("=== Testing FRS Engine ===")
    test_db = Database(":memory:")
    frs = FRSEngine(db=test_db, similarity_threshold=0.70)

    # 1. Verify pre-seeded items
    watchlist = test_db.list_watchlist()
    print(f"[FRS] Watchlist count: {len(watchlist)}")
    names = [w["name"] for w in watchlist]
    print(f"[FRS] Watchlist names: {names}")
    assert any("Suspect-01" in n for n in names), "Suspect-01 missing from watchlist"
    assert any("Suspect-02" in n for n in names), "Suspect-02 missing from watchlist"
    assert any("Sentry-104" in n for n in names), "Sentry-104 missing from watchlist"

    # 2. Test feature vector calculation
    dummy_face = np.full((120, 120, 3), 150, dtype=np.uint8)
    cv2.circle(dummy_face, (60, 60), 40, (100, 150, 200), -1)
    vec = frs.compute_feature_vector(dummy_face)
    assert len(vec) == 216, f"Expected 216-D vector, got {len(vec)}"
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4, f"Vector not L2 normalized: {norm}"
    print(f"[FRS] Feature vector computed successfully: shape={vec.shape}, norm={norm:.4f}")

    # 3. Test face matching
    tgt01 = test_db.get_watchlist_target("TGT-01")
    assert tgt01 is not None
    # Load or generate face for TGT-01
    img_path = Path(DATA_DIR) / "watchlist_faces" / "TGT-01.jpg"
    assert img_path.exists(), "TGT-01 portrait image should exist on disk"
    tgt_img = cv2.imread(str(img_path))
    match = frs.match_face(tgt_img)
    print(f"[FRS] Matching TGT-01 against watchlist: {match}")
    assert match is not None, "TGT-01 should match itself"
    assert match["target_id"] == "TGT-01"
    assert match["similarity"] >= 0.75, f"Expected >= 0.75 similarity, got {match['similarity']}"
    assert match["is_threat"] is True, "Suspect-01 should be marked as threat"

    # Test Authorized personnel
    tgt03 = test_db.get_watchlist_target("TGT-03")
    assert tgt03 is not None
    img_path_03 = Path(DATA_DIR) / "watchlist_faces" / "TGT-03.jpg"
    tgt_img_03 = cv2.imread(str(img_path_03))
    match_03 = frs.match_face(tgt_img_03)
    print(f"[FRS] Matching Sentry-104 (Authorized): {match_03}")
    assert match_03 is not None
    assert match_03["target_id"] == "TGT-03"
    assert match_03["is_threat"] is False, "Sentry-104 should NOT be marked as threat"

    # 4. Test person crop detection
    person_crop = np.zeros((200, 100, 3), dtype=np.uint8)
    # Draw head in upper region
    cv2.circle(person_crop, (50, 40), 25, (120, 160, 210), -1)
    face_crop, rel_box = frs.extract_face(person_crop)
    assert face_crop is not None
    assert rel_box is not None
    print(f"[FRS] Face extracted from person crop: box={rel_box}")

    # 5. Test adding and deleting person
    new_p = frs.add_person(
        name="Suspect-99 (Test Fugitive)",
        category="infiltrator",
        threat_level="CRITICAL",
        notes="Automated test fugitive",
    )
    assert new_p["id"] in frs.watchlist_cache
    print(f"[FRS] Successfully added new suspect: {new_p['id']} - {new_p['name']}")
    frs.delete_person(new_p["id"])
    assert new_p["id"] not in frs.watchlist_cache
    print("[FRS] Successfully deleted suspect ✓")
    print("FRS Engine tests PASSED ✓\n")


def test_anpr_engine():
    print("=== Testing ANPR Engine ===")
    test_db = Database(":memory:")
    anpr = ANPREngine(db=test_db)

    # 1. Verify hotlist pre-seeded items
    hotlist = test_db.list_hotlist()
    print(f"[ANPR] Hotlist count: {len(hotlist)}")
    plates = [h["plate_number"] for h in hotlist]
    print(f"[ANPR] Hotlist plates: {plates}")
    assert "DL01AB1234" in plates, "DL01AB1234 missing from hotlist"
    assert "HR26DQ9911" in plates, "HR26DQ9911 missing from hotlist"
    assert "UP14CZ5050" in plates, "UP14CZ5050 missing from hotlist"

    # 2. Test Levenshtein distance & fuzzy matching
    dist_exact = levenshtein_distance("DL01AB1234", "DL01AB1234")
    assert dist_exact == 0
    sim_exact = normalized_similarity("DL01AB1234", "DL01AB1234")
    assert sim_exact == 1.0

    # 1 character OCR confusion (e.g. B misread as 8)
    dist_fuzzy = levenshtein_distance("DL01A81234", "DL01AB1234")
    assert dist_fuzzy == 1
    sim_fuzzy = normalized_similarity("DL01A81234", "DL01AB1234")
    assert sim_fuzzy == 0.90, f"Expected 0.90, got {sim_fuzzy}"

    # Fuzzy match against hotlist
    match = anpr.match_hotlist("DL01A81234")
    print(f"[ANPR] Fuzzy matching 'DL01A81234': {match}")
    assert match is not None, "DL01A81234 should fuzzy match DL01AB1234"
    assert match["hotlist_plate"] == "DL01AB1234"
    assert match["distance"] == 1
    assert match["similarity"] >= 0.75

    # 3. Test Indian plate syntax formatting
    raw_ocr = "dl01ab1234"
    formatted = anpr._format_indian_plate(raw_ocr)
    assert formatted == "DL01AB1234", f"Expected DL01AB1234, got {formatted}"

    # Positional confusion correction (e.g. '0' at pos 0 should be 'D', 'O' at pos 2 should be '0')
    confused = "0L01AB1234"
    formatted_confused = anpr._format_indian_plate(confused)
    assert formatted_confused == "DL01AB1234", f"Expected DL01AB1234, got {formatted_confused}"
    print(f"[ANPR] Positional syntax correction: '{confused}' -> '{formatted_confused}' ✓")

    # 4. Test plate localization on vehicle crop
    vehicle_crop = np.full((240, 320, 3), 60, dtype=np.uint8)
    # Draw realistic white license plate in lower quadrant
    cv2.rectangle(vehicle_crop, (80, 160), (240, 200), (240, 240, 240), -1)
    cv2.putText(vehicle_crop, "HR26DQ9911", (90, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    frame[100:340, 100:420] = vehicle_crop

    det = {"bbox": [100, 100, 420, 340], "class": "car", "category": "vehicle"}
    v_match = anpr.process_vehicle_detection(frame, det, camera_id="cam_test")
    assert "plate_bbox" in det, "Should attach plate_bbox to detection"
    assert "anpr" in det, "Should attach anpr info to detection"
    print(f"[ANPR] Detected plate: {det['anpr']} | match={v_match}")

    # 5. Verify rolling scans
    scans = anpr.list_scans(10)
    assert len(scans) > 0, "Scan should be recorded in rolling scans"
    print(f"[ANPR] Recent plate scans recorded: count={len(scans)}, latest={scans[0]['plate_number']}")
    print("ANPR Engine tests PASSED ✓\n")


def test_alert_engine_events():
    print("=== Testing Alert Engine FRS & ANPR Alerts ===")
    test_db = Database(":memory:")
    alert_eng = AlertEngine(db=test_db)
    received_alerts = []
    alert_eng.on_new_alert = lambda a: received_alerts.append(a)

    # 1. Trigger WATCHLIST_SUSPECT_DETECTED
    frs_matches = [{
        "matched": True,
        "target_id": "TGT-01",
        "name": "Suspect-01 (Cross-Border Infiltrator)",
        "category": "infiltrator",
        "threat_level": "CRITICAL",
        "similarity": 0.88,
        "confidence_pct": 88.0,
        "is_threat": True,
    }]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    alerts = alert_eng.process_frs_matches("cam_alpha", frs_matches, frame)
    assert len(alerts) == 1, "Should create 1 FRS alert"
    assert alerts[0]["alert_type"] == "WATCHLIST_SUSPECT_DETECTED"
    assert "Suspect-01" in alerts[0]["message"]
    print(f"[ALERT] FRS alert generated: {alerts[0]['alert_type']} - {alerts[0]['message']}")

    # 2. Trigger BLACKLIST_VEHICLE_DETECTED
    anpr_matches = [{
        "matched": True,
        "is_hotlist": 1,
        "hotlist_plate": "DL01AB1234",
        "detected_plate": "DL01AB1234",
        "reason": "Suspected Contraband Carrier",
        "threat_level": "CRITICAL",
        "vehicle_model": "White Bolero Camper",
        "confidence_pct": 94.0,
    }]
    v_alerts = alert_eng.process_anpr_matches("cam_alpha", anpr_matches, frame)
    assert len(v_alerts) == 1, "Should create 1 ANPR alert"
    assert v_alerts[0]["alert_type"] == "BLACKLIST_VEHICLE_DETECTED"
    assert "DL01AB1234" in v_alerts[0]["message"]
    print(f"[ALERT] ANPR alert generated: {v_alerts[0]['alert_type']} - {v_alerts[0]['message']}")
    print("Alert Engine tests PASSED ✓\n")


def test_tactical_hud_annotations():
    print("=== Testing Tactical HUD Annotations ===")
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    detections = [
        {
            "bbox": [100, 100, 200, 350],
            "class": "person",
            "category": "person",
            "confidence": 0.92,
            "track_id": 1,
            "face_bbox": [120, 110, 180, 170],
            "frs_match": {
                "matched": True,
                "target_id": "TGT-01",
                "name": "Suspect-01 (Infiltrator)",
                "category": "infiltrator",
                "threat_level": "CRITICAL",
                "confidence_pct": 89.0,
                "is_threat": True,
            }
        },
        {
            "bbox": [400, 200, 750, 450],
            "class": "car",
            "category": "vehicle",
            "confidence": 0.95,
            "track_id": 2,
            "plate_bbox": [520, 380, 640, 420],
            "anpr": {
                "plate_number": "DL01AB1234",
                "confidence": 0.94,
                "is_hotlist": True,
                "hotlist_match": {
                    "reason": "Suspected Contraband Carrier"
                }
            }
        }
    ]
    annotated = ObjectDetector.annotate_frame(frame, detections)
    assert annotated.shape == (540, 960, 3)
    assert np.any(annotated > 0), "Annotated frame should contain non-black pixels"
    print("[HUD] Frame successfully annotated with FRS and ANPR badges ✓")
    print("HUD Annotation tests PASSED ✓\n")


def test_api_endpoints():
    print("=== Testing FastAPI REST Endpoints ===")
    from main import app
    with TestClient(app) as client:
        # 1. GET /api/frs/watchlist
        r = client.get("/api/frs/watchlist")
        assert r.status_code == 200
        wl = r.json()["watchlist"]
        print(f"[API] GET /api/frs/watchlist returned {len(wl)} items")
        assert len(wl) > 0

        # 2. POST /api/frs/watchlist
        new_suspect = {
            "name": "Target-Alpha (Infiltration Courier)",
            "category": "courier",
            "threat_level": "HIGH",
            "danger_level": "HIGH",
            "notes": "Border test courier",
        }
        r_add = client.post("/api/frs/watchlist", json=new_suspect)
        assert r_add.status_code == 200
        added_tgt = r_add.json()["target"]
        tgt_id = added_tgt["id"]
        print(f"[API] POST /api/frs/watchlist created target: {tgt_id}")

        # 3. DELETE /api/frs/watchlist/{id}
        r_del = client.delete(f"/api/frs/watchlist/{tgt_id}")
        assert r_del.status_code == 200
        print(f"[API] DELETE /api/frs/watchlist/{tgt_id} returned 200 ✓")

        # 4. GET /api/anpr/hotlist
        r_hl = client.get("/api/anpr/hotlist")
        assert r_hl.status_code == 200
        hl = r_hl.json()["hotlist"]
        print(f"[API] GET /api/anpr/hotlist returned {len(hl)} plates")
        assert any(item["plate_number"] == "DL01AB1234" for item in hl)

        # 5. POST /api/anpr/hotlist
        new_plate = {
            "plate_number": "UP99XX0007",
            "vehicle_model": "Black SUV",
            "reason": "Intercept Order #991",
            "threat_level": "CRITICAL",
        }
        r_add_hl = client.post("/api/anpr/hotlist", json=new_plate)
        assert r_add_hl.status_code == 200
        print("[API] POST /api/anpr/hotlist created plate UP99XX0007")

        # 6. DELETE /api/anpr/hotlist/{id}
        r_del_hl = client.delete("/api/anpr/hotlist/UP99XX0007")
        assert r_del_hl.status_code == 200
        print("[API] DELETE /api/anpr/hotlist/UP99XX0007 returned 200 ✓")

        # 7. GET /api/anpr/scans
        r_scans = client.get("/api/anpr/scans")
        assert r_scans.status_code == 200
        scans = r_scans.json()["scans"]
        print(f"[API] GET /api/anpr/scans returned {len(scans)} scans")

        # 8. Test simulation endpoints
        r_sim_frs = client.post("/api/frs/simulate-match")
        assert r_sim_frs.status_code == 200
        assert r_sim_frs.json()["alert"]["alert_type"] == "WATCHLIST_SUSPECT_DETECTED"
        print(f"[API] POST /api/frs/simulate-match triggered: {r_sim_frs.json()['alert']['alert_type']} ✓")

        r_sim_anpr = client.post("/api/anpr/simulate-scan")
        assert r_sim_anpr.status_code == 200
        assert r_sim_anpr.json()["alert"]["alert_type"] == "BLACKLIST_VEHICLE_DETECTED"
        print(f"[API] POST /api/anpr/simulate-scan triggered: {r_sim_anpr.json()['alert']['alert_type']} ✓")

        print("FastAPI REST Endpoints tests PASSED ✓\n")


if __name__ == "__main__":
    test_frs_engine()
    test_anpr_engine()
    test_alert_engine_events()
    test_tactical_hud_annotations()
    test_api_endpoints()
    print("=" * 60)
    print("ALL IBVAP FRS & ANPR TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 60)
