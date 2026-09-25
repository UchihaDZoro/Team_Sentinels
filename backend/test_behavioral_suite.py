"""
Comprehensive Test Suite for IBVAP Tactical Behavioral Analytics
Tests all military-grade capabilities:
- Loitering Detection
- Crawling / Prone Infiltration Detection
- Unattended Baggage / Bag Drop Detection
- Directional Zero-Line Ingress
- Virtual Fence & Alert Engine Integration
"""
import sys
import os
import time
import numpy as np

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from services.behavior_engine import BehaviorEngine
from services.virtual_fence import VirtualFence
from services.alert_engine import AlertEngine
from services.detector import ObjectDetector
from database import Database
from config import DATABASE_PATH


def test_suite():
    print("=" * 60)
    print("  IBVAP TACTICAL BEHAVIORAL ANALYTICS ENGINE TEST SUITE")
    print("=" * 60)

    # ── Test 1: Database Setup & Schema ─────────────────────────
    print("\n[TEST 1] Testing Database Schema and Behavioral Queries...")
    test_db_path = os.path.join(os.path.dirname(__file__), "data", "test_ibvap.db")
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = Database(test_db_path)
    db.add_camera("cam_test", "BOP 14 Sector", "demo_videos/checkpoint_single_person.mp4", "BOP 14")
    
    # Test alert creation with behavioral event
    al = db.add_alert(
        camera_id="cam_test",
        alert_type="LOITERING_DETECTED",
        message="Test loitering alert",
        severity="high",
        details={"track_id": 1, "dwell_time": 22.5}
    )
    assert al["id"] is not None
    assert al["alert_type"] == "LOITERING_DETECTED"

    # Test list_alerts with filtering
    alerts = db.list_alerts(limit=10, alert_type="LOITERING_DETECTED")
    assert len(alerts) == 1
    assert alerts[0]["alert_type"] == "LOITERING_DETECTED"

    breakdown = db.get_behavioral_breakdown()
    assert breakdown["loitering"] == 1
    assert breakdown["crawling"] == 0
    print("✓ Test 1 Passed: Database schema, alerts, and behavioral breakdown functional.")

    # ── Test 2: Virtual Fence & Zero-Line Ingress ───────────────
    print("\n[TEST 2] Testing Virtual Fence & Directional Ingress...")
    fence = VirtualFence()
    # Zone 1: Polygon zone
    # Zone 2: Directional Zero-Line (2 points)
    fence.set_zones("cam_test", [
        {
            "id": "zone_restricted",
            "name": "Barbed Wire Buffer",
            "points": [[200, 200], [500, 200], [500, 400], [200, 400]],
            "type": "zone",
        },
        {
            "id": "zero_line_main",
            "name": "IB Zero Line",
            "points": [[100, 250], [800, 250]],
            "type": "zero_line",
        }
    ])
    assert fence.has_zones("cam_test")
    zones = fence.get_zones("cam_test")
    assert len(zones) == 2

    # Distance to fence
    dist = fence.distance_to_nearest_fence("cam_test", (200, 150))
    assert dist == 50.0

    # Directional ingress check across line [100, 250] -> [800, 250]
    # Crossing from y=200 (foreign side) to y=300 (territory)
    res_ingress = fence.check_directional_ingress("cam_test", (400, 200), (400, 300))
    assert res_ingress["ingress"] is True
    assert res_ingress["zone_id"] == "zero_line_main"

    # Movement on same side (outside polygon and zero-line) does not trigger ingress
    res_no_ingress = fence.check_directional_ingress("cam_test", (400, 150), (450, 160))
    assert res_no_ingress["ingress"] is False
    print("✓ Test 2 Passed: VirtualFence distance heuristics and directional ingress verified.")

    # ── Test 3: Loitering Detection ─────────────────────────────
    print("\n[TEST 3] Testing Loitering Detection (Dwell Time Tracking)...")
    engine = BehaviorEngine(loitering_threshold=2.0)  # low threshold for testing

    t0 = 1000.0
    person_det = {
        "bbox": [250, 250, 300, 350],  # inside Barbed Wire Buffer
        "class": "person",
        "category": "person",
        "confidence": 0.95,
        "track_id": 101,
    }

    # Frame 1: Person enters zone
    events = engine.process_frame("cam_test", [person_det], fence, (540, 960), timestamp=t0)
    assert len(events) == 0  # dwell time is 0s
    assert person_det["dwell_time"] == 0.0

    # Frame 2: Person remains in zone for 1.0s (below threshold of 2.0s)
    events = engine.process_frame("cam_test", [person_det], fence, (540, 960), timestamp=t0 + 1.0)
    assert len(events) == 0
    assert person_det["dwell_time"] == 1.0

    # Frame 3: Person reaches 2.5s (threshold crossed)
    events = engine.process_frame("cam_test", [person_det], fence, (540, 960), timestamp=t0 + 2.5)
    assert len(events) == 1
    assert events[0]["event_type"] == "LOITERING_DETECTED"
    assert events[0]["track_id"] == 101
    assert events[0]["severity"] == "high"
    assert "LOITERING" in events[0]["message"]
    print("✓ Test 3 Passed: Loitering detected accurately upon crossing dwell threshold.")

    # ── Test 4: Crawling / Prone Infiltration Detection ───────────
    print("\n[TEST 4] Testing Crawling / Prone Infiltration Heuristics...")
    engine_crawl = BehaviorEngine(crawling_aspect_ratio=1.15, ground_y_ratio=0.55)

    # Case A: Normal standing person (aspect ratio ~ 0.40)
    standing_det = {
        "bbox": [100, 300, 140, 420],  # w=40, h=120 -> AR = 0.33
        "class": "person",
        "category": "person",
        "confidence": 0.92,
        "track_id": 201,
    }
    events_standing = engine_crawl.process_frame("cam_test", [standing_det], fence, (540, 960), timestamp=100.0)
    assert standing_det["posture"]["is_crawling"] is False
    assert standing_det["posture"]["posture"] == "standing"
    assert len([e for e in events_standing if e["event_type"] == "CRAWLING_INFILTRATION"]) == 0

    # Case B: Crawling infiltrator (prone under fence: w=150, h=50 -> AR = 3.0, lower region)
    crawling_det = {
        "bbox": [200, 380, 350, 430],  # w=150, h=50, cy=405 >= 540*0.55=297
        "class": "person",
        "category": "person",
        "confidence": 0.88,
        "track_id": 202,
    }
    # Feed 2 frames
    events_f1 = engine_crawl.process_frame("cam_test", [crawling_det], fence, (540, 960), timestamp=101.0)
    events_f2 = engine_crawl.process_frame("cam_test", [crawling_det], fence, (540, 960), timestamp=101.2)

    crawl_alerts = [e for e in (events_f1 + events_f2) if e["event_type"] == "CRAWLING_INFILTRATION"]
    assert len(crawl_alerts) == 1
    assert crawl_alerts[0]["severity"] == "critical"
    assert crawl_alerts[0]["track_id"] == 202
    assert crawling_det["posture"]["is_crawling"] is True
    assert "PRONE / CRAWLING" in crawling_det["status_tags"]
    print("✓ Test 4 Passed: Prone crawling infiltration correctly differentiated from standing posture.")

    # ── Test 5: Unattended Baggage / Bag Drop Detection ──────────
    print("\n[TEST 5] Testing Unattended Baggage / Bag Drop Detection...")
    engine_bag = BehaviorEngine(unattended_time_threshold=3.0, unattended_proximity_radius=100.0)

    # Frame 1: Person drops a backpack at (500, 450) and stands next to it at (510, 450)
    bag_det = {
        "bbox": [480, 430, 520, 470],  # centroid (500, 450)
        "class": "backpack",
        "class_id": 24,
        "category": "object",
        "confidence": 0.89,
        "track_id": 301,
    }
    person_nearby = {
        "bbox": [510, 350, 560, 480],  # centroid (535, 415), dist ~ 49px < 100px
        "class": "person",
        "category": "person",
        "confidence": 0.94,
        "track_id": 302,
    }
    events = engine_bag.process_frame("cam_test", [bag_det, person_nearby], fence, (540, 960), timestamp=500.0)
    assert len([e for e in events if e["event_type"] == "UNATTENDED_BAGGAGE"]) == 0

    # Frame 2: Person departs. Backpack remains stationary for 4.0s with no personnel within 100px
    events_after_departure = engine_bag.process_frame("cam_test", [bag_det], fence, (540, 960), timestamp=504.0)
    bag_alerts = [e for e in events_after_departure if e["event_type"] == "UNATTENDED_BAGGAGE"]
    assert len(bag_alerts) == 1
    assert bag_alerts[0]["severity"] == "high"
    assert "UNATTENDED" in bag_alerts[0]["message"]
    print("✓ Test 5 Passed: Stationary baggage triggered unattended alert after person departed.")

    # ── Test 6: Directional Zero-Line Crossing ───────────────────
    print("\n[TEST 6] Testing Directional Zero-Line Ingress...")
    engine_dir = BehaviorEngine()
    # Simulate a target crossing the Zero-Line at y=250 from y=200 down to y=320
    t_start = 600.0
    for i in range(8):
        y_val = 200 + i * 15  # 200, 215, 230, 245, 260, 275, 290, 305
        det = {
            "bbox": [400, y_val - 20, 440, y_val + 20],
            "class": "person",
            "category": "person",
            "confidence": 0.91,
            "track_id": 401,
        }
        evs = engine_dir.process_frame("cam_test", [det], fence, (540, 960), timestamp=t_start + i * 0.1)
        ingress_evs = [e for e in evs if e["event_type"] == "DIRECTIONAL_INGRESS"]
        if ingress_evs:
            break

    assert len(ingress_evs) == 1
    assert ingress_evs[0]["event_type"] == "DIRECTIONAL_INGRESS"
    assert ingress_evs[0]["severity"] == "critical"
    print("✓ Test 6 Passed: Directional ingress detected across border boundary line.")

    # ── Test 7: AlertEngine & Push Integration ──────────────────
    print("\n[TEST 7] Testing AlertEngine Integration with Behavioral Events...")
    pushed_alerts = []
    alert_engine = AlertEngine(db)
    alert_engine.on_new_alert = lambda a: pushed_alerts.append(a)

    dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    created = alert_engine.process_behavioral_events("cam_test", [ingress_evs[0]], dummy_frame)
    assert len(created) == 1
    assert len(pushed_alerts) == 1
    assert pushed_alerts[0]["alert_type"] == "DIRECTIONAL_INGRESS"
    assert os.path.exists(created[0]["snapshot_path"])
    print("✓ Test 7 Passed: AlertEngine processed behavioral events, saved snapshot, and dispatched alert.")

    # ── Test 8: HUD Visual Annotation ───────────────────────────
    print("\n[TEST 8] Testing Tactical HUD Annotation Rendering...")
    annotated = ObjectDetector.annotate_frame(
        dummy_frame,
        [standing_det, crawling_det, bag_det],
        intrusions=[{"detection": crawling_det, "zone_id": "z1", "zone_name": "Perimeter"}],
        fence_zones=fence.get_zone_polygons_for_drawing("cam_test"),
        behavioral_events=created,
    )
    assert annotated.shape == (540, 960, 3)
    assert annotated.dtype == np.uint8
    print("✓ Test 8 Passed: Tactical HUD, bounding boxes, posture pills, and overlays rendered cleanly.")

    # Clean up test database
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("  ALL 8 TACTICAL BEHAVIORAL SUITE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    test_suite()
