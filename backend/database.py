"""
IBVAP Database Layer — SQLite
"""
import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path

_local = threading.local()


class Database:
    """Thread-safe SQLite database wrapper for IBVAP."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    # ── connection (per-thread) ─────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(_local, "conn") or _local.conn is None:
            _local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            _local.conn.row_factory = sqlite3.Row
            _local.conn.execute("PRAGMA journal_mode=WAL")
        return _local.conn

    # ── schema ──────────────────────────────────────────────────
    def _init_schema(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cameras (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                source      TEXT NOT NULL,
                location    TEXT DEFAULT '',
                status      TEXT DEFAULT 'inactive',
                night_mode  INTEGER DEFAULT 0,
                fence_zones TEXT DEFAULT '[]',
                created_at  TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id     TEXT NOT NULL,
                alert_type    TEXT NOT NULL,
                severity      TEXT DEFAULT 'medium',
                message       TEXT NOT NULL,
                details       TEXT DEFAULT '{}',
                snapshot_path TEXT DEFAULT '',
                acknowledged  INTEGER DEFAULT 0,
                created_at    TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (camera_id) REFERENCES cameras(id)
            );

            CREATE TABLE IF NOT EXISTS license_plates (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id        TEXT NOT NULL,
                vehicle_track_id INTEGER DEFAULT -1,
                plate_number     TEXT NOT NULL,
                confidence       REAL DEFAULT 0.0,
                vehicle_type     TEXT DEFAULT 'vehicle',
                snapshot_path    TEXT DEFAULT '',
                created_at       TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (camera_id) REFERENCES cameras(id)
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_camera  ON alerts(camera_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_lp_camera      ON license_plates(camera_id);
            CREATE INDEX IF NOT EXISTS idx_lp_created     ON license_plates(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_lp_number      ON license_plates(plate_number);
        """)
        conn.commit()
        conn.close()

    # ── Camera CRUD ─────────────────────────────────────────────
    def add_camera(self, cam_id: str, name: str, source: str, location: str = "") -> dict:
        c = self._conn()
        c.execute(
            "INSERT OR REPLACE INTO cameras (id, name, source, location, status) VALUES (?,?,?,?,?)",
            (cam_id, name, source, location, "inactive"),
        )
        c.commit()
        return self.get_camera(cam_id)

    def get_camera(self, cam_id: str) -> dict | None:
        row = self._conn().execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
        return dict(row) if row else None

    def list_cameras(self) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM cameras ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def update_camera_status(self, cam_id: str, status: str):
        self._conn().execute("UPDATE cameras SET status=? WHERE id=?", (status, cam_id))
        self._conn().commit()

    def update_camera_source(self, cam_id: str, source: str):
        self._conn().execute("UPDATE cameras SET source=? WHERE id=?", (source, cam_id))
        self._conn().commit()

    def update_camera_name(self, cam_id: str, name: str):
        self._conn().execute("UPDATE cameras SET name=? WHERE id=?", (name, cam_id))
        self._conn().commit()

    def update_fence_zones(self, cam_id: str, zones: list):
        self._conn().execute(
            "UPDATE cameras SET fence_zones=? WHERE id=?",
            (json.dumps(zones), cam_id),
        )
        self._conn().commit()

    def update_night_mode(self, cam_id: str, enabled: bool):
        self._conn().execute(
            "UPDATE cameras SET night_mode=? WHERE id=?",
            (1 if enabled else 0, cam_id),
        )
        self._conn().commit()

    def delete_camera(self, cam_id: str):
        c = self._conn()
        c.execute("DELETE FROM alerts WHERE camera_id=?", (cam_id,))
        c.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
        c.commit()

    def clear_all_cameras(self):
        c = self._conn()
        c.execute("DELETE FROM alerts")
        c.execute("DELETE FROM cameras")
        c.commit()

    # ── Alert CRUD ──────────────────────────────────────────────
    def add_alert(
        self,
        camera_id: str,
        alert_type: str,
        message: str,
        severity: str = "medium",
        details: dict | None = None,
        snapshot_path: str = "",
    ) -> dict:
        c = self._conn()
        c.execute(
            """INSERT INTO alerts (camera_id, alert_type, severity, message, details, snapshot_path)
               VALUES (?,?,?,?,?,?)""",
            (camera_id, alert_type, severity, message, json.dumps(details or {}), snapshot_path),
        )
        c.commit()
        alert_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        row = c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return dict(row)

    def list_alerts(self, limit: int = 100, camera_id: str | None = None) -> list[dict]:
        if camera_id:
            rows = self._conn().execute(
                "SELECT * FROM alerts WHERE camera_id=? ORDER BY created_at DESC LIMIT ?",
                (camera_id, limit),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def acknowledge_alert(self, alert_id: int):
        self._conn().execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
        self._conn().commit()

    def clear_alerts(self, camera_id: str | None = None):
        if camera_id:
            self._conn().execute("DELETE FROM alerts WHERE camera_id=?", (camera_id,))
        else:
            self._conn().execute("DELETE FROM alerts")
        self._conn().commit()

    def get_alert_counts(self) -> dict:
        """Get alert counts grouped by type for analytics."""
        rows = self._conn().execute(
            "SELECT alert_type, COUNT(*) as cnt FROM alerts GROUP BY alert_type"
        ).fetchall()
        return {r["alert_type"]: r["cnt"] for r in rows}

    def get_hourly_alerts(self, hours: int = 24) -> list[dict]:
        rows = self._conn().execute(
            """SELECT strftime('%%Y-%%m-%%d %%H:00', created_at) as hour,
                      alert_type, COUNT(*) as cnt
               FROM alerts
               WHERE created_at >= datetime('now', 'localtime', '-%d hours')
               GROUP BY hour, alert_type
               ORDER BY hour""" % hours
        ).fetchall()
        return [dict(r) for r in rows]

    # ── License Plate CRUD ──────────────────────────────────────
    def add_license_plate(
        self,
        camera_id: str,
        vehicle_track_id: int = -1,
        plate_number: str = "",
        confidence: float = 0.0,
        vehicle_type: str = "vehicle",
        snapshot_path: str = "",
        photo_path: str = "",
    ) -> dict:
        """Record a recognized license plate, deduplicating recent identical plates for the same track."""
        if not snapshot_path and photo_path:
            snapshot_path = photo_path
        c = self._conn()
        # Avoid spamming duplicates within 30 seconds for same vehicle
        existing = c.execute(
            """SELECT id FROM license_plates 
               WHERE camera_id=? AND plate_number=? 
               AND created_at >= datetime('now', 'localtime', '-30 seconds')""",
            (camera_id, plate_number),
        ).fetchone()
        if existing:
            return {"id": existing[0], "status": "duplicate_suppressed"}

        c.execute(
            """INSERT INTO license_plates (camera_id, vehicle_track_id, plate_number, confidence, vehicle_type, snapshot_path)
               VALUES (?,?,?,?,?,?)""",
            (camera_id, vehicle_track_id, plate_number, float(confidence), vehicle_type, snapshot_path),
        )
        c.commit()
        lp_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        row = c.execute("SELECT * FROM license_plates WHERE id=?", (lp_id,)).fetchone()
        return dict(row)

    def list_license_plates(self, limit: int = 100, camera_id: str | None = None, search: str | None = None) -> list[dict]:
        c = self._conn()
        query = "SELECT * FROM license_plates WHERE 1=1"
        params = []
        if camera_id:
            query += " AND camera_id=?"
            params.append(camera_id)
        if search:
            query += " AND plate_number LIKE ?"
            params.append(f"%{search}%")
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = c.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def clear_license_plates(self, camera_id: str | None = None):
        c = self._conn()
        if camera_id:
            c.execute("DELETE FROM license_plates WHERE camera_id=?", (camera_id,))
        else:
            c.execute("DELETE FROM license_plates")
        c.commit()

