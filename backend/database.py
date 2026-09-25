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
        needs_new = False
        if not hasattr(_local, "conn") or _local.conn is None:
            needs_new = True
        else:
            try:
                _local.conn.execute("SELECT 1")
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                needs_new = True

        if needs_new:
            _local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            _local.conn.row_factory = sqlite3.Row
            if self.db_path != ":memory:":
                try:
                    _local.conn.execute("PRAGMA journal_mode=WAL")
                except Exception:
                    pass
        return _local.conn


    # ── schema ──────────────────────────────────────────────────
    def _init_schema(self):
        conn = self._conn()
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

            CREATE INDEX IF NOT EXISTS idx_alerts_camera   ON alerts(camera_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_created  ON alerts(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_alerts_type     ON alerts(alert_type);
            CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

            CREATE TABLE IF NOT EXISTS frs_watchlist (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                category     TEXT NOT NULL,
                danger_level TEXT NOT NULL,
                notes        TEXT DEFAULT '',
                photo_url    TEXT DEFAULT '',
                match_count  INTEGER DEFAULT 0,
                last_seen    TEXT DEFAULT '',
                created_at   TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS anpr_hotlist (
                plate_number TEXT PRIMARY KEY,
                vehicle_model TEXT NOT NULL,
                reason       TEXT NOT NULL,
                danger_level TEXT NOT NULL,
                status       TEXT DEFAULT 'ACTIVE',
                created_at   TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS anpr_scans (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id    TEXT NOT NULL,
                plate_number TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                confidence   REAL NOT NULL,
                is_hotlist   INTEGER DEFAULT 0,
                timestamp    TEXT DEFAULT (datetime('now','localtime'))
            );
        """)
        try:
            conn.execute("ALTER TABLE cameras ADD COLUMN vision_mode TEXT DEFAULT 'normal'")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE frs_watchlist ADD COLUMN features TEXT DEFAULT '[]'")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE frs_watchlist ADD COLUMN threat_level TEXT DEFAULT 'HIGH'")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE frs_watchlist ADD COLUMN image_path TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE anpr_hotlist ADD COLUMN threat_level TEXT DEFAULT 'HIGH'")
        except Exception:
            pass

        # Seed FRS Watchlist with SSB suspects and authorized personnel
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM frs_watchlist")
        if cur.fetchone()[0] == 0:
            initial_watchlist = [
                ("TGT-01", "Suspect-01 (Cross-Border Infiltrator)", "Cross-Border Infiltrator", "CRITICAL", "Armed infiltrator flagged in Birgunj-Raxaul sector", "/data/watchlist_faces/TGT-01.jpg", 2, "BOP-17 Sector IV"),
                ("TGT-02", "Suspect-02 (Contraband Smuggler)", "Contraband Smuggler", "HIGH", "Known contraband and narcotics courier along riverine crossings", "/data/watchlist_faces/TGT-02.jpg", 1, "Transit Gate Alpha"),
                ("TGT-03", "Sentry-104 (Authorized SSB Personnel)", "Authorized Personnel", "AUTHORIZED", "SSB 42nd BN Sentry Patrol Officer - Service ID #SSB-8834", "/data/watchlist_faces/TGT-03.jpg", 14, "Sector HQ Main Gate"),
                ("TGT-4401", "Mohd. Tariq @ Tiger", "Cross-Border Infiltrator", "CRITICAL", "Armed smuggler active on Raxaul-Birgunj corridor", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80", 2, "BOP-17 Sector IV"),
                ("TGT-4402", "Bikram Thapa @ Cobra", "Arms Trafficker", "CRITICAL", "Suspected weapons transport syndicate ringleader", "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&auto=format&fit=crop&q=80", 1, "Transit Gate Alpha"),
                ("TGT-4403", "Kishore Rai", "Contraband Courier", "HIGH", "Known for night-time riverine border crossing", "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=150&auto=format&fit=crop&q=80", 0, "BOP-12 Perimeter"),
                ("TGT-4404", "Maj. R. K. Neogi (SSB)", "Authorized Personnel", "AUTHORIZED", "SSB 42nd BN QRT Patrol Commander", "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&auto=format&fit=crop&q=80", 14, "Sector HQ Main Gate"),
                ("TGT-4405", "Sub-Insp. Anita Gurung", "Authorized Personnel", "AUTHORIZED", "Border Intelligence Liaison Officer", "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=150&auto=format&fit=crop&q=80", 8, "Checkpost Alpha"),
            ]
            cur.executemany(
                "INSERT INTO frs_watchlist (id, name, category, danger_level, notes, photo_url, match_count, last_seen) VALUES (?,?,?,?,?,?,?,?)",
                initial_watchlist
            )

        # Seed ANPR Hotlist with SSB blacklisted vehicles
        cur.execute("SELECT COUNT(*) FROM anpr_hotlist")
        if cur.fetchone()[0] == 0:
            initial_hotlist = [
                ("DL01AB1234", "White Bolero Camper", "Suspected Contraband Carrier", "CRITICAL", "ACTIVE"),
                ("HR26DQ9911", "Grey Toyota Fortuner", "Stolen Vehicle Alert", "HIGH", "ACTIVE"),
                ("UP14CZ5050", "Black Scorpio Classic", "Flagged Transit", "HIGH", "ACTIVE"),
                ("UP 53 AZ 4421", "White Mahindra Bolero", "Smuggling / Unauthorized Frontier Transit", "CRITICAL", "ACTIVE"),
                ("BR 06 BC 8920", "Tata 407 Heavy Truck", "Wanted in Contraband Seizure Case #441", "HIGH", "ACTIVE"),
                ("NL 01 AA 2291", "Pulsar 220 Black", "Reconnaissance scout vehicle", "HIGH", "ACTIVE"),
                ("DL 8C AB 1010", "Toyota Fortuner Gray", "Escort vehicle linked to fake currency network", "CRITICAL", "ACTIVE"),
            ]
            cur.executemany(
                "INSERT INTO anpr_hotlist (plate_number, vehicle_model, reason, danger_level, status) VALUES (?,?,?,?,?)",
                initial_hotlist
            )

        # Seed initial plate scans if empty
        cur.execute("SELECT COUNT(*) FROM anpr_scans")
        if cur.fetchone()[0] == 0:
            initial_scans = [
                ("cam_bop14", "UP 53 AZ 4421", "Mahindra Bolero", 0.96, 1),
                ("cam_alpha", "BR 01 PA 5512", "Maruti Swift", 0.98, 0),
                ("cam_bop17", "NL 01 AA 2291", "Motorcycle", 0.93, 1),
                ("cam_alpha", "DL 3C BB 7811", "Hyundai Creta", 0.97, 0),
                ("cam_bop12", "BR 06 BC 8920", "Tata Truck", 0.94, 1),
                ("cam_hq", "SSB 04 QRT 01", "SSB Patrol Gypsy", 0.99, 0),
            ]
            cur.executemany(
                "INSERT INTO anpr_scans (camera_id, plate_number, vehicle_type, confidence, is_hotlist) VALUES (?,?,?,?,?)",
                initial_scans
            )

        conn.commit()

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

    def update_night_mode(self, cam_id: str, mode: str | bool):
        if isinstance(mode, bool):
            val = "clahe" if mode else "off"
        else:
            val = str(mode).lower().strip()
        self._conn().execute(
            "UPDATE cameras SET night_mode=? WHERE id=?",
            (val, cam_id),
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

    def list_alerts(
        self,
        limit: int = 100,
        camera_id: str | None = None,
        alert_type: str | None = None,
        severity: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        if camera_id:
            query += " AND camera_id=?"
            params.append(camera_id)
        if alert_type:
            query += " AND alert_type=?"
            params.append(alert_type)
        if severity:
            query += " AND severity=?"
            params.append(severity)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn().execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_behavioral_breakdown(self) -> dict:
        """Aggregate alert counts by tactical behavioral categories."""
        row = self._conn().execute("""
            SELECT 
                SUM(CASE WHEN alert_type LIKE '%loitering%' THEN 1 ELSE 0 END) as loitering,
                SUM(CASE WHEN alert_type LIKE '%crawling%' THEN 1 ELSE 0 END) as crawling,
                SUM(CASE WHEN alert_type LIKE '%unattended%' THEN 1 ELSE 0 END) as unattended,
                SUM(CASE WHEN alert_type LIKE '%ingress%' THEN 1 ELSE 0 END) as ingress,
                SUM(CASE WHEN alert_type LIKE '%intrusion%' THEN 1 ELSE 0 END) as intrusion
            FROM alerts
        """).fetchone()
        if row:
            return {k: (row[k] or 0) for k in row.keys()}
        return {"loitering": 0, "crawling": 0, "unattended": 0, "ingress": 0, "intrusion": 0}

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

    def update_vision_mode(self, cam_id: str, mode: str):
        self._conn().execute("UPDATE cameras SET vision_mode=? WHERE id=?", (mode, cam_id))
        self._conn().commit()

    # ── FRS Watchlist CRUD ──────────────────────────────────────
    def list_watchlist(self) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM frs_watchlist ORDER BY danger_level DESC, match_count DESC").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["threat_level"] = d.get("danger_level") or d.get("threat_level", "HIGH")
            d["danger_level"] = d["threat_level"]
            d["photo_url"] = d.get("photo_url") or d.get("image_path", "")
            d["image_path"] = d["photo_url"]
            result.append(d)
        return result

    def get_watchlist_target(self, target_id: str) -> dict | None:
        row = self._conn().execute("SELECT * FROM frs_watchlist WHERE id=?", (target_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["threat_level"] = d.get("danger_level") or d.get("threat_level", "HIGH")
        d["danger_level"] = d["threat_level"]
        d["photo_url"] = d.get("photo_url") or d.get("image_path", "")
        d["image_path"] = d["photo_url"]
        return d

    def add_watchlist_target(
        self,
        target_id: str,
        name: str,
        category: str,
        danger_level: str = "HIGH",
        notes: str = "",
        photo_url: str = "",
        features: str = "[]",
    ) -> dict:
        c = self._conn()
        c.execute(
            """INSERT OR REPLACE INTO frs_watchlist (id, name, category, danger_level, notes, photo_url, features)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (target_id, name, category, danger_level, notes, photo_url, features)
        )
        c.commit()
        return self.get_watchlist_target(target_id)

    def update_watchlist_features(self, target_id: str, features: list):
        c = self._conn()
        c.execute(
            "UPDATE frs_watchlist SET features=? WHERE id=?",
            (json.dumps(features), target_id)
        )
        c.commit()

    def delete_watchlist_target(self, target_id: str):
        c = self._conn()
        c.execute("DELETE FROM frs_watchlist WHERE id=?", (target_id,))
        c.commit()

    def record_frs_match(self, target_id: str, location: str = "") -> dict | None:
        c = self._conn()
        c.execute(
            """UPDATE frs_watchlist
               SET match_count = match_count + 1, last_seen = ?
               WHERE id = ?""",
            (location or datetime.now().strftime("%Y-%m-%d %H:%M:%S"), target_id)
        )
        c.commit()
        return self.get_watchlist_target(target_id)

    # ── ANPR Hotlist & Scans CRUD ───────────────────────────────
    def list_hotlist(self) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM anpr_hotlist ORDER BY created_at DESC").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["threat_level"] = d.get("danger_level") or d.get("threat_level", "HIGH")
            d["danger_level"] = d["threat_level"]
            result.append(d)
        return result

    def get_hotlist_vehicle(self, plate_number: str) -> dict | None:
        plate = plate_number.strip().upper()
        row = self._conn().execute("SELECT * FROM anpr_hotlist WHERE plate_number=?", (plate,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["threat_level"] = d.get("danger_level") or d.get("threat_level", "HIGH")
        d["danger_level"] = d["threat_level"]
        return d

    def add_hotlist_vehicle(
        self,
        plate_number: str,
        vehicle_model: str,
        reason: str,
        danger_level: str = "HIGH",
        status: str = "ACTIVE",
    ) -> dict:
        plate = plate_number.strip().upper()
        c = self._conn()
        c.execute(
            """INSERT OR REPLACE INTO anpr_hotlist (plate_number, vehicle_model, reason, danger_level, status)
               VALUES (?, ?, ?, ?, ?)""",
            (plate, vehicle_model, reason, danger_level, status)
        )
        c.commit()
        return self.get_hotlist_vehicle(plate) or {}

    def delete_hotlist_vehicle(self, plate_number: str):
        plate = plate_number.strip().upper()
        c = self._conn()
        c.execute("DELETE FROM anpr_hotlist WHERE plate_number=?", (plate,))
        c.commit()

    def list_scans(self, limit: int = 50) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM anpr_scans ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def record_scan(
        self,
        camera_id: str,
        plate_number: str,
        vehicle_type: str,
        confidence: float = 0.95,
        is_hotlist: int = 0,
    ) -> dict:
        c = self._conn()
        c.execute(
            """INSERT INTO anpr_scans (camera_id, plate_number, vehicle_type, confidence, is_hotlist)
               VALUES (?, ?, ?, ?, ?)""",
            (camera_id, plate_number.strip().upper(), vehicle_type, confidence, is_hotlist)
        )
        c.commit()
        scan_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        row = c.execute("SELECT * FROM anpr_scans WHERE id=?", (scan_id,)).fetchone()
        return dict(row) if row else {}

