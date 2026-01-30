"""SQLite-backed job store shared by CLI and dashboard."""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


DB_DIR = Path.home() / ".qdrant-finetune"
DB_PATH = DB_DIR / "jobs.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    config TEXT DEFAULT '{}',
    logs TEXT DEFAULT '[]',
    metrics TEXT DEFAULT '[]',
    error TEXT,
    results TEXT,
    created_at TEXT,
    started_at TEXT,
    completed_at TEXT,
    source TEXT DEFAULT 'dashboard'
)
"""


class JobStore:
    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path) if db_path else DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)

    def create_job(
        self,
        job_type: str,
        config: dict | None = None,
        source: str = "dashboard",
    ) -> dict:
        job_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, type, status, config, logs, metrics, created_at, source) VALUES (?,?,?,?,?,?,?,?)",
                (job_id, job_type, "pending", json.dumps(config or {}), "[]", "[]", now, source),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_jobs(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_job(self, job_id: str, **fields):
        allowed = {"status", "error", "results", "started_at", "completed_at", "metrics", "config"}
        sets = []
        vals = []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k in ("results", "metrics", "config"):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        if not sets:
            return
        vals.append(job_id)
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)

    def append_log(self, job_id: str, message: str):
        with self._conn() as conn:
            row = conn.execute("SELECT logs FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return
            logs = json.loads(row["logs"])
            logs.append(message)
            conn.execute("UPDATE jobs SET logs = ? WHERE id = ?", (json.dumps(logs), job_id))

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for field in ("config", "logs", "metrics", "results"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
