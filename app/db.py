"""
SQLite Job Store & Resumable Per-Stage Checkpointing (`app/db.py`)

Persists every job and stage transition both in SQLite (`studio_jobs.db`) and as a standalone
`checkpoint.json` inside the job directory so a restarted Kaggle session can resume from the
exact stage and shot index where it left off.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class JobDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'cinematic',
                    aspect TEXT NOT NULL DEFAULT '9:16',
                    target_duration TEXT NOT NULL DEFAULT 'auto',
                    voice_profile_id TEXT NOT NULL DEFAULT 'default_tamil_narrator',
                    engine TEXT NOT NULL DEFAULT 'wan22',
                    preset TEXT NOT NULL DEFAULT 'balanced',
                    resolution TEXT NOT NULL DEFAULT '480p',
                    status TEXT NOT NULL DEFAULT 'planned',
                    current_stage TEXT NOT NULL DEFAULT 'plan',
                    progress_pct INTEGER NOT NULL DEFAULT 0,
                    eta_sec INTEGER NOT NULL DEFAULT 0,
                    plan_json TEXT NOT NULL DEFAULT '{}',
                    checkpoint_json TEXT NOT NULL DEFAULT '{}',
                    outputs_json TEXT NOT NULL DEFAULT '{}',
                    logs_json TEXT NOT NULL DEFAULT '[]',
                    drive_uri TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_job(
        self,
        job_id: str,
        prompt: str,
        mode: str = "cinematic",
        aspect: str = "9:16",
        target_duration: str = "auto",
        voice_profile_id: str = "default_tamil_narrator",
        engine: str = "wan22",
        preset: str = "balanced",
        resolution: str = "480p",
        plan_dict: Optional[Dict[str, Any]] = None,
        checkpoint_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        plan_str = json.dumps(plan_dict or {}, ensure_ascii=False)
        ckpt_str = json.dumps(checkpoint_dict or {}, ensure_ascii=False)
        logs_str = json.dumps([f"[{now}] Job {job_id} created."], ensure_ascii=False)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    id, prompt, mode, aspect, target_duration, voice_profile_id,
                    engine, preset, resolution, status, current_stage, progress_pct,
                    eta_sec, plan_json, checkpoint_json, outputs_json, logs_json,
                    drive_uri, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id, prompt, mode, aspect, str(target_duration), voice_profile_id,
                    engine, preset, resolution, "review_gate", "review", 30,
                    0, plan_str, ckpt_str, "{}", logs_str,
                    "", "", now, now,
                ),
            )
            conn.commit()
        return self.get_job(job_id) or {}

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        current_stage: Optional[str] = None,
        progress_pct: Optional[int] = None,
        eta_sec: Optional[int] = None,
        plan_dict: Optional[Dict[str, Any]] = None,
        checkpoint_dict: Optional[Dict[str, Any]] = None,
        outputs_dict: Optional[Dict[str, Any]] = None,
        drive_uri: Optional[str] = None,
        error: Optional[str] = None,
        append_log: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return {}

            logs: List[str] = json.loads(row["logs_json"] or "[]")
            if append_log:
                logs.append(f"[{time.strftime('%H:%M:%S')}] {append_log}")
                logs = logs[-120:]  # Keep tail of 120 lines

            new_status = status if status is not None else row["status"]
            new_stage = current_stage if current_stage is not None else row["current_stage"]
            new_pct = progress_pct if progress_pct is not None else row["progress_pct"]
            new_eta = eta_sec if eta_sec is not None else row["eta_sec"]
            new_plan = json.dumps(plan_dict, ensure_ascii=False) if plan_dict is not None else row["plan_json"]
            new_ckpt = json.dumps(checkpoint_dict, ensure_ascii=False) if checkpoint_dict is not None else row["checkpoint_json"]
            new_out = json.dumps(outputs_dict, ensure_ascii=False) if outputs_dict is not None else row["outputs_json"]
            new_drive = drive_uri if drive_uri is not None else row["drive_uri"]
            new_err = error if error is not None else row["error"]

            conn.execute(
                """
                UPDATE jobs SET
                    status = ?, current_stage = ?, progress_pct = ?, eta_sec = ?,
                    plan_json = ?, checkpoint_json = ?, outputs_json = ?, logs_json = ?,
                    drive_uri = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_status, new_stage, new_pct, new_eta,
                    new_plan, new_ckpt, new_out, json.dumps(logs, ensure_ascii=False),
                    new_drive, new_err, now, job_id,
                ),
            )
            conn.commit()
        return self.get_job(job_id) or {}

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return self._deserialize_row(row)

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
            return [self._deserialize_row(r) for r in rows]

    @staticmethod
    def _deserialize_row(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["plan"] = json.loads(d.pop("plan_json") or "{}")
        d["checkpoint"] = json.loads(d.pop("checkpoint_json") or "{}")
        d["outputs"] = json.loads(d.pop("outputs_json") or "{}")
        d["logs"] = json.loads(d.pop("logs_json") or "[]")
        return d
