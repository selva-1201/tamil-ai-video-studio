"""
Stage 7 — Resumable Google Drive Cloud Sync (`pipeline/drive.py`)

Important Design Note:
- Google Cloud Service Accounts CANNOT upload files into personal Google Drive folders
  (`storageQuotaExceeded` error because service accounts have 0 GB personal storage quota).
- This module exclusively uses **OAuth 2.0 user tokens** stored in Kaggle Secrets
  (`RCLONE_GDRIVE_TOKEN` or `GDRIVE_OAUTH_TOKEN`) with `rclone` (primary) and `PyDrive2` (fallback).
- Folder hierarchy in Google Drive:
  - `TamilVideoStudio/<YYYY-MM-DD>/<job_id>/` (checkpoints after each stage + final deliverables)
  - `TamilVideoStudio/voices/` (cloned Tamil voice profiles `ref.wav`, `ref.txt`, `profile.json`)
  - `TamilVideoStudio/models/` (optional weight cache if faster than Hugging Face)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tamil_studio.drive")


def fetch_kaggle_secret(secret_name: str, default: str = "") -> str:
    """
    Fetch secret from Kaggle Secrets (`kaggle_secrets.UserSecretsClient`) or environment variables.
    Never hard-codes tokens in code or notebooks.
    """
    env_val = os.environ.get(secret_name, "").strip()
    if env_val:
        return env_val
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        client = UserSecretsClient()
        val = client.get_secret(secret_name)
        if val:
            return val.strip()
    except Exception:
        pass
    return default


class DriveSyncManager:
    """
    Handles rclone OAuth configuration, background stage checkpoint uploads, and PyDrive2 fallback.
    """

    def __init__(self, config: Dict[str, Any], local_mirror_dir: Path):
        self.config = config
        self.drive_cfg = config.get("drive", {})
        self.remote_name = self.drive_cfg.get("remote_name", "gdrive")
        self.root_folder = self.drive_cfg.get("root_folder", "TamilVideoStudio")
        self.local_mirror_dir = local_mirror_dir / self.root_folder
        self.local_mirror_dir.mkdir(parents=True, exist_ok=True)
        self.rclone_conf_path = Path.home() / ".config" / "rclone" / "rclone.conf"
        self._bg_threads: List[threading.Thread] = []
        self.rclone_ready = self.configure_rclone_from_secrets()

    def configure_rclone_from_secrets(self) -> bool:
        """
        Configure `~/.config/rclone/rclone.conf` from Kaggle Secret `RCLONE_GDRIVE_TOKEN`
        (the JSON token produced by `rclone authorize "drive"`).
        """
        token_json = fetch_kaggle_secret("RCLONE_GDRIVE_TOKEN")
        client_id = fetch_kaggle_secret("GDRIVE_CLIENT_ID")
        client_secret = fetch_kaggle_secret("GDRIVE_CLIENT_SECRET")

        if not token_json:
            return False

        self.rclone_conf_path.parent.mkdir(parents=True, exist_ok=True)
        conf_lines = [
            f"[{self.remote_name}]",
            "type = drive",
            "scope = drive",
            f"token = {token_json}",
        ]
        if client_id:
            conf_lines.append(f"client_id = {client_id}")
        if client_secret:
            conf_lines.append(f"client_secret = {client_secret}")

        self.rclone_conf_path.write_text("\n".join(conf_lines) + "\n", encoding="utf-8")
        os.chmod(self.rclone_conf_path, 0o600)
        return shutil.which("rclone") is not None

    def get_status(self) -> Dict[str, Any]:
        return {
            "rclone_installed": shutil.which("rclone") is not None,
            "oauth_configured": self.rclone_ready,
            "remote": f"{self.remote_name}:{self.root_folder}",
            "local_mirror_path": str(self.local_mirror_dir),
            "active_upload_threads": sum(1 for t in self._bg_threads if t.is_alive()),
        }

    def sync_directory_now(self, src_dir: Path, rel_remote_subpath: str) -> Dict[str, Any]:
        """
        Synchronously copy `src_dir` to:
        1. Local mirror (`local_mirror_dir / rel_remote_subpath`) for instant Library browsing
        2. Google Drive via `rclone copy` (or `PyDrive2` OAuth fallback if configured)
        """
        mirror_dest = self.local_mirror_dir / rel_remote_subpath
        mirror_dest.mkdir(parents=True, exist_ok=True)
        copied_files: List[str] = []

        if src_dir.is_file():
            shutil.copy2(src_dir, mirror_dest / src_dir.name)
            copied_files.append(src_dir.name)
        elif src_dir.is_dir():
            for item in src_dir.iterdir():
                if item.is_file() and not item.name.startswith("_"):
                    shutil.copy2(item, mirror_dest / item.name)
                    copied_files.append(item.name)

        remote_uri = f"{self.remote_name}:{self.root_folder}/{rel_remote_subpath}"
        uploaded_remote = False

        if self.rclone_ready and shutil.which("rclone"):
            try:
                cmd = [
                    "rclone", "copy",
                    str(mirror_dest),
                    remote_uri,
                    "--config", str(self.rclone_conf_path),
                    "--transfers", "4",
                    "--Rclone-fast-list" if False else "--fast-list",
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
                uploaded_remote = True
            except Exception as exc:
                logger.warning("rclone upload warning (tried PyDrive2 fallback next): %s", exc)
                uploaded_remote = self._pydrive2_fallback_upload(mirror_dest, rel_remote_subpath)

        return {
            "remote_uri": remote_uri,
            "mirror_path": str(mirror_dest),
            "uploaded_to_gdrive": uploaded_remote,
            "files": copied_files,
        }

    def _pydrive2_fallback_upload(self, local_dir: Path, rel_subpath: str) -> bool:
        """Fallback uploader using PyDrive2 with OAuth user credentials from Kaggle Secrets."""
        oauth_creds_json = fetch_kaggle_secret("PYDRIVE2_OAUTH_CREDENTIALS")
        if not oauth_creds_json:
            return False
        try:
            from pydrive2.auth import GoogleAuth  # type: ignore
            from pydrive2.drive import GoogleDrive  # type: ignore

            creds_file = Path("/tmp/pydrive2_creds.json")
            creds_file.write_text(oauth_creds_json, encoding="utf-8")
            gauth = GoogleAuth()
            gauth.LoadCredentialsFile(str(creds_file))
            drive = GoogleDrive(gauth)
            for item in local_dir.iterdir():
                if item.is_file():
                    f = drive.CreateFile({"title": f"{self.root_folder}_{rel_subpath.replace('/', '_')}_{item.name}"})
                    f.SetContentFile(str(item))
                    f.Upload()
            return True
        except Exception as exc:
            logger.debug("PyDrive2 fallback skipped: %s", exc)
            return False

    def sync_job_stage_async(self, job_id: str, job_dir: Path, date_str: Optional[str] = None) -> str:
        """
        Launch a non-blocking background thread to sync the current job checkpoint/deliverables
        to `gdrive:TamilVideoStudio/<date>/<job_id>/`.
        """
        date_folder = date_str or dt.date.today().isoformat()
        rel_subpath = f"{date_folder}/{job_id}"

        def _worker() -> None:
            try:
                self.sync_directory_now(job_dir, rel_subpath)
            except Exception as exc:
                logger.warning("Background Drive sync error for job %s: %s", job_id, exc)

        t = threading.Thread(target=_worker, name=f"drive-sync-{job_id}", daemon=True)
        t.start()
        self._bg_threads.append(t)
        return f"{self.remote_name}:{self.root_folder}/{rel_subpath}"

    def sync_voice_profile_async(self, profile_id: str, profile_dir: Path) -> str:
        rel_subpath = f"voices/{profile_id}"

        def _worker() -> None:
            try:
                self.sync_directory_now(profile_dir, rel_subpath)
            except Exception as exc:
                logger.warning("Background voice sync error for %s: %s", profile_id, exc)

        t = threading.Thread(target=_worker, name=f"voice-sync-{profile_id}", daemon=True)
        t.start()
        self._bg_threads.append(t)
        return f"{self.remote_name}:{self.root_folder}/{rel_subpath}"
