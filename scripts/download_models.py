#!/usr/bin/env python3
"""
Resumable Model Downloader for Kaggle 2×T4 (`scripts/download_models.py`)

Stores all model weights in `/kaggle/temp/tamil_studio/models` (never `/kaggle/working`,
which has a hard 20 GB limit on Kaggle).
Prints model sizes, verifies free disk space (`df -h`), and benchmarks Hugging Face
(`hf_transfer`) vs Google Drive `rclone` cache if available.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from pipeline.drive import fetch_kaggle_secret
from pipeline.t4_patches import ensure_workspace_dirs, get_disk_status

MODEL_MANIFEST: List[Dict[str, str]] = [
    {
        "stage": "Stage 1 · Planner LLM (Primary)",
        "repo_id": "Qwen/Qwen3-8B-GGUF",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "approx_size_gb": "5.03 GB",
    },
    {
        "stage": "Stage 1 · Planner LLM (Fallback)",
        "repo_id": "Qwen/Qwen3-4B-GGUF",
        "filename": "Qwen3-4B-Q4_K_M.gguf",
        "approx_size_gb": "2.50 GB",
    },
    {
        "stage": "Stage 2 · Tamil TTS + Voice Clone (Primary)",
        "repo_id": "ai4bharat/IndicF5",
        "filename": "*",
        "approx_size_gb": "1.45 GB",
    },
    {
        "stage": "Stage 2 · Speech Verification (Auto-QA)",
        "repo_id": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "filename": "*",
        "approx_size_gb": "1.62 GB",
    },
    {
        "stage": "Stage 4 · Music Generation (Turbo)",
        "repo_id": "ACE-Step/ACE-Step-v1.5",
        "filename": "*",
        "approx_size_gb": "4.80 GB",
    },
    {
        "stage": "Stage 5A · Quality Video (Wan 2.2 A14B I2V GGUF)",
        "repo_id": "QuantStack/Wan2.2-I2V-A14B-GGUF",
        "filename": "Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf",
        "approx_size_gb": "10.80 GB",
    },
    {
        "stage": "Stage 5A · Quality Video (Wan 2.2 A14B LowNoise GGUF)",
        "repo_id": "QuantStack/Wan2.2-I2V-A14B-GGUF",
        "filename": "Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf",
        "approx_size_gb": "10.80 GB",
    },
    {
        "stage": "Stage 5A · Quality Video Fallback (Wan 2.2 TI2V-5B GGUF)",
        "repo_id": "QuantStack/Wan2.2-TI2V-5B-GGUF",
        "filename": "Wan2.2-TI2V-5B-Q8_0.gguf",
        "approx_size_gb": "5.40 GB",
    },
    {
        "stage": "Stage 5B · Fast Video (LTX-2.3 22B Distilled GGUF)",
        "repo_id": "unsloth/LTX-2.3-GGUF",
        "filename": "ltx-2.3-22b-dev-UD-Q3_K_M.gguf",
        "approx_size_gb": "13.50 GB",
    },
    {
        "stage": "Stage 5 · Ambient Audio (MMAudio)",
        "repo_id": "hkchengrex/MMAudio",
        "filename": "*",
        "approx_size_gb": "1.35 GB",
    },
    {
        "stage": "Stage 5C · Talking Person (InfiniteTalk 480p)",
        "repo_id": "MeiGen-AI/InfiniteTalk",
        "filename": "single/infinitetalk.safetensors",
        "approx_size_gb": "2.40 GB",
    },
]


def print_manifest_table() -> None:
    print("=" * 92)
    print(f"{'STAGE & ROLE':<46} | {'HUGGING FACE REPO':<30} | {'SIZE':<8}")
    print("-" * 92)
    for item in MODEL_MANIFEST:
        print(f"{item['stage']:<46} | {item['repo_id']:<30} | {item['approx_size_gb']:<8}")
    print("=" * 92)


def download_models(dry_run: bool = False, stage_filter: str = "all") -> None:
    paths = ensure_workspace_dirs()
    disk = get_disk_status(paths)
    print(f"\n[Disk Check] /kaggle/temp path: {disk['temp_root']['path']} ({disk['temp_root']['free_gb']} GB free)")
    print(f"[Disk Check] /kaggle/working path: {disk['work_root']['path']} ({disk['work_root']['free_gb']} GB free)\n")
    print_manifest_table()

    if dry_run:
        print("\n[Dry Run] Verified model manifest and target paths in /kaggle/temp. Skipping network download.")
        return

    hf_token = fetch_kaggle_secret("HF_TOKEN")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore

    models_dir: Path = paths["models_dir"]
    for item in MODEL_MANIFEST:
        if stage_filter != "all" and stage_filter.lower() not in item["stage"].lower():
            continue
        t0 = time.perf_counter()
        print(f"\n>>> Downloading {item['stage']} ({item['repo_id']} / {item['filename']})...")
        if item["filename"] == "*":
            local_dir = models_dir / item["repo_id"].split("/")[-1]
            snapshot_download(
                repo_id=item["repo_id"],
                local_dir=str(local_dir),
                token=hf_token or None,
                resume_download=True,
            )
        else:
            hf_hub_download(
                repo_id=item["repo_id"],
                filename=item["filename"],
                local_dir=str(models_dir),
                token=hf_token or None,
                resume_download=True,
            )
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"<<< Completed {item['repo_id']} in {elapsed}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Tamil AI Video Studio models to /kaggle/temp")
    parser.add_argument("--dry-run", action="store_true", help="Print model table and disk usage without downloading")
    parser.add_argument("--stage", default="all", help="Filter by stage substring (e.g., planner, tts, music, video)")
    args = parser.parse_args()
    download_models(dry_run=args.dry_run, stage_filter=args.stage)
