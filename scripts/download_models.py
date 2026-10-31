#!/usr/bin/env python3
"""
Resumable Model Downloader for Kaggle 2×T4 v2 (`scripts/download_models.py`)

Stores all model weights in `/kaggle/temp/tamil_studio/models` (never `/kaggle/working`,
which has a hard 20 GB limit on Kaggle).
Includes Stage 0 VLM (`Qwen2.5-VL-7B-Instruct-GGUF` + `mmproj-F16.gguf`) and Stage 4b
Keyframe Editor (`FLUX.1-Kontext-dev-GGUF` `Q5_K_S` / `Qwen-Image-Edit-GGUF`).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

from pipeline.drive import fetch_kaggle_secret
from pipeline.t4_patches import ensure_workspace_dirs, get_disk_status, verify_models_on_disk

MODEL_MANIFEST: List[Dict[str, str]] = [
    {
        "stage": "Stage 0 · Reference VLM Backbone",
        "repo_id": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "filename": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "approx_size_gb": "4.68 GB",
        "license": "Apache-2.0",
    },
    {
        "stage": "Stage 0 · Reference VLM Vision Projector",
        "repo_id": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "filename": "mmproj-F16.gguf",
        "approx_size_gb": "1.35 GB",
        "license": "Apache-2.0",
    },
    {
        "stage": "Stage 1 · Planner LLM (Primary)",
        "repo_id": "Qwen/Qwen3-8B-GGUF",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "approx_size_gb": "5.03 GB",
        "license": "Apache-2.0",
    },
    {
        "stage": "Stage 1 · Planner LLM (Fallback)",
        "repo_id": "Qwen/Qwen3-4B-GGUF",
        "filename": "Qwen3-4B-Q4_K_M.gguf",
        "approx_size_gb": "2.50 GB",
        "license": "Apache-2.0",
    },
    {
        "stage": "Stage 2 · Tamil TTS + Voice Clone (Primary)",
        "repo_id": "ai4bharat/IndicF5",
        "filename": "*",
        "approx_size_gb": "1.45 GB",
        "license": "MIT",
    },
    {
        "stage": "Stage 2 · Speech Verification (Auto-QA)",
        "repo_id": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "filename": "*",
        "approx_size_gb": "1.62 GB",
        "license": "MIT",
    },
    {
        "stage": "Stage 4 · Music Generation (Turbo)",
        "repo_id": "ACE-Step/ACE-Step-v1.5",
        "filename": "*",
        "approx_size_gb": "4.80 GB",
        "license": "MIT",
    },
    {
        "stage": "Stage 4b · Keyframe Editor (Primary)",
        "repo_id": "unsloth/FLUX.1-Kontext-dev-GGUF",
        "filename": "flux1-kontext-dev-Q5_K_S.gguf",
        "approx_size_gb": "8.29 GB",
        "license": "FLUX.1 [dev] Non-Commercial",
    },
    {
        "stage": "Stage 4b · Keyframe Editor (Commercial Alt)",
        "repo_id": "unsloth/Qwen-Image-Edit-GGUF",
        "filename": "qwen-image-edit-Q4_K_M.gguf",
        "approx_size_gb": "4.95 GB",
        "license": "Apache-2.0",
    },
    {
        "stage": "Stage 5A · Quality Video (Wan 2.2 A14B HighNoise)",
        "repo_id": "QuantStack/Wan2.2-I2V-A14B-GGUF",
        "filename": "Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf",
        "approx_size_gb": "10.80 GB",
        "license": "Apache-2.0",
    },
    {
        "stage": "Stage 5A · Quality Video (Wan 2.2 A14B LowNoise)",
        "repo_id": "QuantStack/Wan2.2-I2V-A14B-GGUF",
        "filename": "Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf",
        "approx_size_gb": "10.80 GB",
        "license": "Apache-2.0",
    },
    {
        "stage": "Stage 5B · Fast Video (LTX-2.3 22B Distilled GGUF)",
        "repo_id": "unsloth/LTX-2.3-GGUF",
        "filename": "ltx-2.3-22b-dev-UD-Q3_K_M.gguf",
        "approx_size_gb": "13.50 GB",
        "license": "LTX-2 Community",
    },
    {
        "stage": "Stage 5 · Ambient Audio (MMAudio)",
        "repo_id": "hkchengrex/MMAudio",
        "filename": "*",
        "approx_size_gb": "1.35 GB",
        "license": "MIT",
    },
    {
        "stage": "Stage 5C · Talking Person (InfiniteTalk 480p)",
        "repo_id": "MeiGen-AI/InfiniteTalk",
        "filename": "single/infinitetalk.safetensors",
        "approx_size_gb": "2.40 GB",
        "license": "Apache-2.0",
    },
]


def print_manifest_table() -> None:
    print("=" * 108)
    print(f"{'STAGE & ROLE':<44} | {'HUGGING FACE REPO':<34} | {'SIZE':<8} | {'LICENSE'}")
    print("-" * 108)
    for item in MODEL_MANIFEST:
        print(f"{item['stage']:<44} | {item['repo_id']:<34} | {item['approx_size_gb']:<8} | {item['license']}")
    print("=" * 108)


def download_models(dry_run: bool = False, stage_filter: str = "all") -> None:
    paths = ensure_workspace_dirs()
    disk = get_disk_status(paths)
    print(
        f"\n[Disk Check] /kaggle/temp: {disk['temp_root']['path']} "
        f"(Folder used: {disk['temp_root']['folder_used_mb']} MB | FS free: {disk['temp_root']['fs_free_gb']} GB)"
    )
    print(
        f"[Disk Check] /kaggle/working: {disk['work_root']['path']} "
        f"(Folder used: {disk['work_root']['folder_used_mb']} MB | FS free: {disk['work_root']['fs_free_gb']} GB)\n"
    )
    print_manifest_table()

    if dry_run:
        statuses = verify_models_on_disk(paths["models_dir"])
        print("\n[Real Disk Verification Status]")
        for st in statuses:
            print(f"  - {st['stage']:<38}: {st['badge_text']} ({st['detail']})")
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
    parser = argparse.ArgumentParser(description="Download Tamil AI Video Studio v2 models to /kaggle/temp")
    parser.add_argument("--dry-run", action="store_true", help="Print model table and real disk status without downloading")
    parser.add_argument("--stage", default="all", help="Filter by stage substring")
    args = parser.parse_args()
    download_models(dry_run=args.dry_run, stage_filter=args.stage)
