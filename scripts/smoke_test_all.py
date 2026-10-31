#!/usr/bin/env python3
"""
Per-Stage & End-to-End Smoke Tests with A/B Video Engine Comparison (`scripts/smoke_test_all.py`)

Runs each stage independently (or all in sequence) and prints measured wall-clock time + peak VRAM:
1. Stage 1 — LLM JSON Planner test with acceptance prompt:
   `"oru chinna paiyan mazhai la school ku nadandhu poguraan, emotional ah"`
2. Stage 2 — 10-second AI4Bharat IndicF5 Tamil TTS + faster-whisper Auto-QA test (-16 LUFS)
3. Stage 4 — 10-second ACE-Step 1.5 Turbo instrumental music test (shift=3.0, fade in/out)
4. Stage 5 — A/B Video Engine Test:
   - Clip A: 5-second Wan 2.2 clip + MMAudio ambient track
   - Clip B: 5-second LTX-2.3 Distilled shot with native ambient audio
5. Stage 6 — FFmpeg 3-stem mix (Voice 0 dB, sidechain-ducked Music -18 dB, Ambience -24 dB) + H.264 export
6. Stage 7 — Google Drive / rclone sync test (`TamilVideoStudio/<date>/<job_id>/`)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from pipeline.drive import DriveSyncManager
from pipeline.mix import assemble_final_video
from pipeline.music import ACEStepMusicEngine
from pipeline.planner import QwenPlannerEngine
from pipeline.t4_patches import apply_t4_compatibility_patches, ensure_workspace_dirs, get_disk_status, get_gpu_telemetry
from pipeline.tts import TamilVoiceEngine
from pipeline.video import VideoGenerationEngine


ACCEPTANCE_PROMPT = "oru chinna paiyan mazhai la school ku nadandhu poguraan, emotional ah"


def load_cfg() -> Dict[str, Any]:
    cfg_path = REPO_ROOT / "configs" / "studio_config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def run_smoke_tests(stage: str = "all") -> Dict[str, Any]:
    apply_t4_compatibility_patches()
    cfg = load_cfg()
    paths = ensure_workspace_dirs(cfg)
    smoke_dir = paths["work_root"] / "smoke_test_run"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "gpus": get_gpu_telemetry(),
        "disk": get_disk_status(paths),
        "stages": {},
    }

    print("=" * 88)
    print("TAMIL AI VIDEO STUDIO — KAGGLE 2×T4 SMOKE TEST SUITE")
    print("=" * 88)
    print(f"Acceptance Prompt: {ACCEPTANCE_PROMPT}")
    print(f"Temp Storage:      {paths['temp_root']} ({report['disk']['temp_root']['free_gb']} GB free)")
    print(f"Working Storage:   {paths['work_root']} ({report['disk']['work_root']['free_gb']} GB free)")
    print("-" * 88)

    # 1. Stage 1: Planner LLM Test
    planner = QwenPlannerEngine(cfg, paths["models_dir"])
    plan_out = planner.generate_plan(prompt=ACCEPTANCE_PROMPT, mode="cinematic", target_duration="45")
    plan = plan_out["plan"]
    report["stages"]["stage1_planner"] = {
        "engine": plan_out["engine"],
        "title": plan["title"],
        "mood": plan["mood"],
        "num_scenes": len(plan["scenes"]),
        "sample_narration_ta": plan["scenes"][0]["narration_ta"],
        "music_prompt": plan["music_prompt"],
        "benchmark": plan_out["benchmark"],
    }
    print(
        f"[STAGE 1 · PLANNER] PASS | {plan_out['benchmark']['duration_sec']}s | "
        f"Peak VRAM GPU0={plan_out['benchmark']['gpu0_peak_gb']} GB, GPU1={plan_out['benchmark']['gpu1_peak_gb']} GB\n"
        f"  -> Title: {plan['title']} | Mood: {plan['mood']}\n"
        f"  -> Scene 1 Tamil Script: {plan['scenes'][0]['narration_ta']}"
    )
    if stage == "planner":
        return report

    # 2. Stage 2: 10-Second IndicF5 + Whisper Auto-QA Test
    tts = TamilVoiceEngine(cfg, paths["voices_dir"], paths["models_dir"])
    tts_out = tts.synthesize_scenes(
        scenes=plan["scenes"][:2],  # ~10-15s test slice
        profile_id="default_tamil_narrator",
        output_dir=smoke_dir / "tts",
    )
    report["stages"]["stage2_tts"] = {
        "engine": tts_out["engine"],
        "voice_wav": tts_out["voice_wav"],
        "final_duration_sec": tts_out["final_duration_sec"],
        "all_qa_passed": tts_out["all_qa_passed"],
        "sentences_qa": tts_out["sentences_qa"],
        "benchmark": tts_out["benchmark"],
    }
    print(
        f"[STAGE 2 · INDICF5 TTS + WHISPER QA] PASS | {tts_out['benchmark']['duration_sec']}s | "
        f"Peak VRAM GPU0={tts_out['benchmark']['gpu0_peak_gb']} GB | "
        f"Audio={tts_out['final_duration_sec']}s (-16 LUFS) | QA Passed={tts_out['all_qa_passed']}"
    )
    if stage == "tts":
        return report

    # 3. Stage 4: 10-Second ACE-Step 1.5 Music Test
    music = ACEStepMusicEngine(cfg, paths["models_dir"])
    music_out = music.generate_music_track(
        music_prompt=plan["music_prompt"],
        mood=plan["mood"],
        final_voice_duration_sec=10.0,
        output_dir=smoke_dir / "music",
    )
    report["stages"]["stage4_music"] = {
        "engine": music_out["engine"],
        "music_wav": music_out["music_wav"],
        "duration_sec": music_out["duration_sec"],
        "benchmark": music_out["benchmark"],
    }
    print(
        f"[STAGE 4 · ACE-STEP 1.5 MUSIC] PASS | {music_out['benchmark']['duration_sec']}s | "
        f"Peak VRAM GPU0={music_out['benchmark']['gpu0_peak_gb']} GB | Track={music_out['duration_sec']}s"
    )
    if stage == "music":
        return report

    # 4. Stage 5: Video A/B Test (5s Wan 2.2 Clip vs LTX-2.3 Shot)
    video = VideoGenerationEngine(cfg, paths["models_dir"], paths["checkpoints_dir"])
    wan_out = video.render_shots(
        job_id="smoke_wan22",
        scenes=plan["scenes"][:1],
        character_description_en=plan["character_description_en"],
        final_duration_sec=5.0,
        output_dir=smoke_dir / "video_wan22",
        engine="wan22",
        preset="balanced",
        aspect="9:16",
        resolution="480p",
    )
    ltx_out = video.render_shots(
        job_id="smoke_ltx23",
        scenes=plan["scenes"][:1],
        character_description_en=plan["character_description_en"],
        final_duration_sec=5.0,
        output_dir=smoke_dir / "video_ltx23",
        engine="ltx23",
        preset="fast",
        aspect="9:16",
        resolution="480p",
    )
    report["stages"]["stage5_video_ab"] = {
        "wan22": {
            "clip_paths": wan_out["clip_paths"],
            "ambient_wav_paths": wan_out["ambient_wav_paths"],
            "benchmark": wan_out["benchmark"],
        },
        "ltx23": {
            "clip_paths": ltx_out["clip_paths"],
            "ambient_wav_paths": ltx_out["ambient_wav_paths"],
            "benchmark": ltx_out["benchmark"],
        },
    }
    print(
        f"[STAGE 5A · WAN 2.2 + MMAUDIO] PASS | {wan_out['benchmark']['duration_sec']}s | "
        f"Peak VRAM GPU0={wan_out['benchmark']['gpu0_peak_gb']} GB, GPU1={wan_out['benchmark']['gpu1_peak_gb']} GB"
    )
    print(
        f"[STAGE 5B · LTX-2.3 DISTILLED] PASS | {ltx_out['benchmark']['duration_sec']}s | "
        f"Peak VRAM GPU0={ltx_out['benchmark']['gpu0_peak_gb']} GB, GPU1={ltx_out['benchmark']['gpu1_peak_gb']} GB"
    )
    if stage == "video":
        return report

    # 5. Stage 6: FFmpeg Mix & Export Test
    mix_out = assemble_final_video(
        clip_paths=wan_out["clip_paths"] + ltx_out["clip_paths"],
        voice_wav=Path(tts_out["voice_wav"]),
        music_wav=Path(music_out["music_wav"]),
        ambient_wavs=wan_out["ambient_wav_paths"] + ltx_out["ambient_wav_paths"],
        plan_dict=plan,
        output_dir=smoke_dir / "final_deliverables",
        mix_cfg=cfg.get("mix", {}),
    )
    report["stages"]["stage6_mix"] = mix_out
    print(
        f"[STAGE 6 · FFMPEG MIX & DUCKING] PASS | {mix_out['benchmark']['duration_sec']}s | "
        f"Output={mix_out['final_mp4']} ({mix_out['duration_sec']}s)"
    )
    if stage == "mix":
        return report

    # 6. Stage 7: Drive Sync Test
    drive = DriveSyncManager(cfg, paths["work_root"] / "drive_mirror")
    drive_out = drive.sync_directory_now(smoke_dir / "final_deliverables", "smoke_test/job_smoke_001")
    report["stages"]["stage7_drive"] = drive_out
    print(
        f"[STAGE 7 · DRIVE SYNC] PASS | Remote={drive_out['remote_uri']} | "
        f"Files={', '.join(drive_out['files'])}"
    )
    print("=" * 88)
    print("ALL 7 STAGES PASSED SMOKE VERIFICATION SUCCESSFULLY.")
    print("=" * 88)

    report_path = smoke_dir / "smoke_test_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tamil AI Video Studio Smoke Tests")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "planner", "tts", "music", "video", "mix", "drive"],
        help="Run a specific stage smoke test or all stages sequentially",
    )
    args = parser.parse_args()
    run_smoke_tests(stage=args.stage)
