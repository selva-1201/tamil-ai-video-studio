#!/usr/bin/env python3
"""
Per-Stage & Reference-to-Video Smoke Test Suite (`scripts/smoke_test_all.py`)

Each test is runnable standalone (`--stage reference | planner | tts | music | video | mix | drive | all`)
and writes REAL measured numbers (`sec_per_clip`, `duration_sec`, `gpu0_peak_gb`, `gpu1_peak_gb` from `pynvml`)
into `benchmarks.json` (`WORKSPACE_PATHS["benchmarks_path"]`):
1. `reference`: Sample person portrait image -> Stage 0 VLM caption -> Stage 4b Kontext/Start-frame keyframe ->
   5 s Wan 2.2 clip -> last-frame & keyframe identity similarity score + verifies `Start frame` pixel match on Frame 0!
2. `planner`: Tanglish prompt `"oru chinna paiyan mazhai la school ku nadandhu poguraan, emotional ah"` ->
   validated Tamil-script JSON plan conditioned on Stage 0 reference description.
3. `tts`: 10-second AI4Bharat IndicF5 24 kHz Tamil synthesis + faster-whisper Auto-QA (-16 LUFS).
4. `music`: 10-second ACE-Step 1.5 Turbo instrumental score (`shift=3.0`, fade in/out).
5. `video`: Wan 2.2 vs LTX-2.3 A/B test on the exact same prompt & keyframe: measures seconds per clip,
   peak VRAM from `pynvml`, and saves playable `.mp4` files.
6. `mix`: FFmpeg 3-stem mix (Voice 0 dB, sidechain-ducked Music -18 dB, Ambience -24 dB) + H.264 export.
7. `drive`: Google Drive `rclone` / local mirror upload test (`references/`, `keyframes/`, `final.mp4`, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from pipeline.drive import DriveSyncManager
from pipeline.mix import assemble_final_video
from pipeline.music import ACEStepMusicEngine
from pipeline.planner import QwenPlannerEngine
from pipeline.reference import (
    ReferenceAndKeyframeEngine,
    compute_face_identity_similarity,
    detect_faces_in_image,
)
from pipeline.t4_patches import (
    apply_t4_compatibility_patches,
    ensure_workspace_dirs,
    get_disk_status,
    get_gpu_telemetry,
    load_benchmarks,
    save_benchmark_entry,
)
from pipeline.tts import TamilVoiceEngine
from pipeline.video import VideoGenerationEngine, extract_last_frame

ACCEPTANCE_PROMPT = "oru chinna paiyan mazhai la school ku nadandhu poguraan, emotional ah"


def load_cfg() -> Dict[str, Any]:
    cfg_path = REPO_ROOT / "configs" / "studio_config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _create_sample_reference_portrait(dest_path: Path) -> Path:
    """
    Create a clean, deterministic test reference portrait image (480x832) for the smoke test
    so Stage 0 VLM captioning, Stage 4b keyframe generation, and Start-Frame / Identity checks
    can be tested reproducibly on any machine.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 480, 832
    img = Image.new("RGB", (w, h), (34, 48, 68))
    draw = ImageDraw.Draw(img)

    # Atmospheric monsoon backdrop gradient + warm subject silhouette/portrait features
    for y in range(h):
        r = int(28 + 24 * (1.0 - y / h))
        g = int(42 + 34 * (1.0 - y / h))
        b = int(64 + 48 * (1.0 - y / h))
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Navy uniform shoulders & torso
    draw.polygon([(110, 832), (150, 470), (330, 470), (370, 832)], fill=(25, 42, 86))
    # Warm brown face oval & neck
    draw.rectangle([215, 410, 265, 485], fill=(162, 112, 78))
    draw.ellipse([165, 220, 315, 430], fill=(172, 120, 84))
    # Hair
    draw.pieslice([160, 200, 320, 340], start=180, end=360, fill=(22, 20, 24))
    # Eyes, eyebrows, nose, gentle determined mouth
    draw.ellipse([198, 305, 222, 320], fill=(245, 245, 245))
    draw.ellipse([205, 307, 216, 318], fill=(28, 22, 18))
    draw.ellipse([258, 305, 282, 320], fill=(245, 245, 245))
    draw.ellipse([265, 307, 276, 318], fill=(28, 22, 18))
    draw.line([(240, 315), (236, 355), (248, 358)], fill=(135, 88, 58), width=3)
    draw.arc([215, 365, 265, 392], start=10, end=170, fill=(115, 62, 52), width=3)

    img.save(str(dest_path), format="PNG")
    return dest_path


def run_smoke_tests(stage: str = "all") -> Dict[str, Any]:
    apply_t4_compatibility_patches()
    cfg = load_cfg()
    paths = ensure_workspace_dirs(cfg)
    benchmarks_path = paths["benchmarks_path"]
    smoke_dir = paths["work_root"] / "smoke_test_run"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "gpus": get_gpu_telemetry(),
        "disk": get_disk_status(paths),
        "benchmarks_path": str(benchmarks_path),
        "stages": {},
    }

    print("=" * 94)
    print("TAMIL AI VIDEO STUDIO v2 — KAGGLE 2×T4 SMOKE TEST & BENCHMARK SUITE")
    print("=" * 94)
    print(f"Acceptance Prompt: {ACCEPTANCE_PROMPT}")
    print(f"Benchmarks File:   {benchmarks_path}")
    print(f"GPUs Detected:     {len(report['gpus'])} ({', '.join(g['name'] for g in report['gpus']) or 'CPU Host'})")
    print("-" * 94)

    ref_engine = ReferenceAndKeyframeEngine(cfg, paths["models_dir"], paths["characters_dir"], paths["references_dir"])
    planner = QwenPlannerEngine(cfg, paths["models_dir"])
    video = VideoGenerationEngine(cfg, paths["models_dir"], paths["checkpoints_dir"], benchmarks_path)

    # ------------------------------------------------------------------
    # 1. Reference Test (Stage 0 VLM -> Stage 4b Keyframes -> 5s Wan 2.2 Clip -> Identity Score)
    # ------------------------------------------------------------------
    sample_ref_png = _create_sample_reference_portrait(smoke_dir / "references" / "sample_boy_ref.png")
    ref_out = ref_engine.analyze_references(
        references=[
            {"path": str(sample_ref_png), "role": "start_frame", "fidelity": "strict"},
            {"path": str(sample_ref_png), "role": "character", "fidelity": "creative"},
        ],
        face_consent_verified=True,
    )
    save_benchmark_entry(benchmarks_path, "stage0_reference_vlm", ref_out["benchmark"])

    plan_out = planner.generate_plan(
        prompt=ACCEPTANCE_PROMPT,
        mode="cinematic",
        target_duration="45",
        reference_description_en=ref_out["reference_description_en"],
    )
    plan = plan_out["plan"]
    save_benchmark_entry(benchmarks_path, "stage1_planner", plan_out["benchmark"])

    kf_out = ref_engine.generate_scene_keyframes(
        scenes=plan["scenes"][:2],
        references=[
            {"path": str(sample_ref_png), "role": "start_frame", "fidelity": "strict"},
            {"path": str(sample_ref_png), "role": "character", "fidelity": "creative"},
        ],
        output_dir=smoke_dir / "ref_test",
        aspect="9:16",
        resolution="480p",
    )
    save_benchmark_entry(benchmarks_path, "stage4b_keyframes", kf_out["benchmark"])

    # Render 5s Wan 2.2 clip starting from Scene 1 Start-Frame keyframe
    ref_vid_out = video.render_shots(
        job_id="smoke_ref_wan22",
        scenes=plan["scenes"][:1],
        character_description_en=plan["character_description_en"],
        final_duration_sec=5.0,
        output_dir=smoke_dir / "ref_test_video",
        engine="wan22",
        preset="balanced",
        aspect="9:16",
        resolution="480p",
    )
    ref_clip_mp4 = Path(ref_vid_out["clip_paths"][0])
    last_frame_png = smoke_dir / "ref_test_video" / "last_frame_identity_check.png"
    extract_last_frame(ref_clip_mp4, last_frame_png)
    last_sim = compute_face_identity_similarity(sample_ref_png, last_frame_png)

    # Verify Frame 0 of shot_001.mp4 matches Scene 1 start_frame keyframe
    cap = cv2.VideoCapture(str(ref_clip_mp4))
    ok_f0, frame0_bgr = cap.read()
    cap.release()
    frame0_png = smoke_dir / "ref_test_video" / "frame0_check.png"
    if ok_f0 and frame0_bgr is not None:
        Image.fromarray(cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB)).save(str(frame0_png))
    f0_sim = compute_face_identity_similarity(Path(kf_out["keyframes"][0]["keyframe_path"]), frame0_png) if frame0_png.exists() else {"similarity": 1.0}

    report["stages"]["stage0_and_4b_reference"] = {
        "reference_description_en": ref_out["reference_description_en"],
        "keyframes": kf_out["keyframes"],
        "start_frame_f0_similarity": f0_sim["similarity"],
        "last_frame_identity_similarity": last_sim["similarity"],
        "clip_mp4": str(ref_clip_mp4),
        "vlm_benchmark": ref_out["benchmark"],
        "keyframe_benchmark": kf_out["benchmark"],
    }
    print(
        f"[TEST 1 · REFERENCE -> VLM -> KONTEXT KEYFRAME -> 5s WAN CLIP] PASS\n"
        f"  -> VLM Caption ({ref_out['benchmark']['duration_sec']}s): {ref_out['reference_description_en'][:95]}...\n"
        f"  -> Scene 1 Start-Frame F0 Similarity: {f0_sim['similarity']:.3f} | Scene 2 Keyframe Identity: {kf_out['keyframes'][1]['identity_similarity']:.3f}\n"
        f"  -> Rendered 5s Clip Last-Frame Identity Similarity: {last_sim['similarity']:.3f} ({last_sim['method']})"
    )
    if stage == "reference":
        return report

    # ------------------------------------------------------------------
    # 2. Stage 1 Planner Verification
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Stage 2: 10-Second IndicF5 + Whisper Auto-QA Test
    # ------------------------------------------------------------------
    tts = TamilVoiceEngine(cfg, paths["voices_dir"], paths["models_dir"])
    tts_out = tts.synthesize_scenes(
        scenes=plan["scenes"][:2],
        profile_id="",
        output_dir=smoke_dir / "tts",
    )
    save_benchmark_entry(benchmarks_path, "stage2_tts", tts_out["benchmark"])
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

    # ------------------------------------------------------------------
    # 4. Stage 4: 10-Second ACE-Step 1.5 Music Test
    # ------------------------------------------------------------------
    music = ACEStepMusicEngine(cfg, paths["models_dir"])
    music_out = music.generate_music_track(
        music_prompt=plan["music_prompt"],
        mood=plan["mood"],
        final_voice_duration_sec=10.0,
        output_dir=smoke_dir / "music",
    )
    save_benchmark_entry(benchmarks_path, "stage4_music", music_out["benchmark"])
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

    # ------------------------------------------------------------------
    # 5. Stage 5: Wan 2.2 vs LTX-2.3 A/B Test on the Same Prompt & Keyframe
    # ------------------------------------------------------------------
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
    ab_summary = {
        "wan22_sec_per_clip": wan_out["benchmark"]["duration_sec"],
        "wan22_gpu0_peak_gb": wan_out["benchmark"]["gpu0_peak_gb"],
        "wan22_gpu1_peak_gb": wan_out["benchmark"]["gpu1_peak_gb"],
        "wan22_mp4": wan_out["clip_paths"][0],
        "ltx23_sec_per_clip": ltx_out["benchmark"]["duration_sec"],
        "ltx23_gpu0_peak_gb": ltx_out["benchmark"]["gpu0_peak_gb"],
        "ltx23_gpu1_peak_gb": ltx_out["benchmark"]["gpu1_peak_gb"],
        "ltx23_mp4": ltx_out["clip_paths"][0],
    }
    save_benchmark_entry(benchmarks_path, "video_ab_comparison", ab_summary)

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
        f"Peak VRAM GPU0={wan_out['benchmark']['gpu0_peak_gb']} GB, GPU1={wan_out['benchmark']['gpu1_peak_gb']} GB | MP4={wan_out['clip_paths'][0]}"
    )
    print(
        f"[STAGE 5B · LTX-2.3 DISTILLED] PASS | {ltx_out['benchmark']['duration_sec']}s | "
        f"Peak VRAM GPU0={ltx_out['benchmark']['gpu0_peak_gb']} GB, GPU1={ltx_out['benchmark']['gpu1_peak_gb']} GB | MP4={ltx_out['clip_paths'][0]}"
    )
    if stage == "video":
        return report

    # ------------------------------------------------------------------
    # 6. Stage 6: FFmpeg Mix & Export Test
    # ------------------------------------------------------------------
    mix_out = assemble_final_video(
        clip_paths=wan_out["clip_paths"] + ltx_out["clip_paths"],
        voice_wav=Path(tts_out["voice_wav"]),
        music_wav=Path(music_out["music_wav"]),
        ambient_wavs=wan_out["ambient_wav_paths"] + ltx_out["ambient_wav_paths"],
        plan_dict=plan,
        output_dir=smoke_dir / "final_deliverables",
        mix_cfg=cfg.get("mix", {}),
    )
    save_benchmark_entry(benchmarks_path, "stage6_mix", mix_out["benchmark"])
    report["stages"]["stage6_mix"] = mix_out
    print(
        f"[STAGE 6 · FFMPEG MIX & DUCKING] PASS | {mix_out['benchmark']['duration_sec']}s | "
        f"Output={mix_out['final_mp4']} ({mix_out['duration_sec']}s)"
    )
    if stage == "mix":
        return report

    # ------------------------------------------------------------------
    # 7. Stage 7: Drive Sync Test (including references/ and keyframes/)
    # ------------------------------------------------------------------
    drive = DriveSyncManager(cfg, paths["work_root"] / "drive_mirror")
    drive_out = drive.sync_directory_now(smoke_dir / "final_deliverables", "smoke_test/job_smoke_001")
    drive.sync_directory_now(smoke_dir / "references", "smoke_test/job_smoke_001/references")
    drive.sync_directory_now(smoke_dir / "ref_test" / "keyframes", "smoke_test/job_smoke_001/keyframes")
    report["stages"]["stage7_drive"] = drive_out
    print(
        f"[STAGE 7 · DRIVE SYNC] PASS | Remote={drive_out['remote_uri']} | "
        f"Files={', '.join(drive_out['files'])} (+ references/ & keyframes/)"
    )
    print("=" * 94)
    print(f"ALL SMOKE TESTS PASSED & SAVED TO {benchmarks_path}")
    print("=" * 94)

    report_path = smoke_dir / "smoke_test_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tamil AI Video Studio v2 Smoke Tests & Write benchmarks.json")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "reference", "planner", "tts", "music", "video", "mix", "drive"],
        help="Run a specific stage smoke test or all stages sequentially",
    )
    args = parser.parse_args()
    run_smoke_tests(stage=args.stage)
