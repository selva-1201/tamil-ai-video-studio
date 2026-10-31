"""
Tamil AI Video Studio v2 — FastAPI Backend (`app/main.py`)

Implements:
- Real `pynvml` per-GPU telemetry (`GET /api/telemetry`), real `shutil.disk_usage` + `du`,
  real binary header model verification (`GET /api/models/verify`), and real `benchmarks.json` ETAs.
- Stage 0 Reference Image upload (`POST /api/references/upload`), Drive reference picker
  (`GET /api/references/drive`), face detection + mandatory consent gate, and VLM captioning.
- Character Profiles (`GET /api/characters`, `POST /api/characters`) paired with Voice Profiles.
- Stage 4b Scene Keyframe generation (`FLUX.1 Kontext dev GGUF` / `Qwen-Image-Edit` / `Start frame` direct crop),
  per-scene keyframe regeneration (`POST /api/keyframes/regenerate`), and custom scene keyframe upload
  (`POST /api/keyframes/upload`) in the Stage 3 Review Gate.
- Strict stage output file verification: a stage is marked `done` ONLY when its real output files exist on disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.auth import verify_studio_auth
from app.db import JobDatabase
from pipeline.drive import DriveSyncManager
from pipeline.mix import assemble_final_video
from pipeline.music import ACEStepMusicEngine
from pipeline.planner import QwenPlannerEngine, VideoPlan
from pipeline.reference import (
    ReferenceAndKeyframeEngine,
    compute_face_identity_similarity,
    detect_faces_in_image,
    preprocess_uploaded_image,
    smart_crop_or_outpaint_to_aspect,
)
from pipeline.t4_patches import (
    aggressive_vram_cleanup,
    apply_t4_compatibility_patches,
    ensure_workspace_dirs,
    get_disk_status,
    get_gpu_telemetry,
    load_benchmarks,
    verify_models_on_disk,
)
from pipeline.tts import TamilVoiceEngine
from pipeline.video import VideoGenerationEngine, build_shot_list, resolve_dimensions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tamil_studio.api")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "studio_config.yaml"
UI_DIR = REPO_ROOT / "ui"


def load_studio_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


STUDIO_CONFIG = load_studio_config()
apply_t4_compatibility_patches()
WORKSPACE_PATHS = ensure_workspace_dirs(STUDIO_CONFIG)

DB = JobDatabase(WORKSPACE_PATHS["work_root"] / "studio_jobs.db")
REF_ENGINE = ReferenceAndKeyframeEngine(
    STUDIO_CONFIG,
    WORKSPACE_PATHS["models_dir"],
    WORKSPACE_PATHS["characters_dir"],
    WORKSPACE_PATHS["references_dir"],
)
PLANNER = QwenPlannerEngine(STUDIO_CONFIG, WORKSPACE_PATHS["models_dir"])
VOICE_ENGINE = TamilVoiceEngine(STUDIO_CONFIG, WORKSPACE_PATHS["voices_dir"], WORKSPACE_PATHS["models_dir"])
MUSIC_ENGINE = ACEStepMusicEngine(STUDIO_CONFIG, WORKSPACE_PATHS["models_dir"])
VIDEO_ENGINE = VideoGenerationEngine(
    STUDIO_CONFIG,
    WORKSPACE_PATHS["models_dir"],
    WORKSPACE_PATHS["checkpoints_dir"],
    WORKSPACE_PATHS["benchmarks_path"],
)
DRIVE_SYNC = DriveSyncManager(STUDIO_CONFIG, WORKSPACE_PATHS["work_root"] / "drive_mirror")

JOB_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
CANCEL_FLAGS: Dict[str, bool] = {}

app = FastAPI(
    title="Tamil AI Video Studio v2 (Kaggle 2×T4)",
    version="2.0.0",
    description="Open-source audio-first & reference-consistent Tamil AI Video Studio on 2× NVIDIA Tesla T4 GPUs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Request Schemas
# ------------------------------------------------------------------------------
class ReferenceItemInput(BaseModel):
    ref_id: str
    path: str
    role: str = Field(default="character", description="start_frame | character | style")
    fidelity: str = Field(default="strict", description="strict | creative")
    face_detected: bool = False


class PlanRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Tanglish, Tamil, or English prompt")
    mode: str = Field(default="cinematic", description="cinematic | talking")
    target_duration: str = Field(default="auto", description="auto | 30 | 45 | 60")
    aspect: str = Field(default="9:16", description="9:16 | 16:9 | 1:1")
    voice_profile_id: str = Field(default="")
    character_id: Optional[str] = Field(default=None)
    mood_hint: Optional[str] = Field(default=None)
    engine: str = Field(default="wan22", description="wan22 | wan22_lora | ltx23")
    preset: str = Field(default="balanced", description="fast | balanced | best")
    resolution: str = Field(default="480p", description="480p | 720p")
    references: List[ReferenceItemInput] = Field(default_factory=list)
    face_consent_verified: bool = Field(default=False)
    outpaint_start_frame: bool = Field(default=True)


class VoicePreviewRequest(BaseModel):
    job_id: Optional[str] = None
    voice_profile_id: str = ""
    text_ta: Optional[str] = None
    updated_plan: Optional[Dict[str, Any]] = None
    reference_description_en: Optional[str] = None


class KeyframeRegenerateRequest(BaseModel):
    job_id: str
    scene_id: int
    visual_prompt_en: Optional[str] = None
    fidelity: str = "strict"


class RenderRequest(BaseModel):
    job_id: str
    updated_plan: Optional[Dict[str, Any]] = None
    reference_description_en: Optional[str] = None
    engine: Optional[str] = None
    preset: Optional[str] = None
    resolution: Optional[str] = None
    preview_5s: bool = False


class SettingsUpdateRequest(BaseModel):
    default_engine: Optional[str] = None
    default_preset: Optional[str] = None
    resolution: Optional[str] = None
    custom_gguf_path: Optional[str] = None
    keyframe_engine: Optional[str] = None
    transformer_device: Optional[str] = None
    aux_device: Optional[str] = None


# ------------------------------------------------------------------------------
# Stage Output File Verification Helper
# ------------------------------------------------------------------------------
def compute_verified_stage_states(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the ordered stage list for the Progress screen (including Stage 0, Stage 3 Review Gate,
    and Stage 4b Keyframes). A stage is marked `done` ONLY if its real output files exist on disk!
    """
    job_id = job["id"]
    job_dir = WORKSPACE_PATHS["jobs_dir"] / job_id
    ckpt = job.get("checkpoint") or {}
    plan = job.get("plan") or {}
    cur_stage = job.get("current_stage", "")
    status = job.get("status", "")

    refs = ckpt.get("references", [])
    has_refs = len(refs) > 0

    plan_file = job_dir / "plan_checkpoint.json"
    voice_file = job_dir / "voice.wav"
    music_file = job_dir / "music.wav"
    kf_dir = job_dir / "keyframes"
    final_mp4 = job_dir / "final.mp4"

    expected_shots = int(ckpt.get("expected_shots", 0))
    rendered_shots = [
        p for p in sorted(job_dir.glob("shot_*.mp4"))
        if p.is_file() and p.stat().st_size > 512
    ]
    n_rendered = len(rendered_shots)
    if expected_shots <= 0 and plan.get("scenes"):
        shots_list = build_shot_list(
            scenes=plan["scenes"],
            character_description_en=plan.get("character_description_en", ""),
            engine="ltx23" if job.get("engine") == "ltx23" else "wan22",
            final_duration_sec=float(ckpt.get("final_duration_sec", plan.get("target_seconds", 45))),
        )
        expected_shots = len(shots_list)

    kf_files = list(kf_dir.glob("scene_*.png")) if kf_dir.exists() else []
    n_scenes = len(plan.get("scenes", []))

    stages: List[Dict[str, Any]] = []
    if has_refs:
        ref_done = bool(ckpt.get("reference_description_en")) and (job_dir / "references").exists()
        stages.append(
            {
                "key": "stage0_ref",
                "name": "Stage 0 · Reference Analysis (VLM)",
                "status": "done" if ref_done else ("running" if cur_stage == "stage0_ref" else "pending"),
                "detail": f"{len(refs)} reference image(s) analyzed" if ref_done else "Waiting for VLM analysis",
            }
        )

    plan_done = plan_file.exists() and len(plan.get("scenes", [])) > 0
    stages.append(
        {
            "key": "plan",
            "name": "Stage 1 · Planner LLM (Qwen3 GGUF)",
            "status": "done" if plan_done else ("running" if cur_stage == "plan" else "pending"),
            "detail": f"{n_scenes} scenes validated in Tamil script" if plan_done else "Pending",
        }
    )

    voice_done = voice_file.exists() and voice_file.stat().st_size > 1024
    stages.append(
        {
            "key": "voice",
            "name": "Stage 2 · Tamil TTS (IndicF5 24 kHz) + Whisper QA",
            "status": "done" if voice_done else ("running" if cur_stage == "voice" else "pending"),
            "detail": (
                f"voice.wav ({ckpt.get('final_duration_sec', 0)}s · -16 LUFS · QA={'PASS' if ckpt.get('all_qa_passed') else 'FLAGGED'})"
                if voice_done
                else "Pending"
            ),
        }
    )

    if has_refs:
        kf_done = len(kf_files) >= max(1, n_scenes)
        stages.append(
            {
                "key": "keyframes",
                "name": "Stage 4b · Scene Keyframe Generation (Kontext / Start-Frame)",
                "status": "done" if kf_done else ("running" if cur_stage == "keyframes" else "pending"),
                "detail": f"{len(kf_files)}/{n_scenes} scene keyframes generated & identity-checked" if kf_done else "Pending",
            }
        )

    review_approved = bool(ckpt.get("review_approved"))
    stages.append(
        {
            "key": "review",
            "name": "Stage 3 · Script, Voice & Keyframe Review Gate",
            "status": "done" if review_approved else ("running" if status == "review_gate" else "pending"),
            "detail": "Approved by user" if review_approved else "Awaiting user review & approval before GPU video render",
        }
    )

    music_done = music_file.exists() and music_file.stat().st_size > 1024
    stages.append(
        {
            "key": "music",
            "name": "Stage 4 · Instrumental Music (ACE-Step 1.5 Turbo)",
            "status": "done" if music_done else ("running" if cur_stage == "music" else "pending"),
            "detail": f"music.wav ({round(music_file.stat().st_size / 1024, 1)} KB)" if music_done else "Pending",
        }
    )

    video_done = expected_shots > 0 and n_rendered >= expected_shots and bool(ckpt.get("video_done"))
    stages.append(
        {
            "key": "video",
            "name": f"Stage 5 · Video Generation ({n_rendered}/{expected_shots} shots)",
            "status": "done" if video_done else ("running" if cur_stage == "video" else "pending"),
            "detail": f"Rendered {n_rendered}/{expected_shots} clips on disk",
            "shots_rendered": n_rendered,
            "shots_total": expected_shots,
        }
    )

    mix_done = final_mp4.exists() and final_mp4.stat().st_size > 2048
    stages.append(
        {
            "key": "mix",
            "name": "Stage 6 · FFmpeg Mix, Sidechain Ducking & Export",
            "status": "done" if mix_done else ("running" if cur_stage == "mix" else "pending"),
            "detail": f"final.mp4 ({round(final_mp4.stat().st_size / (1024 * 1024), 2)} MB)" if mix_done else "Pending",
        }
    )

    drive_done = bool(job.get("drive_uri")) and mix_done
    stages.append(
        {
            "key": "drive",
            "name": "Stage 7 · Google Drive Sync (rclone OAuth)",
            "status": "done" if drive_done else ("running" if cur_stage == "drive" else "pending"),
            "detail": job.get("drive_uri") or "Pending",
        }
    )
    return stages


# ------------------------------------------------------------------------------
# Background Single-Job GPU Worker Thread
# ------------------------------------------------------------------------------
def _gpu_worker_loop() -> None:
    while True:
        task = JOB_QUEUE.get()
        if task is None:
            break
        job_id = task["job_id"]
        preview_5s = bool(task.get("preview_5s", False))
        try:
            _execute_heavy_render_pipeline(job_id, preview_5s=preview_5s)
        except Exception as exc:
            logger.exception("Render pipeline failed for job %s", job_id)
            DB.update_job(
                job_id,
                status="failed",
                error=str(exc),
                append_log=f"ERROR: {exc}",
            )
        finally:
            aggressive_vram_cleanup()
            JOB_QUEUE.task_done()


def _check_cancelled(job_id: str) -> None:
    if CANCEL_FLAGS.get(job_id):
        raise RuntimeError(f"Job {job_id} was cancelled by user (checkpoint saved; resumable).")


def _execute_heavy_render_pipeline(job_id: str, preview_5s: bool = False) -> None:
    job = DB.get_job(job_id)
    if not job:
        return

    CANCEL_FLAGS[job_id] = False
    job_dir = WORKSPACE_PATHS["jobs_dir"] / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = WORKSPACE_PATHS["checkpoints_dir"] / f"{job_id}_checkpoint.json"

    plan = job["plan"]
    ckpt = job["checkpoint"] or {}
    mode = job["mode"]
    engine = job["engine"]
    preset = job["preset"]
    resolution = job["resolution"]
    aspect = job["aspect"]

    voice_wav_path = Path(ckpt.get("voice_wav", job_dir / "voice.wav"))
    final_duration_sec = float(ckpt.get("final_duration_sec", plan.get("target_seconds", 45)))

    # Mode B (Talking person): use Character / Start-frame reference image as face by default
    refs = ckpt.get("references", [])
    char_or_start_ref = next(
        (Path(r["path"]) for r in refs if r.get("role") in ("character", "start_frame") and Path(r["path"]).exists()),
        None,
    )

    if preview_5s and mode == "talking":
        DB.update_job(
            job_id,
            status="running",
            current_stage="video_preview",
            progress_pct=50,
            eta_sec=45,
            append_log="Starting 5-second Talking Person lip-sync preview (InfiniteTalk 480p)...",
        )
        face_img = char_or_start_ref
        if not face_img or not face_img.exists():
            from PIL import Image

            face_img = job_dir / "talking_face_ref.png"
            Image.new("RGB", (480, 832), (28, 28, 32)).save(str(face_img))

        prev_res = VIDEO_ENGINE.render_talking_person(
            face_image_path=face_img,
            voice_wav_path=voice_wav_path,
            prompt_en=plan["scenes"][0]["visual_prompt_en"],
            output_dir=job_dir,
            aspect=aspect,
            preview_5s=True,
        )
        ckpt["preview_5s_mp4"] = prev_res["clip_paths"][0]
        outputs = job.get("outputs") or {}
        outputs["preview_5s_url"] = f"/api/media/{job_id}/talking_preview_5s.mp4"
        DB.update_job(
            job_id,
            status="review_gate",
            current_stage="review",
            progress_pct=45,
            eta_sec=0,
            checkpoint_dict=ckpt,
            outputs_dict=outputs,
            append_log="5-second lip-sync preview ready for inspection.",
        )
        return

    # ------------------------------------------------------------------
    # STAGE 4: Music Generation (ACE-Step 1.5 Turbo)
    # ------------------------------------------------------------------
    _check_cancelled(job_id)
    music_wav_path = job_dir / "music.wav"
    if not (ckpt.get("music_done") and music_wav_path.exists()):
        DB.update_job(
            job_id,
            status="running",
            current_stage="music",
            progress_pct=45,
            eta_sec=12,
            append_log=f"Stage 4: Generating instrumental score via ACE-Step 1.5 Turbo ({final_duration_sec + 2.0:.1f}s)...",
        )
        music_res = MUSIC_ENGINE.generate_music_track(
            music_prompt=plan["music_prompt"],
            mood=plan["mood"],
            final_voice_duration_sec=final_duration_sec,
            output_dir=job_dir,
        )
        ckpt["music_done"] = True
        ckpt["music_wav"] = music_res["music_wav"]
        ckpt["music_benchmark"] = music_res["benchmark"]
        ckpt_file.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
        DRIVE_SYNC.sync_job_stage_async(job_id, job_dir)
        DB.update_job(
            job_id,
            progress_pct=55,
            checkpoint_dict=ckpt,
            append_log=f"Stage 4 complete ({music_res['engine']}). Unloaded ACE-Step from VRAM.",
        )

    # ------------------------------------------------------------------
    # STAGE 5: Video Generation (Wan 2.2 / LTX-2.3 / InfiniteTalk) + MMAudio
    # ------------------------------------------------------------------
    _check_cancelled(job_id)
    shots_preview = build_shot_list(
        scenes=plan["scenes"],
        character_description_en=plan["character_description_en"],
        engine="ltx23" if engine == "ltx23" else "wan22",
        final_duration_sec=final_duration_sec,
    )
    ckpt["expected_shots"] = 1 if mode == "talking" else len(shots_preview)

    if not ckpt.get("video_done"):
        eta_info = VIDEO_ENGINE.estimate_render_eta_seconds(
            num_shots=ckpt["expected_shots"],
            engine=engine,
            preset=preset,
            resolution=resolution,
            mode=mode,
        )
        DB.update_job(
            job_id,
            status="running",
            current_stage="video",
            progress_pct=58,
            eta_sec=int(eta_info["total_eta_sec"] or 0),
            checkpoint_dict=ckpt,
            append_log=(
                f"Stage 5: Rendering {ckpt['expected_shots']} shots with {engine.upper()} "
                f"({resolution}, preset={preset})."
            ),
        )

        def _on_shot_progress(info: Dict[str, Any]) -> None:
            _check_cancelled(job_id)
            shot_id = info["shot_id"]
            total_s = info["total_shots"]
            pct = 58 + int(round(28 * (shot_id / float(max(1, total_s)))))
            rem_shots = max(0, total_s - shot_id)
            per_shot_s = float(eta_info["per_shot_eta_sec"] or info.get("shot_duration_sec") or 0.0)
            rem_eta = int(round(rem_shots * per_shot_s))
            msg = f"Stage 5: Finished shot {shot_id}/{total_s}"
            if info.get("resumed"):
                msg += " (restored from checkpoint)"
            DB.update_job(
                job_id,
                current_stage="video",
                progress_pct=pct,
                eta_sec=rem_eta,
                append_log=msg,
            )

        def _on_clip_ckpt(shot_id: int, clip_path: Path) -> None:
            DRIVE_SYNC.sync_job_stage_async(job_id, job_dir)

        if mode == "talking":
            face_img = char_or_start_ref or (job_dir / "talking_face_ref.png")
            if not face_img.exists():
                from PIL import Image

                Image.new("RGB", (480, 832), (28, 28, 32)).save(str(face_img))
            vid_res = VIDEO_ENGINE.render_talking_person(
                face_image_path=face_img,
                voice_wav_path=voice_wav_path,
                prompt_en=plan["scenes"][0]["visual_prompt_en"],
                output_dir=job_dir,
                aspect=aspect,
                preview_5s=False,
                progress_cb=_on_shot_progress,
            )
        else:
            vid_res = VIDEO_ENGINE.render_shots(
                job_id=job_id,
                scenes=plan["scenes"],
                character_description_en=plan["character_description_en"],
                final_duration_sec=final_duration_sec,
                output_dir=job_dir,
                engine="ltx23" if engine == "ltx23" else "wan22",
                preset="fast" if engine == "wan22_lora" else preset,
                aspect=aspect,
                resolution=resolution,
                face_image_path=char_or_start_ref,
                progress_cb=_on_shot_progress,
                on_clip_checkpoint=_on_clip_ckpt,
            )

        ckpt["video_done"] = True
        ckpt["clip_paths"] = vid_res["clip_paths"]
        ckpt["ambient_wav_paths"] = vid_res["ambient_wav_paths"]
        ckpt["downgrades"] = vid_res.get("downgrades", [])
        ckpt["video_benchmark"] = vid_res.get("benchmark", {})
        ckpt_file.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
        DB.update_job(
            job_id,
            progress_pct=88,
            checkpoint_dict=ckpt,
            append_log=f"Stage 5 complete: {len(vid_res['clip_paths'])}/{ckpt['expected_shots']} clips verified on disk.",
        )

    # ------------------------------------------------------------------
    # STAGE 6: FFmpeg Mix, Ducking & Mastering
    # ------------------------------------------------------------------
    _check_cancelled(job_id)
    DB.update_job(
        job_id,
        current_stage="mix",
        progress_pct=92,
        eta_sec=5,
        append_log="Stage 6: Mixing voice (-16 LUFS), sidechain-ducked music (-18 dB), and ambience (-24 dB)...",
    )
    mix_res = assemble_final_video(
        clip_paths=ckpt["clip_paths"],
        voice_wav=voice_wav_path,
        music_wav=Path(ckpt["music_wav"]),
        ambient_wavs=ckpt["ambient_wav_paths"],
        plan_dict=plan,
        output_dir=job_dir,
        mix_cfg=STUDIO_CONFIG.get("mix", {}),
    )
    ckpt["mix_done"] = True
    ckpt["mix_benchmark"] = mix_res["benchmark"]

    # ------------------------------------------------------------------
    # STAGE 7: Google Drive Sync (including references/ and keyframes/)
    # ------------------------------------------------------------------
    DB.update_job(
        job_id,
        current_stage="drive",
        progress_pct=97,
        eta_sec=3,
        append_log="Stage 7: Syncing final.mp4, voice.wav, music.wav, script.json, thumbnail.jpg, references/, keyframes/ to Drive...",
    )
    rel_base = f"{time.strftime('%Y-%m-%d')}/{job_id}"
    drive_res = DRIVE_SYNC.sync_directory_now(job_dir, rel_base)
    if (job_dir / "references").exists():
        DRIVE_SYNC.sync_directory_now(job_dir / "references", f"{rel_base}/references")
    if (job_dir / "keyframes").exists():
        DRIVE_SYNC.sync_directory_now(job_dir / "keyframes", f"{rel_base}/keyframes")

    outputs = {
        "final_mp4_url": f"/api/media/{job_id}/final.mp4",
        "voice_wav_url": f"/api/media/{job_id}/voice.wav",
        "music_wav_url": f"/api/media/{job_id}/music.wav",
        "script_json_url": f"/api/media/{job_id}/script.json",
        "thumbnail_url": f"/api/media/{job_id}/thumbnail.jpg",
        "duration_sec": mix_res["duration_sec"],
        "drive_uri": drive_res["remote_uri"],
        "uploaded_to_gdrive": drive_res["uploaded_to_gdrive"],
    }
    ckpt_file.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
    DB.update_job(
        job_id,
        status="completed",
        current_stage="drive",
        progress_pct=100,
        eta_sec=0,
        checkpoint_dict=ckpt,
        outputs_dict=outputs,
        drive_uri=drive_res["remote_uri"],
        append_log=f"Render complete! Final video ({mix_res['duration_sec']}s) saved and synced to {drive_res['remote_uri']}.",
    )


_WORKER_THREAD = threading.Thread(target=_gpu_worker_loop, name="studio-gpu-worker", daemon=True)
_WORKER_THREAD.start()


# ------------------------------------------------------------------------------
# Live Telemetry, Model Verification & Reference Endpoints
# ------------------------------------------------------------------------------
@app.get("/api/telemetry")
def api_get_live_telemetry(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    """
    Polled every 1-2 s across all screens.
    Returns real `pynvml` per-GPU stats, real disk usage (`shutil.disk_usage` + `du`),
    and real measured `benchmarks.json` data.
    """
    return {
        "gpus": get_gpu_telemetry(),
        "disk": get_disk_status(WORKSPACE_PATHS),
        "benchmarks": load_benchmarks(WORKSPACE_PATHS["benchmarks_path"]),
        "drive": DRIVE_SYNC.get_status(),
    }


@app.get("/api/models/verify")
def api_verify_models(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    return {
        "models": verify_models_on_disk(WORKSPACE_PATHS["models_dir"], STUDIO_CONFIG),
        "licenses": STUDIO_CONFIG.get("licenses", {}),
    }


@app.post("/api/references/upload")
async def api_upload_reference_image(
    image_file: UploadFile = File(...),
    role: str = Form("character"),
    fidelity: str = Form("strict"),
    _token: str = Depends(verify_studio_auth),
) -> Dict[str, Any]:
    """
    Upload a reference image (up to 4 per job), EXIF-rotate & resize to <= 1280px,
    and run real face detection so the UI can enforce the mandatory consent checkbox.
    """
    ref_id = f"ref_{uuid.uuid4().hex[:8]}"
    raw_tmp = WORKSPACE_PATHS["temp_root"] / f"raw_{ref_id}_{Path(image_file.filename or 'ref.png').name}"
    dest_png = WORKSPACE_PATHS["references_dir"] / f"{ref_id}.png"

    with raw_tmp.open("wb") as f:
        shutil.copyfileobj(image_file.file, f)

    try:
        prep = preprocess_uploaded_image(raw_tmp, dest_png, max_edge=1280)
        face_info = detect_faces_in_image(dest_png)
        # Also mirror to TamilVideoStudio/references/ for Drive picking
        DRIVE_SYNC.sync_directory_now(dest_png, "references")
        return {
            "ref_id": ref_id,
            "filename": dest_png.name,
            "path": str(dest_png),
            "preview_url": f"/api/references/file/{dest_png.name}",
            "width": prep["width"],
            "height": prep["height"],
            "aspect_ratio": prep["aspect_ratio"],
            "role": role,
            "fidelity": fidelity,
            "face_detected": face_info["face_detected"],
            "face_count": face_info["face_count"],
        }
    finally:
        raw_tmp.unlink(missing_ok=True)


@app.get("/api/references/drive")
def api_list_drive_references(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    """List reference images stored in `TamilVideoStudio/references/` (local mirror + Drive)."""
    items: List[Dict[str, Any]] = []
    for folder in (WORKSPACE_PATHS["references_dir"], DRIVE_SYNC.local_mirror_dir / "references"):
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg")):
            if any(x["filename"] == p.name for x in items):
                continue
            dest = WORKSPACE_PATHS["references_dir"] / p.name
            if p.resolve() != dest.resolve() and not dest.exists():
                shutil.copy2(p, dest)
            face_info = detect_faces_in_image(dest)
            items.append(
                {
                    "ref_id": dest.stem,
                    "filename": dest.name,
                    "path": str(dest),
                    "preview_url": f"/api/references/file/{dest.name}",
                    "face_detected": face_info["face_detected"],
                    "face_count": face_info["face_count"],
                }
            )
    return {"references": items}


@app.get("/api/references/file/{filename}")
def api_serve_reference_file(filename: str, _token: str = Depends(verify_studio_auth)) -> FileResponse:
    safe_name = Path(filename).name
    target = WORKSPACE_PATHS["references_dir"] / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Reference image not found")
    return FileResponse(str(target))


@app.get("/api/characters")
def api_list_characters(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    chars = REF_ENGINE.list_characters()
    for c in chars:
        imgs = c.get("image_paths", [])
        c["thumbnail_url"] = (
            f"/api/characters/{c['character_id']}/image/0" if imgs else ""
        )
    return {"characters": chars}


@app.get("/api/characters/{character_id}/image/{idx}")
def api_serve_character_image(character_id: str, idx: int, _token: str = Depends(verify_studio_auth)) -> FileResponse:
    c_dir = WORKSPACE_PATHS["characters_dir"] / Path(character_id).name
    meta_file = c_dir / "character.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Character not found")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    imgs = meta.get("image_paths", [])
    if idx < 0 or idx >= len(imgs) or not Path(imgs[idx]).exists():
        raise HTTPException(status_code=404, detail="Character image not found")
    return FileResponse(imgs[idx])


@app.post("/api/characters")
def api_save_character(
    name: str = Form(...),
    reference_paths_json: str = Form("[]"),
    reference_description_en: str = Form(""),
    voice_profile_id: str = Form(""),
    consent_verified: bool = Form(False),
    _token: str = Depends(verify_studio_auth),
) -> Dict[str, Any]:
    paths_raw = json.loads(reference_paths_json or "[]")
    img_paths = [Path(p) for p in paths_raw if Path(p).exists()]
    try:
        meta = REF_ENGINE.save_character_profile(
            name=name,
            image_paths=img_paths,
            reference_description_en=reference_description_en,
            voice_profile_id=voice_profile_id,
            consent_verified=consent_verified,
        )
        c_dir = WORKSPACE_PATHS["characters_dir"] / meta["character_id"]
        DRIVE_SYNC.sync_directory_now(c_dir, f"characters/{meta['character_id']}")
        return {"character": meta, "characters": api_list_characters(_token=_token)["characters"]}
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


# ------------------------------------------------------------------------------
# Pipeline Endpoints (Stage 0 -> Stage 1 -> Stage 2 -> Stage 4b -> Stage 3 Gate)
# ------------------------------------------------------------------------------
@app.post("/api/plan")
def api_create_plan_voice_and_keyframes(
    req: PlanRequest,
    _token: str = Depends(verify_studio_auth),
) -> Dict[str, Any]:
    """
    Executes:
    - Stage 0 (Reference Analysis VLM, if reference images or character_id are provided)
    - Stage 1 (LLM Planner conditioned on `reference_description_en`)
    - Stage 2 (IndicF5 Tamil TTS + faster-whisper Auto-QA)
    - Stage 4b (Scene Keyframe Generation + Identity Similarity Check)
    Pauses at Stage 3 (Review Gate) with real measured ETA from `benchmarks.json`.
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = WORKSPACE_PATHS["jobs_dir"] / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    refs_input = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in req.references]
    voice_profile_id = req.voice_profile_id
    pre_ref_desc = ""

    # If a saved character profile was selected, load its reference images + paired voice profile
    if req.character_id:
        c_meta_path = WORKSPACE_PATHS["characters_dir"] / Path(req.character_id).name / "character.json"
        if c_meta_path.exists():
            c_meta = json.loads(c_meta_path.read_text(encoding="utf-8"))
            pre_ref_desc = c_meta.get("reference_description_en", "")
            if not voice_profile_id and c_meta.get("voice_profile_id"):
                voice_profile_id = c_meta["voice_profile_id"]
            if not refs_input:
                for p in c_meta.get("image_paths", []):
                    refs_input.append({"ref_id": Path(p).stem, "path": p, "role": "character", "fidelity": "strict"})

    # Stage 0: Reference Analysis & Consent Check
    ref_analysis: Dict[str, Any] = {"reference_description_en": pre_ref_desc, "items": [], "benchmark": {}}
    copied_refs: List[Dict[str, Any]] = []

    if refs_input:
        job_refs_dir = job_dir / "references"
        job_refs_dir.mkdir(parents=True, exist_ok=True)
        for idx, r in enumerate(refs_input[:4], start=1):
            src_p = Path(r["path"])
            if src_p.exists():
                dst_p = job_refs_dir / f"ref_{idx:02d}_{src_p.name}"
                shutil.copy2(src_p, dst_p)
                copied_refs.append({**r, "path": str(dst_p), "url": f"/api/media/{job_id}/references/{dst_p.name}"})

        try:
            ref_analysis = REF_ENGINE.analyze_references(
                references=copied_refs,
                face_consent_verified=req.face_consent_verified or bool(req.character_id),
            )
            if pre_ref_desc and not ref_analysis["reference_description_en"]:
                ref_analysis["reference_description_en"] = pre_ref_desc
        except ValueError as err:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(err))

    # Stage 1: Generate JSON plan conditioned on reference_description_en
    plan_res = PLANNER.generate_plan(
        prompt=req.prompt,
        mode=req.mode,
        target_duration=req.target_duration,
        mood_hint=req.mood_hint,
        word_budget_scale=1.0,
        reference_description_en=ref_analysis.get("reference_description_en") or None,
    )
    plan_dict = plan_res["plan"]
    (job_dir / "plan_checkpoint.json").write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stage 2: Synthesize Tamil voice & run Whisper Auto-QA
    tts_res = VOICE_ENGINE.synthesize_scenes(
        scenes=plan_dict["scenes"],
        profile_id=voice_profile_id,
        output_dir=job_dir,
    )

    if tts_res.get("needs_llm_shortening"):
        plan_res = PLANNER.generate_plan(
            prompt=req.prompt,
            mode=req.mode,
            target_duration=req.target_duration,
            mood_hint=req.mood_hint,
            word_budget_scale=0.78,
            reference_description_en=ref_analysis.get("reference_description_en") or None,
        )
        plan_dict = plan_res["plan"]
        (job_dir / "plan_checkpoint.json").write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        tts_res = VOICE_ENGINE.synthesize_scenes(
            scenes=plan_dict["scenes"],
            profile_id=voice_profile_id,
            output_dir=job_dir,
        )

    # Stage 4b: Generate per-scene keyframes if references were provided
    kf_res: Dict[str, Any] = {"keyframes": [], "benchmark": {}}
    if copied_refs:
        kf_res = REF_ENGINE.generate_scene_keyframes(
            scenes=plan_dict["scenes"],
            references=copied_refs,
            output_dir=job_dir,
            aspect=req.aspect,
            resolution=req.resolution,
            outpaint_start_frame=req.outpaint_start_frame,
        )
        for s in plan_dict["scenes"]:
            sid = int(s["id"])
            s["keyframe_url"] = f"/api/media/{job_id}/keyframes/scene_{sid:02d}.png"

    shots = build_shot_list(
        scenes=plan_dict["scenes"],
        character_description_en=plan_dict["character_description_en"],
        engine="ltx23" if req.engine == "ltx23" else "wan22",
        final_duration_sec=tts_res["final_duration_sec"],
    )
    eta_info = VIDEO_ENGINE.estimate_render_eta_seconds(
        num_shots=1 if req.mode == "talking" else len(shots),
        engine=req.engine,
        preset="fast" if req.engine == "wan22_lora" else req.preset,
        resolution=req.resolution,
        mode=req.mode,
    )

    ckpt = {
        "references": copied_refs,
        "reference_description_en": ref_analysis.get("reference_description_en", ""),
        "reference_items": ref_analysis.get("items", []),
        "reference_benchmark": ref_analysis.get("benchmark", {}),
        "plan_done": True,
        "voice_done": True,
        "voice_wav": tts_res["voice_wav"],
        "sentences_qa": tts_res["sentences_qa"],
        "all_qa_passed": tts_res["all_qa_passed"],
        "final_duration_sec": tts_res["final_duration_sec"],
        "keyframes": kf_res.get("keyframes", []),
        "keyframes_benchmark": kf_res.get("benchmark", {}),
        "planner_benchmark": plan_res["benchmark"],
        "tts_benchmark": tts_res["benchmark"],
        "eta_estimate": eta_info,
        "expected_shots": 1 if req.mode == "talking" else len(shots),
        "review_approved": False,
    }
    DB.create_job(
        job_id=job_id,
        prompt=req.prompt,
        mode=req.mode,
        aspect=req.aspect,
        target_duration=req.target_duration,
        voice_profile_id=voice_profile_id,
        engine=req.engine,
        preset=req.preset,
        resolution=req.resolution,
        plan_dict=plan_dict,
        checkpoint_dict=ckpt,
    )
    DB.update_job(
        job_id,
        outputs_dict={"voice_wav_url": f"/api/media/{job_id}/voice.wav"},
        append_log=(
            f"Stages {'0, ' if copied_refs else ''}1 & 2{' + 4b keyframes' if copied_refs else ''} complete: "
            f"{len(plan_dict['scenes'])} scenes, voice={tts_res['final_duration_sec']}s (-16 LUFS). "
            "Paused at Stage 3 Review Gate."
        ),
    )
    DRIVE_SYNC.sync_job_stage_async(job_id, job_dir)
    out_job = DB.get_job(job_id) or {}
    out_job["verified_stages"] = compute_verified_stage_states(out_job)
    return out_job


@app.post("/api/keyframes/regenerate")
def api_regenerate_scene_keyframe(
    req: KeyframeRegenerateRequest,
    _token: str = Depends(verify_studio_auth),
) -> Dict[str, Any]:
    job = DB.get_job(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = WORKSPACE_PATHS["jobs_dir"] / req.job_id
    plan = job["plan"]
    ckpt = job["checkpoint"]
    refs = ckpt.get("references", [])
    if not refs:
        raise HTTPException(status_code=400, detail="No reference images attached to this job")

    target_scene = next((s for s in plan["scenes"] if int(s["id"]) == int(req.scene_id)), None)
    if not target_scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    if req.visual_prompt_en:
        target_scene["visual_prompt_en"] = req.visual_prompt_en

    for r in refs:
        r["fidelity"] = req.fidelity

    kf_res = REF_ENGINE.generate_scene_keyframes(
        scenes=[target_scene],
        references=refs,
        output_dir=job_dir,
        aspect=job["aspect"],
        resolution=job["resolution"],
    )
    target_scene["keyframe_url"] = f"/api/media/{req.job_id}/keyframes/scene_{int(req.scene_id):02d}.png?t={int(time.time())}"
    updated = DB.update_job(
        req.job_id,
        plan_dict=plan,
        checkpoint_dict=ckpt,
        append_log=f"Regenerated Scene {req.scene_id} keyframe (identity similarity={target_scene.get('identity_similarity', 1.0)}).",
    )
    updated["verified_stages"] = compute_verified_stage_states(updated)
    return updated


@app.post("/api/keyframes/upload")
async def api_upload_custom_scene_keyframe(
    job_id: str = Form(...),
    scene_id: int = Form(...),
    keyframe_file: UploadFile = File(...),
    _token: str = Depends(verify_studio_auth),
) -> Dict[str, Any]:
    job = DB.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = WORKSPACE_PATHS["jobs_dir"] / job_id
    kf_dir = job_dir / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)

    raw_tmp = WORKSPACE_PATHS["temp_root"] / f"custom_kf_{uuid.uuid4().hex[:6]}.png"
    with raw_tmp.open("wb") as f:
        shutil.copyfileobj(keyframe_file.file, f)

    try:
        w, h = resolve_dimensions(aspect=job["aspect"], quality_tier=job["resolution"])
        dest_kf = kf_dir / f"scene_{int(scene_id):02d}.png"
        smart_crop_or_outpaint_to_aspect(raw_tmp, dest_kf, w, h, outpaint_if_aspect_diff=False)
        plan = job["plan"]
        for s in plan["scenes"]:
            if int(s["id"]) == int(scene_id):
                s["keyframe_path"] = str(dest_kf)
                s["keyframe_url"] = f"/api/media/{job_id}/keyframes/scene_{int(scene_id):02d}.png?t={int(time.time())}"
                s["identity_similarity"] = 1.0
        updated = DB.update_job(
            job_id,
            plan_dict=plan,
            append_log=f"Uploaded custom keyframe for Scene {scene_id}.",
        )
        updated["verified_stages"] = compute_verified_stage_states(updated)
        return updated
    finally:
        raw_tmp.unlink(missing_ok=True)


@app.post("/api/voice/profile")
async def api_create_voice_profile(
    audio_file: UploadFile = File(...),
    profile_name: str = Form("My Tamil Voice"),
    consent_verified: bool = Form(False),
    transcript_ta: Optional[str] = Form(None),
    trim_start_sec: Optional[float] = Form(None),
    trim_end_sec: Optional[float] = Form(None),
    _token: str = Depends(verify_studio_auth),
) -> Dict[str, Any]:
    temp_upload = WORKSPACE_PATHS["temp_root"] / f"upload_{uuid.uuid4().hex[:8]}_{audio_file.filename}"
    with temp_upload.open("wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    try:
        meta = VOICE_ENGINE.create_voice_profile(
            raw_audio_path=temp_upload,
            profile_name=profile_name,
            consent_verified=consent_verified,
            user_transcript_ta=transcript_ta,
            trim_start_sec=trim_start_sec,
            trim_end_sec=trim_end_sec,
        )
        profile_dir = WORKSPACE_PATHS["voices_dir"] / meta["profile_id"]
        DRIVE_SYNC.sync_voice_profile_async(meta["profile_id"], profile_dir)
        return {"profile": meta, "profiles": VOICE_ENGINE.list_profiles()}
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    finally:
        temp_upload.unlink(missing_ok=True)


@app.get("/api/voice/profiles")
def api_list_voice_profiles(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    return {"profiles": VOICE_ENGINE.list_profiles()}


@app.post("/api/voice/preview")
def api_voice_preview_or_regenerate(
    req: VoicePreviewRequest,
    _token: str = Depends(verify_studio_auth),
) -> Dict[str, Any]:
    if req.job_id:
        job = DB.get_job(req.job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")
        plan_dict = req.updated_plan or job["plan"]
        validated = VideoPlan.validate_data(plan_dict).to_dict()
        # Preserve keyframe paths on scenes
        old_scenes = {int(s["id"]): s for s in job["plan"].get("scenes", [])}
        for s in validated["scenes"]:
            old_s = old_scenes.get(int(s["id"]), {})
            for k in ("keyframe_path", "keyframe_url", "identity_similarity"):
                if k in old_s:
                    s[k] = old_s[k]

        job_dir = WORKSPACE_PATHS["jobs_dir"] / req.job_id
        tts_res = VOICE_ENGINE.synthesize_scenes(
            scenes=validated["scenes"],
            profile_id=req.voice_profile_id or job["voice_profile_id"],
            output_dir=job_dir,
        )
        ckpt = job["checkpoint"]
        if req.reference_description_en is not None:
            ckpt["reference_description_en"] = req.reference_description_en
        ckpt["voice_wav"] = tts_res["voice_wav"]
        ckpt["sentences_qa"] = tts_res["sentences_qa"]
        ckpt["all_qa_passed"] = tts_res["all_qa_passed"]
        ckpt["final_duration_sec"] = tts_res["final_duration_sec"]
        updated = DB.update_job(
            req.job_id,
            plan_dict=validated,
            checkpoint_dict=ckpt,
            append_log=f"Regenerated Tamil voice ({tts_res['final_duration_sec']}s) after script review edits.",
        )
        updated["verified_stages"] = compute_verified_stage_states(updated)
        return updated

    sample_text = req.text_ta or "வணக்கம், இது என்னுடைய குரல் மாதிரி பரிசோதனை."
    preview_dir = WORKSPACE_PATHS["temp_root"] / "voice_previews"
    tts_res = VOICE_ENGINE.synthesize_scenes(
        scenes=[{"id": 1, "narration_ta": sample_text, "visual_prompt_en": "preview prompt for voice synthesis test", "est_seconds": 5}],
        profile_id=req.voice_profile_id,
        output_dir=preview_dir,
    )
    return {
        "preview_wav_url": f"/api/voice/preview-audio?t={int(time.time())}",
        "sentences_qa": tts_res["sentences_qa"],
        "duration_sec": tts_res["final_duration_sec"],
    }


@app.get("/api/voice/preview-audio")
def api_serve_voice_preview_audio(_token: str = Depends(verify_studio_auth)) -> FileResponse:
    wav_path = WORKSPACE_PATHS["temp_root"] / "voice_previews" / "voice.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Preview audio not found")
    return FileResponse(str(wav_path), media_type="audio/wav")


@app.post("/api/render")
def api_approve_and_render(
    req: RenderRequest,
    _token: str = Depends(verify_studio_auth),
) -> Dict[str, Any]:
    job = DB.get_job(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    plan_dict = req.updated_plan or job["plan"]
    validated_plan = VideoPlan.validate_data(plan_dict).to_dict()
    old_scenes = {int(s["id"]): s for s in job["plan"].get("scenes", [])}
    for s in validated_plan["scenes"]:
        old_s = old_scenes.get(int(s["id"]), {})
        for k in ("keyframe_path", "keyframe_url", "identity_similarity"):
            if k in old_s:
                s[k] = old_s[k]

    ckpt = job["checkpoint"] or {}
    if req.reference_description_en is not None:
        ckpt["reference_description_en"] = req.reference_description_en
    ckpt["review_approved"] = True

    engine = req.engine or job["engine"]
    preset = req.preset or job["preset"]
    resolution = req.resolution or job["resolution"]

    updated = DB.update_job(
        req.job_id,
        status="queued",
        current_stage="music" if not req.preview_5s else "video_preview",
        plan_dict=validated_plan,
        checkpoint_dict=ckpt,
        append_log="Approved at Stage 3 Review Gate. Queued for sequential GPU execution.",
    )
    with DB._lock, DB._connect() as conn:
        conn.execute(
            "UPDATE jobs SET engine = ?, preset = ?, resolution = ? WHERE id = ?",
            (engine, preset, resolution, req.job_id),
        )
        conn.commit()

    CANCEL_FLAGS[req.job_id] = False
    JOB_QUEUE.put({"job_id": req.job_id, "preview_5s": req.preview_5s})
    out_job = DB.get_job(req.job_id) or updated
    out_job["verified_stages"] = compute_verified_stage_states(out_job)
    return out_job


@app.get("/api/jobs")
def api_list_jobs(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    jobs = DB.list_jobs()
    for j in jobs:
        j["verified_stages"] = compute_verified_stage_states(j)
    return {"jobs": jobs, "gpus": get_gpu_telemetry()}


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str, _token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    job = DB.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job["verified_stages"] = compute_verified_stage_states(job)
    return job


@app.get("/api/jobs/{job_id}/events")
async def api_job_events_sse(
    job_id: str,
    _token: str = Depends(verify_studio_auth),
) -> StreamingResponse:
    async def _event_generator() -> AsyncGenerator[str, None]:
        while True:
            job = DB.get_job(job_id)
            if not job:
                payload = {"error": "Job not found"}
                yield f"data: {json.dumps(payload)}\n\n"
                break

            payload = {
                "job_id": job["id"],
                "status": job["status"],
                "stage": job["current_stage"],
                "progress_pct": job["progress_pct"],
                "eta_sec": job["eta_sec"],
                "verified_stages": compute_verified_stage_states(job),
                "vram": get_gpu_telemetry(),
                "log_tail": job["logs"][-25:],
                "outputs": job["outputs"],
                "downgrades": job["checkpoint"].get("downgrades", []),
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if job["status"] in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: str, _token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    job = DB.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    CANCEL_FLAGS[job_id] = True
    updated = DB.update_job(
        job_id,
        status="cancelled",
        append_log="Cancellation requested by user. Completed stage/shot checkpoints preserved.",
    )
    updated["verified_stages"] = compute_verified_stage_states(updated)
    return updated


@app.get("/api/library")
def api_get_library(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    """
    Return ONLY real completed videos whose `final.mp4` genuinely exists on disk (size > 0).
    Never returns mock or seeded videos.
    """
    real_videos: List[Dict[str, Any]] = []
    for j in DB.list_jobs(limit=100):
        final_mp4 = WORKSPACE_PATHS["jobs_dir"] / j["id"] / "final.mp4"
        if j["status"] == "completed" and final_mp4.exists() and final_mp4.stat().st_size > 1024:
            j["file_size_mb"] = round(final_mp4.stat().st_size / (1024 * 1024), 2)
            real_videos.append(j)
    return {
        "videos": real_videos,
        "drive_status": DRIVE_SYNC.get_status(),
    }


@app.get("/api/settings")
def api_get_settings(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    return {
        "config": STUDIO_CONFIG,
        "gpus": get_gpu_telemetry(),
        "disk": get_disk_status(WORKSPACE_PATHS),
        "drive": DRIVE_SYNC.get_status(),
        "models": verify_models_on_disk(WORKSPACE_PATHS["models_dir"], STUDIO_CONFIG),
        "benchmarks": load_benchmarks(WORKSPACE_PATHS["benchmarks_path"]),
        "licenses": STUDIO_CONFIG.get("licenses", {}),
    }


@app.post("/api/settings")
def api_update_settings(
    req: SettingsUpdateRequest,
    _token: str = Depends(verify_studio_auth),
) -> Dict[str, Any]:
    if req.default_engine:
        STUDIO_CONFIG.setdefault("video", {})["default_engine"] = req.default_engine
    if req.default_preset:
        STUDIO_CONFIG.setdefault("video", {})["default_preset"] = req.default_preset
    if req.custom_gguf_path is not None:
        STUDIO_CONFIG.setdefault("planner", {})["custom_gguf_path"] = req.custom_gguf_path
    if req.keyframe_engine:
        STUDIO_CONFIG.setdefault("keyframe_edit", {})["primary_engine"] = req.keyframe_engine
    if req.transformer_device:
        STUDIO_CONFIG.setdefault("hardware", {}).setdefault("default_placement", {})[
            "transformer_device"
        ] = req.transformer_device
    if req.aux_device:
        STUDIO_CONFIG.setdefault("hardware", {}).setdefault("default_placement", {})["aux_device"] = req.aux_device
    return api_get_settings(_token=_token)


@app.post("/api/settings/clear-cache")
def api_clear_cache(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    aggressive_vram_cleanup()
    return {
        "cleared": True,
        "gpus": get_gpu_telemetry(),
        "disk": get_disk_status(WORKSPACE_PATHS),
    }


@app.get("/api/media/{job_id}/{subpath:path}")
def api_serve_job_media(
    job_id: str,
    subpath: str,
    _token: str = Depends(verify_studio_auth),
) -> FileResponse:
    job_root = (WORKSPACE_PATHS["jobs_dir"] / Path(job_id).name).resolve()
    target = (job_root / subpath).resolve()
    if not str(target).startswith(str(job_root)) or not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File {subpath} not found for job {job_id}")
    return FileResponse(str(target))


if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_spa_index() -> HTMLResponse:
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Tamil AI Video Studio v2 Backend Running</h1>")
