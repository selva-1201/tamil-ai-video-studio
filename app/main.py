"""
Tamil AI Video Studio — FastAPI Backend (`app/main.py`)

Implements all required endpoints + SSE real-time progress + single-GPU-job-at-a-time worker queue:
- POST /api/plan              (Stage 1 LLM Planner + Stage 2 IndicF5 TTS + Whisper Auto-QA -> pauses at Stage 3 Review Gate)
- POST /api/voice/profile     (Voice note upload -> 24 kHz mono -> denoise -> 8-12s trim -> Whisper draft transcript -> Drive sync)
- POST /api/voice/preview     (Preview sentence or regenerate voice-only in Stage 3 after script edits)
- POST /api/render            (Approve & Render: runs Stage 4 Music -> Stage 5 Video + MMAudio -> Stage 6 Mix -> Stage 7 Drive)
- GET  /api/jobs              (List all jobs & checkpoint states)
- GET  /api/jobs/{id}/events  (Server-Sent Events: stage, %, ETA, dual-T4 VRAM usage, log tail)
- GET  /api/library           (Finished videos from local/Drive mirror with inline playback & re-render)
- POST /api/jobs/{id}/cancel  (Cancel running job cleanly; resumable from last finished stage/clip)
- GET/POST /api/settings      (Runtime settings, model status, GPU telemetry, disk space, cache cleanup)
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
from pipeline.t4_patches import (
    aggressive_vram_cleanup,
    apply_t4_compatibility_patches,
    ensure_workspace_dirs,
    get_disk_status,
    get_gpu_telemetry,
)
from pipeline.tts import TamilVoiceEngine
from pipeline.video import VideoGenerationEngine, build_shot_list

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
PLANNER = QwenPlannerEngine(STUDIO_CONFIG, WORKSPACE_PATHS["models_dir"])
VOICE_ENGINE = TamilVoiceEngine(STUDIO_CONFIG, WORKSPACE_PATHS["voices_dir"], WORKSPACE_PATHS["models_dir"])
MUSIC_ENGINE = ACEStepMusicEngine(STUDIO_CONFIG, WORKSPACE_PATHS["models_dir"])
VIDEO_ENGINE = VideoGenerationEngine(
    STUDIO_CONFIG, WORKSPACE_PATHS["models_dir"], WORKSPACE_PATHS["checkpoints_dir"]
)
DRIVE_SYNC = DriveSyncManager(STUDIO_CONFIG, WORKSPACE_PATHS["work_root"] / "drive_mirror")

# Single-GPU-job-at-a-time queue + cancellation flags
JOB_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
CANCEL_FLAGS: Dict[str, bool] = {}

app = FastAPI(
    title="Tamil AI Video Studio (Kaggle 2×T4)",
    version="1.0.0",
    description="Open-source audio-first Tamil AI Video Studio running on 2× NVIDIA Tesla T4 GPUs.",
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
class PlanRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Tanglish, Tamil, or English prompt")
    mode: str = Field(default="cinematic", description="cinematic | talking")
    target_duration: str = Field(default="auto", description="auto | 30 | 45 | 60")
    aspect: str = Field(default="9:16", description="9:16 | 16:9 | 1:1")
    voice_profile_id: str = Field(default="default_tamil_narrator")
    mood_hint: Optional[str] = Field(default=None)
    engine: str = Field(default="wan22", description="wan22 | ltx23")
    preset: str = Field(default="balanced", description="fast | balanced | best")
    resolution: str = Field(default="480p", description="480p | 720p")
    face_image_path: Optional[str] = Field(default=None)


class VoicePreviewRequest(BaseModel):
    job_id: Optional[str] = None
    voice_profile_id: str = "default_tamil_narrator"
    text_ta: Optional[str] = None
    updated_plan: Optional[Dict[str, Any]] = None


class RenderRequest(BaseModel):
    job_id: str
    updated_plan: Optional[Dict[str, Any]] = None
    engine: Optional[str] = None
    preset: Optional[str] = None
    resolution: Optional[str] = None
    preview_5s: bool = False


class SettingsUpdateRequest(BaseModel):
    default_engine: Optional[str] = None
    default_preset: Optional[str] = None
    resolution: Optional[str] = None
    custom_gguf_path: Optional[str] = None
    transformer_device: Optional[str] = None
    aux_device: Optional[str] = None


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

    # If Mode B 5-second lip-sync preview was requested, run only 5s preview and return to review_gate
    if preview_5s and mode == "talking":
        DB.update_job(
            job_id,
            status="running",
            current_stage="video_preview",
            progress_pct=50,
            eta_sec=45,
            append_log="Starting 5-second Talking Person lip-sync preview (InfiniteTalk 480p)...",
        )
        face_img = Path(ckpt.get("face_image_path", "")) if ckpt.get("face_image_path") else None
        if not face_img or not face_img.exists():
            # Generate a neutral portrait placeholder if user didn't upload one yet
            from PIL import Image, ImageDraw

            face_img = job_dir / "talking_face_ref.png"
            img = Image.new("RGB", (480, 832), (28, 28, 32))
            d = ImageDraw.Draw(img)
            d.ellipse([(150, 220), (330, 460)], outline=(220, 220, 220), width=4)
            d.text((155, 500), "Tamil Speaker Reference", fill=(220, 220, 220))
            img.save(str(face_img))

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
            progress_pct=40,
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
            progress_pct=42,
            eta_sec=15,
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
            progress_pct=52,
            checkpoint_dict=ckpt,
            append_log=f"Stage 4 complete ({music_res['engine']}). Unloaded ACE-Step from VRAM.",
        )

    # ------------------------------------------------------------------
    # STAGE 5: Video Generation (Wan 2.2 / LTX-2.3 / InfiniteTalk) + MMAudio
    # ------------------------------------------------------------------
    _check_cancelled(job_id)
    if not ckpt.get("video_done"):
        shots_preview = build_shot_list(
            scenes=plan["scenes"],
            character_description_en=plan["character_description_en"],
            engine=engine,
            final_duration_sec=final_duration_sec,
        )
        eta_info = VIDEO_ENGINE.estimate_render_eta_seconds(
            num_shots=len(shots_preview),
            engine=engine,
            preset=preset,
            mode=mode,
        )
        DB.update_job(
            job_id,
            status="running",
            current_stage="video",
            progress_pct=55,
            eta_sec=eta_info["total_eta_sec"],
            append_log=(
                f"Stage 5: Rendering {len(shots_preview)} shots with {engine.upper()} "
                f"({resolution}, preset={preset}). {eta_info['gpu_placement']}"
            ),
        )

        def _on_shot_progress(info: Dict[str, Any]) -> None:
            _check_cancelled(job_id)
            shot_id = info["shot_id"]
            total_s = info["total_shots"]
            pct = 55 + int(round(30 * (shot_id / float(max(1, total_s)))))
            rem_shots = max(0, total_s - shot_id)
            rem_eta = rem_shots * eta_info["per_shot_eta_sec"]
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
            face_img = Path(ckpt.get("face_image_path", "")) if ckpt.get("face_image_path") else (job_dir / "talking_face_ref.png")
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
                engine=engine,
                preset=preset,
                aspect=aspect,
                resolution=resolution,
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
            progress_pct=86,
            checkpoint_dict=ckpt,
            append_log="Stage 5 complete. All video clips and ambient tracks saved & unloaded from VRAM.",
        )

    # ------------------------------------------------------------------
    # STAGE 6: FFmpeg Mix, Ducking & Mastering
    # ------------------------------------------------------------------
    _check_cancelled(job_id)
    DB.update_job(
        job_id,
        current_stage="mix",
        progress_pct=90,
        eta_sec=8,
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
    # STAGE 7: Google Drive Sync
    # ------------------------------------------------------------------
    DB.update_job(
        job_id,
        current_stage="drive",
        progress_pct=96,
        eta_sec=4,
        append_log="Stage 7: Syncing final.mp4, voice.wav, music.wav, script.json, thumbnail.jpg to Google Drive...",
    )
    drive_res = DRIVE_SYNC.sync_directory_now(job_dir, f"{time.strftime('%Y-%m-%d')}/{job_id}")
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


# Start background GPU worker thread
_WORKER_THREAD = threading.Thread(target=_gpu_worker_loop, name="studio-gpu-worker", daemon=True)
_WORKER_THREAD.start()


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.post("/api/plan")
def api_create_plan_and_voice(
    req: PlanRequest,
    _token: str = Depends(verify_studio_auth),
) -> Dict[str, Any]:
    """
    Executes Stage 1 (LLM Planner: Tanglish/English -> validated Tamil JSON plan)
    followed by Stage 2 (IndicF5 voice synthesis + faster-whisper Auto-QA).
    If synthesized voice exceeds 60 s, automatically asks the Planner LLM to shorten once.
    Pauses at Stage 3 (Script & Voice Review Gate) and returns ETA + QA metrics.
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = WORKSPACE_PATHS["jobs_dir"] / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Generate JSON plan
    plan_res = PLANNER.generate_plan(
        prompt=req.prompt,
        mode=req.mode,
        target_duration=req.target_duration,
        mood_hint=req.mood_hint,
        word_budget_scale=1.0,
    )
    plan_dict = plan_res["plan"]

    # Stage 2: Synthesize Tamil voice & run Whisper Auto-QA
    tts_res = VOICE_ENGINE.synthesize_scenes(
        scenes=plan_dict["scenes"],
        profile_id=req.voice_profile_id,
        output_dir=job_dir,
    )

    # If raw voice length > 60s, ask Planner LLM to shorten and re-synthesize
    if tts_res.get("needs_llm_shortening"):
        logger.info("Voice duration %.2fs > 60s; requesting shortened plan from LLM.", tts_res["raw_duration_sec"])
        plan_res = PLANNER.generate_plan(
            prompt=req.prompt,
            mode=req.mode,
            target_duration=req.target_duration,
            mood_hint=req.mood_hint,
            word_budget_scale=0.78,
        )
        plan_dict = plan_res["plan"]
        tts_res = VOICE_ENGINE.synthesize_scenes(
            scenes=plan_dict["scenes"],
            profile_id=req.voice_profile_id,
            output_dir=job_dir,
        )

    shots = build_shot_list(
        scenes=plan_dict["scenes"],
        character_description_en=plan_dict["character_description_en"],
        engine=req.engine,
        final_duration_sec=tts_res["final_duration_sec"],
    )
    eta_info = VIDEO_ENGINE.estimate_render_eta_seconds(
        num_shots=len(shots),
        engine=req.engine,
        preset=req.preset,
        mode=req.mode,
    )

    ckpt = {
        "plan_done": True,
        "voice_done": True,
        "voice_wav": tts_res["voice_wav"],
        "sentences_qa": tts_res["sentences_qa"],
        "all_qa_passed": tts_res["all_qa_passed"],
        "final_duration_sec": tts_res["final_duration_sec"],
        "planner_benchmark": plan_res["benchmark"],
        "tts_benchmark": tts_res["benchmark"],
        "eta_estimate": eta_info,
        "face_image_path": req.face_image_path,
    }
    record = DB.create_job(
        job_id=job_id,
        prompt=req.prompt,
        mode=req.mode,
        aspect=req.aspect,
        target_duration=req.target_duration,
        voice_profile_id=req.voice_profile_id,
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
            f"Stage 1 & 2 complete: {len(plan_dict['scenes'])} scenes, "
            f"voice={tts_res['final_duration_sec']}s (-16 LUFS), Auto-QA passed={tts_res['all_qa_passed']}. "
            "Waiting at Stage 3 Review Gate."
        ),
    )
    DRIVE_SYNC.sync_job_stage_async(job_id, job_dir)
    return DB.get_job(job_id) or record


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
    """
    Create a cloned Tamil voice profile from an uploaded voice note, save `ref.wav` + `ref.txt`,
    and sync to `gdrive:TamilVideoStudio/voices/`.
    """
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
    """
    Allows:
    1. Regenerating voice-only in Stage 3 after the user edits the Tamil script in the Review screen.
    2. Synthesizing a single preview sentence in the Voices screen.
    """
    if req.job_id:
        job = DB.get_job(req.job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")
        plan_dict = req.updated_plan or job["plan"]
        # Validate updated plan through VideoPlan
        validated = VideoPlan.validate_data(plan_dict).to_dict()
        job_dir = WORKSPACE_PATHS["jobs_dir"] / req.job_id
        tts_res = VOICE_ENGINE.synthesize_scenes(
            scenes=validated["scenes"],
            profile_id=req.voice_profile_id or job["voice_profile_id"],
            output_dir=job_dir,
        )
        ckpt = job["checkpoint"]
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
        return updated

    # Standalone sentence preview for Voices tab
    sample_text = req.text_ta or "வணக்கம், இது என்னுடைய குரல் மாதிரி பரிசோதனை."
    preview_dir = WORKSPACE_PATHS["temp_root"] / "voice_previews"
    tts_res = VOICE_ENGINE.synthesize_scenes(
        scenes=[{"id": 1, "narration_ta": sample_text, "visual_prompt_en": "preview", "est_seconds": 5}],
        profile_id=req.voice_profile_id,
        output_dir=preview_dir,
    )
    return {
        "preview_wav_url": "/api/voice/preview-audio",
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
    """
    Stage 3 Gate -> Starts heavy GPU generation (Stages 4, 5, 6, 7) only after the user clicks
    'Approve & Render' (or 'Preview 5 s' in Mode B).
    """
    job = DB.get_job(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    plan_dict = req.updated_plan or job["plan"]
    validated_plan = VideoPlan.validate_data(plan_dict).to_dict()

    engine = req.engine or job["engine"]
    preset = req.preset or job["preset"]
    resolution = req.resolution or job["resolution"]

    updated = DB.update_job(
        req.job_id,
        status="queued",
        current_stage="music" if not req.preview_5s else "video_preview",
        plan_dict=validated_plan,
        append_log="Approved at Stage 3 Review Gate. Queued for GPU execution.",
    )
    with DB._lock, DB._connect() as conn:
        conn.execute(
            "UPDATE jobs SET engine = ?, preset = ?, resolution = ? WHERE id = ?",
            (engine, preset, resolution, req.job_id),
        )
        conn.commit()

    CANCEL_FLAGS[req.job_id] = False
    JOB_QUEUE.put({"job_id": req.job_id, "preview_5s": req.preview_5s})
    return DB.get_job(req.job_id) or updated


@app.get("/api/jobs")
def api_list_jobs(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    return {"jobs": DB.list_jobs(), "gpus": get_gpu_telemetry()}


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str, _token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    job = DB.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.get("/api/jobs/{job_id}/events")
async def api_job_events_sse(
    job_id: str,
    _token: str = Depends(verify_studio_auth),
) -> StreamingResponse:
    """
    Server-Sent Events (SSE) stream reporting live:
    - `stage`, `status`, `progress_pct`, `eta_sec`
    - `vram` telemetry for both T4 GPUs (`cuda:0` and `cuda:1`)
    - `log_tail` and `outputs`
    """
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
                "vram": get_gpu_telemetry(),
                "log_tail": job["logs"][-20:],
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
    return DB.update_job(
        job_id,
        status="cancelled",
        append_log="Cancellation requested by user. Completed stage/shot checkpoints preserved.",
    )


@app.get("/api/library")
def api_get_library(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    completed = [j for j in DB.list_jobs(limit=100) if j["status"] == "completed"]
    return {
        "videos": completed,
        "drive_status": DRIVE_SYNC.get_status(),
    }


@app.get("/api/settings")
def api_get_settings(_token: str = Depends(verify_studio_auth)) -> Dict[str, Any]:
    return {
        "config": STUDIO_CONFIG,
        "gpus": get_gpu_telemetry(),
        "disk": get_disk_status(WORKSPACE_PATHS),
        "drive": DRIVE_SYNC.get_status(),
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
    return {"cleared": True, "gpus": get_gpu_telemetry(), "disk": get_disk_status(WORKSPACE_PATHS)}


@app.get("/api/media/{job_id}/{filename}")
def api_serve_job_media(
    job_id: str,
    filename: str,
    _token: str = Depends(verify_studio_auth),
) -> FileResponse:
    safe_name = Path(filename).name
    target = WORKSPACE_PATHS["jobs_dir"] / job_id / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File {safe_name} not found for job {job_id}")
    return FileResponse(str(target))


if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_spa_index() -> HTMLResponse:
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Tamil AI Video Studio Backend Running</h1>")
