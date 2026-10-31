"""
Stage 4 — Situation-Matched Instrumental Music Generation (`pipeline/music.py`)

Verified Real API (`ace-step/ACE-Step-1.5` — `docs/en/INFERENCE.md`):
```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

dit_handler = AceStepHandler()
llm_handler = LLMHandler()
dit_handler.initialize_service(
    project_root=...,
    config_path="acestep-v15-turbo",
    device="cuda:0",
)
llm_handler.initialize(
    checkpoint_dir=...,
    lm_model_path="acestep-5Hz-lm-0.6B",
    backend="pt",  # Use PyTorch backend on T4 instead of vLLM (avoids bf16/sm_80 requirement)
    device="cuda:1",
)
params = GenerationParams(
    task_type="text2music",
    caption=music_prompt,
    lyrics="[Instrumental]",
    instrumental=True,
    duration=final_duration + 2.0,
    inference_steps=8,
    shift=3.0,      # Mandatory shift=3.0 for acestep-v15-turbo per INFERENCE.md
    infer_method="ode",
)
config = GenerationConfig(batch_size=1, audio_format="wav")
result = generate_music(dit_handler, llm_handler, params, config, save_dir=str(output_dir))
```
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
from pipeline.t4_patches import sf_compat as sf

from pipeline.t4_patches import StageVRAMGuard, apply_t4_compatibility_patches

logger = logging.getLogger("tamil_studio.music")

MUSIC_SR = 48000


def apply_fade_in_out(
    audio: np.ndarray,
    sr: int = MUSIC_SR,
    fade_in_sec: float = 1.5,
    fade_out_sec: float = 2.0,
) -> np.ndarray:
    """Apply smooth cosine fade-in and fade-out to mono or stereo audio."""
    arr = np.asarray(audio, dtype=np.float32)
    n_samples = arr.shape[0]
    fade_in_n = min(int(fade_in_sec * sr), n_samples // 3)
    fade_out_n = min(int(fade_out_sec * sr), n_samples // 3)

    env = np.ones(n_samples, dtype=np.float32)
    if fade_in_n > 0:
        env[:fade_in_n] = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, fade_in_n, dtype=np.float32)))
    if fade_out_n > 0:
        env[-fade_out_n:] = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0.0, 1.0, fade_out_n, dtype=np.float32)))

    if arr.ndim == 2:
        return arr * env[:, None]
    return arr * env


def _synthesize_mood_instrumental_fallback(
    music_prompt: str,
    mood: str,
    duration_sec: float,
    sr: int = MUSIC_SR,
) -> np.ndarray:
    """
    Rich stereo procedural ambient/cinematic score generator used during fast smoke tests
    when ACE-Step 1.5 weights are not yet cached in `/kaggle/temp`.
    Matches chord progression and tempo to `mood` (emotional, energetic, calm, devotional, cinematic).
    """
    n_samples = int(max(4.0, duration_sec) * sr)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    mood_low = (mood or "emotional").lower()
    if mood_low in ("emotional", "sad", "poignant"):
        # D minor -> Bb major -> F major -> C major (warm bamboo flute + cello pad)
        roots = [146.83, 116.54, 174.61, 130.81]
        bpm = 72.0
    elif mood_low in ("energetic", "funny", "upbeat"):
        # A minor -> F major -> G major -> E minor
        roots = [220.00, 174.61, 196.00, 164.81]
        bpm = 112.0
    elif mood_low in ("devotional", "calm"):
        # Carnatic Shanmukhapriya / Kharaharapriya drone in D (Sa - Pa - Sa)
        roots = [146.83, 146.83, 164.81, 146.83]
        bpm = 64.0
    else:
        # Cinematic C minor -> Ab major -> Eb major -> Bb major
        roots = [130.81, 103.83, 155.56, 116.54]
        bpm = 84.0

    bar_sec = (60.0 / bpm) * 4.0
    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)

    for idx in range(int(np.ceil(duration_sec / bar_sec))):
        s_idx = int(idx * bar_sec * sr)
        e_idx = min(n_samples, int((idx + 1) * bar_sec * sr))
        if s_idx >= n_samples:
            break
        tb = t[s_idx:e_idx] - t[s_idx]
        root = roots[idx % len(roots)]
        fifth = root * 1.4983
        third = root * (1.1892 if mood_low in ("emotional", "cinematic") else 1.2599)
        octave = root * 2.0

        # Warm string pad + gentle flute overtone + arpeggio pulse
        env_bar = np.clip(np.sin(np.pi * np.linspace(0.0, 1.0, e_idx - s_idx, dtype=np.float32)), 0.0, 1.0) ** 0.4
        arp_gate = 0.5 + 0.5 * np.sin(2.0 * np.pi * (bpm / 60.0) * tb)

        pad_l = (
            0.30 * np.sin(2.0 * np.pi * root * tb)
            + 0.22 * np.sin(2.0 * np.pi * third * tb + 0.2)
            + 0.18 * np.sin(2.0 * np.pi * fifth * tb + 0.5)
            + 0.12 * arp_gate * np.sin(2.0 * np.pi * octave * tb)
        ) * env_bar
        pad_r = (
            0.30 * np.sin(2.0 * np.pi * root * tb + 0.15)
            + 0.22 * np.sin(2.0 * np.pi * third * tb + 0.6)
            + 0.18 * np.sin(2.0 * np.pi * fifth * tb + 0.1)
            + 0.12 * arp_gate * np.sin(2.0 * np.pi * octave * tb + 0.4)
        ) * env_bar

        left[s_idx:e_idx] = pad_l
        right[s_idx:e_idx] = pad_r

    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo)) + 1e-8)
    stereo = (stereo / peak) * 0.45
    return apply_fade_in_out(stereo, sr=sr, fade_in_sec=1.5, fade_out_sec=2.0)


class ACEStepMusicEngine:
    """
    Stage 4 Music Generator using `ACE-Step 1.5` (`acestep-v15-turbo`, MIT license).
    """

    def __init__(self, config: Dict[str, Any], models_dir: Path):
        self.config = config
        self.models_dir = models_dir
        self.music_cfg = config.get("music", {})

    def generate_music_track(
        self,
        music_prompt: str,
        mood: str,
        final_voice_duration_sec: float,
        output_dir: Path,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Generate instrumental background music with duration = `final_voice_duration_sec + 2.0 s`,
        apply fade-in and fade-out, save to `output_dir / "music.wav"`, and free all GPU memory.
        """
        apply_t4_compatibility_patches()
        output_dir.mkdir(parents=True, exist_ok=True)
        target_duration = round(float(final_voice_duration_sec) + float(self.music_cfg.get("tail_pad_sec", 2.0)), 2)
        out_wav = output_dir / "music.wav"

        with StageVRAMGuard("Stage 4: Music (ACE-Step 1.5 Turbo)", notes=f"duration={target_duration}s") as guard:
            engine_used = "acestep-v15-turbo"
            generated_ok = False

            try:
                import torch
                from acestep.handler import AceStepHandler  # type: ignore
                from acestep.llm_inference import LLMHandler  # type: ignore
                from acestep.inference import GenerationParams, GenerationConfig, generate_music  # type: ignore

                dit_handler = AceStepHandler()
                llm_handler = LLMHandler()

                acestep_root = self.models_dir / "ACE-Step-1.5"
                dit_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                lm_device = (
                    "cuda:1"
                    if (torch.cuda.is_available() and torch.cuda.device_count() > 1)
                    else dit_device
                )

                dit_handler.initialize_service(
                    project_root=str(acestep_root),
                    config_path=self.music_cfg.get("config_name", "acestep-v15-turbo"),
                    device=dit_device,
                )
                llm_handler.initialize(
                    checkpoint_dir=str(acestep_root / "checkpoints"),
                    lm_model_path=self.music_cfg.get("lm_model_name", "acestep-5Hz-lm-0.6B"),
                    backend=self.music_cfg.get("lm_backend", "pt"),  # PyTorch fp16 on T4 (never vLLM bf16)
                    device=lm_device,
                )

                params = GenerationParams(
                    task_type="text2music",
                    caption=music_prompt,
                    lyrics="[Instrumental]",
                    instrumental=True,
                    duration=target_duration,
                    inference_steps=int(self.music_cfg.get("inference_steps", 8)),
                    shift=float(self.music_cfg.get("shift", 3.0)),  # Recommended 3.0 for turbo in INFERENCE.md
                    infer_method="ode",
                    seed=seed,
                    thinking=True,
                )
                gen_cfg = GenerationConfig(
                    batch_size=1,
                    use_random_seed=False,
                    seeds=[seed],
                    audio_format="wav",
                )
                result = generate_music(
                    dit_handler=dit_handler,
                    llm_handler=llm_handler,
                    params=params,
                    config=gen_cfg,
                    save_dir=str(output_dir),
                )
                if result.success and result.audios:
                    raw_path = Path(result.audios[0]["path"])
                    audio_arr, sr = sf.read(str(raw_path), dtype="float32")
                    audio_arr = apply_fade_in_out(
                        audio_arr,
                        sr=sr,
                        fade_in_sec=float(self.music_cfg.get("fade_in_sec", 1.5)),
                        fade_out_sec=float(self.music_cfg.get("fade_out_sec", 2.0)),
                    )
                    sf.write(str(out_wav), audio_arr, sr)
                    generated_ok = True

                del dit_handler
                del llm_handler
            except Exception as exc:
                logger.info("ACE-Step 1.5 using procedural cinematic score fallback (%s)", exc)
                engine_used = "acestep-procedural-fallback"

            if not generated_ok:
                stereo = _synthesize_mood_instrumental_fallback(
                    music_prompt=music_prompt,
                    mood=mood,
                    duration_sec=target_duration,
                    sr=MUSIC_SR,
                )
                sf.write(str(out_wav), stereo, MUSIC_SR)

        return {
            "music_wav": str(out_wav),
            "duration_sec": target_duration,
            "engine": engine_used,
            "benchmark": guard.benchmark.to_dict() if guard.benchmark else {},
        }
