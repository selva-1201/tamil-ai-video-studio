"""
T4 (Turing sm_75) Hardware Compatibility Layer & Sequential VRAM Guard.

Why this module is essential on Kaggle 2×T4:
1. T4 GPUs (compute capability 7.5) do NOT have native bfloat16 tensor cores or FP8 support,
   and FlashAttention-2 refuses to run on sm_75 (requires sm_80+).
2. Upstream video/audio repos (Wan 2.2, InfiniteTalk, LTX-2.3, ACE-Step 1.5) hardcode
   `torch.bfloat16` or `import flash_attn`.
3. This module installs a transparent `flash_attn` SDPA/xFormers shim into `sys.modules`
   and redirects `torch.cuda.amp.autocast(dtype=torch.bfloat16)` to `torch.float16` on sm_75.
4. Enforces strict sequential model execution:
   load -> run -> unload -> gc.collect() + torch.cuda.empty_cache()
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import time
import types
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tamil_studio.t4")

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False


@dataclass
class GPUTelemetry:
    index: int
    name: str
    allocated_gb: float
    reserved_gb: float
    peak_gb: float
    total_gb: float
    free_gb: float
    compute_capability: str


@dataclass
class StageBenchmark:
    stage: str
    duration_sec: float
    gpu0_peak_gb: float
    gpu1_peak_gb: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_turing_or_older() -> bool:
    """Return True if primary CUDA device is Turing (sm_75, e.g. Tesla T4) or older."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return True
    major, minor = torch.cuda.get_device_capability(0)
    return major < 8


def _install_flash_attn_sdpa_shim() -> None:
    """
    Install a synthetic `flash_attn` module backed by PyTorch SDPA (`F.scaled_dot_product_attention`)
    so Wan 2.2 and InfiniteTalk do not fail on `import flash_attn` on Tesla T4 (sm_75).
    """
    if "flash_attn" in sys.modules:
        return

    shim = types.ModuleType("flash_attn")
    shim.__version__ = "2.7.4.t4_sdpa_shim"

    def flash_attn_func(
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        **kwargs: Any,
    ) -> "torch.Tensor":
        # flash_attn expects [B, S, H, D]; PyTorch SDPA expects [B, H, S, D]
        if q.dim() == 4:
            q_t = q.transpose(1, 2).to(torch.float16)
            k_t = k.transpose(1, 2).to(torch.float16)
            v_t = v.transpose(1, 2).to(torch.float16)
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            )
            return out.transpose(1, 2).to(q.dtype)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)

    def flash_attn_varlen_func(
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        cu_seqlens_q: "torch.Tensor",
        cu_seqlens_k: "torch.Tensor",
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        **kwargs: Any,
    ) -> "torch.Tensor":
        # Unpack varlen batch slices and run SDPA per sequence on T4
        outputs = []
        batch_size = len(cu_seqlens_q) - 1
        for i in range(batch_size):
            qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
            ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
            qi = q[qs:qe].unsqueeze(0).transpose(1, 2).to(torch.float16)
            ki = k[ks:ke].unsqueeze(0).transpose(1, 2).to(torch.float16)
            vi = v[ks:ke].unsqueeze(0).transpose(1, 2).to(torch.float16)
            oi = F.scaled_dot_product_attention(
                qi, ki, vi, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
            )
            outputs.append(oi.transpose(1, 2).squeeze(0).to(q.dtype))
        return torch.cat(outputs, dim=0)

    shim.flash_attn_func = flash_attn_func  # type: ignore
    shim.flash_attn_varlen_func = flash_attn_varlen_func  # type: ignore

    interface_mod = types.ModuleType("flash_attn.flash_attn_interface")
    interface_mod.flash_attn_func = flash_attn_func  # type: ignore
    interface_mod.flash_attn_varlen_func = flash_attn_varlen_func  # type: ignore

    sys.modules["flash_attn"] = shim
    sys.modules["flash_attn.flash_attn_interface"] = interface_mod
    logger.info("Installed T4 SDPA shim for flash_attn (sm_75 compatible).")


def apply_t4_compatibility_patches() -> Dict[str, Any]:
    """
    Apply global PyTorch settings and patches for Kaggle 2×T4 (sm_75).
    - Disables TF32/bf16 assumptions
    - Enables Flash/Mem-Efficient SDPA fallback kernels for sm_75 fp16
    - Redirects torch.autocast(dtype=torch.bfloat16) -> torch.float16 on Turing
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _install_flash_attn_sdpa_shim()

    status: Dict[str, Any] = {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "device_count": 0,
        "patched_bf16_autocast": False,
        "sdpa_enabled": False,
    }

    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return status

    status["cuda_available"] = True
    status["device_count"] = torch.cuda.device_count()

    # Enable memory-efficient SDPA on T4
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        status["sdpa_enabled"] = True

    if is_turing_or_older():
        orig_autocast = torch.amp.autocast

        class _T4SafeAutocast(orig_autocast):  # type: ignore
            def __init__(self, device_type: str = "cuda", dtype: Any = None, **kwargs: Any):
                if dtype == torch.bfloat16:
                    dtype = torch.float16
                super().__init__(device_type=device_type, dtype=dtype, **kwargs)

        torch.amp.autocast = _T4SafeAutocast  # type: ignore
        torch.cuda.amp.autocast = lambda dtype=torch.float16, **kw: _T4SafeAutocast("cuda", dtype=dtype, **kw)  # type: ignore
        status["patched_bf16_autocast"] = True
        logger.info("Patched torch.amp.autocast(bfloat16 -> float16) for Turing T4 GPUs.")

    return status


def aggressive_vram_cleanup() -> None:
    """Force garbage collection and release all cached CUDA blocks across all GPUs."""
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            with torch.cuda.device(idx):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    gc.collect()


def get_gpu_telemetry() -> List[Dict[str, Any]]:
    """Query live VRAM telemetry for both Kaggle T4 GPUs."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return [
            {
                "index": 0,
                "name": "Tesla T4 (Simulated / CPU Mode)",
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "peak_gb": 0.0,
                "total_gb": 15.0,
                "free_gb": 15.0,
                "compute_capability": "7.5",
            },
            {
                "index": 1,
                "name": "Tesla T4 (Simulated / CPU Mode)",
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "peak_gb": 0.0,
                "total_gb": 15.0,
                "free_gb": 15.0,
                "compute_capability": "7.5",
            },
        ]

    gpus: List[Dict[str, Any]] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_gb = round(props.total_memory / (1024 ** 3), 2)
        alloc_gb = round(torch.cuda.memory_allocated(idx) / (1024 ** 3), 2)
        res_gb = round(torch.cuda.memory_reserved(idx) / (1024 ** 3), 2)
        peak_gb = round(torch.cuda.max_memory_allocated(idx) / (1024 ** 3), 2)
        free_gb = round(max(0.0, total_gb - res_gb), 2)
        gpus.append(
            asdict(
                GPUTelemetry(
                    index=idx,
                    name=props.name,
                    allocated_gb=alloc_gb,
                    reserved_gb=res_gb,
                    peak_gb=peak_gb,
                    total_gb=total_gb,
                    free_gb=free_gb,
                    compute_capability=f"{props.major}.{props.minor}",
                )
            )
        )
    return gpus


class StageVRAMGuard:
    """
    Context manager enforcing sequential stage execution:
      1. Resets peak VRAM counters on entry
      2. Measures wall-clock duration
      3. Records peak VRAM on cuda:0 and cuda:1
      4. Runs aggressive_vram_cleanup() on exit so the next stage starts with ~0 GB used
    """

    def __init__(self, stage_name: str, notes: str = ""):
        self.stage_name = stage_name
        self.notes = notes
        self.start_time: float = 0.0
        self.benchmark: Optional[StageBenchmark] = None

    def __enter__(self) -> "StageVRAMGuard":
        aggressive_vram_cleanup()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(idx)
        self.start_time = time.perf_counter()
        logger.info(">>> [STAGE START] %s", self.stage_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = round(time.perf_counter() - self.start_time, 2)
        gpu0_peak = 0.0
        gpu1_peak = 0.0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu0_peak = round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 2)
            if torch.cuda.device_count() > 1:
                gpu1_peak = round(torch.cuda.max_memory_allocated(1) / (1024 ** 3), 2)

        self.benchmark = StageBenchmark(
            stage=self.stage_name,
            duration_sec=elapsed,
            gpu0_peak_gb=gpu0_peak,
            gpu1_peak_gb=gpu1_peak,
            notes=self.notes,
        )
        aggressive_vram_cleanup()
        logger.info(
            "<<< [STAGE END] %s | %.2fs | Peak VRAM GPU0=%.2f GB, GPU1=%.2f GB",
            self.stage_name,
            elapsed,
            gpu0_peak,
            gpu1_peak,
        )


def ensure_workspace_dirs(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """
    Ensure model weights and temp files live in /kaggle/temp (avoiding the 20 GB /kaggle/working limit)
    while final outputs live in /kaggle/working. Falls back cleanly on non-Kaggle machines.
    """
    is_kaggle = Path("/kaggle").exists() and os.access("/kaggle", os.W_OK)
    if is_kaggle:
        temp_root = Path("/kaggle/temp/tamil_studio")
        work_root = Path("/kaggle/working/tamil_studio_out")
    else:
        base = Path((config or {}).get("environment", {}).get("fallback_root", "./workspace_data")).resolve()
        temp_root = base / "temp"
        work_root = base / "working"

    paths = {
        "temp_root": temp_root,
        "work_root": work_root,
        "models_dir": temp_root / "models",
        "checkpoints_dir": temp_root / "checkpoints",
        "voices_dir": temp_root / "voices",
        "jobs_dir": work_root / "jobs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths


def get_disk_status(paths: Optional[Dict[str, Path]] = None) -> Dict[str, Any]:
    """Return free/used/total GB for temp and working directories (equivalent to df -h check)."""
    if paths is None:
        paths = ensure_workspace_dirs()
    out: Dict[str, Any] = {}
    for key in ("temp_root", "work_root"):
        target = paths[key]
        usage = shutil.disk_usage(target)
        out[key] = {
            "path": str(target),
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "used_gb": round(usage.used / (1024 ** 3), 2),
            "free_gb": round(usage.free / (1024 ** 3), 2),
        }
    return out


import wave
import numpy as np

try:
    import soundfile as _real_sf  # type: ignore
    _HAS_SF = True
except ImportError:
    _real_sf = None
    _HAS_SF = False


class _SoundFileShim:
    @staticmethod
    def read(path, dtype="float32"):
        if _HAS_SF:
            data, sr = _real_sf.read(str(path), dtype=dtype)
            return np.asarray(data, dtype=np.float32), int(sr)
        with wave.open(str(path), "rb") as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if sw == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        if n_ch > 1:
            arr = arr.reshape(-1, n_ch)
        return arr, int(sr)

    @staticmethod
    def write(path, data, samplerate):
        if _HAS_SF:
            _real_sf.write(str(path), data, samplerate)
            return
        arr = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
        pcm16 = (arr * 32767.0).astype(np.int16)
        n_ch = 1 if pcm16.ndim == 1 else pcm16.shape[1]
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(n_ch)
            wf.setsampwidth(2)
            wf.setframerate(int(samplerate))
            wf.writeframes(pcm16.tobytes())


sf_compat = _SoundFileShim()
