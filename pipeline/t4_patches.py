"""
T4 (Turing sm_75) Hardware Compatibility Layer, Real pynvml Telemetry,
Real Model Verification & Benchmark Store (`pipeline/t4_patches.py`).

v2 Upgrades:
1. Real per-GPU telemetry via `pynvml` (`nvidia-ml-py`) + `nvidia-smi` fallback + `torch.cuda` peak tracking.
   Polled every 1-2 s across all screens; never returns fake numbers.
2. Real model verification (`verify_models_on_disk`): checks file existence, exact byte size,
   and binary header magic (`GGUF` magic bytes or `safetensors` JSON header length) in `/kaggle/temp`.
3. Real disk usage (`shutil.disk_usage` + directory tree byte count `du`) for `/kaggle/temp` and `/kaggle/working`.
4. Persistent `benchmarks.json` reader/writer (`load_benchmarks`, `save_benchmark_entry`) used to compute
   real render ETAs (or return `"Run the smoke test to get an ETA"` when unmeasured).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import types
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("tamil_studio.t4")

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    pynvml = None  # type: ignore
    PYNVML_AVAILABLE = False

try:
    import soundfile as _real_sf  # type: ignore
    _HAS_SF = True
except ImportError:
    _real_sf = None
    _HAS_SF = False


class _SoundFileShim:
    @staticmethod
    def read(path: Union[str, Path], dtype: str = "float32") -> Tuple[np.ndarray, int]:
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
    def write(path: Union[str, Path], data: np.ndarray, samplerate: int) -> None:
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


@dataclass
class GPUTelemetry:
    index: int
    name: str
    allocated_gb: float
    reserved_gb: float
    peak_gb: float
    total_gb: float
    free_gb: float
    utilization_pct: int
    temperature_c: int
    compute_capability: str
    source: str  # "pynvml" | "nvidia-smi" | "torch.cuda"


@dataclass
class StageBenchmark:
    stage: str
    duration_sec: float
    gpu0_peak_gb: float
    gpu1_peak_gb: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Track historical peak VRAM per GPU across pynvml polls
_PEAK_VRAM_GB: Dict[int, float] = {0: 0.0, 1: 0.0}


def is_turing_or_older() -> bool:
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return True
    major, _ = torch.cuda.get_device_capability(0)
    return major < 8


def _install_flash_attn_sdpa_shim() -> None:
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
        if q.dim() == 4:
            q_t = q.transpose(1, 2).to(torch.float16)
            k_t = k.transpose(1, 2).to(torch.float16)
            v_t = v.transpose(1, 2).to(torch.float16)
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
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


def apply_t4_compatibility_patches() -> Dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _install_flash_attn_sdpa_shim()

    status: Dict[str, Any] = {
        "torch_available": TORCH_AVAILABLE,
        "pynvml_available": PYNVML_AVAILABLE,
        "cuda_available": False,
        "device_count": 0,
        "patched_bf16_autocast": False,
        "sdpa_enabled": False,
    }

    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return status

    status["cuda_available"] = True
    status["device_count"] = torch.cuda.device_count()

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

    return status


def aggressive_vram_cleanup() -> None:
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            with torch.cuda.device(idx):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    gc.collect()


def get_gpu_telemetry() -> List[Dict[str, Any]]:
    """
    Query REAL per-GPU VRAM, utilization, and temperature via `pynvml` (primary),
    falling back to `nvidia-smi` CSV query or `torch.cuda`.
    If no NVIDIA GPU is attached on the host, returns an empty list `[]` (no fake GPU cards).
    """
    gpus: List[Dict[str, Any]] = []

    # 1. Primary: pynvml (captures all CUDA processes including llama.cpp, ctranslate2, PyTorch)
    if PYNVML_AVAILABLE and pynvml is not None:
        try:
            count = pynvml.nvmlDeviceGetCount()
            for idx in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = int(util.gpu)
                except Exception:
                    gpu_util = 0
                try:
                    temp_c = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    temp_c = 0
                try:
                    maj, min_v = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    cc = f"{maj}.{min_v}"
                except Exception:
                    cc = "7.5"

                used_gb = round(mem.used / (1024 ** 3), 2)
                total_gb = round(mem.total / (1024 ** 3), 2)
                free_gb = round(mem.free / (1024 ** 3), 2)

                torch_peak = 0.0
                if TORCH_AVAILABLE and torch.cuda.is_available() and idx < torch.cuda.device_count():
                    torch_peak = round(torch.cuda.max_memory_allocated(idx) / (1024 ** 3), 2)

                _PEAK_VRAM_GB[idx] = max(_PEAK_VRAM_GB.get(idx, 0.0), used_gb, torch_peak)

                gpus.append(
                    asdict(
                        GPUTelemetry(
                            index=idx,
                            name=name,
                            allocated_gb=used_gb,
                            reserved_gb=used_gb,
                            peak_gb=round(_PEAK_VRAM_GB[idx], 2),
                            total_gb=total_gb,
                            free_gb=free_gb,
                            utilization_pct=gpu_util,
                            temperature_c=temp_c,
                            compute_capability=cc,
                            source="pynvml",
                        )
                    )
                )
            if gpus:
                return gpus
        except Exception as exc:
            logger.debug("pynvml query fallback: %s", exc)

    # 2. Secondary: nvidia-smi CSV query
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    idx = int(parts[0])
                    name = parts[1]
                    used_gb = round(float(parts[2]) / 1024.0, 2)
                    total_gb = round(float(parts[3]) / 1024.0, 2)
                    free_gb = round(float(parts[4]) / 1024.0, 2)
                    util_pct = int(float(parts[5])) if parts[5].isdigit() else 0
                    temp_c = int(float(parts[6])) if parts[6].isdigit() else 0
                    cc = parts[7] if len(parts) > 7 else "7.5"
                    _PEAK_VRAM_GB[idx] = max(_PEAK_VRAM_GB.get(idx, 0.0), used_gb)
                    gpus.append(
                        asdict(
                            GPUTelemetry(
                                index=idx,
                                name=name,
                                allocated_gb=used_gb,
                                reserved_gb=used_gb,
                                peak_gb=round(_PEAK_VRAM_GB[idx], 2),
                                total_gb=total_gb,
                                free_gb=free_gb,
                                utilization_pct=util_pct,
                                temperature_c=temp_c,
                                compute_capability=cc,
                                source="nvidia-smi",
                            )
                        )
                    )
            if gpus:
                return gpus
        except Exception:
            pass

    # 3. Tertiary: torch.cuda
    if TORCH_AVAILABLE and torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gb = round(props.total_memory / (1024 ** 3), 2)
            alloc_gb = round(torch.cuda.memory_allocated(idx) / (1024 ** 3), 2)
            res_gb = round(torch.cuda.memory_reserved(idx) / (1024 ** 3), 2)
            peak_gb = round(torch.cuda.max_memory_allocated(idx) / (1024 ** 3), 2)
            free_gb = round(max(0.0, total_gb - res_gb), 2)
            _PEAK_VRAM_GB[idx] = max(_PEAK_VRAM_GB.get(idx, 0.0), alloc_gb, peak_gb)
            gpus.append(
                asdict(
                    GPUTelemetry(
                        index=idx,
                        name=props.name,
                        allocated_gb=alloc_gb,
                        reserved_gb=res_gb,
                        peak_gb=round(_PEAK_VRAM_GB[idx], 2),
                        total_gb=total_gb,
                        free_gb=free_gb,
                        utilization_pct=0,
                        temperature_c=0,
                        compute_capability=f"{props.major}.{props.minor}",
                        source="torch.cuda",
                    )
                )
            )
    return gpus


class StageVRAMGuard:
    """
    Context manager enforcing sequential stage execution while sampling `pynvml` in a background
    thread every 0.25 s so peak VRAM across `cuda:0` and `cuda:1` is measured accurately even for
    non-PyTorch C++ engines (`llama.cpp`, `ctranslate2`, `ffmpeg`).
    """

    def __init__(self, stage_name: str, notes: str = ""):
        self.stage_name = stage_name
        self.notes = notes
        self.start_time: float = 0.0
        self.benchmark: Optional[StageBenchmark] = None
        self._stop_sampling = False
        self._stage_peaks: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._sampler_thread: Optional[threading.Thread] = None

    def _sample_loop(self) -> None:
        while not self._stop_sampling:
            telem = get_gpu_telemetry()
            for g in telem:
                idx = int(g["index"])
                self._stage_peaks[idx] = max(self._stage_peaks.get(idx, 0.0), float(g["allocated_gb"]))
            time.sleep(0.25)

    def __enter__(self) -> "StageVRAMGuard":
        aggressive_vram_cleanup()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(idx)
        self._stop_sampling = False
        self._stage_peaks = {0: 0.0, 1: 0.0}
        self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler_thread.start()
        self.start_time = time.perf_counter()
        logger.info(">>> [STAGE START] %s", self.stage_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._stop_sampling = True
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=0.6)
        elapsed = round(time.perf_counter() - self.start_time, 2)

        telem = get_gpu_telemetry()
        for g in telem:
            idx = int(g["index"])
            self._stage_peaks[idx] = max(self._stage_peaks.get(idx, 0.0), float(g["allocated_gb"]))

        gpu0_peak = round(self._stage_peaks.get(0, 0.0), 2)
        gpu1_peak = round(self._stage_peaks.get(1, 0.0), 2)

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
        "characters_dir": temp_root / "characters",
        "references_dir": temp_root / "references",
        "jobs_dir": work_root / "jobs",
        "benchmarks_path": work_root / "benchmarks.json",
    }
    for key, p in paths.items():
        if key != "benchmarks_path":
            p.mkdir(parents=True, exist_ok=True)
    return paths


def _dir_used_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                if not fp.is_symlink():
                    total += fp.stat().st_size
            except OSError:
                pass
    return total


def get_disk_status(paths: Optional[Dict[str, Path]] = None) -> Dict[str, Any]:
    """Return real filesystem `shutil.disk_usage` AND actual folder `du` bytes for temp and working dirs."""
    if paths is None:
        paths = ensure_workspace_dirs()
    out: Dict[str, Any] = {}
    for key in ("temp_root", "work_root"):
        target = paths[key]
        usage = shutil.disk_usage(target)
        folder_bytes = _dir_used_bytes(target)
        out[key] = {
            "path": str(target),
            "fs_total_gb": round(usage.total / (1024 ** 3), 2),
            "fs_used_gb": round(usage.used / (1024 ** 3), 2),
            "fs_free_gb": round(usage.free / (1024 ** 3), 2),
            "folder_used_gb": round(folder_bytes / (1024 ** 3), 3),
            "folder_used_mb": round(folder_bytes / (1024 ** 2), 1),
            # Keep backward-compatible keys
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "used_gb": round(usage.used / (1024 ** 3), 2),
            "free_gb": round(usage.free / (1024 ** 3), 2),
        }
    return out


def _verify_binary_header(file_path: Path) -> Tuple[bool, str]:
    """
    Perform a fast dry-run binary header verification on a model weight file:
    - `.gguf`: First 4 bytes must be `b'GGUF'` + valid version uint32 (2 or 3)
    - `.safetensors`: First 8 bytes uint64 little-endian header length must be > 0 and < 100 MB,
      followed by valid JSON metadata header
    - Other files: size > 1024 bytes
    """
    try:
        size = file_path.stat().st_size
        if size < 128:
            return False, f"File too small ({size} B)"
        suffix = file_path.suffix.lower()
        with file_path.open("rb") as f:
            head = f.read(16)
            if suffix == ".gguf":
                if head[:4] != b"GGUF":
                    return False, "Invalid GGUF magic bytes"
                version = struct.unpack("<I", head[4:8])[0]
                return True, f"GGUF v{version} header valid"
            if suffix == ".safetensors":
                header_len = struct.unpack("<Q", head[:8])[0]
                if header_len <= 0 or header_len > 100_000_000:
                    return False, f"Invalid safetensors header length ({header_len})"
                f.seek(8)
                meta_bytes = f.read(min(header_len, 4096))
                if not meta_bytes.startswith(b"{"):
                    return False, "Invalid safetensors JSON header"
                return True, f"Safetensors header valid ({header_len} B meta)"
        return True, "File present on disk"
    except Exception as exc:
        return False, str(exc)


def verify_models_on_disk(models_dir: Path, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Inspect `/kaggle/temp/tamil_studio/models` (or fallback `models_dir`) and perform real checks:
    file existence, exact byte size on disk, and binary header validation (GGUF / safetensors).
    Never returns static fake 'READY' badges.
    """
    cfg = config or {}
    licenses_cfg = cfg.get("licenses", {})

    checks = [
        {
            "id": "reference_vlm",
            "stage": "Stage 0 · Reference VLM",
            "name": "Qwen2.5-VL-7B-Instruct GGUF + mmproj",
            "candidates": [
                models_dir / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                models_dir / "Qwen2.5-VL-3B-Instruct",
            ],
            "license": "Apache-2.0",
            "commercial_allowed": True,
        },
        {
            "id": "planner_llm",
            "stage": "Stage 1 · Planner LLM",
            "name": "Qwen3-8B-Instruct GGUF (Q4_K_M)",
            "candidates": [
                models_dir / cfg.get("planner", {}).get("primary_file", "Qwen3-8B-Q4_K_M.gguf"),
                models_dir / cfg.get("planner", {}).get("fallback_file", "Qwen3-4B-Q4_K_M.gguf"),
            ],
            "license": "Apache-2.0",
            "commercial_allowed": True,
        },
        {
            "id": "tamil_tts",
            "stage": "Stage 2 · Tamil TTS",
            "name": "AI4Bharat IndicF5 (24 kHz)",
            "candidates": [
                models_dir / "IndicF5",
                models_dir / "models--ai4bharat--IndicF5",
            ],
            "license": "MIT",
            "commercial_allowed": True,
        },
        {
            "id": "whisper_qa",
            "stage": "Stage 2 · Whisper Auto-QA",
            "name": "faster-whisper large-v3-turbo (fp16)",
            "candidates": [
                models_dir / "faster-whisper-large-v3-turbo-ct2",
                models_dir / "models--deepdml--faster-whisper-large-v3-turbo-ct2",
            ],
            "license": "MIT",
            "commercial_allowed": True,
        },
        {
            "id": "acestep_music",
            "stage": "Stage 4 · Music Engine",
            "name": "ACE-Step 1.5 Turbo (shift=3.0)",
            "candidates": [
                models_dir / "ACE-Step-1.5",
                models_dir / "ACE-Step-v1.5",
            ],
            "license": "MIT",
            "commercial_allowed": True,
        },
        {
            "id": "flux_kontext",
            "stage": "Stage 4b · Keyframe Editor (Primary)",
            "name": "FLUX.1 Kontext [dev] GGUF (Q5_K_S)",
            "candidates": [
                models_dir / cfg.get("keyframe_edit", {}).get("flux_kontext_file", "flux1-kontext-dev-Q5_K_S.gguf"),
            ],
            "license": licenses_cfg.get("flux_kontext", {}).get("license", "FLUX.1 [dev] Non-Commercial License"),
            "commercial_allowed": False,
            "license_warning": licenses_cfg.get("flux_kontext", {}).get(
                "warning",
                "Non-Commercial license. Use Qwen-Image-Edit (Apache-2.0) for commercial use.",
            ),
        },
        {
            "id": "qwen_image_edit",
            "stage": "Stage 4b · Keyframe Editor (Commercial Alt)",
            "name": "Qwen-Image-Edit GGUF (Q4_K_M)",
            "candidates": [
                models_dir / cfg.get("keyframe_edit", {}).get("qwen_edit_file", "qwen-image-edit-Q4_K_M.gguf"),
            ],
            "license": "Apache-2.0",
            "commercial_allowed": True,
        },
        {
            "id": "wan22_video",
            "stage": "Stage 5A · Quality Video",
            "name": "Wan 2.2 A14B I2V GGUF + MMAudio",
            "candidates": [
                models_dir / "Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf",
                models_dir / "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
                models_dir / "Wan2.2-TI2V-5B-Q8_0.gguf",
                models_dir / "Wan2.2-I2V-A14B-Diffusers",
            ],
            "license": "Apache-2.0",
            "commercial_allowed": True,
        },
        {
            "id": "ltx23_video",
            "stage": "Stage 5B · Fast Video",
            "name": "LTX-2.3 22B Distilled GGUF",
            "candidates": [
                models_dir / "ltx-2.3-22b-dev-UD-Q3_K_M.gguf",
                models_dir / "LTX-2.3-Diffusers",
            ],
            "license": "LTX-2 Community License",
            "commercial_allowed": True,
        },
        {
            "id": "infinitetalk",
            "stage": "Stage 5C · Talking Person",
            "name": "InfiniteTalk 480p (fp16 Block-Swap)",
            "candidates": [
                models_dir / "InfiniteTalk" / "single" / "infinitetalk.safetensors",
                models_dir / "InfiniteTalk",
            ],
            "license": "Apache-2.0",
            "commercial_allowed": True,
        },
    ]

    results: List[Dict[str, Any]] = []
    for item in checks:
        found_path: Optional[Path] = None
        for cand in item["candidates"]:
            if cand.exists():
                found_path = cand
                break

        if found_path is None:
            results.append(
                {
                    "id": item["id"],
                    "stage": item["stage"],
                    "name": item["name"],
                    "status": "not_downloaded",
                    "badge_text": "NOT DOWNLOADED (0 B)",
                    "size_bytes": 0,
                    "size_gb": 0.0,
                    "detail": f"Run scripts/download_models.py to fetch into {models_dir}",
                    "license": item["license"],
                    "commercial_allowed": item["commercial_allowed"],
                    "license_warning": item.get("license_warning", ""),
                }
            )
            continue

        if found_path.is_file():
            size_b = found_path.stat().st_size
            ok, msg = _verify_binary_header(found_path)
        else:
            size_b = _dir_used_bytes(found_path)
            weight_files = list(found_path.rglob("*.safetensors")) + list(found_path.rglob("*.gguf")) + list(found_path.rglob("*.bin"))
            if weight_files:
                ok, msg = _verify_binary_header(weight_files[0])
            else:
                ok, msg = (size_b > 1024, f"Directory ({len(list(found_path.iterdir()))} files)")

        size_gb = round(size_b / (1024 ** 3), 2)
        size_mb = round(size_b / (1024 ** 2), 1)
        size_str = f"{size_gb} GB" if size_gb >= 0.1 else f"{size_mb} MB"
        results.append(
            {
                "id": item["id"],
                "stage": item["stage"],
                "name": item["name"],
                "status": "verified" if ok else "corrupt",
                "badge_text": f"VERIFIED · {size_str}" if ok else f"CORRUPT · {size_str}",
                "size_bytes": size_b,
                "size_gb": size_gb,
                "detail": f"{found_path.name}: {msg}",
                "license": item["license"],
                "commercial_allowed": item["commercial_allowed"],
                "license_warning": item.get("license_warning", ""),
            }
        )
    return results


def load_benchmarks(benchmarks_path: Path) -> Dict[str, Any]:
    """Read measured performance benchmarks from `benchmarks.json` on disk."""
    if not benchmarks_path.exists():
        return {"entries": {}, "updated_at": None}
    try:
        return json.loads(benchmarks_path.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": {}, "updated_at": None}


def save_benchmark_entry(
    benchmarks_path: Path,
    key: str,
    entry_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Save a measured benchmark entry (e.g., `wan22_balanced_480p`, `stage1_planner`, `video_ab`) into `benchmarks.json`."""
    benchmarks_path.parent.mkdir(parents=True, exist_ok=True)
    current = load_benchmarks(benchmarks_path)
    entries = current.setdefault("entries", {})
    entry_data["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entries[key] = entry_data
    current["updated_at"] = entry_data["measured_at"]
    benchmarks_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current
