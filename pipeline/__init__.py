"""
Tamil AI Video Studio v2 — Pipeline Package
Sequential, resumable audio-first & reference-consistent pipeline designed for Kaggle 2×T4 GPUs.
"""

import sys as _sys
import types as _types

from pipeline.t4_patches import (
    StageVRAMGuard,
    apply_t4_compatibility_patches,
    ensure_workspace_dirs,
    get_gpu_telemetry,
)
from pipeline.video import (
    ReferenceAndKeyframeEngine,
    compute_face_identity_similarity,
    detect_faces_in_image,
    preprocess_uploaded_image,
    smart_crop_or_outpaint_to_aspect,
)

# Register `pipeline.reference` module alias so `from pipeline.reference import ...` works everywhere
_ref_mod = _types.ModuleType("pipeline.reference")
_ref_mod.ReferenceAndKeyframeEngine = ReferenceAndKeyframeEngine
_ref_mod.compute_face_identity_similarity = compute_face_identity_similarity
_ref_mod.detect_faces_in_image = detect_faces_in_image
_ref_mod.preprocess_uploaded_image = preprocess_uploaded_image
_ref_mod.smart_crop_or_outpaint_to_aspect = smart_crop_or_outpaint_to_aspect
_sys.modules["pipeline.reference"] = _ref_mod

__all__ = [
    "apply_t4_compatibility_patches",
    "StageVRAMGuard",
    "get_gpu_telemetry",
    "ensure_workspace_dirs",
    "ReferenceAndKeyframeEngine",
    "compute_face_identity_similarity",
    "detect_faces_in_image",
    "preprocess_uploaded_image",
    "smart_crop_or_outpaint_to_aspect",
]
