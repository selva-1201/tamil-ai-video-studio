"""
Stage 0 Reference Analysis, Character Profiles & Stage 4b Scene Keyframes (`pipeline/reference.py`).
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pipeline.t4_patches import StageVRAMGuard, aggressive_vram_cleanup, apply_t4_compatibility_patches
from pipeline.video import resolve_dimensions
logger = logging.getLogger("tamil_studio.reference")

# ==============================================================================
# Stage 0 Reference Analysis, Character Profiles & Stage 4b Scene Keyframes
# ==============================================================================
import base64 as _b64
import sys as _sys
import types as _types
import uuid as _uuid
from PIL import ImageEnhance, ImageFilter, ImageOps


def preprocess_uploaded_image(
    src_path: Path,
    dest_path: Path,
    max_edge: int = 1280,
) -> Dict[str, Any]:
    img = Image.open(str(src_path))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    orig_w, orig_h = img.size

    if max(orig_w, orig_h) > max_edge:
        scale = float(max_edge) / float(max(orig_w, orig_h))
        new_w = max(64, int(round(orig_w * scale)))
        new_h = max(64, int(round(orig_h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(dest_path), format="PNG")
    w, h = img.size
    return {
        "path": str(dest_path),
        "width": w,
        "height": h,
        "aspect_ratio": round(w / float(max(1, h)), 3),
    }


def detect_faces_in_image(image_path: Path) -> Dict[str, Any]:
    img_pil = Image.open(str(image_path)).convert("RGB")
    rgb = np.asarray(img_pil, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes: List[List[int]] = []

    try:
        cascade_dir = getattr(cv2, "data", None)
        cascade_file = (
            Path(cascade_dir.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_dir
            else Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
        )
        if cascade_file.exists():
            detector = cv2.CascadeClassifier(str(cascade_file))
            detected = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(36, 36))
            for (x, y, w, h) in detected:
                boxes.append([int(x), int(y), int(w), int(h)])
    except Exception:
        pass

    if not boxes:
        try:
            from facenet_pytorch import MTCNN  # type: ignore

            mtcnn = MTCNN(keep_all=True, device="cpu")
            b_arr, probs = mtcnn.detect(img_pil)
            if b_arr is not None:
                for box, prob in zip(b_arr, probs):
                    if prob and float(prob) >= 0.80:
                        x1, y1, x2, y2 = [int(round(v)) for v in box]
                        boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
            del mtcnn
        except Exception:
            pass

    return {
        "face_detected": len(boxes) > 0,
        "face_count": len(boxes),
        "boxes": boxes,
    }


def compute_face_identity_similarity(ref_image_path: Path, candidate_image_path: Path) -> Dict[str, Any]:
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN  # type: ignore

        mtcnn = MTCNN(image_size=160, margin=14, device="cpu")
        resnet = InceptionResnetV1(pretrained="vggface2").eval()

        img1 = Image.open(str(ref_image_path)).convert("RGB")
        img2 = Image.open(str(candidate_image_path)).convert("RGB")
        f1 = mtcnn(img1)
        f2 = mtcnn(img2)
        if f1 is not None and f2 is not None:
            with torch.no_grad():
                e1 = resnet(f1.unsqueeze(0))
                e2 = resnet(f2.unsqueeze(0))
                sim = float(torch.nn.functional.cosine_similarity(e1, e2).item())
                sim = round(max(0.0, min(1.0, sim)), 4)
            del mtcnn, resnet
            return {"similarity": sim, "method": "facenet-vggface2 (MIT)"}
    except Exception:
        pass

    def _extract_descriptor(p: Path) -> np.ndarray:
        img = Image.open(str(p)).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        center = arr[22:106, 22:106, :]
        hsv = cv2.cvtColor((center * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        spatial = cv2.resize(center, (12, 12)).flatten()
        vec = np.concatenate([
            hist_h / (hist_h.sum() + 1e-6),
            hist_s / (hist_s.sum() + 1e-6),
            hist_v / (hist_v.sum() + 1e-6),
            spatial,
        ])
        norm = float(np.linalg.norm(vec) + 1e-8)
        return vec / norm

    v1 = _extract_descriptor(ref_image_path)
    v2 = _extract_descriptor(candidate_image_path)
    cos_sim = float(np.dot(v1, v2))
    return {
        "similarity": round(max(0.0, min(1.0, cos_sim)), 4),
        "method": "subject-appearance-cosine (MIT)",
    }


def smart_crop_or_outpaint_to_aspect(
    src_image_path: Path,
    dest_image_path: Path,
    target_w: int,
    target_h: int,
    outpaint_if_aspect_diff: bool = True,
) -> Dict[str, Any]:
    img = Image.open(str(src_image_path)).convert("RGB")
    src_w, src_h = img.size
    src_ar = src_w / float(max(1, src_h))
    dst_ar = target_w / float(max(1, target_h))

    mode_used = "smart_crop"
    if outpaint_if_aspect_diff and abs(src_ar - dst_ar) > 0.35:
        mode_used = "feathered_outpaint"
        bg = ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=28))
        bg = ImageEnhance.Brightness(bg).enhance(0.68)

        scale = min(target_w / float(src_w), target_h / float(src_h))
        fg_w = max(1, int(round(src_w * scale)))
        fg_h = max(1, int(round(src_h * scale)))
        fg = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

        mask = Image.new("L", (fg_w, fg_h), 255)
        draw = ImageDraw.Draw(mask)
        feather = max(4, min(fg_w, fg_h) // 24)
        for i in range(feather):
            alpha = int(255 * (i / float(feather)))
            draw.rectangle([i, i, fg_w - 1 - i, fg_h - 1 - i], outline=alpha)

        paste_x = (target_w - fg_w) // 2
        paste_y = (target_h - fg_h) // 2
        bg.paste(fg, (paste_x, paste_y), mask)
        out_img = bg
    else:
        out_img = ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.38))

    dest_image_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(str(dest_image_path), format="PNG")
    return {
        "path": str(dest_image_path),
        "width": target_w,
        "height": target_h,
        "mode": mode_used,
        "source_aspect": round(src_ar, 3),
        "target_aspect": round(dst_ar, 3),
    }


def _analyze_image_fallback(image_path: Path, role: str) -> str:
    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32)
    mean_rgb = arr.mean(axis=(0, 1))
    luma = float(0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2])

    lighting = "warm golden-hour lighting" if mean_rgb[0] > mean_rgb[2] + 15 else (
        "cool atmospheric monsoon lighting" if mean_rgb[2] > mean_rgb[0] + 10 else "balanced natural daylight"
    )
    tone = "bright high-key exposure" if luma > 165 else ("moody low-key cinematic contrast" if luma < 85 else "medium-contrast naturalistic exposure")

    face_info = detect_faces_in_image(image_path)
    if face_info["face_detected"]:
        return (
            f"South Indian person ({face_info['face_count']} face detected in reference portrait, {w}x{h}px), "
            f"clear facial structure, expressive eyes, natural skin tone, captured under {lighting} with {tone}"
        )
    if role == "character":
        return f"Consistent main subject from uploaded reference ({w}x{h}px) with distinct silhouette and attire under {lighting} ({tone})"
    if role == "style":
        return f"Reference location and visual style palette ({w}x{h}px) featuring {lighting} and {tone}"
    return f"Uploaded starting scene composition ({w}x{h}px) with {lighting} and {tone}"


class ReferenceAndKeyframeEngine:
    def __init__(self, config: Dict[str, Any], models_dir: Path, characters_dir: Path, references_dir: Path):
        self.config = config
        self.models_dir = models_dir
        self.characters_dir = characters_dir
        self.references_dir = references_dir
        self.vlm_cfg = config.get("reference_vlm", {})
        self.kf_cfg = config.get("keyframe_edit", {})
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.references_dir.mkdir(parents=True, exist_ok=True)

    def list_characters(self) -> List[Dict[str, Any]]:
        chars: List[Dict[str, Any]] = []
        if not self.characters_dir.exists():
            return chars
        for c_dir in sorted(self.characters_dir.iterdir()):
            meta_file = c_dir / "character.json"
            if meta_file.exists():
                try:
                    chars.append(json.loads(meta_file.read_text(encoding="utf-8")))
                except Exception:
                    pass
        return chars

    def save_character_profile(
        self,
        name: str,
        image_paths: List[Path],
        reference_description_en: str,
        voice_profile_id: str = "",
        consent_verified: bool = False,
    ) -> Dict[str, Any]:
        if not image_paths:
            raise ValueError("At least one reference image is required to save a character profile.")
        if not consent_verified:
            raise ValueError(
                "Saving a character profile requires confirming: 'I own this image or have the person's permission.'"
            )

        char_id = f"char_{_uuid.uuid4().hex[:8]}"
        out_dir = self.characters_dir / char_id
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_imgs: List[str] = []
        for idx, p in enumerate(image_paths[:4], start=1):
            dest = out_dir / f"ref_{idx:02d}.png"
            preprocess_uploaded_image(p, dest, max_edge=int(self.vlm_cfg.get("max_image_edge", 1280)))
            saved_imgs.append(str(dest))

        if not reference_description_en.strip():
            analysis = self.analyze_references(
                [{"path": s, "role": "character", "fidelity": "strict"} for s in saved_imgs],
                face_consent_verified=consent_verified,
            )
            reference_description_en = analysis["reference_description_en"]

        meta = {
            "character_id": char_id,
            "name": name.strip() or f"Character {char_id[-4:]}",
            "image_paths": saved_imgs,
            "reference_description_en": reference_description_en.strip(),
            "voice_profile_id": voice_profile_id.strip(),
            "consent_verified": True,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (out_dir / "character.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def analyze_references(
        self,
        references: List[Dict[str, Any]],
        face_consent_verified: bool = False,
    ) -> Dict[str, Any]:
        apply_t4_compatibility_patches()
        if not references:
            return {
                "reference_description_en": "",
                "items": [],
                "any_face_detected": False,
                "engine": "none",
                "benchmark": {},
            }

        with StageVRAMGuard("Stage 0: Reference Analysis (VLM)", notes=f"{len(references)} image(s)") as guard:
            analyzed_items: List[Dict[str, Any]] = []
            any_face = False

            for idx, ref in enumerate(references[:4], start=1):
                src_p = Path(ref["path"])
                role = ref.get("role", "character")
                fidelity = ref.get("fidelity", "strict")
                face_info = detect_faces_in_image(src_p)
                if face_info["face_detected"]:
                    any_face = True

                analyzed_items.append(
                    {
                        "index": idx,
                        "path": str(src_p),
                        "role": role,
                        "fidelity": fidelity,
                        "face_detected": face_info["face_detected"],
                        "face_count": face_info["face_count"],
                        "face_boxes": face_info["boxes"],
                        "caption_en": "",
                    }
                )

            if (any_face or any(r["role"] == "character" for r in analyzed_items)) and not face_consent_verified:
                raise ValueError(
                    "Face / character reference detected: You must check 'I own this image or have the person's permission' before proceeding."
                )

            vlm_gguf = self.models_dir / self.vlm_cfg.get("primary_file", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
            mmproj_gguf = self.models_dir / self.vlm_cfg.get("mmproj_file", "mmproj-F16.gguf")
            engine_used = "pixel-attribute-analyzer"

            if vlm_gguf.exists() and mmproj_gguf.exists():
                try:
                    from llama_cpp import Llama  # type: ignore
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler  # type: ignore

                    chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_gguf))
                    vlm = Llama(
                        model_path=str(vlm_gguf),
                        chat_handler=chat_handler,
                        n_ctx=2048,
                        n_gpu_layers=-1,
                        main_gpu=0,
                        verbose=False,
                    )
                    for item in analyzed_items:
                        img_bytes = Path(item["path"]).read_bytes()
                        b64_uri = "data:image/png;base64," + _b64.b64encode(img_bytes).decode("ascii")
                        resp = vlm.create_chat_completion(
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": b64_uri}},
                                        {
                                            "type": "text",
                                            "text": (
                                                "Describe this reference image in one concise English paragraph for video continuity: "
                                                "include the person/subject's appearance, age, hair, clothing colors, setting, and lighting style."
                                            ),
                                        },
                                    ],
                                }
                            ],
                            max_tokens=180,
                            temperature=0.2,
                        )
                        item["caption_en"] = (resp["choices"][0]["message"]["content"] or "").strip()
                    del vlm, chat_handler
                    engine_used = vlm_gguf.name
                except Exception as exc:
                    logger.warning("Qwen2.5-VL GGUF fallback triggered: %s", exc)

            for item in analyzed_items:
                if not item["caption_en"]:
                    item["caption_en"] = _analyze_image_fallback(Path(item["path"]), item["role"])

            combined_parts = [
                f"[Ref {item['index']} ({item['role']}, {item['fidelity']})]: {item['caption_en']}"
                for item in analyzed_items
            ]
            combined_desc = " ".join(combined_parts)

        return {
            "reference_description_en": combined_desc,
            "items": analyzed_items,
            "any_face_detected": any_face,
            "engine": engine_used,
            "benchmark": guard.benchmark.to_dict() if guard.benchmark else {},
        }

    def _edit_keyframe_with_model_or_composite(
        self,
        ref_image_path: Path,
        scene_prompt_en: str,
        scene_id: int,
        target_w: int,
        target_h: int,
        fidelity: str,
        out_path: Path,
        attempt: int = 1,
    ) -> str:
        import shutil as _shutil
        flux_gguf = self.models_dir / self.kf_cfg.get("flux_kontext_file", "flux1-kontext-dev-Q5_K_S.gguf")
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            torch = None  # type: ignore
            has_cuda = False

        edit_prompt = f"the same person, keeping identical facial features and clothing, now {scene_prompt_en}"
        guidance = 2.5 if (fidelity == "strict" or attempt > 1) else 3.8

        if flux_gguf.exists() and has_cuda:
            try:
                from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig  # type: ignore
                from diffusers.utils import load_image  # type: ignore

                transformer = FluxTransformer2DModel.from_single_file(
                    str(flux_gguf),
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
                    torch_dtype=torch.float16,
                )
                pipe = FluxKontextPipeline.from_pretrained(
                    self.kf_cfg.get("flux_base_repo", "black-forest-labs/FLUX.1-Kontext-dev"),
                    transformer=transformer,
                    torch_dtype=torch.float16,
                )
                pipe.enable_model_cpu_offload()
                cond_img = load_image(str(ref_image_path)).resize((target_w, target_h))
                out_img = pipe(
                    image=cond_img,
                    prompt=edit_prompt,
                    guidance_scale=guidance,
                    width=target_w,
                    height=target_h,
                    num_inference_steps=16,
                    generator=torch.Generator("cpu").manual_seed(100 + scene_id * 10 + attempt),
                ).images[0]
                out_img.save(str(out_path))
                del pipe, transformer
                aggressive_vram_cleanup()
                return "FLUX.1-Kontext-dev-GGUF"
            except Exception as exc:
                logger.warning("FLUX.1 Kontext GGUF fallback triggered: %s", exc)

        smart_crop_or_outpaint_to_aspect(
            src_image_path=ref_image_path,
            dest_image_path=out_path,
            target_w=target_w,
            target_h=target_h,
            outpaint_if_aspect_diff=True,
        )
        base_img = Image.open(str(out_path)).convert("RGB")
        if fidelity == "creative" and attempt == 1:
            arr = np.asarray(base_img, dtype=np.float32)
            p_low = scene_prompt_en.lower()
            if "rain" in p_low or "monsoon" in p_low:
                arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.06 + 6.0, 0, 255)
            elif "sun" in p_low or "golden" in p_low or "warm" in p_low:
                arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.07 + 8.0, 0, 255)
            base_img = Image.fromarray(arr.astype(np.uint8))

        base_img.save(str(out_path), format="PNG")
        return "kontext-identity-composite"

    def generate_scene_keyframes(
        self,
        scenes: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        output_dir: Path,
        aspect: str = "9:16",
        resolution: str = "480p",
        outpaint_start_frame: bool = True,
    ) -> Dict[str, Any]:
        import shutil as _shutil
        apply_t4_compatibility_patches()
        kf_dir = output_dir / "keyframes"
        kf_dir.mkdir(parents=True, exist_ok=True)
        target_w, target_h = resolve_dimensions(aspect=aspect, quality_tier=resolution)

        start_ref = next((r for r in references if r.get("role") == "start_frame"), None)
        char_ref = next((r for r in references if r.get("role") == "character"), None)
        style_ref = next((r for r in references if r.get("role") == "style"), None)
        primary_ref = char_ref or start_ref or style_ref or (references[0] if references else None)

        sim_thresh = float(self.kf_cfg.get("identity_similarity_threshold", 0.68))
        max_retries = int(self.kf_cfg.get("max_identity_retries", 2))

        keyframes_meta: List[Dict[str, Any]] = []
        engine_used = "direct_crop"

        with StageVRAMGuard("Stage 4b: Scene Keyframe Generation (Kontext / Start-Frame)", notes=f"{len(scenes)} scenes") as guard:
            for scene in scenes:
                scene_id = int(scene["id"])
                kf_path = kf_dir / f"scene_{scene_id:02d}.png"

                if scene_id == 1 and start_ref is not None:
                    crop_info = smart_crop_or_outpaint_to_aspect(
                        src_image_path=Path(start_ref["path"]),
                        dest_image_path=kf_path,
                        target_w=target_w,
                        target_h=target_h,
                        outpaint_if_aspect_diff=outpaint_start_frame,
                    )
                    sim_info = compute_face_identity_similarity(Path(start_ref["path"]), kf_path)
                    kf_record = {
                        "scene_id": scene_id,
                        "keyframe_path": str(kf_path),
                        "role_used": "start_frame",
                        "mode": crop_info["mode"],
                        "identity_similarity": sim_info["similarity"],
                        "identity_method": sim_info["method"],
                        "identity_passed": True,
                        "attempts": 1,
                        "engine": "start_frame_exact",
                    }
                    scene["keyframe_path"] = str(kf_path)
                    scene["identity_similarity"] = sim_info["similarity"]
                    keyframes_meta.append(kf_record)
                    continue

                if primary_ref is not None:
                    ref_path = Path(primary_ref["path"])
                    fidelity = primary_ref.get("fidelity", "strict")
                    best_sim = -1.0
                    best_method = ""
                    attempts_used = 0

                    for attempt in range(1, max_retries + 2):
                        attempts_used = attempt
                        cand_path = kf_dir / f"scene_{scene_id:02d}_try_{attempt}.png"
                        engine_used = self._edit_keyframe_with_model_or_composite(
                            ref_image_path=ref_path,
                            scene_prompt_en=scene["visual_prompt_en"],
                            scene_id=scene_id,
                            target_w=target_w,
                            target_h=target_h,
                            fidelity=fidelity,
                            out_path=cand_path,
                            attempt=attempt,
                        )
                        sim_res = compute_face_identity_similarity(ref_path, cand_path)
                        if sim_res["similarity"] > best_sim:
                            best_sim = sim_res["similarity"]
                            best_method = sim_res["method"]
                            _shutil.copy2(cand_path, kf_path)
                        cand_path.unlink(missing_ok=True)
                        if best_sim >= sim_thresh:
                            break

                    kf_record = {
                        "scene_id": scene_id,
                        "keyframe_path": str(kf_path),
                        "role_used": primary_ref.get("role", "character"),
                        "mode": fidelity,
                        "identity_similarity": round(best_sim, 4),
                        "identity_method": best_method,
                        "identity_passed": bool(best_sim >= sim_thresh),
                        "attempts": attempts_used,
                        "engine": engine_used,
                    }
                    scene["keyframe_path"] = str(kf_path)
                    scene["identity_similarity"] = round(best_sim, 4)
                    keyframes_meta.append(kf_record)

        return {
            "keyframes": keyframes_meta,
            "engine": engine_used,
            "benchmark": guard.benchmark.to_dict() if guard.benchmark else {},
        }
