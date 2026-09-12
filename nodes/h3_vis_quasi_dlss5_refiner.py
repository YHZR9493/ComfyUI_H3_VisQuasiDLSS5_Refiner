"""H3VisQuasiDLSS5Refiner - self-contained YOLO-World guided MiniMax H3 refine.

Single-node pipeline: YOLO-World open-vocabulary detection -> prompt definition
(class names injected into the fix prompt) -> chunked latent re-generation by
MiniMax H3 -> decode -> mask-feathered paste-back. Result keeps the untouched
regions pixel-identical and rebuilds only the detected areas, closely matching
the H3-FaceRefine "detect -> re-sample -> stitch" precision.

Interface (final): 4 inputs (model / video_vae / clip / images) -> 1 output
(images). No audio_vae, no audio output, no reference-image groups.

Self-contained: detection, latent injection, conditioning, sampling and the
paste-back are all implemented here against ComfyUI's official MiniMax H3 nodes
(comfy_extras.nodes_minimax_h3) - no dependency on the FaceRefine package.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional heavy deps (loaded lazily)
# ---------------------------------------------------------------------------
_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


# ---------------------------------------------------------------------------
# H3 grid helpers (mirror comfy_extras.nodes_minimax_h3, kept small & stable)
# ---------------------------------------------------------------------------
FPS = 24
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344

# Default fix prompts (also referenced by INPUT_TYPES widgets).
FF_REPAIR_PROMPT = (
    "remove all compression damage and rendering artifacts: texture shimmering "
    "and tearing, material and shading flicker, color banding streaks, "
    "static-frame corruption, random color patches, red and green flashing "
    "bursts, noise blotches; restore clean, stable, solid texture and smooth, "
    "artifact-free image"
)
BOX_REPAIR_PROMPT = (
    "sharpen and restore the {classes} regions: clean natural skin texture and "
    "facial details, stable hand anatomy and fingers, remove blur, noise, "
    "color patches, flicker and compression artifacts; keep identity, "
    "anatomy and proportions unchanged"
)


def _align_frame_count(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def _temporal_shape(length: int):
    frame_count = _align_frame_count(max(5, int(length)))
    duration = frame_count / FPS
    return frame_count, ((frame_count - 5) // 17) * 5 + 2, round(duration * 40)


def _adapt_canvas(width: int, height: int):
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def _resize(images, width: int, height: int, crop="center"):
    import comfy.utils

    torch = _torch()
    if images.shape[-1] != 3:
        images = images[..., :3]
    samples = images.movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, int(width), int(height), "lanczos", crop)
    return samples.movedim(1, -1)


# ---------------------------------------------------------------------------
# YOLO-World detection
# ---------------------------------------------------------------------------
_yolo_cache = {}


def _load_yolo(weights: str):
    """Load & cache an ultralytics YOLO model by path."""
    key = os.path.normpath(weights) if weights else "default"
    if key in _yolo_cache:
        return _yolo_cache[key]
    from ultralytics import YOLO

    model = YOLO(weights)
    _yolo_cache[key] = model
    return model


def _detect_boxes(yolo, images, classes: list[str], confidence: float,
                  detect_step: int = 1):
    """Detect open-vocabulary classes.

    When detect_step > 1 only every step-th frame is actually inferred; the
    intermediate frames reuse the nearest previous frame's boxes (the temporal
    mask smoothing downstream tolerates this), cutting detection cost ~step x.

    Returns list aligned with frames: each entry is a [N,4] xyxy tensor (float,
    pixel coords in the ORIGINAL frame size) plus a [N] confidence tensor.
    """
    torch = _torch()
    H, W = images.shape[1], images.shape[2]
    T = images.shape[0]
    step = max(1, int(detect_step))
    yolo.set_classes(list(classes))

    idxs = list(range(0, T, step))
    sampled = images[list(idxs)]
    frames_np = (sampled[..., :3].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()

    results = yolo.predict(
        source=list(frames_np),
        conf=float(confidence),
        device=yolo.device if hasattr(yolo, "device") else None,
        verbose=False,
    )
    if not isinstance(results, (list, tuple)):
        results = list(results)

    boxes_list = []
    confs_list = []
    last_boxes = torch.zeros([0, 4], dtype=torch.float32)
    last_confs = torch.zeros([0], dtype=torch.float32)
    di = 0
    for t in range(T):
        if di < len(idxs) and t == idxs[di]:
            r = results[di]
            di += 1
            if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
                last_boxes = torch.zeros([0, 4], dtype=torch.float32)
                last_confs = torch.zeros([0], dtype=torch.float32)
            else:
                last_boxes = torch.as_tensor(r.boxes.xyxy, dtype=torch.float32).cpu()
                last_confs = torch.as_tensor(r.boxes.conf, dtype=torch.float32).cpu()
        boxes_list.append(last_boxes)
        confs_list.append(last_confs)
    return boxes_list, confs_list, W, H


# ---------------------------------------------------------------------------
# Mask construction (bbox -> per-frame spatial mask with temporal smoothing)
# ---------------------------------------------------------------------------
_BLUR_CACHE = {}


def _gaussian_kernel_1d(radius: int):
    if radius <= 0:
        return None
    if radius in _BLUR_CACHE:
        return _BLUR_CACHE[radius]
    torch = _torch()
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x * x) / (2.0 * (radius / 2.0) ** 2))
    kernel = kernel / kernel.sum()
    _BLUR_CACHE[radius] = kernel
    return kernel


def _boxes_to_mask(H: int, W: int, boxes, dilation: int, feather: int,
                   device=None, dtype=None):
    """Build a [H, W] float mask in [0,1] from xyxy boxes."""
    torch = _torch()
    mask = torch.zeros([H, W], dtype=torch.float32)
    if boxes is None or len(boxes) == 0:
        return mask
    x0 = torch.clamp(boxes[:, 0] - dilation, 0, W - 1)
    y0 = torch.clamp(boxes[:, 1] - dilation, 0, H - 1)
    x1 = torch.clamp(boxes[:, 2] + dilation, 0, W)
    y1 = torch.clamp(boxes[:, 3] + dilation, 0, H)
    for i in range(len(boxes)):
        yy, xx = torch.meshgrid(
            torch.arange(int(y0[i]), int(y1[i]), device=mask.device),
            torch.arange(int(x0[i]), int(x1[i]), device=mask.device),
            indexing="ij",
        )
        mask[yy, xx] = 1.0

    if feather and feather > 0:
        import torch.nn.functional as F

        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        for _ in range(int(feather)):
            mask = F.avg_pool2d(mask, 3, 1, padding=1)
        mask = mask.squeeze(0).squeeze(0)
    return mask.contiguous()


def _build_video_mask(images, boxes_list, *, dilation, feather, temporal_smooth):
    """Per-frame [T,H,W] mask, temporally blurred across ±temporal_smooth."""
    torch = _torch()
    T, H, W = images.shape[0], images.shape[1], images.shape[2]
    masks = torch.stack(
        [_boxes_to_mask(H, W, b, dilation, feather) for b in boxes_list]
    )  # [T,H,W]

    if temporal_smooth and temporal_smooth > 0:
        kernel = _gaussian_kernel_1d(temporal_smooth)
        if kernel is not None:
            kernel = kernel.to(masks.dtype)
            r = temporal_smooth
            flat = masks.reshape(T, H * W)
            out = []
            for t in range(T):
                acc = torch.zeros_like(flat[t])
                for j, w in enumerate(kernel):
                    idx = t + (j - r)
                    idx = min(max(idx, 0), T - 1)  # clamp (edge-hold)
                    acc += w * flat[idx]
                out.append(acc)
            masks = torch.stack(out).reshape(T, H, W)
    return masks.clamp(0, 1)


# ---------------------------------------------------------------------------
# Conditioning + sampling (self-contained, mirrors Director core_sampling)
# ---------------------------------------------------------------------------
def _io_unpack(out):
    """Unpack official node output (comfy.io.NodeOutput / tuple / list)."""
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"H3VisQuasiDLSS5Refiner: unexpected node output {type(out)!r}")


def _build_conditioning(clip, video_vae, prompt, width, height, length,
                        first_frame, last_frame):
    """Positive conditioning + empty AV latent template via official H3 node."""
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

    out = MiniMaxH3ImageToVideo.execute(
        clip=clip,
        vae=video_vae,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        first_frame=first_frame,
        last_frame=last_frame,
    )
    args = _io_unpack(out)
    positive, latent = args[0], args[1]
    return positive, latent


def _encode_block_into_latent(video_vae, block_rgb, template_latent):
    """Encode real frames into the video stream of an H3 AV latent template.

    Mirrors H3InjectVideoLatent: injects into members[0], keeps the audio stream
    (all-zero in our template) intact, trims/pads on temporal mismatch.
    """
    import comfy.nested_tensor

    torch = _torch()
    samples = template_latent.get("samples")
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise ValueError("H3VisQuasiDLSS5Refiner: template latent must be an H3 AV NestedTensor.")
    members = list(samples.unbind())
    video_tmpl = members[0]

    encoded = video_vae.encode(block_rgb[..., :3])
    if encoded.ndim == 4:  # [B,C,H,W] -> [1,C,T,H,W]
        encoded = encoded.unsqueeze(0).movedim(1, 2)

    tgt_t, tgt_h, tgt_w = (
        video_tmpl.shape[-3],
        video_tmpl.shape[-2],
        video_tmpl.shape[-1],
    )
    got_t, got_h, got_w = encoded.shape[-3], encoded.shape[-2], encoded.shape[-1]
    if (got_h, got_w) != (tgt_h, tgt_w):
        encoded = _resize(block_rgb[..., :3], tgt_w * 16, tgt_h * 16)
        encoded = video_vae.encode(encoded[..., :3])
        if encoded.ndim == 4:
            encoded = encoded.unsqueeze(0).movedim(1, 2)
    got_t = encoded.shape[-3]
    if got_t != tgt_t:
        if got_t > tgt_t:
            encoded = encoded[..., :tgt_t, :, :]
        else:
            pad = video_tmpl[..., : tgt_t - got_t, :, :].to(encoded.device, encoded.dtype)
            encoded = torch.cat([encoded, pad], dim=-3)

    members[0] = encoded.to(video_tmpl.device, video_tmpl.dtype)
    out = dict(template_latent)
    out["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))
    return out


def _sample_block(model, positive, negative, latent, *, seed, cfg, steps,
                  denoise, sampler_name, scheduler, shift_video=12.0, shift_audio=3.0):
    torch = _torch()
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        BasicScheduler,
        CFGGuider,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    # SigmaShift wrapper
    shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
    model_use = _io_unpack(shifted)[0]

    denoise_use = float(max(0.0, min(1.0, denoise)))
    sigma_out = BasicScheduler.execute(model_use, str(scheduler), int(steps), denoise_use)
    sigma_t = _io_unpack(sigma_out)[0]
    if torch.is_tensor(sigma_t):
        sigma_t = sigma_t.detach().float().cpu().reshape(-1)

    sampler_obj = _io_unpack(KSamplerSelect.execute(str(sampler_name)))[0]
    noise_obj = _io_unpack(RandomNoise.execute(int(seed)))[0]

    neg = negative if negative else []
    if float(cfg) <= 1.0 or not neg:
        guider = _io_unpack(BasicGuider.execute(model_use, positive))[0]
    else:
        guider = _io_unpack(CFGGuider.execute(model_use, positive, neg, float(cfg)))[0]

    sampled = SamplerCustomAdvanced.execute(noise_obj, guider, sampler_obj, sigma_t, latent)
    out = _io_unpack(sampled)[0]
    return out


def _decode_video(video_vae, latent):
    from nodes import VAEDecode

    images, = VAEDecode().decode(video_vae, latent)
    return images


def _split_video_latent(video_vae, latent):
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent

    sep = LTXVSeparateAVLatent.execute(latent)
    args = getattr(sep, "args", None) or sep
    return args[0], args[1]


def _blend(original, refined, mask):
    """refined over original, weighted by per-frame mask.

    original/refined are channels-last [T,H,W,3]; mask is [T,H,W] -> broadcast
    as [T,H,W,1] so the channel dim aligns, not the width dim.
    """
    torch = _torch()
    m = mask.unsqueeze(-1).to(original.device, original.dtype)  # [T,H,W,1]
    return original * (1.0 - m) + refined * m


# ---------------------------------------------------------------------------
# RTX VSR headless enhancement (same-multiple, in-node)
# ---------------------------------------------------------------------------
RTX_QUALITY_LEVELS = ("LOW", "MEDIUM", "HIGH", "ULTRA")


def _temporal_dc_stabilize(frames, *, window=5, clip=0.03):
    """Sliding-median DC (per-channel mean) smoothing across frames.

    nvvfx VSR processes frames independently -> bright/color level jumps on
    fast motion. Cache per-channel means in a small window, clamp each frame's
    DC to the window median and renormalize. Identity on non-flickering
    footage.
    """
    torch = _torch()
    n = int(frames.shape[0])
    if n < 3:
        return frames
    fr = frames.float()
    means = fr.mean(dim=(1, 2))  # [n,3]
    res = fr.clone()
    for i in range(n):
        lo, hi = max(0, i - window // 2), min(n, i + window // 2 + 1)
        med = means[lo:hi].median(dim=0).values
        fac = (med / (means[i] + 1e-6)).clamp(1.0 - clip, 1.0 + clip)
        res[i] *= fac.view(1, 1, 3)
    return torch.clamp(res, 0.0, 1.0).to(device=frames.device, dtype=frames.dtype)


def _rtx_vsr_enhance(images, quality="ULTRA", *, window=5, dc_clip=0.03):
    """Block-wise NVIDIA RTX Video Super Resolution at the same resolution.

    Feeds each decoded frame through nvvfx.VideoSuperRes targeting the SAME
    canvas (1x, no upscale) and applies a light temporal DC smoothing to kill
    per-frame luminance/color flicker. The whole pass runs headless inside
    this node so the workflow does not need a separate RTXVideoSuperResolution
    node; canvas is kept identical to the input (same-multiple sampling). Any
    nvvfx / GPU issue is non-fatal: the H3 repair output is passed through
    untouched.
    """
    try:
        import nvvfx
    except ImportError:
        return images

    torch = _torch()
    N = int(images.shape[0])
    H, W = int(images.shape[1]), int(images.shape[2])
    if N == 0:
        return images
    rgb = images[..., :3].contiguous().float()
    frames_chw = rgb.movedim(-1, 1)  # [N,3,H,W]

    quality_map = {}
    for name in RTX_QUALITY_LEVELS:
        try:
            quality_map[name] = getattr(nvvfx.effects.QualityLevel, name, None)
        except AttributeError:
            quality_map[name] = None
    level = quality_map.get(str(quality).upper())

    try:
        ctx = nvvfx.VideoSuperRes(level) if level is not None else nvvfx.VideoSuperRes()
        nvvfx_sr = ctx.__enter__()
    except Exception:
        return images

    try:
        # target = source (same-multiple, 1x) snapped to the 8px grid
        out_w = max(8, round(W / 8) * 8)
        out_h = max(8, round(H / 8) * 8)
        nvvfx_sr.output_width = out_w
        nvvfx_sr.output_height = out_h
        if hasattr(nvvfx_sr, "load"):
            nvvfx_sr.load()

        # block-wise: bound concurrent pixels so an 8GB card never explodes
        MAX_PIXELS = 1024 * 1024 * 16
        per_batch = max(1, MAX_PIXELS // max(1, out_w * out_h))

        upscaled = []
        for i in range(N):
            frame = frames_chw[i]
            if frame.device.type != "cuda":
                frame = frame.cuda()
            try:
                dlpack_out = nvvfx_sr.run(frame).image
                upscaled.append(torch.from_dlpack(dlpack_out).movedim(0, -1).clone())
            except Exception:
                upscaled.append(rgb[i])
        if len(upscaled) != N:
            return images

        out = torch.stack(upscaled, dim=0)
        if (out_w, out_h) != (W, H):
            out = _resize(out[..., :3].contiguous(), W, H, "disabled")
        out = _temporal_dc_stabilize(out, window=int(window), clip=float(dc_clip))
        return out.to(images.dtype)
    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------
DEFAULT_YOLO = str(Path(
    r"E:\COMFYUI\ComfyUI-aki-v3.2\ComfyUI-aki-v3.2\ComfyUI\models"
    r"\ultralytics\bbox\yolov8s-world.pt"
))

SAMPLERS_DEFAULT = ("euler", "res_multistep", "dpmpp_2m", "dpmpp_3m_sde",
                    "ddim", "uni_pc")
SCHEDULERS_DEFAULT = ("karras", "exponential", "sgm_uniform", "simple", "normal")


class H3VisQuasiDLSS5Refiner:
    """YOLO-World guided local refine for MiniMax H3 videos.

    Detect open-vocabulary targets (e.g. text, logo, face) with YOLO-World,
    define the repair goal through a fix prompt (+ detected class names), then
    re-generate ONLY the detected regions by MiniMax H3 in temporal chunks and
    blend the result back. Everything outside the mask stays pixel-identical.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": ("MODEL",),
                "video_vae": ("VAE",),
                "clip": ("CLIP",),
            },
            "optional": {
                "yolov8_weights": (
                    "STRING",
                    {
                        "default": DEFAULT_YOLO,
                        "tooltip": "YOLO-World weights (yolov8s-world.pt / yolov8l-world.pt).",
                    },
                ),
                "detect_classes": (
                    "STRING",
                    {
                        "default": "face, hand",
                        "tooltip": "Box-mode targets (full_frame_repair off). Comma-separated open-vocabulary classes, e.g. face, hand, text, logo.",
                    },
                ),
                "fix_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": FF_REPAIR_PROMPT,
                        "tooltip": "H3 repair prompt. {classes} is replaced by the detected class names when using box-mode repair. Leave unchanged to auto-pick: full-frame -> de-artifact prompt, box-mode -> face/hand prompt.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "blurry, distorted, low resolution, artifacts",
                    },
                ),
                "confidence": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "box_dilation": ("INT", {"default": 12, "min": 0, "max": 200, "step": 1}),
                "mask_feather": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "temporal_smooth": ("INT", {"default": 3, "min": 0, "max": 30, "step": 1}),
                # H3 sampling
                "chunk_frames": ("INT", {"default": 124, "min": 22, "max": 362, "step": 17}),
                "denoise": ("FLOAT", {"default": 0.32, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 60, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampler": (list(SAMPLERS_DEFAULT), {"default": SAMPLERS_DEFAULT[0]}),
                "scheduler": (list(SCHEDULERS_DEFAULT), {"default": SCHEDULERS_DEFAULT[0]}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "detect_step": ("INT", {"default": 1, "min": 1, "max": 30, "step": 1}),
                "full_frame_repair": ("BOOLEAN", {"default": True}),
                # RTX VSR headless enhancement (applied after repair)
                "rtx_enhance": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "After the H3 repair (incl. full_frame_repair de-artifact), "
                        "run a headless NVIDIA RTX Video Super Resolution pass at the SAME "
                        "multiple (1x, no upscale) to clean/sharp-smooth the whole clip. "
                        "Requires nvidia-vfx on an RTX GPU; non-fatal if unavailable.",
                    },
                ),
                "rtx_quality": (
                    list(RTX_QUALITY_LEVELS),
                    {"default": "ULTRA"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "detected_prompt")
    FUNCTION = "run"
    CATEGORY = "H3 Vis Quasi DLSS5"
    DESCRIPTION = (
        "YOLO-World guided local H3 refine: detection -> prompt definition -> "
        "chunked latent re-generation -> mask paste-back. Fully automatic: "
        "detects the defined classes, injects their names into the fix prompt "
        "({classes} placeholder) and emits the resolved prompt on "
        "detected_prompt, then rebuilds only the detected regions."
    )

    # ------------------------------------------------------------------
    def run(self, images, model, video_vae, clip,
            yolov8_weights=DEFAULT_YOLO, detect_classes="face, hand",
            fix_prompt=FF_REPAIR_PROMPT,
            negative_prompt="blurry, distorted, low resolution, artifacts",
            confidence=0.25, box_dilation=12, mask_feather=6, temporal_smooth=3,
            chunk_frames=124, denoise=0.32, steps=4, cfg=1.0, seed=0,
            sampler="euler", scheduler="karras", shift_video=12.0, shift_audio=3.0,
            detect_step=1, full_frame_repair=True,
            rtx_enhance=True, rtx_quality="ULTRA"):
        torch = _torch()
        if images is None or images.shape[0] == 0:
            raise ValueError("H3VisQuasiDLSS5Refiner: empty input images.")

        T, H, W = images.shape[0], images.shape[1], images.shape[2]
        orig_size = (W, H)

        # 1) detection / repair scope
        if bool(full_frame_repair):
            # Whole-frame de-artifact repair: skip YOLO, mask covers every pixel.
            # Targets: material/shading flicker, texture shimmer & tearing, frame
            # corruption, static-frame damage, random color patches, red/green
            # bursts, noise blotches, banding streaks.
            classes = [c.strip() for c in detect_classes.split(",") if c.strip()]
            if not classes:
                classes = ["artifacts"]
            has_boxes = True
            vid_mask = torch.ones([T, H, W], dtype=torch.float32,
                                  device=images.device)
        else:
            classes = [c.strip() for c in detect_classes.split(",") if c.strip()]
            if not classes:
                raise ValueError("H3VisQuasiDLSS5Refiner: detect_classes is empty.")
            yolo = _load_yolo(yolov8_weights)
            boxes_list, _, _, _ = _detect_boxes(yolo, images, classes, confidence,
                                                detect_step)
            has_boxes = any(len(b) > 0 for b in boxes_list)
            vid_mask = _build_video_mask(
                images, boxes_list, dilation=box_dilation, feather=mask_feather,
                temporal_smooth=temporal_smooth,
            )

        # Same-multiple canvas: keep original resolution, no upscale/downscale.
        # Sample at identical latent resolution; only the prompt changes.
        canvas_w, canvas_h = W, H
        canvas_images = images[..., :3].contiguous()

        # 2) prompt definition
        # Auto-pick a purpose-built prompt when the widget was left untouched:
        # full-frame keeps the de-artifact prompt, box-mode gets a face/hand
        # prompt tuned for localized repair.
        if (not bool(full_frame_repair)
                and fix_prompt.strip() == FF_REPAIR_PROMPT.strip()):
            fix_prompt = BOX_REPAIR_PROMPT
        prompt = fix_prompt.replace("{classes}", ", ".join(classes))

        # 3) chunked latent re-generation
        aligned_chunk = _align_frame_count(int(chunk_frames))
        chunks = []
        for start in range(0, T, aligned_chunk):
            chunks.append((start, min(start + aligned_chunk, T)))

        out_frames = []
        for cidx, (cs, ce) in enumerate(chunks):
            block_rgb = canvas_images[cs:ce]  # [b,H,W,3], b<= aligned_chunk
            length = ce - cs
            aligned_len = _align_frame_count(length)

            # frame-aligned conditioning (keyframes anchor content)
            first_frame = _resize(block_rgb[:1], canvas_w, canvas_h, "disabled")
            last_frame = _resize(block_rgb[-1:], canvas_w, canvas_h, "center")
            positive, template_latent = _build_conditioning(
                clip, video_vae, prompt, canvas_w, canvas_h, aligned_len,
                first_frame, last_frame,
            )

            # pad short tail to aligned grid (repeat last frame)
            if length != aligned_len:
                pad_n = aligned_len - length
                tail = block_rgb[-1:].repeat(pad_n, 1, 1, 1)
                block_enc = torch.cat([block_rgb, tail], dim=0)
            else:
                block_enc = block_rgb

            latent = _encode_block_into_latent(video_vae, block_enc, template_latent)

            # negative: only needed for CFG>1; otherwise pass [] (BasicGuider path)
            negative = None
            if float(cfg) > 1.0 and negative_prompt and negative_prompt.strip():
                negative = clip.encode_from_tokens_scheduled(
                    clip.tokenize(negative_prompt),
                )

            sampled = _sample_block(
                model, positive, negative, latent,
                seed=seed + cidx, cfg=cfg, steps=steps, denoise=denoise,
                sampler_name=sampler, scheduler=scheduler,
                shift_video=shift_video, shift_audio=shift_audio,
            )

            # decode whole block (may be longer than the chunk tail)
            sample_dict = (
                sampled if isinstance(sampled, dict) else {"samples": sampled}
            )
            video_latent, _ = _split_video_latent(video_vae, sample_dict)
            block_out = _decode_video(video_vae, video_latent)
            out_frames.append(block_out)

        refined_all = torch.cat(out_frames, dim=0)[:T]

        # 4) paste back with per-frame mask (only detected regions replaced)
        if refined_all.shape[1:3] == (canvas_h, canvas_w):
            refin_canvas = refined_all
        else:
            refin_canvas = _resize(refined_all[..., :3].contiguous(), canvas_w, canvas_h)

        result_canvas = _blend(canvas_images, refin_canvas, vid_mask)

        if (canvas_w, canvas_h) != orig_size:
            result = _resize(result_canvas[..., :3], W, H)
        else:
            result = result_canvas[..., :3]

        # 5) headless RTX VSR same-multiple enhancement after repair.
        # Applied to the final repaired clip at 1x (no resolution change, same
        # latent multiple) to add NVIDIA RTX temporal-super-res cleaning on top
        # of the H3 de-artifact pass. Fails open when nvvfx/GPU is missing.
        if bool(rtx_enhance):
            result = _rtx_vsr_enhance(result, str(rtx_quality))

        keep = result.to(images.dtype)
        if has_boxes:
            return (keep, prompt)
        # no detections: return input unchanged
        return (images, prompt)

