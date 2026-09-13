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


def _resample_blocks(model, clip, video_vae, video, canvas_w, canvas_h, prompt,
                     negative_prompt, *, chunk_frames, denoise, steps, cfg, seed,
                     sampler, scheduler, shift_video, shift_audio):
    """Chunked latent re-generation of `video` (channels-last [T,H,W,3]) at a
    fixed canvas. Encodes real frames into the H3 AV latent template (img2img
    start point), samples each temporal chunk, decodes and returns the refined
    clip [T,H,W,3] trimmed back to T. Works for both whole frames and the
    enlarged crops used by the FaceRefine-style repair.
    """
    torch = _torch()
    T = video.shape[0]
    aligned_chunk = _align_frame_count(int(chunk_frames))
    chunks = []
    for start in range(0, T, aligned_chunk):
        chunks.append((start, min(start + aligned_chunk, T)))

    out_frames = []
    for cidx, (cs, ce) in enumerate(chunks):
        block_rgb = video[cs:ce]  # [b,H,W,3]
        length = ce - cs
        aligned_len = _align_frame_count(length)

        first_frame = _resize(block_rgb[:1], canvas_w, canvas_h, "disabled")
        last_frame = _resize(block_rgb[-1:], canvas_w, canvas_h, "center")
        positive, template_latent = _build_conditioning(
            clip, video_vae, prompt, canvas_w, canvas_h, aligned_len,
            first_frame, last_frame,
        )

        if length != aligned_len:
            pad_n = aligned_len - length
            tail = block_rgb[-1:].repeat(pad_n, 1, 1, 1)
            block_enc = torch.cat([block_rgb, tail], dim=0)
        else:
            block_enc = block_rgb

        latent = _encode_block_into_latent(video_vae, block_enc, template_latent)

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

        sample_dict = sampled if isinstance(sampled, dict) else {"samples": sampled}
        video_latent, _ = _split_video_latent(video_vae, sample_dict)
        out_frames.append(_decode_video(video_vae, video_latent))

    return torch.cat(out_frames, dim=0)[:T]


def _blend(original, refined, mask):
    """refined over original, weighted by per-frame mask.

    original/refined are channels-last [T,H,W,3]; mask is [T,H,W] -> broadcast
    as [T,H,W,1] so the channel dim aligns, not the width dim.
    """
    torch = _torch()
    m = mask.unsqueeze(-1).to(original.device, original.dtype)  # [T,H,W,1]
    return original * (1.0 - m) + refined * m


# ---------------------------------------------------------------------------
# FaceRefine-style enlarged-crop repair (face / hand)
# ---------------------------------------------------------------------------
def _colour_match_patch(src, ref, strength):
    """Align per-channel mean/std of `src` patch to `ref` region.

    Kill the seam contrast jump between the regenerated crop and the untouched
    surrounding footage (same idea as FaceRefine's colour_match).
    """
    torch = _torch()
    if src.shape[:2] != ref.shape[:2]:
        src = _resize(src.unsqueeze(0), int(ref.shape[1]), int(ref.shape[0]),
                      "center")[0]
    s = src.float()
    r = ref.float()
    s_mean = s.mean(dim=(0, 1))
    r_mean = r.mean(dim=(0, 1))
    s_std = s.std(dim=(0, 1), unbiased=False) + 1e-6
    r_std = r.std(dim=(0, 1), unbiased=False) + 1e-6
    matched = (s - s_mean) / s_std * r_std + r_mean
    out = s * (1.0 - strength) + matched * strength
    return out.clamp(0, 1).to(dtype=src.dtype)


def _build_crop_video(images, boxes_list, canvas: int):
    """Build one enlarged-crop video from per-frame detected boxes.

    Every detected face/hand region of a frame is lanczos-resized into a
    fixed-size slot of a shared square canvas (grid layout, stable across the
    whole clip). Empty excess slots are filled with a down-scaled copy of the
    frame so the H3 re-sample always sees natural footage everywhere (those
    filler slots are never pasted back). Returns the crop video
    [T,C,C,3] (C snapped to a 16-multiple) plus a per-frame slot map and the
    resulting canvas size.
    """
    torch = _torch()
    T, H, W = images.shape[0], images.shape[1], images.shape[2]
    n_max = max((len(b) for b in boxes_list), default=0)
    n_max = max(1, n_max)
    cols = min(4, n_max)
    rows = math.ceil(n_max / cols)
    # 16-aligned square layout: slot pitch snapped, so the crop canvas stays a
    # clean multiple for the H3 latent grid.
    slot = max(16, (int(canvas) // max(cols, rows)) // 16 * 16)
    canvas_final = max(16, (slot * max(cols, rows)) // 16 * 16)
    slot = canvas_final // max(cols, rows)  # re-derive after final snap

    crop_video = torch.zeros([T, canvas_final, canvas_final, 3],
                             dtype=images.dtype, device=images.device)
    slot_map = []
    for t in range(T):
        boxes_t = boxes_list[t]
        n_t = len(boxes_t)
        frame_slots = []
        for k in range(n_max):
            sx = (k % cols) * slot
            sy = (k // cols) * slot
            if k < n_t:
                b = boxes_t[k]
                x0, y0 = int(max(0, b[0])), int(max(0, b[1]))
                x1, y1 = int(min(W, b[2])), int(min(H, b[3]))
                if x1 > x0 and y1 > y0:
                    reg = _resize(images[t, y0:y1, x0:x1].unsqueeze(0),
                                  slot, slot, "center")
                    crop_video[t, sy:sy + slot, sx:sx + slot] = reg[0]
                    frame_slots.append({
                        "x": sx, "y": sy, "w": slot, "h": slot,
                        "bx0": x0, "by0": y0, "bx1": x1, "by1": y1,
                    })
                    continue
            # filler slot (no box on this frame / degenerate box): downscale
            # the whole frame so the crop video has natural content there.
            fill = _resize(images[t:t + 1].contiguous(), slot, slot, "center")
            crop_video[t, sy:sy + slot, sx:sx + slot] = fill[0]
        slot_map.append(frame_slots)
    return crop_video, slot_map, canvas_final


def _paste_crops_back(base_images, refined_crops, slot_map, *, mask,
                      colour_match=0.8, dilation=0):
    """Warp regenerated crop slots back onto their detected boxes.

    base_images: [T,H,W,3]; untouched regions stay pixel-identical.
    refined_crops: [T,C,C,3] regenerated enlarged crops.
    slot_map: per-frame list of {x,y,w,h,bx0,by0,bx1,by1} (original-film boxes).
    mask: [T,H,W] blend weight built from the same boxes (dilation+feather).
    dilation: overlap the pasted patch by this many px beyond every box edge,
      so the feathered blend mask only ever interpolates across the *buffer*
      band around each box, never reaching inside the regenerated core --
      without this the mask (<1 at the box edge) would eat away part of the
      repaired content and leave a translucent seam.
    """
    torch = _torch()
    H, W = base_images.shape[1], base_images.shape[2]
    dx = max(0, int(dilation))
    result = base_images.clone()
    for t, frame_slots in enumerate(slot_map):
        base_t = base_images[t]
        for sl in frame_slots:
            patch = refined_crops[t, sl["y"]:sl["y"] + sl["h"],
                                  sl["x"]:sl["x"] + sl["w"]]
            rx0 = max(0, int(sl["bx0"]) - dx)
            ry0 = max(0, int(sl["by0"]) - dx)
            rx1 = min(W, int(sl["bx1"]) + dx)
            ry1 = min(H, int(sl["by1"]) + dx)
            bw = rx1 - rx0
            bh = ry1 - ry0
            if bw <= 0 or bh <= 0:
                continue
            patch = _resize(patch.unsqueeze(0).contiguous(), bw, bh, "center")[0]
            if colour_match and colour_match > 0:
                ref = base_t[ry0:ry1, rx0:rx1]
                patch = _colour_match_patch(patch, ref, float(colour_match))
            result[t, ry0:ry1, rx0:rx1] = patch
    return _blend(base_images, result, mask)


# ---------------------------------------------------------------------------
# RTX VSR headless enhancement (in-node, optional upscale kept)
# ---------------------------------------------------------------------------
RTX_QUALITY_LEVELS = ("LOW", "MEDIUM", "HIGH", "ULTRA")


def _temporal_dc_stabilize(frames, *, window=5, clip=0.03, dc_chunk=16):
    """Sliding-median DC (per-channel mean) smoothing across frames.

    nvvfx VSR processes frames independently -> bright/color level jumps on
    fast motion. Cache per-channel means in a small window, clamp each frame's
    DC to the window median and renormalize. Identity on non-flickering
    footage.

    VRAM-safe: per-frame channel means are accumulated block by block (only a
    `dc_chunk`-sized float window is ever materialised at once) and the final
    correction is applied in-place per block on a single clone of the input --
    no full-batch float copy + clone + clamp triple allocation, which spiked
    an extra ~2x the clip on 8GB cards and was the OOM trigger.
    """
    torch = _torch()
    n = int(frames.shape[0])
    if n < 3:
        return frames
    dev = frames.device
    dt = frames.dtype
    block = max(1, int(dc_chunk))
    # 1) per-frame channel means, accumulated chunk by chunk (peak = one chunk)
    means = torch.empty(n, 3, dtype=torch.float32, device=dev)
    for c0 in range(0, n, block):
        b = frames[c0:c0 + block].float()
        means[c0:c0 + block].copy_(b.mean(dim=(1, 2)))
        del b
    # 2) DC correction coefficients (tiny [n,3] tensor)
    coeff = torch.empty_like(means)
    for i in range(n):
        lo, hi = max(0, i - window // 2), min(n, i + window // 2 + 1)
        med = means[lo:hi].median(dim=0).values
        coeff[i].copy_(
            (med / (means[i] + 1e-6)).clamp(1.0 - clip, 1.0 + clip)
        )
    # 3) apply in-place per block on one clone (peak = clone + one block)
    res = frames.clone()
    if dt in (torch.float16, torch.float32, torch.bfloat16):
        for c0 in range(0, n, block):
            blk = res[c0:c0 + block]
            blk.mul_(coeff[c0:c0 + block].view(-1, 1, 1, 3))
            torch.clamp_(blk, 0.0, 1.0)
    else:
        # integer inputs: convert per block (small temp) and write back
        for c0 in range(0, n, block):
            blk = res[c0:c0 + block]
            blk.copy_(
                torch.clamp(
                    blk.float() * coeff[c0:c0 + block].view(-1, 1, 1, 3),
                    0.0, 1.0,
                )
            )
    return res


def _rtx_vsr_enhance(images, quality="ULTRA", *, scale=1.5, keep_upscaled=False,
                     detail_strength=0.85, window=5, dc_clip=0.03):
    """Headless NVIDIA RTX Video Super Resolution.

    Feeds every frame through nvvfx.VideoSuperRes targeting `scale` x
    enlargement (snapped to the 8px nvvfx grid). Behaviour of the result:
      * keep_upscaled=True  -> the enlarged clip is output at the upscaled
        resolution (max detail, larger output, slower downstream).
      * keep_upscaled=False -> the enlarged super-res clip is shrunk back to
        the original resolution AND the super-res detail (SR reconstruction
        minus a plain upsample of the input) is injected back during the
        shrink. This keeps the output size identical to the input (no extra
        downstream cost) while retaining most of the sharpness the AI rebuild
        produced -- a content-aware detail-preserving downscale, far sharper
        than a plain lanczos shrink-back.
    A light temporal DC smoothing kills per-frame flicker. Fully in-node;
    any nvvfx / GPU issue is non-fatal and passes the input through.
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
        # target `scale` x upscale snapped to the 8px grid.
        _SCALE = float(max(1.0, min(2.0, scale)))
        out_w = max(8, round(W * _SCALE / 8) * 8)
        out_h = max(8, round(H * _SCALE / 8) * 8)
        nvvfx_sr.output_width = out_w
        nvvfx_sr.output_height = out_h
        if hasattr(nvvfx_sr, "load"):
            nvvfx_sr.load()

        # VRAM-safe streaming: preallocate one output clip and write each
        # super-res frame straight into it. The old code collected every
        # enlarged frame into a python list and then torch.stack()ed them,
        # materialising TWO full upscaled clips at once (for 640x1152x248 @1.5x
        # that was ~10 GB alone) -- the OOM trigger. Now the peak is a single
        # output clip + one enlarged frame, regardless of clip length.
        if bool(keep_upscaled):
            out = torch.empty(N, out_h, out_w, 3, dtype=torch.float32,
                              device=rgb.device)
        else:
            out = torch.empty(N, H, W, 3, dtype=torch.float32,
                              device=rgb.device)
        for i in range(N):
            frame = frames_chw[i].contiguous()
            if frame.device.type != "cuda":
                frame = frame.cuda()
            try:
                dlpack_out = nvvfx_sr.run(frame).image
                sr_i = torch.from_dlpack(dlpack_out).movedim(0, -1).clone()
            except Exception:
                # per-frame fallback: upscaled copy of the original frame
                sr_i = _resize(rgb[i].unsqueeze(0), out_w, out_h, "center")[0]
            del frame
            if bool(keep_upscaled):
                out[i].copy_(sr_i)
            else:
                # ---- detail-preserving shrink-back to the original frame ----
                # Plain shrink loses the high frequencies the SR rebuild added.
                # Instead: (1) lanczos downscale the SR frame (anti-aliased
                # base), (2) SR-specific detail = SR frame - plain upsample of
                # the original, (3) downscale that detail too and inject it
                # back, so the sharpening signal survives the shrink.
                base = _resize(sr_i.unsqueeze(0), W, H, "disabled")[0]
                up_ref = _resize(rgb[i].unsqueeze(0), out_w, out_h, "disabled")[0]
                out[i].copy_(
                    torch.clamp(
                        base + float(detail_strength) * (sr_i - up_ref),
                        0.0, 1.0,
                    )
                )
                del base, up_ref
            del sr_i

        out = _temporal_dc_stabilize(out, window=int(window), clip=float(dc_clip))
        return out.to(images.dtype)
    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# De-fog / de-haze post filter (optional, "evil-cult dehaze" style)
# ---------------------------------------------------------------------------
def _unload_comfy_models():
    """Best-effort: evict the loaded ComfyUI model stack (H3 + VAE) and clear
    the CUDA cache before the VRAM-hungry RTX VSR post pass.

    Root cause of the 8GB OOM: MiniMax H3 (12.99GB staged) + H3VideoVAE +
    Krea2T all stayed resident after sampling ("0 models unloaded" in the
    logs), so the post-pass allocation for a full-clip tensor (4.82GB) had
    zero free VRAM left. Freeing the model stack first gives the VSR pass
    several GB to work in. Pure optimization: any failure is ignored.
    """
    try:
        from comfy import model_management
    except Exception:
        return False
    try:
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        return True
    except Exception:
        return False


def _defog(images, strength: float = 0.5, block: int = 16):
    """Haze-removal look: build an enhanced layer from the frames, then
    hard-light blend it back over the original.

    Per-pipeline (all on the enhancement layer):
      1) brightness up
      2) glow / highlight down (soft shoulder compression)
      3) local sharpening up (unsharp mask, box-blur pyramid)
      4) shadow lift (dark areas weakened / de-clipped)
    final blend: hard_light(original, layer) instead of soft light, for a
    punchier local-contrast match with the 'DLSS5' refiner look; then lerp
    by `strength`.
    Pure torch, processed in T-blocks to stay VRAM-safe on 8GB cards.
    """
    if strength is None or float(strength) <= 0.0:
        return images
    torch = _torch()
    import torch.nn.functional as F

    strength = float(min(1.0, max(0.0, strength)))
    x = images[..., :3].float()
    T = x.shape[0]
    if T == 0:
        return images

    block = max(1, int(block))

    def _blur_soft(t):
        """Box-blur pyramid approximation of a gaussian (channels-last)."""
        b = t.movedim(-1, 1)                      # [T,3,H,W]
        for _ in range(3):
            b = F.avg_pool2d(b, 3, 1, padding=1)  # kernel3/stride1/pad1 -> size kept
        return b.movedim(1, -1)

    def _hard_light(base, blend):
        """Photoshop hard light blend on [0,1]: stronger contrast punch than
        soft light. Semantics invert the layer as the 'light source':
          blend <= 0.5 : multiply  -> out = 2 * base * blend   (darken)
          blend >  0.5 : screen    -> out = 1 - 2*(1-base)*(1-blend) (lighten)
        Unlike multiply alone it also lightens highlights, so it punches
        local contrast without uniformly crushing brightness."""
        m = (blend <= 0.5).float()
        dark = 2.0 * base * blend
        light = 1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
        return base * m + light * (1.0 - m)

    def _defog_block(xb):
        # 1) brightness up
        layer = xb * 1.10
        # 2) glow / highlight down
        glow = (layer - 0.72).clamp_min(0.0)
        layer = layer - glow * 0.45
        # 3) sharpening up (stronger: 0.6 -> 0.8)
        layer = layer + (layer - _blur_soft(layer)) * 0.8
        # 4) shadow lift (stronger: gain 0.5 -> 0.65)
        shadow = (0.35 - layer).clamp_min(0.0)
        layer = layer + shadow * 0.65
        layer = layer.clamp(0.0, 1.0)
        # 5) hard-light blend + strength lerp
        out = _hard_light(xb, layer)
        return xb + (out - xb) * strength

    outs = []
    for s in range(0, T, block):
        outs.append(_defog_block(x[s:s + block]))
    out = torch.cat(outs, dim=0).clamp(0.0, 1.0)
    return out.to(images.dtype)


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
                # FaceRefine-style enlarged-crop repair (box-mode, full_frame_repair off)
                "crop_repair": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Independent of full_frame_repair. FaceRefine-style "
                        "repair: detected face/hand regions are scaled up onto a "
                        "larger canvas, regenerated by H3 at high resolution, then "
                        "warped back with colour matching. Stacks on top of the "
                        "whole-frame pass when full_frame_repair is also on; with "
                        "full_frame_repair off it is the sole repair path. Turn "
                        "off to fall back to the low-res in-place re-generation.",
                    },
                ),
                "crop_canvas": ("INT", {"default": 512, "min": 256, "max": 1440, "step": 32}),
                "crop_denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "colour_match": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                # RTX VSR headless enhancement (applied after repair)
                "rtx_enhance": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "After the H3 repair (incl. full_frame_repair de-artifact), "
                        "run a headless NVIDIA RTX Video Super Resolution pass at the "
                        "rtx_scale multiple. Requires nvidia-vfx on an RTX GPU; non-fatal "
                        "if unavailable.",
                    },
                ),
                "rtx_unload_models": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload the H3 / VAE model stack and clear the CUDA "
                        "cache right before the RTX VSR pass, so the VRAM-hungry "
                        "post-enhancement has room to work on 8GB cards (the OOM root "
                        "cause was the H3 stack staying resident). Any node that still "
                        "needs those models downstream will just reload them.",
                    },
                ),
                "rtx_quality": (
                    list(RTX_QUALITY_LEVELS),
                    {"default": "ULTRA"},
                ),
                "rtx_scale": (
                    "FLOAT",
                    {
                        "default": 1.5, "min": 1.0, "max": 2.0, "step": 0.05,
                        "tooltip": "RTX VSR upscale factor (1.0 = no enlargement, "
                        "pure temporal clean at the same resolution).",
                    },
                ),
                "rtx_keep_upscaled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "False (default): after RTX upscale the clip is shrunk "
                        "back to the input resolution and the super-res detail is injected "
                        "back during the shrink -- output size identical to input (fast "
                        "downstream), sharpness close to the enlarged result. True: output "
                        "the upscaled resolution directly for max detail (larger output, "
                        "slower downstream).",
                    },
                ),
                "rtx_detail_strength": (
                    "FLOAT",
                    {
                        "default": 0.85, "min": 0.0, "max": 1.5, "step": 0.05,
                        "tooltip": "How much of the SR-specific detail (super-res rebuild "
                        "minus plain upsample) is injected back when shrinking to the "
                        "input resolution. Only used when rtx_keep_upscaled=False.",
                    },
                ),
                # De-fog / de-haze post look (optional, multiply-blend layer)
                "defog_enabled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "De-fog ('evil-cult dehaze') post look: brighten + glow-down "
                        "+ sharpen + shadow-lift an enhancement layer, then multiply-blend it "
                        "back over the repaired clip. Pure torch, applied after RTX enhance.",
                    },
                ),
                "defog_strength": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "detected_prompt")
    FUNCTION = "run"
    CATEGORY = "H3 Vis Quasi DLSS5"
    DESCRIPTION = (
        "YOLO-World guided H3 refine: detection -> prompt definition -> "
        "chunked latent re-generation -> mask paste-back. Fully automatic: "
        "detects the defined classes, injects their names into the fix prompt "
        "({classes} placeholder) and emits the resolved prompt on "
        "detected_prompt, then rebuilds only the detected regions. "
        "full_frame_repair (whole-frame de-artifact re-generation) and "
        "crop_repair (FaceRefine-style: face/hand regions scaled up onto a "
        "larger canvas, regenerated at high resolution, warped back with "
        "colour matching) are independent switches and stack when both are on."
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
            crop_repair=True, crop_canvas=512, crop_denoise=0.55, colour_match=0.8,
            rtx_enhance=True, rtx_unload_models=True,
            rtx_quality="ULTRA", rtx_scale=1.5,
            rtx_keep_upscaled=False, rtx_detail_strength=0.85,
            defog_enabled=False, defog_strength=0.5):
        torch = _torch()
        if images is None or images.shape[0] == 0:
            raise ValueError("H3VisQuasiDLSS5Refiner: empty input images.")

        T, H, W = images.shape[0], images.shape[1], images.shape[2]
        orig_size = (W, H)

        # 1) detection / repair scope
        # full_frame_repair and crop_repair are independent switches that
        # stack: whole-frame de-artifact re-generation first (if enabled),
        # then high-res crop repair of the detected face/hand boxes on top
        # (if enabled). YOLO only runs when box data is actually needed.
        classes = [c.strip() for c in detect_classes.split(",") if c.strip()]
        need_boxes = (not bool(full_frame_repair)) or bool(crop_repair)
        if need_boxes:
            if not classes:
                raise ValueError("H3VisQuasiDLSS5Refiner: detect_classes is empty.")
            yolo = _load_yolo(yolov8_weights)
            boxes_list, _, _, _ = _detect_boxes(yolo, images, classes, confidence,
                                                detect_step)
            has_boxes = any(len(b) > 0 for b in boxes_list)
            box_mask = _build_video_mask(
                images, boxes_list, dilation=box_dilation, feather=mask_feather,
                temporal_smooth=temporal_smooth,
            )
        else:
            # pure whole-frame mode: no YOLO needed
            if not classes:
                classes = ["artifacts"]
            boxes_list = []
            has_boxes = True
            box_mask = None

        did_repair = False
        if bool(full_frame_repair):
            frame_mask = torch.ones([T, H, W], dtype=torch.float32,
                                    device=images.device)
        else:
            frame_mask = None

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

        # 3) chunked latent re-generation. Two independent stages that stack:
        #    a) whole-frame de-artifact re-generation (full_frame_repair)
        #    b) FaceRefine-style enlarged-crop repair of detected face/hand
        #       boxes (crop_repair), applied on top of (a) when both are on.
        # 3a) whole-frame repair (full_frame_repair, independent of crop_repair)
        if bool(full_frame_repair):
            refined_all = _resample_blocks(
                model, clip, video_vae, canvas_images, canvas_w, canvas_h,
                prompt, negative_prompt,
                chunk_frames=chunk_frames, denoise=denoise,
                steps=steps, cfg=cfg, seed=seed, sampler=sampler,
                scheduler=scheduler, shift_video=shift_video, shift_audio=shift_audio,
            )
            if refined_all.shape[1:3] != (canvas_h, canvas_w):
                refined_all = _resize(refined_all[..., :3].contiguous(),
                                     canvas_w, canvas_h)
            result_canvas = _blend(canvas_images, refined_all, frame_mask)
            did_repair = True
        else:
            result_canvas = canvas_images

        # 3b) FaceRefine-style enlarged-crop repair (face / hand), works on the
        # result of 3a when full_frame_repair is also on.
        if bool(crop_repair) and has_boxes and box_mask is not None and any(
                len(b) > 0 for b in boxes_list):
            crop_canvas_snap = max(256, int(crop_canvas))
            crop_video, slot_map, crop_canvas_snap = _build_crop_video(
                result_canvas, boxes_list, crop_canvas_snap)
            refined_crops = _resample_blocks(
                model, clip, video_vae, crop_video,
                crop_canvas_snap, crop_canvas_snap, prompt, negative_prompt,
                chunk_frames=chunk_frames, denoise=float(crop_denoise),
                steps=steps, cfg=cfg, seed=seed, sampler=sampler,
                scheduler=scheduler, shift_video=shift_video, shift_audio=shift_audio,
            )
            # suppress flicker the re-sample can introduce between crop frames
            refined_crops = _temporal_dc_stabilize(refined_crops, window=5,
                                                   clip=0.02)
            result_canvas = _paste_crops_back(
                result_canvas, refined_crops, slot_map,
                mask=box_mask, colour_match=colour_match,
                dilation=int(box_dilation),
            )
            did_repair = True
        elif (not bool(crop_repair) and not bool(full_frame_repair)
                and has_boxes and box_mask is not None):
            # legacy in-place box repair (original box-mode behaviour, low denoise)
            refined_all = _resample_blocks(
                model, clip, video_vae, canvas_images, canvas_w, canvas_h,
                prompt, negative_prompt,
                chunk_frames=chunk_frames, denoise=denoise,
                steps=steps, cfg=cfg, seed=seed, sampler=sampler,
                scheduler=scheduler, shift_video=shift_video, shift_audio=shift_audio,
            )
            if refined_all.shape[1:3] != (canvas_h, canvas_w):
                refined_all = _resize(refined_all[..., :3].contiguous(),
                                     canvas_w, canvas_h)
            result_canvas = _blend(result_canvas, refined_all, box_mask)
            did_repair = True

        # 4) project back to original frame size
        if (canvas_w, canvas_h) != orig_size:
            result = _resize(result_canvas[..., :3], W, H)
        else:
            result = result_canvas[..., :3]

        # 5) headless RTX VSR enhancement after repair. Applied to the final
        # repaired clip at rtx_scale. keep_upscaled=False (default) shrinks the
        # super-res clip back to the input resolution while injecting the SR
        # detail back (detail-preserving downscale: same output size, sharpness
        # close to the enlarged result). keep_upscaled=True outputs the
        # enlarged resolution directly for max detail. The H3 stack is evicted
        # beforehand so the VRAM-hungry pass fits in 8GB. Fails open when
        # nvvfx/GPU is missing.
        if bool(rtx_enhance):
            if bool(rtx_unload_models):
                _unload_comfy_models()
            result = _rtx_vsr_enhance(
                result, str(rtx_quality), scale=float(rtx_scale),
                keep_upscaled=bool(rtx_keep_upscaled),
                detail_strength=float(rtx_detail_strength),
            )

        # 6) optional de-fog / de-haze look (multiply-blend enhanced layer).
        if bool(defog_enabled):
            result = _defog(result, float(defog_strength))

        keep = result.to(images.dtype)
        if did_repair or bool(rtx_enhance) or bool(defog_enabled):
            return (keep, prompt)
        # no repair/enhance pass active: return input unchanged
        return (images, prompt)

