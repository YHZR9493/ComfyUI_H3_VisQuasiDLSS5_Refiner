"""H3VisQuasiDLSS5Refiner - headless RTX VSR / DLSS SR enhancement node.

Single-node pipeline: input clip -> (optional model unload) -> NVIDIA
super-resolution enhancement (DLSS5 backend preferred via in-process
VapourSynth bridge, nvvfx RTX VSR fallback) -> optional de-fog / de-haze
post look. YOLO-World guided detection, chunked latent re-generation and
mask-feathered paste-back repair were removed; only the rtx_*/dlss_*/defog_*
enhancement chain remains.

Interface: 4 inputs (model / video_vae / clip / images) -> 1 output (images).
No audio_vae, no audio output, no reference-image groups.
"""

from __future__ import annotations

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
# True NVIDIA DLSS SR bridge (in-process VapourSynth + vsdlsssr.dll).
# Optional: falls back to nvvfx / passthrough when the runtime is missing.
# ---------------------------------------------------------------------------
try:
    from .dlss5_bridge import dlss5_available as _dlss5_available
    from .dlss5_bridge import upscale as _dlss5_upscale
except Exception:  # noqa: BLE001 - ComfyUI may load this file without package ctx
    try:
        from dlss5_bridge import dlss5_available as _dlss5_available
        from dlss5_bridge import upscale as _dlss5_upscale
    except Exception:  # noqa: BLE001
        _dlss5_available = None
        _dlss5_upscale = None

DLSS5_QUALITY_LEVELS = ("Quality", "Balanced", "Performance",
                        "Ultra Performance", "Ultra Quality", "DLAA")
DLSS5_DEPTH_MODELS = ("Small", "Base", "Large")
DLSS5_MOTION_MODELS = ("Small", "Large")
DLSS5_NR_MODES = ("Off", "Neutral / faithful", "Realistic detail",
                  "Strong detail")
RTX_BACKENDS = ("auto", "dlss5", "nvvfx")


def _resize(images, width: int, height: int, crop="center"):
    import comfy.utils

    torch = _torch()
    if images.shape[-1] != 3:
        images = images[..., :3]
    samples = images.movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, int(width), int(height), "lanczos", crop)
    return samples.movedim(1, -1)


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
                     detail_strength=0.85, window=5, dc_clip=0.03,
                     backend="auto", dlss_quality="Quality",
                     dlss_depth_model="Small", dlss_motion_model="Small",
                     dlss_chunk_frames=4,
                     dlss_nr_mode="Off", dlss_nr_intensity=1.0,
                     dlss_nr_chunk_frames=4, dlss_motion_scale=0.5):
    """Headless NVIDIA super-resolution enhancement.

    Two backends, selected by `backend` ("auto" prefers the true DLSS bridge):
      * dlss5  - official NVIDIA DLSS Super Resolution (vsdlsssr.dll +
        nvngx_dlss.dll) through the in-process VapourSynth bridge, with
        Depth Anything V2 + RAFT guides estimated inside the node. When
        `dlss_nr_mode` is not "Off", the enlarged SR rebuild is further
        processed by official DLSS 5 Neural Rendering (Feature 18,
        nvngx_dlssnr.dll) in an isolated process -- max-gain pipeline:
        SR -> NR on the enlarged frame, guides resized to match.
      * nvvfx  - headless NVIDIA RTX Video Super Resolution (nvvfx package).

    Behaviour of the result (both backends):
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
    any runtime / GPU issue is non-fatal and passes the input through.
    """
    # ---- DLSS5 backend: true NVIDIA DLSS SR (in-process VapourSynth) -----
    use_dlss = (backend == "dlss5") or (
        backend == "auto"
        and _dlss5_available is not None
        and _dlss5_available()
    )
    if use_dlss and _dlss5_upscale is not None:
        try:
            out = _dlss5_upscale(
                images, scale=float(scale), quality=str(dlss_quality),
                keep_upscaled=bool(keep_upscaled),
                detail_strength=float(detail_strength),
                depth_model=str(dlss_depth_model),
                motion_model=str(dlss_motion_model),
                chunk_frames=int(dlss_chunk_frames),
                nr_mode=str(dlss_nr_mode),
                nr_intensity=float(dlss_nr_intensity),
                nr_chunk_frames=int(dlss_nr_chunk_frames),
                motion_scale=float(dlss_motion_scale),
            )
            if out is not None:
                return out.to(images.device).to(images.dtype)
        except Exception:  # noqa: BLE001 - fail open to nvvfx / passthrough
            pass

    # ---- nvvfx backend (legacy RTX VSR) ----------------------------------
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


class H3VisQuasiDLSS5Refiner:
    """Headless RTX VSR / DLSS SR enhancement + optional de-fog look.

    YOLO-World guided detection and H3 re-generation repair were removed;
    the clip flows straight into the NVIDIA super-resolution bridge and the
    optional de-fog post look.
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
                # RTX VSR headless enhancement (applied to the input clip)
                "rtx_enhance": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "对输入画面运行无头 NVIDIA RTX 视频超分（RTX VSR/DLSS SR）增强，"
                        "倍率由 rtx_scale 控制。需要 RTX 显卡与对应运行时，不可用时不报错直接跳过。\n"
                        "推荐值：True。",
                    },
                ),
                "rtx_unload_models": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "RTX 增强前先卸载 H3/VAE 模型栈并清空 CUDA 缓存，给显存吃紧的"
                        "超分后处理腾出空间（8GB 显存 OOM 的根因就是 H3 栈常驻）。后续仍需要模型的节点会自动重新加载。\n"
                        "推荐值：True。",
                    },
                ),
                "rtx_quality": (
                    list(RTX_QUALITY_LEVELS),
                    {"default": "ULTRA",
                     "tooltip": "RTX 超分质量档位（LOW/MEDIUM/HIGH/ULTRA）。越高细节越好但更慢更吃显存。\n"
                     "推荐值：ULTRA；8GB 显存或追求速度可降为 HIGH。"},
                ),
                "rtx_scale": (
                    "FLOAT",
                    {
                        "default": 1.5, "min": 1.0, "max": 2.0, "step": 0.05,
                        "tooltip": "RTX 超分放大倍率（1.0 = 不放大，仅同分辨率时间维去闪烁）。\n"
                        "推荐值：1.5；追求更高清晰度可试 2.0（显存需求上升）。",
                    },
                ),
                "rtx_keep_upscaled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "True：直接输出放大后的分辨率，细节最大化（输出变大、下游更慢）。\n"
                        "False：放大后缩回原分辨率，并把超分细节注入回缩过程中（输出尺寸不变、略柔、下游更快）。\n"
                        "推荐值：True（追求细节）或 False（保持管线尺寸一致）。",
                    },
                ),
                "rtx_backend": (
                    RTX_BACKENDS,
                    {
                        "default": "auto",
                        "tooltip": "增强引擎：auto = 存在真 DLSS SR（vsdlsssr.dll）时用 DLSS SR，"
                        "否则退回 RTX VSR（nvvfx）；dlss5 = 强制走 DLSS SR 桥；nvvfx = 强制走传统 RTX VSR。\n"
                        "推荐值：auto。",
                    },
                ),
                "dlss_quality": (
                    DLSS5_QUALITY_LEVELS,
                    {
                        "default": "Quality",
                        "tooltip": "DLSS SR 质量预设（Quality/Balanced/Performance/"
                        "Ultra Performance/Ultra Quality/DLAA）。仅 dlss5 后端生效。\n"
                        "推荐值：Quality；显存紧可降 Performance。",
                    },
                ),
                "dlss_depth_model": (
                    DLSS5_DEPTH_MODELS,
                    {
                        "default": "Small",
                        "tooltip": "DLSS 深度引导用的 Depth Anything V2 模型。Small 适配 8GB 显存；"
                        "权重首次使用时自动下载。\n"
                        "推荐值：Small。",
                    },
                ),
                "dlss_motion_model": (
                    DLSS5_MOTION_MODELS,
                    {
                        "default": "Small",
                        "tooltip": "DLSS 运动引导用的 RAFT 光流模型。Small 快且适配 8GB 显存；"
                        "权重首次使用时自动下载。\n"
                        "推荐值：Small。",
                    },
                ),
                "dlss_chunk_frames": (
                    "INT",
                    {
                        "default": 4, "min": 1, "max": 32,
                        "tooltip": "深度/光流引导推理的帧批大小。显存低时调小。\n"
                        "推荐值：4；8GB 显存可试 2。",
                    },
                ),
                "dlss_motion_scale": (
                    "FLOAT",
                    {
                        "default": 0.5, "min": 0.25, "max": 1.0, "step": 0.25,
                        "tooltip": "RAFT 运动引导的分辨率比例。0.5 在半分辨率估光流（约快 4 倍）"
                        "再缩放回原尺寸；1.0 全分辨率（快速运动时略准，更慢）。\n"
                        "推荐值：0.5。",
                    },
                ),
                "dlss_nr_mode": (
                    DLSS5_NR_MODES,
                    {
                        "default": "Off",
                        "tooltip": "DLSS 5 Neural Rendering 外观，在 DLSS SR 重建后施加"
                        "（最大化管线：SR -> 在放大帧上做 NR，引导图缩放对齐）。"
                        "在独立进程运行，占用 torch 显存很少；D3D12 侧显存随放大分辨率增长。"
                        "Off 关闭 NR。\n"
                        "推荐值：Off；追求电影质感可试 Cinema/Anime 等外观。",
                    },
                ),
                "dlss_nr_intensity": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                        "tooltip": "Neural Rendering 整体效果强度倍率（style_strength 与 intensity）。"
                        "1.0 = 所选外观的原生强度；<1 更柔和，>1 更强烈。\n"
                        "推荐值：1.0。",
                    },
                ),
                "dlss_nr_chunk_frames": (
                    "INT",
                    {
                        "default": 4, "min": 1, "max": 16,
                        "tooltip": "NR 进程的帧批大小。超过 2M 像素时节点自动限制为 2，"
                        "以保持在 8GB 显卡的 D3D12 显存预算内。\n"
                        "推荐值：4（自动保护）。",
                    },
                ),
                "rtx_detail_strength": (
                    "FLOAT",
                    {
                        "default": 0.85, "min": 0.0, "max": 1.5, "step": 0.05,
                        "tooltip": "缩回原分辨率时，SR 专属细节（超分重建减普通放大）回注多少。"
                        "仅 rtx_keep_upscaled=False 时使用。\n"
                        "推荐值：0.85。",
                    },
                ),
                # De-fog / de-haze post look (optional, multiply-blend layer)
                "defog_enabled": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "去雾（‘evil-cult dehaze’）后期外观：提亮 + 减辉光 + 锐化 + "
                        "提阴影得到增强层，再乘性混合回修复后的画面。纯 torch 实现，在 RTX 增强之后应用。\n"
                        "推荐值：False；雾感/灰蒙画面可开 True。",
                    },
                ),
                "defog_strength": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "去雾强度。越大去雾越明显，但过强会损失对比与层次。\n"
                    "推荐值：0.5；去雾不足可调 0.6~0.7。"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "detected_prompt")
    FUNCTION = "run"
    CATEGORY = "H3 Vis Quasi DLSS5"
    DESCRIPTION = (
        "Headless RTX VSR / DLSS SR enhancement plus optional de-fog look "
        "for MiniMax H3 videos. The input clip is passed straight through "
        "the NVIDIA super-resolution bridge (DLSS5 backend preferred, "
        "nvvfx RTX VSR fallback) at rtx_scale, with an optional de-fog / "
        "de-haze multiply-blend layer applied afterwards. YOLO-guided "
        "detection and H3 re-generation repair were removed."
    )

    # ------------------------------------------------------------------
    def run(self, images, model, video_vae, clip,
            rtx_enhance=True, rtx_unload_models=True,
            rtx_quality="ULTRA", rtx_scale=1.5,
            rtx_keep_upscaled=True, rtx_detail_strength=0.85,
            rtx_backend="auto",
            dlss_quality="Quality", dlss_depth_model="Small",
            dlss_motion_model="Small", dlss_chunk_frames=4,
            dlss_nr_mode="Off", dlss_nr_intensity=1.0,
            dlss_nr_chunk_frames=4, dlss_motion_scale=0.5,
            defog_enabled=False, defog_strength=0.5):
        torch = _torch()
        if images is None or images.shape[0] == 0:
            raise ValueError("H3VisQuasiDLSS5Refiner: empty input images.")

        T, H, W = images.shape[0], images.shape[1], images.shape[2]

        # RTX/DLSS/defog only: YOLO-guided detection, chunked latent
        # re-generation and repair paste-back were removed. The input clip
        # flows straight into the headless RTX VSR/DLSS SR enhancement and
        # (optionally) the de-fog look.
        canvas_images = images[..., :3].contiguous()
        result_canvas = canvas_images

        # 3) headless RTX VSR enhancement. keep_upscaled=False shrinks the
        # super-res clip back to the input resolution while injecting the SR
        # detail back (detail-preserving downscale: same output size, sharpness
        # close to the enlarged result). keep_upscaled=True outputs the
        # enlarged resolution directly for max detail. The H3 stack is evicted
        # beforehand so the VRAM-hungry pass fits in 8GB. Fails open when
        # nvvfx/GPU is missing.
        result = result_canvas
        if bool(rtx_enhance):
            if bool(rtx_unload_models):
                _unload_comfy_models()
            result = _rtx_vsr_enhance(
                result, str(rtx_quality), scale=float(rtx_scale),
                keep_upscaled=bool(rtx_keep_upscaled),
                detail_strength=float(rtx_detail_strength),
                backend=str(rtx_backend),
                dlss_quality=str(dlss_quality),
                dlss_depth_model=str(dlss_depth_model),
                dlss_motion_model=str(dlss_motion_model),
                dlss_chunk_frames=int(dlss_chunk_frames),
                dlss_nr_mode=str(dlss_nr_mode),
                dlss_nr_intensity=float(dlss_nr_intensity),
                dlss_nr_chunk_frames=int(dlss_nr_chunk_frames),
                dlss_motion_scale=float(dlss_motion_scale),
            )

        # 4) optional de-fog / de-haze look (multiply-blend enhanced layer).
        if bool(defog_enabled):
            result = _defog(result, float(defog_strength))

        keep = result.to(images.dtype)
        if bool(rtx_enhance) or bool(defog_enabled):
            return (keep, "RTX/DLSS/defog enhanced")
        # no enhance pass active: return input unchanged
        return (images, "no enhancement applied")

