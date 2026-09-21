"""dlss5_native.py - True NVIDIA DLSS 5 Neural Rendering (Feature 18) bridge.

References the sibling ``ComfyUI-DLSS5-Enhancer`` package and reuses its native
worker client (``dlss5.session.DlssSession``) plus temporal motion guidance
(``dlss5.motion.TemporalGuide``) and frame conversions (``dlss5.imaging``), so
``H3VisQuasiDLSS5Refiner`` gets exactly the same DLSS5 enhancement as the
DLSS5-Enhancer nodes: NGX feature 18 evaluated on the RGBA frame stream by the
native ``nvngx.dll --video`` worker, guided by DIS optical flow.

Contract:
    dlss5_native_enhance(images, ...) -> torch.Tensor [N, oh, ow, 3] float32 0..1
    Returns None when the sibling package / runtime is unavailable; callers
    fail open to their next backend (nvvfx / passthrough).
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy import of the sibling ComfyUI-DLSS5-Enhancer package (once).
# ---------------------------------------------------------------------------
_STATE: dict = {}


def _load_dlss5():
    """Import the DLSS5-Enhancer core modules; None when unavailable."""
    cached = _STATE.get("mods")
    if cached is not None:
        return cached
    if _STATE.get("failed"):
        return None
    enhancer_root = (
        Path(__file__).resolve().parent.parent.parent / "ComfyUI-DLSS5-Enhancer"
    )
    if not (enhancer_root / "dlss5").is_dir():
        _STATE["failed"] = True
        return None
    try:
        if str(enhancer_root) not in sys.path:
            sys.path.insert(0, str(enhancer_root))
        from dlss5 import (  # noqa: F401
            DLSS_MODEL_PRESETS,
            MOTION_MODES,
            NR_PRESETS,
            NR_STYLES,
            UPSCALING_LABELS,
        )
        from dlss5.imaging import fit_frame, resize_alpha, rgba_to_tensor, tensor_to_rgba
        from dlss5.motion import TemporalGuide
        from dlss5.paths import find_runtime
        from dlss5.session import DlssSession
        from dlss5.settings import DlssOptions, SessionConfig
    except Exception:  # noqa: BLE001 - optional bridge, never break the node
        _STATE["failed"] = True
        return None
    mods = {
        "DlssOptions": DlssOptions,
        "DlssSession": DlssSession,
        "SessionConfig": SessionConfig,
        "TemporalGuide": TemporalGuide,
        "find_runtime": find_runtime,
        "fit_frame": fit_frame,
        "resize_alpha": resize_alpha,
        "rgba_to_tensor": rgba_to_tensor,
        "tensor_to_rgba": tensor_to_rgba,
        "UPSCALING_LABELS": UPSCALING_LABELS,
        "NR_PRESETS": NR_PRESETS,
        "NR_STYLES": NR_STYLES,
        "DLSS_MODEL_PRESETS": DLSS_MODEL_PRESETS,
        "MOTION_MODES": MOTION_MODES,
    }
    _STATE["mods"] = mods
    return mods


def _torch():
    import torch

    return torch


def dlss5_native_available() -> bool:
    """True when the sibling package is importable and a runtime is found."""
    mods = _load_dlss5()
    if mods is None:
        return False
    try:
        mods["find_runtime"]()
        return True
    except Exception:  # noqa: BLE001
        return False


def _shrink_detail(t, width, height, source, detail_strength):
    """Detail-preserving shrink of an enhanced rebuild back to (w, h).

    Mirrors the vsdlsssr bridge: bicubic downscale (anti-aliased base) plus the
    downscaled detail (rebuild - plain upsample of the source) injected back.
    """
    torch = _torch()
    import torch.nn.functional as F

    def _resize(x, tw, th):
        if x.shape[2] == th and x.shape[1] == tw:
            return x
        return F.interpolate(
            x.permute(0, 3, 1, 2), size=(th, tw),
            mode="bicubic", align_corners=False,
        ).permute(0, 2, 3, 1)

    base = _resize(t, width, height)
    src = source.to(t.device) if source.device != t.device else source
    ref = _resize(src, t.shape[2], t.shape[1])
    detail = _resize(t - ref, width, height)
    return (base + float(detail_strength) * detail).clamp(0.0, 1.0)


def dlss5_native_enhance(
    images,
    *,
    upscaling_mode: str = "1.5x (Quality)",
    nr_preset: str = "Default",
    nr_style: str = "Default",
    nr_intensity: float = 1.0,
    local_tone_strength: float = 1.0,
    local_structure_strength: float = 1.5,
    skin_structure_strength: float = 2.0,
    automatic_mask: bool = True,
    dlss_model_preset: str = "M",
    motion_mode: str = "auto",
    scene_change_threshold: float = 0.24,
    warmup_frames: int = 0,
    flow_width: int = 640,
    runtime_dir: str = "",
    keep_upscaled: bool = True,
    detail_strength: float = 0.85,
    verify_neural_rendering: bool = True,
    progress_callback=None,
):
    """Run an IMAGE batch through the native DLSS 5 neural renderer.

    Batch order is temporal order (same contract as DLSS5EnhanceImages). The
    worker output is at the upscaling factor; with keep_upscaled=False it is
    shrunk back to the input resolution while injecting the neural detail back.

    Returns torch.Tensor [N, oh, ow, 3] float32 0..1, or None when the runtime
    is unavailable. Raises on real worker failures so the caller can fall back.
    """
    mods = _load_dlss5()
    if mods is None:
        return None
    torch = _torch()
    DlssOptions = mods["DlssOptions"]
    DlssSession = mods["DlssSession"]
    SessionConfig = mods["SessionConfig"]
    TemporalGuide = mods["TemporalGuide"]
    find_runtime = mods["find_runtime"]
    fit_frame = mods["fit_frame"]
    resize_alpha = mods["resize_alpha"]
    rgba_to_tensor = mods["rgba_to_tensor"]
    tensor_to_rgba = mods["tensor_to_rgba"]

    count, height, width, channels = images.shape
    if count == 0:
        return None
    keep_alpha = channels == 4
    rgb = images[..., :3].contiguous().float()

    options = DlssOptions.create(
        upscaling_mode=upscaling_mode,
        nr_preset=nr_preset,
        nr_style=nr_style,
        nr_intensity=nr_intensity,
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        skin_structure_strength=skin_structure_strength,
        automatic_mask=bool(automatic_mask),
        dlss_model_preset=dlss_model_preset,
        motion_mode=motion_mode,
        scene_change_threshold=scene_change_threshold,
        warmup_frames=warmup_frames,
        flow_width=flow_width,
    )
    config = SessionConfig(options=options, runtime_dir=runtime_dir)
    layout = find_runtime(config.runtime_override)

    session = None
    try:
        session = DlssSession(
            layout,
            options,
            input_width=width,
            input_height=height,
            frame_count=count,
        )
        guide = TemporalGuide(
            session.render_width,
            session.render_height,
            flow_width=options.flow_width,
            scene_change_threshold=options.scene_change_threshold,
            enabled=options.wants_motion(count),
        )
        out_h, out_w = session.output_height, session.output_width
        result = torch.empty(
            (count, out_h, out_w, 4 if keep_alpha else 3),
            dtype=torch.float32,
        )

        for index in range(count):
            rgba = fit_frame(
                tensor_to_rgba(rgb[index]),
                session.render_width,
                session.render_height,
            )
            motion = guide.process(rgba)
            enhanced, _pts = session.submit(
                index=index,
                rgba=rgba,
                motion=motion.motion,
                reset=motion.reset,
                pts=index,
            )
            if keep_alpha:
                enhanced[..., 3] = resize_alpha(
                    rgba[..., 3], session.output_width, session.output_height
                )
            result[index] = rgba_to_tensor(enhanced, keep_alpha=keep_alpha)
            if progress_callback is not None:
                progress_callback(1)
    finally:
        if session is not None:
            try:
                session.close()
            except BaseException:  # noqa: BLE001 - surface worker errors
                session.abort()
                raise

    if verify_neural_rendering and session is not None:
        try:
            session.feature_report()
        except RuntimeError:
            # Feature-18 evidence missing -> the frames may be plain upscaled.
            raise

    if not keep_upscaled:
        return _shrink_detail(
            result[..., :3], width, height, rgb, detail_strength
        ).to(images.dtype)
    return result[..., :3].to(images.dtype)
