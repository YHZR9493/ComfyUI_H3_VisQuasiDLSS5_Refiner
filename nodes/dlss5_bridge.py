"""dlss5_bridge.py - True NVIDIA DLSS Super Resolution bridge.

In-process VapourSynth + vsdlsssr.dll (official NVIDIA DLSS SR runtime,
nvngx_dlss.dll) with Depth Anything V2 (depth guide) + RAFT (motion guide)
generated inside this module. Integrated into H3VisQuasiDLSS5Refiner as the
`dlss5` enhancement backend, replacing the pseudo RTX VSR (nvvfx) pass.

Contract:
    upscale(images, scale, quality, keep_upscaled, detail_strength)
        -> torch.Tensor [N, oh, ow, 3] float32 0..1
    Returns None when the runtime / guides are unavailable; callers fail open.

Guides follow the HECer/ComfyUI-DLSS5 convention:
    depth : [N, H, W] float32, roughly 0..1 (sequence-normalized)
    mvec  : [N, H, W, 2] float32, pixel displacement in the INPUT resolution
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Path / runtime discovery
# ---------------------------------------------------------------------------
_RUNTIME_CACHE: dict[str, Path | None] = {}


def _find_runtime_dir() -> Path | None:
    """Locate the directory containing vsdlsssr.dll + nvngx_dlss.dll."""
    cached = _RUNTIME_CACHE.get("dir", "unset")
    if cached != "unset":
        return cached
    env = os.environ.get("DLSS5_RT_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    # sibling ComfyUI-DLSS5 clone (HECer)
    parent = Path(__file__).resolve().parent.parent.parent
    candidates.append(parent / "ComfyUI-DLSS5" / "runtime")
    # fall back to any runtime/ directory two levels up
    candidates.append(parent / "ComfyUI-DLSS5" / "runtime" / "dlssg")
    found: Path | None = None
    for c in candidates:
        if c.is_dir() and (c / "vsdlsssr.dll").is_file() and (
            c / "nvngx_dlss.dll"
        ).is_file():
            found = c
            break
    _RUNTIME_CACHE["dir"] = found
    return found


# ---------------------------------------------------------------------------
# Lazy heavy imports (torch / vapoursynth / transformers / torchvision)
# ---------------------------------------------------------------------------
_STATE: dict = {}
_LOCK = threading.Lock()

DLSS_QUALITY = {
    "Quality": 2,
    "Balanced": 1,
    "Performance": 0,
    "Ultra Performance": 3,
    "Ultra Quality": 4,
    "DLAA": 5,
}

DEPTH_MODELS = {
    "Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def dlss5_available() -> bool:
    """True when the DLSS SR runtime is present and vapoursynth imports."""
    if _find_runtime_dir() is None:
        return False
    try:
        import vapoursynth  # noqa: F401
        return True
    except Exception:
        return False


def _torch():
    import torch
    return torch


def _vs_core(rt_dir: Path):
    """Get the shared VapourSynth core with vsdlsssr loaded once."""
    with _LOCK:
        vs = _STATE.get("vs")
        if vs is None:
            import vapoursynth
            vs = vapoursynth
            _STATE["vs"] = vs
        if not _STATE.get("plugin"):
            core = vs.core
            if not hasattr(core, "dlsssr"):
                core.std.LoadPlugin(str((rt_dir / "vsdlsssr.dll").resolve()))
            _STATE["plugin"] = True
        return vs


def _depth_estimator(model_name: str, device, half: bool = True):
    key = ("depth", model_name, str(device), bool(half))
    with _LOCK:
        if key not in _STATE:
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )
            model_id = DEPTH_MODELS[model_name]
            processor = AutoImageProcessor.from_pretrained(
                model_id, use_fast=False
            )
            network = (
                AutoModelForDepthEstimation.from_pretrained(model_id)
                .eval()
                .to(device)
            )
            if half and str(device).startswith("cuda"):
                network = network.half()
            _STATE[key] = (processor, network)
        return _STATE[key]


def _motion_estimator(model_name: str, device, half: bool = True):
    key = ("raft", model_name, str(device), bool(half))
    with _LOCK:
        if key not in _STATE:
            from torchvision.models.optical_flow import (
                raft_large,
                raft_small,
                Raft_Large_Weights,
                Raft_Small_Weights,
            )
            if model_name == "Large":
                weights = Raft_Large_Weights.DEFAULT
                network = raft_large(weights=weights, progress=True)
            else:
                weights = Raft_Small_Weights.DEFAULT
                network = raft_small(weights=weights, progress=True)
            network = network.eval().to(device)
            if half and str(device).startswith("cuda"):
                network = network.half()
            _STATE[key] = (network, weights.transforms())
        return _STATE[key]


# ---------------------------------------------------------------------------
# Guide estimation
# ---------------------------------------------------------------------------
def _estimate_depth(rgb, device, model_name="Small", chunk_frames=4,
                    half=True):
    """rgb: [N,H,W,3] float32 cpu. Returns depth [N,H,W] 0..1."""
    torch = _torch()
    import torch.nn.functional as F

    processor, network = _depth_estimator(model_name, device, half=half)
    use_half = half and str(device).startswith("cuda")
    source = rgb.detach().cpu().float()
    N, H, W = source.shape[0], source.shape[1], source.shape[2]
    chunks = []
    with torch.inference_mode():
        for start in range(0, N, chunk_frames):
            frames = source[start : start + chunk_frames]
            inputs = processor(
                images=[
                    (f.numpy().clip(0, 1) * 255).round().astype("uint8")
                    for f in frames
                ],
                return_tensors="pt",
            )
            inputs = {
                k: (v.half() if use_half else v).to(device)
                for k, v in inputs.items()
            }
            part = network(**inputs).predicted_depth[:, None].float()
            part = F.interpolate(
                part, size=(H, W), mode="bicubic", align_corners=False
            )[:, 0]
            chunks.append(part.detach().cpu())
            del inputs, part
    depth = torch.cat(chunks, dim=0)
    # sequence-wide normalization (stable range, less temporal flicker)
    sample = depth[:, ::8, ::8].flatten()
    low, high = torch.quantile(sample, torch.tensor([0.02, 0.98]))
    depth = (depth - low) / (high - low).clamp_min(1e-6)
    return depth.clamp(0.0, 1.0)


def _estimate_motion(rgb, device, model_name="Small", chunk_frames=2,
                     guide_scale=0.5, half=True):
    """rgb: [N,H,W,3] float32 cpu. Returns mvec [N,H,W,2] pixel displacement.

    guide_scale < 1 estimates optical flow at a reduced resolution (e.g. 0.5 =
    quarter the pixels) and rescales the displacement back to the input frame
    size. RAFT cost scales with input pixels, so this cuts the motion-guide
    forward time by ~guide_scale^-2 while staying accurate enough for the DLSS
    SR temporal guidance (fast large motion slightly underestimates; SR keeps
    its spatial detail either way).
    """
    torch = _torch()
    import torch.nn.functional as F

    network, transform = _motion_estimator(model_name, device, half=half)
    use_half = half and str(device).startswith("cuda")
    source = rgb.detach().cpu().float().permute(0, 3, 1, 2)  # [N,3,H,W]
    N, H, W = source.shape[0], source.shape[2], source.shape[3]
    gs = float(guide_scale)
    if gs < 1.0:
        work = F.interpolate(
            source, scale_factor=gs, mode="bilinear", align_corners=False
        )
    else:
        work = source
    Hw, Ww = work.shape[2], work.shape[3]
    pad_h = max(128, ((Hw + 7) // 8) * 8)
    pad_w = max(128, ((Ww + 7) // 8) * 8)
    mvec = torch.zeros(N, H, W, 2, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(1, N, chunk_frames):
            stop = min(start + chunk_frames, N)
            current = F.interpolate(
                work[start:stop], size=(pad_h, pad_w),
                mode="bilinear", align_corners=False,
            )
            previous = F.interpolate(
                work[start - 1 : stop - 1], size=(pad_h, pad_w),
                mode="bilinear", align_corners=False,
            )
            current, previous = transform(current, previous)
            if use_half:
                current, previous = current.half(), previous.half()
            current, previous = current.to(device), previous.to(device)
            flow = network(current, previous)[-1]  # [B,2,pad_h,pad_w]
            flow = flow.float()
            flow = F.interpolate(
                flow, size=(H, W), mode="bilinear", align_corners=False
            ).cpu()
            mvec[start:stop, :, :, 0] = flow[:, 0] * (W / pad_w)
            mvec[start:stop, :, :, 1] = flow[:, 1] * (H / pad_h)
            del current, previous, flow
    return mvec


# ---------------------------------------------------------------------------
# VapourSynth DLSS SR execution
# ---------------------------------------------------------------------------
def _run_dlss_sr(rgb, depth, mvec, rt_dir: Path, scale: int, quality: int):
    """Run official DLSS SR on rgb [N,H,W,3]; returns [N, oh, ow, 3] float32."""
    torch = _torch()
    import numpy as np

    vs = _vs_core(rt_dir)
    core = vs.core
    N, H, W = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    color = rgb.detach().cpu().numpy()
    depth_np = depth.detach().cpu().numpy()
    mvec_np = mvec.detach().cpu().numpy()

    blank = core.std.BlankClip(
        width=W, height=H, length=N, format=vs.RGBS, color=[0.0, 0.0, 0.0]
    )
    db = core.std.BlankClip(
        width=W, height=H, length=N, format=vs.GRAYS, color=[0.0]
    )
    mb = core.std.BlankClip(
        width=W, height=H, length=N, format=vs.RGBS, color=[0.0, 0.0, 0.0]
    )

    def uc(n, f):
        o = f.copy()
        for q in range(3):
            np.asarray(o[q])[:, :W] = color[n, :, :, q]
        return o

    def ud(n, f):
        o = f.copy()
        np.asarray(o[0])[:, :W] = depth_np[n]
        return o

    def um(n, f):
        o = f.copy()
        np.asarray(o[0])[:, :W] = mvec_np[n, :, :, 0]
        np.asarray(o[1])[:, :W] = mvec_np[n, :, :, 1]
        return o

    source = core.std.ModifyFrame(blank, blank, uc)
    z = core.std.ModifyFrame(db, db, ud)
    mv = core.std.ModifyFrame(mb, mb, um)
    up = core.dlsssr.Upscale(
        source, depth=z, mvec=mv, scale=int(scale), quality=int(quality)
    )
    ow, oh = up.width, up.height
    out = torch.empty(N, oh, ow, 3, dtype=torch.float32)
    for i in range(N):
        f = up.get_frame(i)
        for q in range(3):
            out[i, :, :, q] = torch.from_numpy(
                np.asarray(f[q])[:, :ow].copy()
            ).clamp(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def upscale(
    images,
    scale: float = 1.5,
    quality: str = "Quality",
    keep_upscaled: bool = True,
    detail_strength: float = 0.85,
    depth_model: str = "Small",
    motion_model: str = "Small",
    chunk_frames: int = 4,
    device=None,
    nr_mode: str = "Off",
    nr_intensity: float = 1.0,
    nr_chunk_frames: int = 4,
    motion_scale: float = 0.5,
    half: bool = True,
):
    """DLSS SR (+ optional DLSS 5 Neural Rendering) enhancement of images.

    DLSS only exposes integer scale (2x here). When the requested `scale`
    target is smaller, the 2x rebuild is shrunk with the SR detail preserved;
    keep_upscaled=True outputs the target enlargement directly (never shrink
    back to the input resolution unless scale == 1.0). Returns None on any
    runtime failure so callers can fall back to nvvfx / passthrough.

    nr_mode selects the Neural Rendering look applied AFTER the SR rebuild
    (max-gain path, matching HECer "Upscale + neural rendering"): the guides
    are resized to the SR output resolution and NR runs in an isolated
    process, so the 8GB VRAM budget only pays for the depth/motion estimate.
    """
    torch = _torch()
    import torch.nn.functional as F

    rt_dir = _find_runtime_dir()
    if rt_dir is None:
        return None
    try:
        import vapoursynth  # noqa: F401
    except Exception:
        return None

    rgb = images[..., :3].contiguous().float()
    N, H, W = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])
    if N == 0:
        return None

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    target_w = max(8, round(W * float(scale) / 8) * 8)
    target_h = max(8, round(H * float(scale) / 8) * 8)

    # guides at input resolution
    depth = _estimate_depth(rgb, device, depth_model, chunk_frames,
                            half=half)
    mvec = _estimate_motion(rgb, device, motion_model, max(2, chunk_frames // 2),
                            guide_scale=motion_scale, half=half)

    sr = _run_dlss_sr(
        rgb, depth, mvec, rt_dir, scale=2,
        quality=DLSS_QUALITY.get(str(quality), 2),
    )
    if sr is None:
        return None
    sr = sr.to(rgb.device) if rgb.is_cuda else sr

    def _resize(t, w, h):
        if t.shape[2] == h and t.shape[1] == w:
            return t
        return F.interpolate(
            t.permute(0, 3, 1, 2), size=(h, w), mode="bicubic", align_corners=False
        ).permute(0, 2, 3, 1)

    def _shrink_detail(t, w, h):
        # detail-preserving shrink of an enhanced rebuild back to (w, h)
        base = _resize(t, w, h)
        src = rgb.to(t.device) if rgb.device != t.device else rgb
        ref = _resize(src, t.shape[2], t.shape[1])
        detail = _resize(t - ref, w, h)
        return (base + float(detail_strength) * detail).clamp(0.0, 1.0)

    # ---- Optional DLSS 5 Neural Rendering on the enlarged rebuild ----
    nr_on = str(nr_mode) != "Off" and dlss_nr_available()
    if nr_on:
        nr_w, nr_h = (target_w, target_h) if bool(keep_upscaled) else (W, H)
        frame = _resize(sr, nr_w, nr_h)
        d_nr = F.interpolate(
            depth.detach().cpu().float()[:, None],
            size=(nr_h, nr_w), mode="bilinear", align_corners=False,
        )[:, 0]
        m_nr = _resize_mvec_scaled(mvec.detach().cpu().float(), nr_w, nr_h)
        enhanced = nr_enhance(
            frame.detach().cpu().float(), d_nr, m_nr,
            mode=str(nr_mode), intensity=float(nr_intensity),
            chunk_frames=int(nr_chunk_frames),
        )
        if enhanced is not None:
            if not bool(keep_upscaled):
                return _shrink_detail(enhanced, W, H)
            return enhanced.to(sr.device).clamp(0.0, 1.0)
    del depth, mvec

    if bool(keep_upscaled) and (target_w != W or target_h != H):
        # enlarged output at the requested multiple (never shrunk to input)
        return _resize(sr, target_w, target_h).clamp(0.0, 1.0)

    if not bool(keep_upscaled):
        # detail-preserving shrink back to the original resolution
        return _shrink_detail(sr, W, H)

    # keep_upscaled=True with target == input resolution (scale=1.0):
    # return the 2x rebuild shrunk with detail injection at original size
    return _shrink_detail(sr, W, H)


# ---------------------------------------------------------------------------
# DLSS 5 Neural Rendering (NR) -- Feature 18 via HECer bridge_runner
# ---------------------------------------------------------------------------
# Reuses the HECer/ComfyUI-DLSS5 isolated VapourSynth runner:
#   python -u bridge_runner.py in.npy out.npy --plugin vsdlssnr.dll
#       --snippet nvngx_dlssnr.dll --settings <json> --depth d.npy --mvec m.npy
# NR runs in a separate process (D3D12 side) and never touches torch VRAM.
NR_LOOKS = {
    # mode -> (style, style_strength, local_structure, skin_structure)
    "Neutral / faithful": (0, 0.70, 1.00, -1.00),
    "Realistic detail": (2, 0.85, 1.20, 0.10),
    "Strong detail": (2, 1.00, 1.45, 0.30),
}
_NR_CACHE: dict[str, Path | None] = {}


def _dlss5_package_dir() -> Path | None:
    """Sibling HECer ComfyUI-DLSS5 package directory."""
    parent = Path(__file__).resolve().parent.parent.parent
    pkg = parent / "ComfyUI-DLSS5"
    return pkg if pkg.is_dir() else None


def _find_nr_paths() -> tuple[Path, Path, Path, Path] | None:
    """Locate (python, vsdlssnr.dll, nvngx_dlssnr.dll, bridge_runner.py)."""
    cached = _NR_CACHE.get("paths", "unset")
    if cached != "unset":
        return cached
    pkg = _dlss5_package_dir()
    if pkg is None:
        _NR_CACHE["paths"] = None
        return None
    runtime = pkg / "runtime"

    def first(*cands: Path) -> Path | None:
        return next((c for c in cands if c.is_file()), None)

    env = os.environ
    python = (
        Path(env["DLSS5_PYTHON"])
        if env.get("DLSS5_PYTHON")
        else first(
            pkg / "test-env" / "Scripts" / "python.exe",
            Path(__import__("sys").executable),
        )
    )
    plugin = first(
        *(Path(env["DLSS5_PLUGIN"]) if env.get("DLSS5_PLUGIN") else []),
        runtime / "vsdlssnr.dll",
        pkg / "test-env" / "Lib" / "site-packages" / "vapoursynth" / "plugins" / "vsdlssnr.dll",
    )
    snippet = first(
        *(Path(env["DLSS5_SNIPPET"]) if env.get("DLSS5_SNIPPET") else []),
        runtime / "nvngx_dlssnr.dll",
        pkg / "test-env" / "Lib" / "site-packages" / "vapoursynth" / "plugins" / "nvngx_dlssnr.dll",
    )
    runner = pkg / "bridge_runner.py"
    if not (python and plugin and snippet and runner.is_file()):
        _NR_CACHE["paths"] = None
        return None
    _NR_CACHE["paths"] = (python, plugin, snippet, runner)
    return _NR_CACHE["paths"]


def dlss_nr_available() -> bool:
    """True when the HECer NR runtime (vsdlssnr + nvngx_dlssnr) is present."""
    return _find_nr_paths() is not None


def _run_nr_process(
    python: Path, runner: Path, plugin: Path, snippet: Path,
    settings: dict, input_path: Path, output_path: Path,
    depth_path: Path | None, motion_path: Path | None, timeout=None,
) -> None:
    """Run bridge_runner.py and raise RuntimeError on nonzero exit."""
    import json
    import subprocess

    command = [
        str(python), "-u", str(runner), str(input_path), str(output_path),
        "--plugin", str(plugin), "--snippet", str(snippet),
        "--settings", json.dumps(settings),
    ]
    if depth_path is not None:
        command += ["--depth", str(depth_path), "--mvec", str(motion_path)]
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("DLSS NR timed out") from None
    if proc.returncode:
        raise RuntimeError(
            "DLSS NR failed:\n" + (proc.stdout or "") + "\n" + (proc.stderr or "")
        )


def _resize_mvec_scaled(mvec, w, h):
    """Bilinear-resize [N,H,W,2] pixel-displacement mvec and scale values."""
    torch = _torch()
    import torch.nn.functional as F

    src_h, src_w = mvec.shape[1], mvec.shape[2]
    if src_w == w and src_h == h:
        return mvec
    scaled = F.interpolate(
        mvec.float().permute(0, 3, 1, 2), size=(h, w),
        mode="bilinear", align_corners=False,
    ).permute(0, 2, 3, 1)
    scaled[..., 0] *= w / src_w
    scaled[..., 1] *= h / src_h
    return scaled


def nr_enhance(
    images,
    depth,
    mvec,
    mode: str = "Realistic detail",
    intensity: float = 1.0,
    chunk_frames: int = 4,
    temp_dir: Path | None = None,
):
    """DLSS 5 Neural Rendering (Feature 18) on images [N,H,W,3] float32 cpu.

    depth: [N,H,W] 0..1; mvec: [N,H,W,2] pixel displacement (same size as
    images). Runs in an isolated VapourSynth process so torch VRAM is not
    consumed by the NR runtime itself. Returns torch tensor or None on failure.
    """
    import tempfile

    paths = _find_nr_paths()
    if paths is None:
        return None
    python, plugin, snippet, runner = paths
    if mode not in NR_LOOKS:
        return None
    style, style_strength, local_structure, skin_structure = NR_LOOKS[mode]

    torch = _torch()
    import numpy as np

    rgb = images[..., :3].contiguous().float()
    N, H, W = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])
    if N == 0:
        return None
    d = depth.detach().cpu().float()
    m = mvec.detach().cpu().float()
    if d.shape != (N, H, W) or m.shape != (N, H, W, 2):
        return None
    # VRAM/process safety: adaptive chunk size for high resolutions
    if chunk_frames > 2 and H * W > 2_000_000:
        chunk_frames = 2

    settings = {
        "style": int(style),
        "style_strength": float(style_strength),
        "intensity": float(intensity),
        "local_structure": float(local_structure),
        "skin_structure": float(skin_structure),
        "auto_mask": 1,
        "preset": 0,
        "feature_id": 18,
        "depth_inverted": False,
    }
    chunks = []
    with tempfile.TemporaryDirectory(prefix="marvis-dlss5-nr-",
                                     dir=str(temp_dir) if temp_dir else None) as td:
        tmp = Path(td)
        for start in range(0, N, chunk_frames):
            stop = min(start + chunk_frames, N)
            in_path, out_path = tmp / f"in_{start}.npy", tmp / f"out_{start}.npy"
            np.save(in_path, rgb[start:stop].numpy())
            depth_path = motion_path = None
            if d is not None:
                depth_path, motion_path = tmp / f"d_{start}.npy", tmp / f"m_{start}.npy"
                np.save(depth_path, d[start:stop].numpy())
                np.save(motion_path, m[start:stop].numpy())
            _run_nr_process(
                python, runner, plugin, snippet, settings,
                in_path, out_path, depth_path, motion_path, timeout=None,
            )
            chunks.append(torch.from_numpy(np.load(out_path)))
            for p in (in_path, out_path, depth_path, motion_path):
                if p is not None and p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
    return torch.cat(chunks, dim=0).clamp(0.0, 1.0)
