"""ComfyUI H3 Vis Quasi DLSS5 Refiner — headless RTX VSR / DLSS SR enhancer.

Enhancement-only node for MiniMax H3 video pipelines: the input clip flows
straight through an NVIDIA super-resolution bridge (true DLSS SR preferred,
nvvfx RTX VSR fallback) at rtx_scale, optionally applies DLSS 5 Neural
Rendering after the SR rebuild, then an optional de-fog / de-haze post look.
YOLO-guided detection and H3 re-generation repair were removed.

Standalone: only depends on ComfyUI official core (comfy / comfy_extras /
nodes) and optional NVIDIA runtimes; falls back open when unavailable.
"""

from .nodes.h3_vis_quasi_dlss5_refiner import H3VisQuasiDLSS5Refiner

# New canonical id, plus the legacy Director id so existing workflows that
# referenced H3YoloWorldDefineRefine keep loading without edits.
NODE_CLASS_MAPPINGS = {
    "H3VisQuasiDLSS5Refiner": H3VisQuasiDLSS5Refiner,
    "H3YoloWorldDefineRefine": H3VisQuasiDLSS5Refiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3VisQuasiDLSS5Refiner": "H3 Vis Quasi DLSS5 Refiner",
    "H3YoloWorldDefineRefine": "H3 Vis Quasi DLSS5 Refiner",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
