"""ComfyUI H3 Vis Quasi DLSS5 Refiner — standalone YOLO-World guided H3 refine.

Moved out of ComfyUI_MiniMaxH3_Director as an independent node package
(functionality unchanged). Detects open-vocabulary classes with YOLO-World,
injects the class names into a repair prompt, re-generates only the detected
regions with MiniMax H3 in temporal chunks and blends them back; optionally
follows up with a headless same-multiple NVIDIA RTX VSR pass.

Standalone: only depends on ComfyUI official core (comfy / comfy_extras /
nodes), ultralytics and nvvfx — no Director dependency.
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
