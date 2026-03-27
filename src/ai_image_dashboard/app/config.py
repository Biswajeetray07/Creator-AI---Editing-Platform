"""
Centralized configuration for the AI Image Dashboard.
"""
import os

# ═══════════════════════════════════════════════════════════════
# Path Resolution
# ═══════════════════════════════════════════════════════════════
DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(DASHBOARD_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

# Sibling project directories
SUPER_RES_DIR = os.path.join(SRC_DIR, "ai_super_resolution")
TEXT_REMOVAL_DIR = os.path.join(SRC_DIR, "ai_text_removal")
COLOR_CORRECTION_DIR = os.path.join(SRC_DIR, "hybrid_color_correction")
OBJECT_REMOVAL_DIR = os.path.join(SRC_DIR, "object_removal_ai")

# ═══════════════════════════════════════════════════════════════
# Tool Definitions
# ═══════════════════════════════════════════════════════════════
TOOLS = {
    "super_resolution": {
        "name": "AI Super Resolution",
        "icon": "🔍",
        "description": "Upscale images 2x–4x with RealESRGAN + GFPGAN face enhancement",
        "color": "#6C5CE7",
    },
    "text_removal": {
        "name": "AI Text Removal",
        "icon": "🔤",
        "description": "Detect and remove text from images using EasyOCR + Big-LaMa inpainting",
        "color": "#00B894",
    },
    "color_correction": {
        "name": "Color Correction",
        "icon": "🎨",
        "description": "Auto exposure + color balance (Zero-DCE++) with detail refinement (Restormer)",
        "color": "#FDCB6E",
    },
    "object_removal": {
        "name": "Object Removal",
        "icon": "✂️",
        "description": "Describe an object to auto-detect and remove it with GroundingDINO + SAM + LaMa",
        "color": "#E17055",
    },
    "background_removal": {
        "name": "Background Removal",
        "icon": "🖼️",
        "description": "Adaptive 5-stage pipeline — BiRefNet + SAM + MODNet with scene-aware routing",
        "color": "#0984E3",
    },
}
