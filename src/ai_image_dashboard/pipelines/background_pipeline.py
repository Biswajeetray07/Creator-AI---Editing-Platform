"""
Background Removal Pipeline — 5-Stage Adaptive Engine
Uses BiRefNet (SOD) + SAM ViT-B (segmentation) + MODNet (alpha matting)
in a unified pipeline that adapts processing based on scene complexity.

Replaces the previous rembg/U²-Net implementation with a production-grade
Photoroom-style adaptive pipeline.
"""
import sys
import os
import numpy as np

# Inject background_removal package into sys.path
_BG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "background_removal")
_BG_DIR = os.path.abspath(_BG_DIR)
if _BG_DIR not in sys.path:
    sys.path.insert(0, _BG_DIR)


def get_pipeline():
    """
    Returns the BackgroundRemovalEngine (designed for @st.cache_resource).
    All 3 models (~1.2GB total) are loaded into VRAM once and reused.
    """
    from inference.engine import BackgroundRemovalEngine
    engine = BackgroundRemovalEngine(load_sam=True)
    engine.load()
    return engine


def process(pipeline, image_rgb: np.ndarray, mode: str = "auto",
            return_metrics: bool = False) -> np.ndarray:
    """
    Remove background from an RGB image using the 5-stage adaptive pipeline.

    Args:
        pipeline: Cached BackgroundRemovalEngine instance
        image_rgb: Input image (H, W, 3) uint8 RGB
        mode: Pipeline mode — "auto" (adaptive), "simple" (portrait), "complex" (product)
        return_metrics: If True, returns (rgba, metrics) tuple instead of just rgba

    Returns:
        RGBA image with background removed (H, W, 4) uint8
        OR (RGBA, metrics_dict) if return_metrics=True
    """
    # Map mode to force_mode parameter
    force_mode = None if mode == "auto" else mode

    result = pipeline.run(image_rgb, force_mode=force_mode)

    if return_metrics:
        metrics = result["metrics"]
        metrics["scene_type"] = result["scene_type"]
        return result["rgba"], metrics

    return result["rgba"]
