"""
Super Resolution Pipeline Wrapper
Delegates to: src/ai_super_resolution/sr_pipeline.SuperResolutionPipeline
"""
import sys
import os
import yaml
import numpy as np

# Inject sibling project into sys.path
_SR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ai_super_resolution")
_SR_DIR = os.path.abspath(_SR_DIR)
if _SR_DIR not in sys.path:
    sys.path.insert(0, _SR_DIR)

_PROJECT_ROOT = os.path.abspath(os.path.join(_SR_DIR, "..", ".."))


def _load_config():
    config_path = os.path.join(_SR_DIR, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["paths"]["weights_dir"] = os.path.join(_PROJECT_ROOT, config["paths"]["weights_dir"])
    return config


def get_pipeline():
    """Lazily creates the SuperResolutionPipeline (designed for @st.cache_resource)."""
    from sr_pipeline import SuperResolutionPipeline
    config = _load_config()
    return SuperResolutionPipeline(config)


def process(pipeline, image_rgb: np.ndarray, scale: int = 4,
            mode: str = "fast") -> np.ndarray:
    """
    Run super-resolution on an RGB image.

    Args:
        pipeline: Cached SuperResolutionPipeline instance
        image_rgb: Input image (H, W, 3) uint8 RGB
        scale: Upscale factor (2 or 4)
        mode: "fast", "balanced", or "hd"

    Returns:
        Upscaled RGB image
    """
    return pipeline.run(image_rgb, scale=scale, mode=mode)
