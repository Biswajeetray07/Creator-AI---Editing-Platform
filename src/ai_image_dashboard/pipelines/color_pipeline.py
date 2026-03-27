"""
Color Correction Pipeline Wrapper
Delegates to: src/hybrid_color_correction/pipeline/enhance.HybridEnhancer
"""
import sys
import os
import numpy as np
import torch

# Inject sibling project into sys.path
_CC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "hybrid_color_correction")
_CC_DIR = os.path.abspath(_CC_DIR)
if _CC_DIR not in sys.path:
    sys.path.insert(0, _CC_DIR)


def get_pipeline():
    """Lazily creates the HybridEnhancer (designed for @st.cache_resource)."""
    from hc_pipeline.enhance import HybridEnhancer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HybridEnhancer(device=device)


def process(pipeline, image_rgb: np.ndarray, use_restormer: bool = True) -> np.ndarray:
    """
    Run hybrid color correction on an RGB image.
    
    Args:
        pipeline: Cached HybridEnhancer instance
        image_rgb: Input image (H, W, 3) uint8 RGB
        use_restormer: Enable Restormer detail refinement
    
    Returns:
        Color-corrected RGB image
    """
    from hc_pipeline.preprocess import preprocess_image
    from hc_pipeline.postprocess import postprocess_tensor

    # Preprocess: resize to max 512px and convert to tensor
    tensor = preprocess_image(image_rgb, max_size=512)
    
    # Enhance
    out_tensor = pipeline.enhance(tensor, use_restormer=use_restormer)
    
    # Postprocess: tensor -> uint8 numpy
    return postprocess_tensor(out_tensor)
