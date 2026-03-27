"""
Text Removal Pipeline Wrapper
Delegates to: src/ai_text_removal/tr_pipeline.TextRemovalPipeline
"""
import sys
import os
import numpy as np

# Inject sibling project into sys.path
_TR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ai_text_removal")
_TR_DIR = os.path.abspath(_TR_DIR)
if _TR_DIR not in sys.path:
    sys.path.insert(0, _TR_DIR)


def get_pipeline():
    """Lazily creates the TextRemovalPipeline (designed for @st.cache_resource)."""
    from tr_pipeline import TextRemovalPipeline
    return TextRemovalPipeline()


def process(pipeline, image_rgb: np.ndarray) -> np.ndarray:
    """
    Remove text from an RGB image.
    
    Args:
        pipeline: Cached TextRemovalPipeline instance
        image_rgb: Input image (H, W, 3) uint8 RGB
    
    Returns:
        Cleaned RGB image with text removed
    """
    result = pipeline.run(image_rgb, enable_diffusion=True)
    return result["result"]
