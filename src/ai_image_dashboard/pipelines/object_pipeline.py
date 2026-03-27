"""
Object Removal Pipeline Wrapper
Delegates to: src/object_removal_ai/main_pipeline.ObjectRemovalPipeline
"""
import sys
import os
import numpy as np

# Inject sibling project into sys.path
_OR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "object_removal_ai")
_OR_DIR = os.path.abspath(_OR_DIR)
if _OR_DIR not in sys.path:
    sys.path.insert(0, _OR_DIR)


def get_pipeline():
    """Lazily creates the ObjectRemovalPipeline (designed for @st.cache_resource)."""
    from main_pipeline import ObjectRemovalPipeline
    return ObjectRemovalPipeline()


def process(pipeline, image_rgb: np.ndarray, prompt: str = "object") -> np.ndarray:
    """
    Remove described objects from an RGB image.
    
    Args:
        pipeline: Cached ObjectRemovalPipeline instance
        image_rgb: Input image (H, W, 3) uint8 RGB
        prompt: Text prompt describing what to remove (e.g. "person", "car")
    
    Returns:
        Cleaned RGB image with objects removed
    """
    result = pipeline.run(image_rgb, prompt=prompt)
    return result["result"]
