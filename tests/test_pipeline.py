"""
Integration tests for the Object Removal Pipeline.

Usage:
    python -m pytest tests/test_pipeline.py -v
    OR
    python tests/test_pipeline.py
"""

import os
import sys
import cv2
import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from object_removal_ai.main_pipeline import ObjectRemovalPipeline, resolve_weight_path
from object_removal_ai.pipeline.mask_refiner import MaskRefiner
from object_removal_ai.pipeline.edge_extractor import EdgeExtractor
from object_removal_ai.pipeline.postprocess import PostProcessor


def test_mask_refiner():
    """Test that MaskRefiner properly dilates and smooths a mask."""
    refiner = MaskRefiner(dilate_kernel=15, dilate_iter=3, close_kernel=9, blur_kernel=5)
    
    # Create a small binary mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255  # 40x40 white square
    
    refined = refiner(mask)
    
    assert refined.shape == (100, 100), f"Shape mismatch: {refined.shape}"
    assert refined.dtype == np.uint8, f"Dtype mismatch: {refined.dtype}"
    assert refined.sum() > mask.sum(), "Refined mask should be larger after dilation"
    print("✅ test_mask_refiner passed")


def test_edge_extractor():
    """Test that EdgeExtractor produces edges and zeroes the masked region."""
    extractor = EdgeExtractor(canny_low=50, canny_high=150)
    
    # Create a test image with clear edges (gradient)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 50:] = 255  # Black/White vertical split
    
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # Mask the center
    
    edges = extractor(image, mask)
    
    assert edges.shape == (100, 100), f"Shape mismatch: {edges.shape}"
    assert edges[50, 50] == 0, "Edges inside mask should be zero"
    print("✅ test_edge_extractor passed")


def test_post_processor_alpha():
    """Test alpha blending mode."""
    processor = PostProcessor(use_poisson=False)
    
    original = np.full((100, 100, 3), 128, dtype=np.uint8)  # Gray
    inpainted = np.full((100, 100, 3), 200, dtype=np.uint8)  # Light
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    
    result = processor(original, inpainted, mask)
    
    assert result.shape == (100, 100, 3), f"Shape mismatch: {result.shape}"
    assert result.dtype == np.uint8
    print("✅ test_post_processor_alpha passed")


def test_resolve_weight_path():
    """Test that weight path resolution works for universal and local paths."""
    # Absolute paths should pass through
    abs_path = os.path.abspath("model.pt")
    p = resolve_weight_path(abs_path)
    assert p == abs_path
    
    # HuggingFace model IDs should pass through
    p = resolve_weight_path("runwayml/stable-diffusion-inpainting")
    assert p == "runwayml/stable-diffusion-inpainting"
    
    print("✅ test_resolve_weight_path passed")


def test_pipeline_with_mask():
    """Integration test: run pipeline with a user-provided mask on a synthetic image."""
    # Create synthetic test image (gradient background with a white square "object")
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Gradient background
    for i in range(256):
        image[i, :] = [i, 128, 255 - i]
    # White square "object"
    image[80:180, 80:180] = [255, 255, 255]
    
    # Create mask for the object
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[80:180, 80:180] = 255
    
    # Save test files
    os.makedirs("test_output", exist_ok=True)
    cv2.imwrite("test_output/synthetic_input.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("test_output/synthetic_mask.png", mask)
    
    try:
        pipeline = ObjectRemovalPipeline()
        result = pipeline.run(image, mask=mask)
        
        assert result["result"].shape == image.shape[:2] + (3,), "Output shape mismatch"
        assert result["mask"] is not None, "Mask artifact should be present"
        
        # The object area should NOT be pure white anymore after inpainting
        result_region = result["result"][100:160, 100:160]
        assert not np.all(result_region == 255), "Inpainted region should not be pure white"
        
        cv2.imwrite("test_output/synthetic_clean.jpg", cv2.cvtColor(result["result"], cv2.COLOR_RGB2BGR))
        print("✅ test_pipeline_with_mask passed")
        print(f"   Output saved: test_output/synthetic_clean.jpg")
    except FileNotFoundError as e:
        print(f"⚠️  test_pipeline_with_mask skipped (missing weights): {e}")
    except Exception as e:
        print(f"❌ test_pipeline_with_mask failed: {e}")
        raise


if __name__ == "__main__":
    print("=" * 50)
    print("  Running Object Removal Pipeline Tests")
    print("=" * 50)
    
    test_mask_refiner()
    test_edge_extractor()
    test_post_processor_alpha()
    test_resolve_weight_path()
    test_pipeline_with_mask()
    
    print("\n" + "=" * 50)
    print("  All tests completed!")
    print("=" * 50)
