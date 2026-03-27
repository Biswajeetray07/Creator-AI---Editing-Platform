import os
import sys
import numpy as np
import cv2

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
ai_text_dir = os.path.join(workspace_root, "ai_text_removal")
if ai_text_dir not in sys.path:
    sys.path.insert(0, ai_text_dir)

from ai_text_removal.pipeline import TextRemovalPipeline
from ai_text_removal.pipelines.inpaint import LaMaInpainter
from ai_text_removal.utils.mask_utils import refine_mask

def test_mask_generation():
    print("Running test: Mask Generation (Soft Edge)...")
    # Simulate a hard binary mask
    h, w = 200, 200
    hard_mask = np.zeros((h, w), dtype=np.uint8)
    hard_mask[50:150, 50:150] = 255
    
    # Run through refiner
    soft_mask = refine_mask(hard_mask, dilate_kernel=11, dilate_iter=3, blur_kernel=15)
    
    # Check if soft
    unique_vals = np.unique(soft_mask)
    assert len(unique_vals) > 2, "Mask should be soft, but only found binary-like values."
    print("✅ Mask generation is properly softened.")

def test_inpaint_routing():
    print("Running test: Inpaint Routing logic...")
    # Mock LaMaInpainter simply logs if no models load.
    # Instantiate without models to just test the flow
    inpainter = LaMaInpainter(model_path="", std_model_path="", device="cpu")
    
    # The image is 1000x1000, 1M pixels
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    # Case 1: Mask is 2% (Standard LaMa)
    small_mask = np.zeros((1000, 1000), dtype=np.uint8)
    small_mask[0:100, 0:200] = 255 # 20,000 px = 2%
    
    # Case 2: Mask is 10% (Big LaMa)
    large_mask = np.zeros((1000, 1000), dtype=np.uint8)
    large_mask[0:300, 0:333] = 255 # ~100k px = 10%
    
    try:
        # We expect fallback cv2 to trigger, we just want to ensure
        # it doesn't crash computing the mask_ratio.
        res1 = inpainter(img, small_mask, multi_scale=False)
        res2 = inpainter(img, large_mask, multi_scale=False)
        assert res1.shape == img.shape
        assert res2.shape == img.shape
        print("✅ Inpaint logic processes varying mask sizes successfully.")
    except Exception as e:
        print(f"❌ Inpaint routing failed: {e}")
        raise

if __name__ == "__main__":
    test_mask_generation()
    test_inpaint_routing()
    print("\nAll unit tests passed successfully!")
