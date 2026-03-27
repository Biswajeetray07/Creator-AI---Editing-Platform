"""
AI Text Removal Pipeline — Inference CLI

Pipeline:
  Image → EasyOCR(CRAFT) Detection → Polygon Mask → Refine Mask → 
  Edge Extraction → Big-LaMa Inpaint → SD Diffusion Refinement → 
  Post-Processing → Clean Image
"""
import os
import sys
import argparse
import yaml
import time

# Ensure project root is in sys.path for robust imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from tr_utils.image_utils import load_image, save_image, resize_image
from tr_pipelines.detect_text import TextDetector
from tr_pipelines.segment_mask import TextMaskGenerator
from tr_pipelines.refine_mask import MaskRefiner
from tr_pipelines.edge_extract import EdgeExtractor
from tr_pipelines.inpaint import LaMaInpainter
from tr_pipelines.diffusion_refine import DiffusionRefiner
from tr_pipelines.post_process import PostProcessor


def resolve_weight_path(p: str, script_dir: str) -> str:
    """Resolve a weight path: try local, then universal weights dir."""
    if not p or os.path.isabs(p) or p.startswith("runwayml/"):
        return p
    # Try relative to script
    local = os.path.abspath(os.path.join(script_dir, p))
    if os.path.exists(local):
        return local
    # Try universal weights folder
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    universal = os.path.join(workspace_root, "weights", os.path.basename(p))
    if os.path.exists(universal):
        return universal
    return local


def main():
    parser = argparse.ArgumentParser(description="AI Text Removal Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path")
    parser.add_argument("--config", "-c", type=str,
                        default=os.path.join(script_dir, "configs", "config.yaml"),
                        help="Config file path")
    parser.add_argument("--save-artifacts", action="store_true",
                        help="Save intermediate masks and edge maps")
    parser.add_argument("--no-diffusion", action="store_true",
                        help="Skip diffusion refinement for faster processing")
    args = parser.parse_args()

    # ── Load Config ──
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = config["pipeline"]["device"]
    weights = config["paths"]["weights"]
    params = config["parameters"]

    # Resolve weight paths
    inpaint_path = resolve_weight_path(weights.get("inpainting", ""), script_dir)
    std_inpaint_path = resolve_weight_path(weights.get("std_inpainting", "lama.pt"), script_dir)
    diffusion_id = weights.get("diffusion", "runwayml/stable-diffusion-inpainting")

    start_time = time.time()
    print("=" * 60)
    print("  AI Text Removal Pipeline")
    print("=" * 60)

    # ── Step 1: Initialize Modules ──
    print("\n[1/7] Initializing modules...")
    detector = TextDetector(device=device)
    mask_gen = TextMaskGenerator()

    mr = params.get("mask_refinement", {})
    refiner = MaskRefiner(
        dilate_kernel=mr.get("dilate_kernel", 7),
        dilate_iter=mr.get("dilate_iterations", 3),
        blur_kernel=mr.get("gaussian_blur", 3),
    )

    edge_params = params.get("edge_extraction", {})
    edge_extractor = EdgeExtractor(
        canny_low=edge_params.get("canny_low", 30),
        canny_high=edge_params.get("canny_high", 100),
    )

    print("[1.1/7] Loading LaMa inpainter models...")
    inpainter = LaMaInpainter(
        model_path=inpaint_path, 
        std_model_path=std_inpaint_path,
        device=device
    )

    diff_params = params.get("diffusion", {})
    diffusion_refiner = None
    if not args.no_diffusion and diff_params.get("strength", 0.0) > 0:
        print("[1.2/7] Loading Diffusion Refiner...")
        diffusion_refiner = DiffusionRefiner(model_id=diffusion_id, device=device)

    pp = params.get("post_processing", {})
    post_processor = PostProcessor(use_poisson=pp.get("use_poisson", True))

    # ── Step 2: Load & Preprocess ──
    print("\n[2/7] Loading image...")
    image = load_image(args.input)
    # Preserve the un-resized original for full-res compositing
    true_original = image.copy()
    
    max_size = config["pipeline"].get("image_size", 1024)
    image = resize_image(image, max_size=max_size)
    print(f"      Image size: {image.shape[1]}x{image.shape[0]}")

    # ── Step 3: Detect Text ──
    print("\n[3/7] Detecting text regions...")
    polygons = detector(image)
    print(f"      Found {len(polygons)} text regions.")

    if len(polygons) == 0:
        print("\n      ⚠️  No text detected. Saving original image.")
        out_dir = config["paths"]["output"].get("dir", "text_output")
        os.makedirs(out_dir, exist_ok=True)
        name, ext = os.path.splitext(os.path.basename(args.input))
        out_path = args.output or os.path.join(out_dir, f"{name}_clean{ext}")
        save_image(image, out_path)
        print(f"  → Saved: {out_path}")
        return

    # ── Step 4: Generate Text Mask ──
    print("\n[4/7] Generating text mask (polygon fill)...")
    mask = mask_gen(image, polygons)

    # ── Step 5: Refine Mask ──
    print("[5/7] Refining mask (dilation + smoothing)...")
    refined_mask = refiner(mask)

    # ── Step 6: Extract Structure ──
    print("[6/7] Extracting edge structure map...")
    edge_map = edge_extractor(image, refined_mask)

    # ── Step 7: Inpaint ──
    inp = params.get("inpainting", {})
    print("[7/7] Running structure-guided inpainting (Big-LaMa)...")
    clean_image = inpainter(
        image, refined_mask,
        edge_map=edge_map,
        multi_scale=inp.get("multi_scale", True),
        scales=inp.get("scales", [256, 512, 768]),
    )

    # ── Step 7.5: Diffusion Refinement (optional) ──
    if diffusion_refiner is not None and diff_params.get("strength", 0.0) > 0:
        print("      Running generative refinement (Stable Diffusion)...")
        import torch
        torch.cuda.empty_cache()
        clean_image = diffusion_refiner(
            image=clean_image,
            mask=refined_mask,
            prompt=diff_params.get("prompt", "clean background, no text"),
            negative_prompt=diff_params.get("negative_prompt", "text, artifacts"),
            steps=diff_params.get("num_inference_steps", 25),
            strength=diff_params.get("strength", 0.30),
        )

    # ── Step 8: Post-Process ──
    print("      Post-processing (native-res full compositing + color match)...")
    final_image = post_processor(
        true_original=true_original,
        inpainted=clean_image, 
        mask=refined_mask,
        alpha_mask=None
    )

    # ── Save ──
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  ✅ Text removal completed in {elapsed:.2f}s")
    print(f"{'=' * 60}")

    out_dir = config["paths"]["output"].get("dir", "text_output")
    os.makedirs(out_dir, exist_ok=True)

    name, ext = os.path.splitext(os.path.basename(args.input))
    out_path = args.output or os.path.join(out_dir, f"{name}_clean{ext}")

    if args.save_artifacts:
        import cv2
        cv2.imwrite(os.path.join(out_dir, f"{name}_mask.png"), refined_mask)
        cv2.imwrite(os.path.join(out_dir, f"{name}_edges.png"), edge_map)
        print(f"  → Mask:  {os.path.join(out_dir, f'{name}_mask.png')}")
        print(f"  → Edges: {os.path.join(out_dir, f'{name}_edges.png')}")

    save_image(final_image, out_path)
    print(f"  → Clean: {out_path}")


if __name__ == "__main__":
    main()
