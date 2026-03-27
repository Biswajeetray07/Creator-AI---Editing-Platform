import os
import sys
import argparse
import yaml
import time
import cv2
import numpy as np
import torch

# Ensure project root is in sys.path for robust imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from or_utils.image_utils import load_image, save_image, resize_max
from or_models.groundingdino_detector import GroundingDINODetector
from or_models.sam_segmenter import MobileSamSegmenter
from or_models.midas_depth import MidasDepthEstimator
from or_models.lama_inpainter import LaMaInpainter
from or_models.diffusion_refiner import DiffusionRefiner
from or_pipeline.mask_refiner import MaskRefiner
from or_pipeline.edge_extractor import EdgeExtractor
from or_pipeline.context_expansion import ContextExpander
from or_pipeline.postprocess import PostProcessor
from or_models.yolo_detector import YoloDetector


# ═════════════════════════════════════════════════════════════════════
# Path Resolution
# ═════════════════════════════════════════════════════════════════════
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for weights in Text Removal/weights as the universal location
WEIGHTS_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", "weights"))


def resolve_weight_path(p: str) -> str:
    """Resolve a weight path: try local, then universal weights folder."""
    if not p or os.path.isabs(p) or p.startswith("runwayml/"):
        return p
    # 1. Try relative to project dir
    local = os.path.abspath(os.path.join(PROJECT_DIR, p))
    if os.path.exists(local):
        return local
    # 2. Try universal weights folder
    universal = os.path.join(WEIGHTS_DIR, os.path.basename(p))
    if os.path.exists(universal):
        return universal
    # 3. Return local (will fail at model load with clear error)
    return local


# ═════════════════════════════════════════════════════════════════════
# ObjectRemovalPipeline — Reusable Class
# ═════════════════════════════════════════════════════════════════════
class ObjectRemovalPipeline:
    """
    Production-quality object removal pipeline.
    
    Usage:
        pipeline = ObjectRemovalPipeline("configs/model_config.yaml")
        result = pipeline.run(image, prompt="person")        # auto-detect
        result = pipeline.run(image, mask=user_mask)          # user-provided mask
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(PROJECT_DIR, "configs", "model_config.yaml")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.config = config
        self.device = "cuda"
        self.params = config["parameters"]
        
        # Resolve weight paths
        weights = {k: resolve_weight_path(v) for k, v in config["paths"]["weights"].items()}
        self.weights = weights
        
        print("=" * 60)
        print("  AI Object Removal Pipeline — Initializing")
        print("=" * 60)
        
        # ── Detector (GroundingDINO → YOLOv8 fallback) ──
        self.detector = None
        try:
            print("[Init] Loading GroundingDINO...")
            self.detector = GroundingDINODetector(
                config_path=self.params.get("detection", {}).get("config_path", ""),
                model_path=weights.get("detection", ""),
                device=self.device
            )
        except Exception as e:
            print(f"[Init] GroundingDINO failed: {e}")
            print("[Init] Falling back to YOLOv8...")
            yolo_path = weights.get("yolo", resolve_weight_path("yolov8n.pt"))
            self.detector = YoloDetector(model_path=yolo_path, device=self.device)
        
        # ── Segmenter (MobileSAM) ──
        print("[Init] Loading MobileSAM...")
        self.segmenter = MobileSamSegmenter(model_path=weights["segmentation"], device=self.device)
        
        # ── Mask Refiner ──
        mask_params = self.params["mask_refinement"]
        self.mask_refiner = MaskRefiner(
            dilate_kernel=mask_params["dilate_kernel"],
            dilate_iter=mask_params["dilate_iterations"],
            close_kernel=mask_params["close_kernel"],
            blur_kernel=mask_params["gaussian_blur"],
        )
        
        # ── Edge Extractor ──
        edge_params = self.params["edge_extraction"]
        self.edge_extractor = EdgeExtractor(
            canny_low=edge_params["canny_low"],
            canny_high=edge_params["canny_high"]
        )
        
        # ── Inpainter (LaMa) ──
        print("[Init] Loading LaMa inpainter...")
        self.inpainter = LaMaInpainter(model_path=weights["inpainting"], device=self.device)
        
        # ── Diffusion Refiner ──
        diffusion_params = self.params.get("diffusion", {})
        self.diffusion_refiner = None
        if diffusion_params.get("strength", 0.0) > 0.0:
            print("[Init] Loading Diffusion Refiner...")
            self.diffusion_refiner = DiffusionRefiner(
                model_id=weights.get("diffusion", "runwayml/stable-diffusion-inpainting"),
                device=self.device
            )
        
        # ── Post-Processor ──
        self.post_processor = PostProcessor(
            use_poisson=self.params.get("post_processing", {}).get("use_poisson", True)
        )
        
        print("=" * 60)
        print("  Pipeline ready!")
        print("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    def run(
        self,
        image: np.ndarray,
        prompt: str = None,
        mask: np.ndarray = None,
        save_artifacts: bool = False,
        output_dir: str = "output",
    ) -> dict:
        """
        Run object removal.
        
        Args:
            image: RGB image (H, W, 3), uint8
            prompt: Text prompt for auto-detection (e.g. "person", "deer")
            mask: User-provided binary mask (H, W), 255 = region to remove
            save_artifacts: Whether to save intermediate masks/edges
            output_dir: Directory for saving artifacts
        
        Returns:
            dict with keys: 'result' (cleaned image), 'mask', 'edge_map', 'boxes'
        """
        start_time = time.time()
        
        # Preserve the un-resized original for full-res compositing
        true_original = image.copy()
        
        image_size = self.config["pipeline"]["image_size"]
        image = resize_max(image, image_size)
        original = image.copy()
        
        artifacts = {"boxes": [], "mask": None, "edge_map": None}
        
        # ── Step 1: Get mask (auto-detect or user-provided) ──
        if mask is not None:
            print("[1/5] Using user-provided mask.")
            # Resize mask to match image
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            raw_mask = mask
        elif prompt:
            print(f"[1/5] Auto-detecting: '{prompt}'...")
            box_t = self.params.get("detection", {}).get("box_threshold", 0.3)
            boxes = self.detector(image, text_prompt=prompt, box_threshold=box_t)
            artifacts["boxes"] = boxes
            print(f"      Found {len(boxes)} object(s).")
            
            if not boxes:
                print("      No objects detected. Returning original.")
                return {"result": original, **artifacts}
            
            print("[2/5] Segmenting objects (MobileSAM)...")
            raw_mask = self.segmenter(image, boxes)
        else:
            raise ValueError("Either 'prompt' or 'mask' must be provided.")
        
        # ── Step 2: Refine mask ──
        print("[3/5] Refining mask & calculating context...")
        refined_mask, alpha_mask = self.mask_refiner(raw_mask)
        artifacts["mask"] = refined_mask
        artifacts["alpha_mask"] = alpha_mask
        
        # Calculate object size percentage for dynamic routing
        mask_area = np.count_nonzero(refined_mask)
        total_area = refined_mask.shape[0] * refined_mask.shape[1]
        area_pct = (mask_area / total_area) * 100
        print(f"      Object covers {area_pct:.1f}% of image.")
        if area_pct > 15.0:
            print("      Large object detected -> Routing to Big-LaMa capability.")
        else:
            print("      Small object detected -> Standard inpainting logic.")
        
        # ── Step 3: Extract edges ──
        print("[4/5] Extracting edge structure...")
        edge_map = self.edge_extractor(image, refined_mask)
        artifacts["edge_map"] = edge_map
        
        # ── Step 4: Inpaint ──
        inpaint_params = self.params.get("inpainting", {})
        print("[5/5] Inpainting (LaMa)...")
        clean_image = self.inpainter(
            image, refined_mask,
            edge_map=edge_map,
            multi_scale=inpaint_params.get("multi_scale", True),
            scales=inpaint_params.get("scales", [256, 512, max(image.shape[:2])]), # Added native resolution to scales
        )
        
        # ── Step 4.5: Diffusion Generative Refinement ──
        diffusion_params = self.params.get("diffusion", {})
        if self.diffusion_refiner is not None and diffusion_params.get("strength", 0.0) > 0.0:
            print("[5.5/5] Generative Texture Refinement (Stable Diffusion)...")
            
            # Free VRAM before diffusion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            clean_image = self.diffusion_refiner(
                image=clean_image,
                mask=refined_mask,
                prompt=diffusion_params.get("prompt", "clean natural background, high quality photo"),
                negative_prompt=diffusion_params.get("negative_prompt", "artifacts, blurry, distorted"),
                steps=diffusion_params.get("num_inference_steps", 20),
                strength=diffusion_params.get("strength", 0.35)
            )
            
            # Free VRAM after diffusion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # ── Step 5: Post-process ──
        print("      Post-processing (native-res full compositing + color match)...")
        # Pass the original un-resized image and the working-res modified mask/patch
        final_image = self.post_processor(
            true_original=true_original,  # Important: The un-scaled input image
            inpainted=clean_image, 
            mask=refined_mask, 
            alpha_mask=alpha_mask
        )
        
        elapsed = time.time() - start_time
        print(f"\n      ✅ Done in {elapsed:.2f}s")
        
        # Save artifacts if requested
        if save_artifacts:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "mask.png"), refined_mask)
            # Save alpha mask visualizing the gradient
            cv2.imwrite(os.path.join(output_dir, "alpha_mask.png"), (alpha_mask * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(output_dir, "edges.png"), edge_map)
        
        return {"result": final_image, **artifacts}


# ═════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AI Object Removal Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path")
    parser.add_argument("--config", "-c", type=str,
                        default=os.path.join(PROJECT_DIR, "configs", "model_config.yaml"),
                        help="Config file")
    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="Text prompt describing what to remove (e.g., 'person', 'deer')")
    parser.add_argument("--mask", "-m", type=str, default=None,
                        help="Path to user-provided binary mask (white=remove)")
    parser.add_argument("--save-artifacts", action="store_true",
                        help="Save intermediate masks and edge maps")
    args = parser.parse_args()

    if not args.prompt and not args.mask:
        parser.error("Either --prompt or --mask must be provided.")

    # Initialize pipeline
    pipeline = ObjectRemovalPipeline(config_path=args.config)

    # Load image
    image = load_image(args.input)

    # Load user mask if provided
    user_mask = None
    if args.mask:
        user_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if user_mask is None:
            raise FileNotFoundError(f"Could not load mask: {args.mask}")

    # Run
    out_dir = pipeline.config["paths"]["output"].get("dir", "output")
    result = pipeline.run(
        image, prompt=args.prompt, mask=user_mask,
        save_artifacts=args.save_artifacts, output_dir=out_dir,
    )

    # Save output
    os.makedirs(out_dir, exist_ok=True)
    name, ext = os.path.splitext(os.path.basename(args.input))
    out_path = args.output if args.output else os.path.join(out_dir, f"{name}_clean{ext}")
    save_image(result["result"], out_path)


if __name__ == "__main__":
    main()
