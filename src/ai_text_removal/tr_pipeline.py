"""
Production AI Text Removal Pipeline.

Pipeline:  EasyOCR → Polygon Mask → Gentle Refinement →
           Big-LaMa (structure) → SD Inpainting (texture) →
           STRICT Masked Compositing

GUARANTEE: Non-text pixels are pixel-perfect identical to the original.
"""
import os
import cv2
import yaml
import numpy as np

from tr_utils.image_utils import load_image, resize_image
from tr_pipelines.detect_text import TextDetector
from tr_pipelines.segment_mask import TextMaskGenerator
from tr_pipelines.refine_mask import MaskRefiner
from tr_pipelines.inpaint import LaMaInpainter
from tr_pipelines.diffusion_refine import DiffusionRefiner


class TextRemovalPipeline:
    def __init__(self, config_path: str = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if config_path is None:
            config_path = os.path.join(script_dir, "configs", "config.yaml")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.device = config["pipeline"]["device"]
        self.params = config["parameters"]
        
        weights = config["paths"]["weights"]
        inpaint_path = weights.get("inpainting", "big-lama.pt")
        std_inpaint_path = weights.get("std_inpainting", "lama.pt")
        
        workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        
        def resolve_model_path(path):
            local_path = os.path.join(script_dir, path)
            universal_path = os.path.join(workspace_root, "weights", os.path.basename(path))
            if os.path.exists(universal_path):
                return universal_path
            elif os.path.exists(local_path):
                return local_path
            return None
            
        inpaint_path = resolve_model_path(inpaint_path)
        std_inpaint_path = resolve_model_path(std_inpaint_path)

        print("[TextRemovalPipeline] Initializing models...")
        
        # Stage 1: Detection
        self.detector = TextDetector(device=self.device)
        self.mask_gen = TextMaskGenerator()
        
        mr = self.params.get("mask_refinement", {})
        self.refiner = MaskRefiner(
            dilate_kernel=mr.get("dilate_kernel", 7),
            dilate_iter=mr.get("dilate_iterations", 2),
            blur_kernel=mr.get("gaussian_blur", 5),
        )

        # Stage 2: LaMa Structure Reconstruction
        self.inpainter = LaMaInpainter(
            model_path=inpaint_path, 
            std_model_path=std_inpaint_path,
            device=self.device
        )

        # Stage 3: SD Texture Refinement (optional)
        self.diffusion = None
        diff_params = self.params.get("diffusion", {})
        if diff_params.get("strength", 0.0) > 0:
            diff_id = weights.get("diffusion", "runwayml/stable-diffusion-inpainting")
            self.diffusion = DiffusionRefiner(model_id=diff_id, device=self.device)

        print("[TextRemovalPipeline] ✅ Ready.")

    def run(self, image_rgb: np.ndarray, enable_diffusion: bool = True) -> dict:
        """
        Run text removal pipeline.
        
        STRICT GUARANTEES:
        1. Non-text pixels are pixel-perfect identical to the original
        2. Only detected text regions are modified
        3. No global blur, no global noise, no full-image regeneration
        """
        original = image_rgb.copy()
        h_orig, w_orig = original.shape[:2]
        print(f"[Pipeline] Input: {w_orig}×{h_orig}", flush=True)
        
        # ═══════════════════════════════════════════════════
        # STEP 1: Detect text regions
        # ═══════════════════════════════════════════════════
        max_detect = 1024
        detect_scale = 1.0
        if max(h_orig, w_orig) > max_detect:
            detect_scale = max_detect / max(h_orig, w_orig)
            detect_img = cv2.resize(image_rgb, None, fx=detect_scale, fy=detect_scale, 
                                     interpolation=cv2.INTER_AREA)
        else:
            detect_img = image_rgb
            
        polygons = self.detector(detect_img)
        if len(polygons) == 0:
            print("[Pipeline] No text detected.", flush=True)
            return {"result": original, "mask": np.zeros((h_orig, w_orig), dtype=np.uint8)}

        # Scale polygons back to original resolution
        if detect_scale < 1.0:
            polygons = [(np.array(p, dtype=np.float64) / detect_scale).astype(np.int32) 
                        for p in polygons]

        print(f"[Pipeline] Detected {len(polygons)} text regions.", flush=True)

        # ═══════════════════════════════════════════════════
        # STEP 2: Create binary mask at native resolution
        # ═══════════════════════════════════════════════════
        raw_mask = self.mask_gen(original, polygons)
        refined = self.refiner(raw_mask)
        _, binary_mask = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)
        
        mask_pct = np.count_nonzero(binary_mask > 127) / (h_orig * w_orig) * 100
        print(f"[Pipeline] Mask: {mask_pct:.1f}% coverage", flush=True)
        
        # DEBUG: Verify mask integrity
        print(f"[Pipeline] Mask min={binary_mask.min()}, max={binary_mask.max()}, "
              f"shape={binary_mask.shape}", flush=True)

        # ═══════════════════════════════════════════════════
        # STEP 3: LaMa structural inpainting (native resolution)
        # ═══════════════════════════════════════════════════
        print("[Pipeline] Running LaMa inpainting...", flush=True)
        lama_output = self.inpainter(original, binary_mask)

        # ═══════════════════════════════════════════════════
        # STEP 4: SD Inpainting refinement (MASKED ONLY)
        # ═══════════════════════════════════════════════════
        diff_params = self.params.get("diffusion", {})
        sd_strength = diff_params.get("strength", 0.0)
        
        if enable_diffusion and self.diffusion is not None and sd_strength > 0:
            print(f"[Pipeline] Running SD Inpainting (strength={sd_strength})...", flush=True)
            # SD refines the LaMa output — only inside the mask
            sd_output = self.diffusion(
                image=lama_output,          # ← LaMa output as base
                mask=binary_mask,           # ← restrict to masked region
                prompt=diff_params.get("prompt", "clean natural background, realistic texture"),
                negative_prompt=diff_params.get("negative_prompt", "text, letters, artifacts"),
                steps=diff_params.get("num_inference_steps", 25),
                strength=sd_strength,
            )
            inpainted = sd_output
        else:
            print("[Pipeline] SD refinement disabled. Using LaMa output directly.", flush=True)
            inpainted = lama_output

        # ═══════════════════════════════════════════════════
        # STEP 5: STRICT MASKED COMPOSITING
        # Non-masked pixels = pixel-perfect original
        # ═══════════════════════════════════════════════════
        result = self._strict_composite(original, inpainted, binary_mask)
        
        # Verification
        non_masked = binary_mask == 0
        changed = np.count_nonzero(np.any(result != original, axis=-1))
        masked_px = np.count_nonzero(binary_mask > 127)
        identical = np.array_equal(result[non_masked], original[non_masked])
        print(f"[Pipeline] Changed: {changed}px / Masked: {masked_px}px / "
              f"Non-masked identical: {identical}", flush=True)
        print(f"[Pipeline] ✅ Text removal complete.", flush=True)
        
        return {"result": result, "mask": binary_mask}

    @staticmethod
    def _strict_composite(original: np.ndarray, inpainted: np.ndarray, 
                          mask: np.ndarray) -> np.ndarray:
        """
        STRICT masked compositing.
        
        - Interior mask pixels: 100% from inpainted (LaMa/SD output)
        - Thin border ring (3px): alpha-blended for seamless edges
        - Exterior pixels: 100% from original (pixel-perfect)
        """
        result = original.copy()
        binary = (mask > 127).astype(np.uint8)
        
        # Erode to find safe interior (fully inside text region)
        erode_k = np.ones((5, 5), np.uint8)
        interior = cv2.erode(binary, erode_k, iterations=1)
        
        # Border ring = mask - eroded interior
        border = binary - interior
        
        # 1. Hard-copy interior from inpainted
        interior_bool = interior.astype(bool)
        result[interior_bool] = inpainted[interior_bool]
        
        # 2. Alpha-blend the thin border for seamless transition
        border_f = cv2.GaussianBlur(border.astype(np.float32), (5, 5), 1.0)
        if border_f.max() > 0:
            border_f = border_f / border_f.max()
        
        border_area = border_f > 0.01
        if np.any(border_area):
            bf3 = np.expand_dims(border_f, -1)
            blended = (original.astype(np.float32) * (1.0 - bf3) + 
                       inpainted.astype(np.float32) * bf3)
            # Apply only where border > 0
            for c in range(3):
                result[:, :, c] = np.where(border_area, 
                                            blended[:, :, c].astype(np.uint8),
                                            result[:, :, c])
        
        # 3. ABSOLUTE SAFETY: force non-masked = original
        non_masked = ~binary.astype(bool)
        result[non_masked] = original[non_masked]
        
        return result
