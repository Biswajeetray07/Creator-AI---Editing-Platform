"""
Stable Diffusion Inpainting Refiner — MASKED ONLY.

Uses StableDiffusionInpaintPipeline with mask_image to refine 
ONLY the masked regions. Non-masked pixels are untouched by design.
"""
import numpy as np
import cv2
import torch
from PIL import Image

try:
    from diffusers import StableDiffusionInpaintPipeline
except ImportError:
    StableDiffusionInpaintPipeline = None


class DiffusionRefiner:
    """
    Stable Diffusion Inpainting for texture refinement.
    
    CORRECT USAGE:
    - Uses StableDiffusionInpaintPipeline (NOT img2img)
    - Passes mask_image to restrict generation to masked region
    - Low strength (0.25-0.35) to preserve LaMa structure  
    - Non-masked pixels are NEVER modified
    """

    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting", device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.pipe = None

        if StableDiffusionInpaintPipeline is None:
            print("[DiffusionRefiner] diffusers not installed. Skipping.")
            return

        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            
            # Memory optimization
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            
            print(f"[DiffusionRefiner] ✅ Loaded StableDiffusionInpaintPipeline on {self.device}")
        except Exception as e:
            print(f"[DiffusionRefiner] Failed to load: {e}")

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "clean natural background, realistic texture, seamless",
        negative_prompt: str = "text, letters, watermark, artifacts, blurry",
        steps: int = 25,
        strength: float = 0.30,
    ) -> np.ndarray:
        """
        Refine inpainted image using SD Inpainting (masked only).

        Args:
            image:  RGB uint8 (H, W, 3) — the LaMa output
            mask:   Binary uint8 (H, W), 255 = region to refine
            strength: 0.25-0.35 recommended. >0.5 WILL destroy the image.
        Returns:
            Refined RGB uint8 (H, W, 3) — only masked region modified
        """
        if self.pipe is None:
            print("[DiffusionRefiner] No pipeline loaded. Skipping refinement.")
            return image

        # Clamp strength to safe range
        strength = max(0.15, min(strength, 0.40))

        h, w = image.shape[:2]
        
        # Scale down for SD to prevent CUDA OOM on smaller GPUs (e.g., 4GB VRAM)
        # We limit the max dimension to 512 (SD's native training resolution)
        max_dim = 512
        scale = min(max_dim / max(h, w), 1.0)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # SD requires dimensions divisible by 8
        new_w = max(8, (new_w // 8) * 8)
        new_h = max(8, (new_h // 8) * 8)

        # Convert to PIL
        pil_image = Image.fromarray(image).resize((new_w, new_h), Image.Resampling.LANCZOS)
        pil_mask = Image.fromarray(mask).resize((new_w, new_h), Image.Resampling.NEAREST)

        # DEBUG: Verify mask integrity
        mask_np = np.array(pil_mask)
        mask_coverage = np.mean(mask_np > 127) * 100
        print(f"[DiffusionRefiner] Mask: min={mask_np.min()}, max={mask_np.max()}, "
              f"coverage={mask_coverage:.1f}%, strength={strength}", flush=True)
        
        if mask_coverage > 50:
            print(f"[DiffusionRefiner] ⚠ Mask covers {mask_coverage:.1f}% — too large! Reducing strength.", flush=True)
            strength = 0.20
        
        if mask_coverage < 1:
            print("[DiffusionRefiner] Mask is nearly empty. Skipping refinement.", flush=True)
            return image

        # Run SD Inpainting — CORRECT: passes mask_image 
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,           # ← LaMa output as base
            mask_image=pil_mask,       # ← CRITICAL: restricts to masked region
            height=new_h,
            width=new_w,
            strength=strength,         # ← Low strength preserves LaMa structure
            guidance_scale=6.0,
            num_inference_steps=steps,
        ).images[0]

        # Convert back to numpy at original resolution
        result_np = np.array(result.resize((w, h), Image.Resampling.LANCZOS))

        return result_np
