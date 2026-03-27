import os
import numpy as np
import cv2
import torch
from PIL import Image

class DiffusionRefiner:
    """
    Stable Diffusion Inpainting Refiner.
    Enhances LaMa output with generative textures for photorealism.
    Loaded lazily to save VRAM when not needed.
    """

    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting", device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_id = model_id
        self.pipe = None

        try:
            from diffusers import StableDiffusionInpaintPipeline
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.pipe = self.pipe.to(self.device)
            # Enable memory-efficient attention if available
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            print(f"[DiffusionRefiner] Loaded {model_id} on {self.device}.")
        except Exception as e:
            print(f"[DiffusionRefiner] Failed to load: {e}. Diffusion refinement disabled.")

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "clean natural background, high quality photo",
        negative_prompt: str = "artifacts, blurry, distorted",
        steps: int = 30,
        strength: float = 0.75,
    ) -> np.ndarray:
        """
        Args:
            image:           RGB image (H, W, 3), uint8
            mask:            Binary mask (H, W), 255 = region to refine
            prompt:          Conditioning text for generation
            negative_prompt: Negative conditioning
            steps:           Number of diffusion steps
            strength:        How much to alter the image (0.0 = none, 1.0 = full)
        Returns:
            Refined RGB image (H, W, 3), uint8
        """
        if self.pipe is None:
            return image

        h, w = image.shape[:2]

        # Convert to PIL and ensure dimensions are multiples of 8 (required for SD)
        # We resize slightly to the nearest multiple of 8 to avoid aspect ratio distortion
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8

        # If it's too large for standard SD Inpainting, maybe bound it, but pipeline already resize_max(512)
        pil_image = Image.fromarray(image).resize((new_w, new_h), Image.Resampling.LANCZOS)
        pil_mask = Image.fromarray(mask).resize((new_w, new_h), Image.Resampling.NEAREST)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            height=new_h,
            width=new_w,
            num_inference_steps=steps,
            strength=strength,
        ).images[0]

        # Resize back to original exact dimensions
        result_np = np.array(result.resize((w, h), Image.Resampling.LANCZOS))
        
        # Only overwrite the masked area to preserve original image sharp pixels (like people's faces)
        # We use a soft blur on the mask to ensure seamless blending without hard circles
        soft_mask = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
        soft_mask_3d = np.expand_dims(soft_mask, axis=-1)
        
        final_image = (image.astype(np.float32) * (1 - soft_mask_3d) + result_np.astype(np.float32) * soft_mask_3d)
        return np.clip(final_image, 0, 255).astype(np.uint8)
