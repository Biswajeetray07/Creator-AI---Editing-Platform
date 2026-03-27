"""
Stable Diffusion x4 Upscaler — Texture Reconstruction Module

Uses StableDiffusionUpscalePipeline to generate photorealistic textures
on top of the GAN+Transformer refined output.

This is the final refinement stage before color matching and post-processing.
Auto-downloads the model on first use (~1.7GB).
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger("creator_ai.diffusion_refiner")

try:
    from diffusers import StableDiffusionUpscalePipeline
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False


class DiffusionTextureRefiner:
    """
    Stable Diffusion x4 Upscaler for photorealistic texture generation.

    Unlike the inpainting variant used in text/object removal, this uses
    StableDiffusionUpscalePipeline which is specifically designed for
    super-resolution with texture generation.

    Key features:
    - Generates realistic textures (skin pores, fabric weave, hair strands)
    - Fills in detail that GANs hallucinate or over-smooth
    - Operates at low inference steps (20) for speed
    - Uses FP16 + attention slicing for 4GB VRAM compatibility
    """

    MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"

    def __init__(self, weights_dir: str = None, num_inference_steps: int = 20,
                 guidance_scale: float = 4.0, device: str = "cuda"):
        """
        Args:
            weights_dir: Optional local cache path for model weights
            num_inference_steps: Diffusion steps (20 = fast, 50 = max quality)
            guidance_scale: CFG scale (4.0 = subtle, 7.5 = strong)
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.pipe = None
        self._weights_dir = weights_dir

        if not HAS_DIFFUSERS:
            logger.info("[DiffusionRefiner] ⚠ diffusers not installed. Texture refinement disabled.")
            return

        # Don't load the model at init time — lazy load on first use
        # This saves VRAM when other models need to run first
        logger.info(f"[DiffusionRefiner] Ready (will load {self.MODEL_ID} on first use)")

    def _ensure_loaded(self):
        """Lazy-load the diffusion model on first actual use."""
        if self.pipe is not None:
            return True

        if not HAS_DIFFUSERS:
            return False

        try:
            logger.info(f"[DiffusionRefiner] Loading {self.MODEL_ID}...")
            cache_dir = os.path.join(self._weights_dir, "sd_upscaler") if self._weights_dir else None

            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=cache_dir,
            )
            self.pipe = self.pipe.to(self.device)

            # Memory optimizations for 4GB VRAM
            self.pipe.enable_attention_slicing(1)
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers not available, attention slicing is enough

            logger.info(f"[DiffusionRefiner] ✅ Loaded on {self.device}")
            return True
        except Exception as e:
            logger.warning(f"[DiffusionRefiner] ⚠ Failed to load: {e}. Texture refinement disabled.")
            self.pipe = None
            return False

    @property
    def available(self) -> bool:
        """Check if diffusers is installed (model loads lazily)."""
        return HAS_DIFFUSERS

    @torch.no_grad()
    def refine(self, image_bgr: np.ndarray, prompt: str = "high quality, detailed, sharp") -> np.ndarray:
        """
        Apply diffusion-based texture enhancement to a BGR image.

        The SD x4 upscaler expects a LOW-RES input and produces 4x output.
        We process the large image in tiles (512x512) to avoid OOM.
        For each tile:
        1. Downscale by 4x
        2. Feed to SD upscaler
        3. Blend SD output with original tile

        Args:
            image_bgr: BGR uint8 (H, W, 3) — already upscaled image
            prompt: Text conditioning for texture generation

        Returns:
            Texture-enhanced BGR uint8 image (same resolution)
        """
        if not self._ensure_loaded():
            logger.info("[DiffusionRefiner] Not available, returning input unchanged.")
            return image_bgr

        try:
            h, w = image_bgr.shape[:2]
            
            # Import our TileEngine to process this large image in chunks
            from stages.tile_engine import TileEngine

            # 512x512 tile is ideal. Downscaled 4x = 128x128 input to SD Upscaler.
            tile_engine = TileEngine(tile_size=512, overlap=64)
            
            tiles = tile_engine.split(image_bgr)
            processed_tiles = []
            
            logger.info(f"[DiffusionRefiner] Processing {w}×{h} in {len(tiles)} tiles (512x512) to prevent OOM...")
            
            for i, t in enumerate(tiles):
                tile_bgr = t['tile']
                th, tw = tile_bgr.shape[:2]
                tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)

                # Downscale by 4x to create "low-res" input for SD upscaler
                small_h = max(32, th // 4)
                small_w = max(32, tw // 4)
                # Ensure divisible by 8
                small_h = (small_h // 8) * 8
                small_w = (small_w // 8) * 8

                small_img = cv2.resize(tile_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
                pil_small = Image.fromarray(small_img)

                logger.info(f"        Tile {i+1}/{len(tiles)}")

                # Run SD upscaler
                result = self.pipe(
                    prompt=prompt,
                    image=pil_small,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    noise_level=20,  # Low noise = preserve structure, add texture
                ).images[0]

                result_np = np.array(result)

                # Resize SD output to match original tile dimensions
                sd_resized = cv2.resize(result_np, (tw, th), interpolation=cv2.INTER_LANCZOS4)

                # Blend: use SD for texture detail, original for structure
                blend_weight = 0.3
                blended = cv2.addWeighted(
                    tile_rgb, 1.0 - blend_weight,
                    sd_resized, blend_weight,
                    0
                )
                
                processed_tiles.append({
                    "tile": cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
                    "x": t["x"],
                    "y": t["y"],
                    "w": tw,
                    "h": th,
                })
                
            
            # Fuse back together
            output_bgr = tile_engine.fuse(processed_tiles, (h, w, 3))
            logger.info(f"[DiffusionRefiner] ✅ Texture-enhanced {w}×{h} image.")
            return output_bgr

        except torch.cuda.OutOfMemoryError:
            logger.info("[DiffusionRefiner] ⚠ CUDA OOM even with tiles. Skipping texture refinement.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return image_bgr
        except Exception as e:
            logger.error(f"[DiffusionRefiner] Error: {e}. Returning input unchanged.")
            return image_bgr

    def unload(self):
        """Free GPU memory by unloading the diffusion pipeline."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[DiffusionRefiner] Unloaded from GPU.")
