"""
SwinIR Transformer Refinement Module

Applies SwinIR (Image Restoration Using Swin Transformer) after
GAN upscaling to recover fine details and fix over-smoothing.

Uses spandrel for model loading (same pattern as upsampler.py).
Auto-downloads weights on first use.
"""
import os
import cv2
import torch
import numpy as np
import logging

logger = logging.getLogger("creator_ai.swinir_refiner")

try:
    from spandrel import ModelLoader
    HAS_SPANDREL = True
except ImportError:
    HAS_SPANDREL = False

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


def _download_swinir_weights(weights_dir: str, model_name: str) -> str:
    """Download SwinIR weights from HuggingFace if not present."""
    weight_path = os.path.join(weights_dir, f"{model_name}.pth")
    if os.path.exists(weight_path):
        return weight_path

    if not HAS_HF_HUB:
        logger.info("[SwinIR] huggingface_hub not installed. Cannot download weights.")
        return None

    logger.info(f"[SwinIR] Downloading {model_name}.pth from HuggingFace...")
    try:
        # SwinIR large model for real-world SR x4
        downloaded = hf_hub_download(
            repo_id="eugenesiow/SwinIR",
            filename=f"{model_name}.pth",
            local_dir=weights_dir,
            local_dir_use_symlinks=False,
        )
        if downloaded != weight_path and os.path.exists(downloaded):
            os.rename(downloaded, weight_path)
        logger.info(f"[SwinIR] Saved weights to {weight_path}")
        return weight_path
    except Exception as e:
        logger.info(f"[SwinIR] HuggingFace download failed: {e}")
        # Fallback: try direct URL
        try:
            from urllib.request import urlretrieve

            url = (
                "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
                "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
            )
            logger.info(f"[SwinIR] Trying direct download from GitHub releases...")
            urlretrieve(url, weight_path)
            logger.info(f"[SwinIR] Saved weights to {weight_path}")
            return weight_path
        except Exception as e2:
            logger.info(f"[SwinIR] Direct download also failed: {e2}")
            return None


class SwinIRRefiner:
    """
    SwinIR-based detail refinement after GAN upscaling.

    SwinIR uses Swin Transformer blocks with shifted window attention
    to recover fine details that GANs over-smooth. It excels at:
    - Restoring texture detail (skin, fabric, hair)
    - Sharpening edges without ringing artifacts
    - Reducing GAN hallucination artifacts

    Operates as a 1x refinement pass (no resolution change).
    """

    def __init__(self, weights_dir: str, model_name: str = "SwinIR_large_x4",
                 tile_size: int = 512, device: str = "cuda"):
        """
        Args:
            weights_dir: Directory containing/to-download SwinIR weights
            model_name: Weight filename (without .pth)
            tile_size: Tile size for processing (larger = more VRAM)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tile_size = tile_size
        self.tile_pad = 16

        if not HAS_SPANDREL:
            logger.info("[SwinIR] ⚠ spandrel not installed. SwinIR refinement disabled.")
            return

        # Download weights if needed
        weight_path = _download_swinir_weights(weights_dir, model_name)
        self._weight_path = weight_path
        if weight_path is None or not os.path.exists(weight_path):
            logger.info("[SwinIR] ⚠ Weights not available. SwinIR refinement disabled.")
            return

        try:
            loader = ModelLoader(device=self.device)
            self.model = loader.load_from_file(weight_path)
            self.model.eval()

            # Use FP32 (default) to avoid scalar type Half expected errors with spandrel
            # if self.device.type == "cuda":
            #     self.model.model.half()

            logger.info(f"[SwinIR] ✅ Loaded {model_name} via spandrel on {self.device}")
        except Exception as e:
            logger.warning(f"[SwinIR] ⚠ Failed to load: {e}. SwinIR refinement disabled.")
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _load_model(self):
        """Reload model from saved weight path (after unload)."""
        if not self._weight_path or not os.path.exists(self._weight_path):
            return
        try:
            loader = ModelLoader(device=self.device)
            self.model = loader.load_from_file(self._weight_path)
            self.model.eval()
            logger.info(f"[SwinIR] ✅ Reloaded on {self.device}")
        except Exception as e:
            logger.warning(f"[SwinIR] ⚠ Reload failed: {e}")
            self.model = None

    @torch.no_grad()
    def refine(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Refine a BGR image using SwinIR for detail recovery.

        This is a 1x pass — the output has the same resolution as input.
        SwinIR acts as a detail enhancer / denoiser on the upscaled image.

        Args:
            image_bgr: BGR uint8 (H, W, 3)

        Returns:
            Refined BGR uint8 image (same resolution)
        """
        if not self.available:
            # Try to reload if we previously unloaded
            if self._weight_path and not self.model:
                self._load_model()
            if not self.available:
                logger.info("[SwinIR] Not available, returning input unchanged.")
                return image_bgr

        try:
            h, w = image_bgr.shape[:2]

            # BGR -> RGB, uint8 -> float32, HWC -> BCHW
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0) # Keep on CPU for tile loop!

            # if self.device.type == "cuda":
            #     img_t = img_t.half()

            # Tiled inference for VRAM safety
            if h > self.tile_size or w > self.tile_size:
                output_t = self._tiled_inference(img_t)
            else:
                output_t = self.model(img_t)

            # BCHW -> HWC, clamp, uint8
            # (output_t is already on CPU if it came from _tiled_inference)
            output = output_t.squeeze(0).float().clamp(0, 1).cpu().numpy()
            output = (output.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            # Ensure same size as input (SwinIR may change resolution slightly)
            out_h, out_w = output_bgr.shape[:2]
            if abs(out_h - h) > 2 or abs(out_w - w) > 2:
                output_bgr = cv2.resize(output_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

            logger.info(f"[SwinIR] ✅ Refined {w}×{h} image.")
            return output_bgr

        except Exception as e:
            logger.error(f"[SwinIR] Error: {e}. Returning input unchanged.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return image_bgr

    def _tiled_inference(self, img_t: torch.Tensor) -> torch.Tensor:
        """Process image in tiles to avoid GPU OOM."""
        _, c, h, w = img_t.shape
        tile = self.tile_size
        pad = self.tile_pad

        # Keep massive output tensor on CPU to save ~400MB of VRAM
        output = torch.zeros_like(img_t)

        tiles_y = max(1, (h + tile - 1) // tile)
        tiles_x = max(1, (w + tile - 1) // tile)
        total_tiles = tiles_y * tiles_x
        current_tile = 0

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                current_tile += 1
                logger.info(f"        Tile {current_tile}/{total_tiles}")
                
                y0 = ty * tile
                x0 = tx * tile
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)

                # Add padding
                y0p = max(y0 - pad, 0)
                x0p = max(x0 - pad, 0)
                y1p = min(y1 + pad, h)
                x1p = min(x1 + pad, w)

                input_tile = img_t[:, :, y0p:y1p, x0p:x1p].to(self.device)
                
                # Use autocast to halve activation memory and run 2x faster on RTX cards
                with torch.amp.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
                    output_tile = self.model(input_tile)

                # Remove padding from output and move back to CPU
                oy0 = y0 - y0p
                ox0 = x0 - x0p
                oy1 = oy0 + (y1 - y0)
                ox1 = ox0 + (x1 - x0)

                output[:, :, y0:y1, x0:x1] = output_tile[:, :, oy0:oy1, ox0:ox1].cpu()

         # Clear the carriage return line
        return output

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[SwinIR] Unloaded from GPU.")
