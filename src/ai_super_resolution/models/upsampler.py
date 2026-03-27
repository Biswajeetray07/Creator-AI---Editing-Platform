"""
Module 7 — RealESRGAN Upsampler (spandrel-based)

Uses spandrel to load RealESRGAN weights directly, bypassing the
basicsr dependency that's incompatible with Python 3.13.
"""
import os
import cv2
import torch
import numpy as np
import logging

logger = logging.getLogger("creator_ai.upsampler")

try:
    from spandrel import ModelLoader
    HAS_SPANDREL = True
except ImportError:
    HAS_SPANDREL = False


class Upsampler:
    """
    RealESRGAN-based upsampler using spandrel for model loading.

    spandrel auto-detects the model architecture (RRDBNet, SRVGGNet, etc.)
    from the .pth weights file — no manual architecture config needed.

    """

    def __init__(self, weights_dir: str, model_name: str = "RealESRGAN_x4plus",
                 tile_size: int = 0, tile_pad: int = 10, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.netscale = 4
        self.tile_size = tile_size
        self.tile_pad = tile_pad

        model_path = os.path.join(weights_dir, f"{model_name}.pth")

        if not os.path.exists(model_path):
            try:
                os.makedirs(weights_dir, exist_ok=True)
                from urllib.request import urlretrieve
                url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth"
                logger.info(f"[Upsampler] Downloading {model_name} weights to {model_path}...")
                urlretrieve(url, model_path)
                logger.info("[Upsampler] Download complete.")
            except Exception as e:
                logger.warning(f"[Upsampler] ⚠ Failed to download weights: {e}. Using Lanczos fallback.")
                return

        if not HAS_SPANDREL:
            logger.info("[Upsampler] ⚠ spandrel not installed. Using Lanczos fallback.")
            return

        try:
            loader = ModelLoader(device=self.device)
            self.model = loader.load_from_file(model_path)
            self.model.eval()

            # Use half precision on CUDA
            if self.device.type == "cuda":
                self.model.model.half()

            # Detect scale from model
            if hasattr(self.model, 'scale'):
                self.netscale = self.model.scale
            logger.info(f"[Upsampler] Loaded {model_name} via spandrel on {self.device} (scale={self.netscale}x)")
        except Exception as e:
            logger.warning(f"[Upsampler] ⚠ Failed to load model: {e}. Using Lanczos fallback.")
            self.model = None

    @property
    def upsampler(self):
        """Compatibility property for face_enhancer that checks if model is loaded."""
        return self.model

    def enhance(self, image_bgr: np.ndarray, outscale: int = 4) -> np.ndarray:
        """
        Upscale a BGR image using the loaded AI model.

        Args:
            image_bgr: BGR uint8 (H, W, 3)
            outscale: Output scale factor (2 or 4)

        Returns:
            Upscaled BGR image uint8
        """
        if self.model is None:
            return self._lanczos_fallback(image_bgr, outscale)

        try:
            return self._inference(image_bgr, outscale)
        except Exception as e:
            logger.info(f"[Upsampler] Error during inference: {e}. Using Lanczos fallback.")
            return self._lanczos_fallback(image_bgr, outscale)

    def _inference(self, image_bgr: np.ndarray, outscale: int) -> np.ndarray:
        """Run the actual AI super-resolution inference."""
        h, w = image_bgr.shape[:2]

        # BGR -> RGB, uint8 -> float32 [0,1], HWC -> BCHW
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        img_t = img_t.to(self.device)

        if self.device.type == "cuda":
            img_t = img_t.half()

        # Tiled inference for large images
        if self.tile_size > 0 and (h > self.tile_size or w > self.tile_size):
            output_t = self._tiled_inference(img_t)
        else:
            with torch.no_grad():
                output_t = self.model(img_t)

        # BCHW -> HWC, float32 [0,1] -> uint8, RGB -> BGR
        output = output_t.squeeze(0).float().clamp(0, 1).cpu().numpy()
        output = (output.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # If model scale != requested scale, resize to target
        out_h, out_w = output_bgr.shape[:2]
        target_h, target_w = h * outscale, w * outscale

        if abs(out_h - target_h) > 2 or abs(out_w - target_w) > 2:
            output_bgr = cv2.resize(output_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return output_bgr

    def _tiled_inference(self, img_t: torch.Tensor) -> torch.Tensor:
        """Process image in tiles to avoid GPU OOM on large images."""
        _, c, h, w = img_t.shape
        tile = self.tile_size
        pad = self.tile_pad
        scale = self.netscale

        out_h, out_w = h * scale, w * scale
        output = torch.zeros((1, c, out_h, out_w), dtype=img_t.dtype, device=img_t.device)

        tiles_y = (h + tile - 1) // tile
        tiles_x = (w + tile - 1) // tile

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Input tile with padding
                y0 = ty * tile
                x0 = tx * tile
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)

                y0p = max(y0 - pad, 0)
                x0p = max(x0 - pad, 0)
                y1p = min(y1 + pad, h)
                x1p = min(x1 + pad, w)

                input_tile = img_t[:, :, y0p:y1p, x0p:x1p]

                with torch.no_grad():
                    output_tile = self.model(input_tile)

                # Remove padding from output
                oy0 = (y0 - y0p) * scale
                ox0 = (x0 - x0p) * scale
                oy1 = oy0 + (y1 - y0) * scale
                ox1 = ox0 + (x1 - x0) * scale

                output[:, :, y0*scale:y1*scale, x0*scale:x1*scale] = output_tile[:, :, oy0:oy1, ox0:ox1]

        return output

    def _lanczos_fallback(self, image: np.ndarray, scale: int) -> np.ndarray:
        """Bicubic/Lanczos fallback when GPU inference is not available."""
        h, w = image.shape[:2]
        logger.info(f"[Upsampler] WARNING: Using Lanczos fallback (no AI enhancement)")
        return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
