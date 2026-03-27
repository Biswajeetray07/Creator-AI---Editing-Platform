"""
MODNet Alpha Matting Inference
Provides the ModNetMatting class for running alpha matting predictions.
Key design: processes the FULL image (not a tight crop) for global context.
"""
import torch
import torch.nn.functional as F
import numpy as np
import logging

from bg_models.modnet.modnet_loader import load_modnet_model
from bg_models.modnet.config import (
    DEVICE, REF_SIZE, MAX_INFERENCE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD,
)

logger = logging.getLogger(__name__)


class ModNetMatting:
    """
    MODNet alpha matting inference wrapper.
    Caps inference at 2048px to preserve hair details on 4K+ images.
    """

    def __init__(self):
        self.model = None
        self._loaded = False

    def load(self):
        """Load MODNet model."""
        self.model = load_modnet_model()
        self._loaded = True
        logger.info("ModNetMatting ready.")

    def ensure_loaded(self):
        if not self._loaded:
            self.load()

    def _preprocess(self, image_rgb: np.ndarray) -> tuple:
        """
        Preprocess image for MODNet inference.
        - Scale to ref_size, pad to div-by-32, normalize (x − 0.5) / 0.5

        Args:
            image_rgb: Input image (H, W, 3) uint8

        Returns:
            (tensor, padding_info) for postprocessing
        """
        h, w = image_rgb.shape[:2]

        # Cap at max inference size (preserves hair details on 4K+)
        if max(h, w) > MAX_INFERENCE_SIZE:
            scale = MAX_INFERENCE_SIZE / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
        elif min(h, w) < REF_SIZE:
            scale = REF_SIZE / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            new_h, new_w = h, w

        # Pad to divisible by 32
        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32
        padded_h = new_h + pad_h
        padded_w = new_w + pad_w

        # Resize
        import cv2
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Normalize: (x / 255.0 - 0.5) / 0.5
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - NORMALIZE_MEAN) / NORMALIZE_STD

        # HWC → CHW
        tensor = tensor.transpose(2, 0, 1)
        tensor = torch.from_numpy(tensor).unsqueeze(0).float()

        # Pad
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        return tensor, {
            "original_size": (h, w),
            "resized_size": (new_h, new_w),
            "padded_size": (padded_h, padded_w),
        }

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Run MODNet alpha matting on the full image.

        Args:
            image_rgb: Input image (H, W, 3) uint8 RGB

        Returns:
            Alpha matte (H, W) in [0.0, 1.0]
        """
        self.ensure_loaded()

        tensor, meta = self._preprocess(image_rgb)
        tensor = tensor.to(device=DEVICE, dtype=next(self.model.parameters()).dtype)

        with torch.no_grad():
            # MODNet inference mode
            alpha = self.model(tensor, inference=True)  # [1, 1, H, W]

        # Remove padding
        orig_h, orig_w = meta["original_size"]
        res_h, res_w = meta["resized_size"]
        alpha = alpha[:, :, :res_h, :res_w]

        # Resize back to original
        alpha = F.interpolate(
            alpha, size=(orig_h, orig_w),
            mode="bilinear", align_corners=False,
        )

        alpha = alpha.squeeze().cpu().numpy()
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)
