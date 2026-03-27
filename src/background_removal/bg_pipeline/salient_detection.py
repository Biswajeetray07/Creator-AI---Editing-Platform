"""
Stage 2 — Salient Object Detection (BiRefNet)
Wraps the BiRefNet model for use in the pipeline.
"""
import numpy as np
import time
import logging
from PIL import Image

from bg_models.sod.birefnet_model import BiRefNetModel

logger = logging.getLogger(__name__)


class SalientDetector:
    """
    Stage 2: Runs BiRefNet to produce a soft probability map.
    Output is [H, W] float32 in [0.0, 1.0] — NOT a binary mask.
    """

    def __init__(self):
        self.model = BiRefNetModel()

    def load(self):
        """Eagerly load the model."""
        self.model.ensure_loaded()

    def predict(self, image_rgb: np.ndarray) -> tuple:
        """
        Run salient object detection on an RGB image.

        Args:
            image_rgb: Input image (H, W, 3) uint8 RGB

        Returns:
            (soft_mask, latency_ms) where:
                soft_mask: np.ndarray [H, W] in [0.0, 1.0]
                latency_ms: float — inference time in milliseconds
        """
        start = time.time()

        pil_image = Image.fromarray(image_rgb)
        soft_mask = self.model.predict_from_pil(pil_image)

        latency_ms = (time.time() - start) * 1000
        logger.info(f"Stage 2 (BiRefNet SOD): {latency_ms:.0f}ms")

        return soft_mask, latency_ms
