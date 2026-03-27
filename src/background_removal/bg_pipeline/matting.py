"""
Stage 4 — Alpha Matting (MODNet)
Uses a Soft Trimap Architecture derived from BiRefNet's confidence map.

Key design: MODNet runs on the FULL image (not a tight crop) so it has
global context to distinguish sky from hair.
"""
import numpy as np
import time
import logging

from bg_models.modnet.modnet_infer import ModNetMatting
from bg_models.modnet.config import FOREGROUND_THRESHOLD, BACKGROUND_THRESHOLD

logger = logging.getLogger(__name__)


class AlphaMatting:
    """
    Stage 4: Produces fine-grained alpha transparency using MODNet.

    Soft Trimap zones:
        - Definite Foreground (mask ≥ 0.92): 100% opaque
        - Edge Zone / Uncertain (0.08 < mask < 0.92): MODNet decides
        - Definite Background (mask ≤ 0.08): 0% opaque

    Final alpha: combined = inner_body + (MODNet_alpha × edge_zone)
    """

    def __init__(self):
        self.modnet = ModNetMatting()

    def load(self):
        """Eagerly load MODNet."""
        self.modnet.ensure_loaded()

    def _build_soft_trimap(self, mask: np.ndarray) -> dict:
        """
        Build a soft trimap from BiRefNet's confidence map.

        Args:
            mask: Soft mask [H, W] in [0.0, 1.0]

        Returns:
            dict with 'foreground', 'background', 'uncertain' boolean masks
        """
        foreground = mask >= FOREGROUND_THRESHOLD    # Definite FG (inner body)
        background = mask <= BACKGROUND_THRESHOLD    # Definite BG
        uncertain = ~foreground & ~background        # Edge zone

        logger.debug(
            f"Trimap zones — FG: {foreground.sum()}, "
            f"BG: {background.sum()}, Uncertain: {uncertain.sum()}"
        )

        return {
            "foreground": foreground,
            "background": background,
            "uncertain": uncertain,
        }

    def matte(self, image_rgb: np.ndarray, mask: np.ndarray) -> tuple:
        """
        Run alpha matting with soft trimap.

        Args:
            image_rgb: Input image (H, W, 3) uint8 RGB
            mask: Confidence mask [H, W] in [0.0, 1.0] (from BiRefNet or SAM+BiRefNet)

        Returns:
            (alpha, latency_ms) where:
                alpha: Combined alpha matte [H, W] in [0.0, 1.0]
                latency_ms: float
        """
        start = time.time()

        # Build soft trimap
        trimap = self._build_soft_trimap(mask)

        # Run MODNet on full image
        modnet_alpha = self.modnet.predict(image_rgb)

        # Combine using trimap zones
        # inner_body is 100% opaque
        # edge_zone uses MODNet alpha
        # background is 0%
        combined = np.zeros_like(mask, dtype=np.float32)
        combined[trimap["foreground"]] = 1.0
        combined[trimap["uncertain"]] = modnet_alpha[trimap["uncertain"]]
        combined[trimap["background"]] = 0.0

        # Ensure values stay in [0, 1]
        combined = np.clip(combined, 0.0, 1.0)

        latency_ms = (time.time() - start) * 1000
        logger.info(f"Stage 4 (Alpha Matting): {latency_ms:.0f}ms")

        return combined, latency_ms
