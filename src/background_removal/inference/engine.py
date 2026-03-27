"""
BackgroundRemovalEngine — Production-grade background removal using BiRefNet.

Uses the official BiRefNet inference pipeline:
    1. BiRefNet → high-quality soft probability mask
    2. Refinement → small-island removal + edge smoothing
    3. Compositing → RGBA with transparent background + edge defringe
"""
import numpy as np
import cv2
import time
import logging
import sys
import os

_BG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BG_DIR not in sys.path:
    sys.path.insert(0, _BG_DIR)

from bg_pipeline.salient_detection import SalientDetector

logger = logging.getLogger(__name__)


class BackgroundRemovalEngine:
    """
    Production-grade background removal engine using BiRefNet.

    Usage:
        engine = BackgroundRemovalEngine()
        result = engine.run(image_rgb)
        rgba = result["rgba"]
    """

    def __init__(self, load_sam: bool = False):
        self.salient_detector = SalientDetector()
        self._loaded = False

    def load(self):
        """Load models into VRAM."""
        logger.info("Loading Background Removal Engine...")
        start = time.time()
        self.salient_detector.load()
        load_time = time.time() - start
        logger.info(f"Engine loaded in {load_time:.1f}s")
        self._loaded = True

    def ensure_loaded(self):
        if not self._loaded:
            self.load()

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine BiRefNet's soft mask into a clean alpha matte.

        Steps:
            1. Threshold sharpening — push uncertain pixels toward 0/1
            2. Small island removal (foreground noise)
            3. Small hole filling (background holes in foreground)
            4. Gaussian edge smoothing
        """
        alpha = mask.copy().astype(np.float32)

        # 1. Threshold sharpen — clean up boundary uncertainty
        #    Pixels > 0.5 are foreground, < 0.5 are background
        strength = 8.0
        x = strength * (alpha - 0.5)
        alpha = 1.0 / (1.0 + np.exp(-x))
        
        # CRITICAL FIX: The sigmoid function never reaches true 0.0 or 1.0.
        # This meant the background hovered around ~2% opacity causing dirty patches!
        # We must clamp definite background and foreground to exact absolute values.
        alpha[alpha < 0.05] = 0.0
        alpha[alpha > 0.95] = 1.0
        
        alpha = alpha.astype(np.float32)

        # 2. Remove small foreground islands (noise)
        binary_fg = (alpha > 0.5).astype(np.uint8)
        min_area = max(200, int(mask.shape[0] * mask.shape[1] * 0.001))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_fg, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                alpha[labels == i] = 0.0

        # 3. Fill small background holes inside foreground
        binary_bg = (alpha <= 0.5).astype(np.uint8)
        num_labels_bg, labels_bg, stats_bg, _ = cv2.connectedComponentsWithStats(binary_bg, connectivity=8)
        for i in range(1, num_labels_bg):
            if stats_bg[i, cv2.CC_STAT_AREA] < min_area:
                alpha[labels_bg == i] = 1.0

        # 4. Subtle edge smoothing
        smoothed = cv2.GaussianBlur(alpha, (5, 5), 0.8)
        edge_zone = (alpha > 0.05) & (alpha < 0.95)
        alpha[edge_zone] = smoothed[edge_zone]

        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def _defringe(self, image_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Remove background color bleed from semi-transparent edges.
        This prevents halos around the subject.
        """
        result = image_rgb.copy().astype(np.float32)
        edge_mask = (alpha > 0.02) & (alpha < 0.95)

        if edge_mask.sum() == 0:
            return result.astype(np.uint8)

        # Estimate background color from definite background pixels
        bg_mask = alpha < 0.05
        if bg_mask.sum() > 50:
            bg_color = np.median(result[bg_mask], axis=0)
        else:
            # Fallback: use image border pixels
            border = np.concatenate([
                result[0, :], result[-1, :], result[:, 0], result[:, -1]
            ])
            bg_color = np.median(border, axis=0)

        # Unpremultiply: remove background color contribution from edge pixels
        for c in range(3):
            a = np.clip(alpha, 0.01, 1.0)
            defringed = (result[:, :, c] - bg_color[c] * (1.0 - a)) / a
            result[:, :, c] = np.where(edge_mask, np.clip(defringed, 0, 255), result[:, :, c])

        return np.clip(result, 0, 255).astype(np.uint8)

    def _composite_rgba(self, image_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Create final RGBA output with transparent background."""
        defringed = self._defringe(image_rgb, alpha)
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        rgba = np.dstack([defringed, alpha_uint8])
        return rgba

    def run(self, image_rgb: np.ndarray, force_mode: str = None) -> dict:
        """
        Run background removal on an RGB image.

        Args:
            image_rgb: Input image (H, W, 3) uint8 RGB
            force_mode: Ignored (kept for API compatibility)

        Returns:
            dict with "rgba", "alpha", "soft_mask", "scene_type", "metrics"
        """
        self.ensure_loaded()
        total_start = time.time()
        metrics = {}
        h, w = image_rgb.shape[:2]

        # Stage 1: BiRefNet salient object detection → soft mask [0.0, 1.0]
        soft_mask, birefnet_ms = self.salient_detector.predict(image_rgb)
        metrics["stage2_birefnet_ms"] = birefnet_ms

        logger.info(
            f"BiRefNet mask stats: min={soft_mask.min():.3f}, max={soft_mask.max():.3f}, "
            f"mean={soft_mask.mean():.3f}, shape={soft_mask.shape}"
        )

        # Stage 2: Refine mask → clean alpha
        refine_start = time.time()
        alpha = self._refine_mask(soft_mask)
        metrics["stage4_matting_ms"] = (time.time() - refine_start) * 1000
        metrics["stage3_routing_ms"] = 0.0
        metrics["stage3b_sam_ms"] = 0.0

        # Stage 3: Composite RGBA
        composite_start = time.time()
        rgba = self._composite_rgba(image_rgb, alpha)
        metrics["stage5_postprocess_ms"] = (time.time() - composite_start) * 1000

        total_ms = (time.time() - total_start) * 1000
        metrics["total_ms"] = total_ms

        logger.info(f"Background removal complete: {total_ms:.0f}ms ({h}x{w})")

        return {
            "rgba": rgba,
            "alpha": alpha,
            "soft_mask": soft_mask,
            "scene_type": "BIREFNET",
            "scene_details": {},
            "metrics": metrics,
        }

    def run_simple(self, image_rgb: np.ndarray) -> dict:
        return self.run(image_rgb)

    def run_complex(self, image_rgb: np.ndarray) -> dict:
        return self.run(image_rgb)
