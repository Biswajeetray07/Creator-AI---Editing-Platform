"""
Stage 5 — Postprocessing
Sharpen, guided filter, defringe, and composite the final RGBA output.
"""
import numpy as np
import cv2
import time
import logging

logger = logging.getLogger(__name__)


class Postprocessor:
    """
    Stage 5: Refine the alpha matte and produce the final RGBA composite.

    Steps:
        1. Sigmoid sharpening (strength=1.8) — counteracts MODNet softness
        2. Island removal — connected-component cleaning of artifacts
        3. Color despill/defringe — removes background color bleed from edges
        4. Guided filter (radius=2, ε=1e-6) — edge-aware smoothing
        5. Interior enforcement — re-enforces definite interior ≥ 50% opaque
        6. RGBA composite — merges defringed RGB + refined alpha
    """

    # Configuration
    SHARPEN_STRENGTH = 1.8
    GUIDED_FILTER_RADIUS = 2
    GUIDED_FILTER_EPS = 1e-6
    ISLAND_MIN_AREA = 100
    INTERIOR_THRESHOLD = 0.5

    def _sigmoid_sharpen(self, alpha: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid sharpening to counteract MODNet's natural softness.
        Maps alpha values through a steep sigmoid centered at 0.5.
        """
        # Sigmoid: 1 / (1 + exp(-strength * (x - 0.5)))
        x = self.SHARPEN_STRENGTH * (alpha - 0.5)
        sharpened = 1.0 / (1.0 + np.exp(-x))
        return sharpened.astype(np.float32)

    def _remove_islands(self, alpha: np.ndarray) -> np.ndarray:
        """Remove small disconnected islands (artifacts) from the alpha mask."""
        binary = (alpha > 0.5).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # Keep only components larger than threshold
        cleaned = alpha.copy()
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < self.ISLAND_MIN_AREA:
                cleaned[labels == i] = 0.0

        return cleaned

    def _color_despill(self, image_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Remove background color bleed from semi-transparent edges.
        Non-destructive: only affects pixels with alpha < 1.0.
        """
        result = image_rgb.copy().astype(np.float32)
        h, w = alpha.shape

        # Edge zone: pixels with partial transparency
        edge_mask = (alpha > 0.05) & (alpha < 0.95)

        if edge_mask.sum() == 0:
            return result.astype(np.uint8)

        # Estimate background color from low-alpha pixels
        very_low_alpha = alpha < 0.15
        if very_low_alpha.sum() > 100:
            bg_color = np.mean(result[very_low_alpha], axis=0)
        else:
            # Fallback: use image border pixels
            border_pixels = np.concatenate([
                result[0, :], result[-1, :], result[:, 0], result[:, -1]
            ])
            bg_color = np.mean(border_pixels, axis=0)

        # Remove background color contribution from edge pixels
        # Formula: defringed = (pixel - bg_color * (1 - alpha)) / alpha
        for c in range(3):
            channel = result[:, :, c]
            a = np.clip(alpha, 0.01, 1.0)  # avoid division by zero
            defringed = (channel - bg_color[c] * (1.0 - a)) / a
            channel[edge_mask] = defringed[edge_mask]
            result[:, :, c] = np.clip(channel, 0, 255)

        return result.astype(np.uint8)

    def _guided_filter(self, guide: np.ndarray, src: np.ndarray) -> np.ndarray:
        """
        Edge-aware smoothing using guided filter.
        Falls back to Gaussian blur if cv2.ximgproc is not available.
        """
        try:
            guided = cv2.ximgproc.guidedFilter(
                guide=guide.astype(np.float32),
                src=src.astype(np.float32),
                radius=self.GUIDED_FILTER_RADIUS,
                eps=self.GUIDED_FILTER_EPS,
            )
            return guided
        except (AttributeError, cv2.error):
            # Fallback: Gaussian blur
            logger.debug("cv2.ximgproc not available, using Gaussian blur fallback")
            ksize = self.GUIDED_FILTER_RADIUS * 2 + 1
            return cv2.GaussianBlur(src, (ksize, ksize), 0)

    def _enforce_interior(self, alpha: np.ndarray, original_mask: np.ndarray) -> np.ndarray:
        """Re-enforce that definite interior stays ≥ 50% opaque."""
        interior = original_mask >= 0.92
        result = alpha.copy()
        result[interior] = np.maximum(result[interior], self.INTERIOR_THRESHOLD)
        return result

    def process(self, image_rgb: np.ndarray, alpha: np.ndarray,
                original_mask: np.ndarray = None) -> tuple:
        """
        Full postprocessing pipeline.

        Args:
            image_rgb: Original image (H, W, 3) uint8 RGB
            alpha: Alpha matte [H, W] in [0.0, 1.0]
            original_mask: Original BiRefNet mask for interior enforcement

        Returns:
            (rgba_image, latency_ms) where:
                rgba_image: Final RGBA image (H, W, 4) uint8
                latency_ms: float
        """
        start = time.time()

        # Step 1: Sigmoid sharpening
        alpha = self._sigmoid_sharpen(alpha)

        # Step 2: Island removal
        alpha = self._remove_islands(alpha)

        # Step 3: Color despill/defringe
        defringed_rgb = self._color_despill(image_rgb, alpha)

        # Step 4: Guided filter (edge-aware smoothing)
        gray_guide = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        alpha = self._guided_filter(gray_guide, alpha)
        alpha = np.clip(alpha, 0.0, 1.0)

        # Step 5: Interior enforcement
        if original_mask is not None:
            alpha = self._enforce_interior(alpha, original_mask)

        # Step 6: RGBA composite
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        rgba = np.dstack([defringed_rgb, alpha_uint8])  # (H, W, 4)

        latency_ms = (time.time() - start) * 1000
        logger.info(f"Stage 5 (Postprocessing): {latency_ms:.0f}ms")

        return rgba, latency_ms
