"""
Stage 3b — SAM Box-Prompted Segmentation (COMPLEX path)
Extracts bounding boxes from BiRefNet's coarse mask and prompts SAM
for pixel-perfect binary masks per subject.
"""
import numpy as np
import cv2
import time
import logging

from bg_models.sam.sam_predictor import SamSegmenter

logger = logging.getLogger(__name__)


class SegmentationRefiner:
    """
    Stage 3b: SAM-based segmentation for COMPLEX scenes.
    
    Workflow:
        1. Extract individual bounding boxes from BiRefNet's coarse mask
        2. Prompt SAM with each bounding box → pixel-perfect binary mask per subject
        3. Union all per-subject masks
        4. Failsafe: union with BiRefNet binary mask (ensures no subjects are lost)
    """

    def __init__(self):
        self.sam = SamSegmenter()

    def load(self):
        """Eagerly load SAM model."""
        self.sam.ensure_loaded()

    def _extract_bboxes(self, soft_mask: np.ndarray, min_area_ratio: float = 0.005) -> list:
        """
        Extract bounding boxes from BiRefNet's coarse mask using connected components.

        Args:
            soft_mask: Soft probability mask [H, W] in [0.0, 1.0]
            min_area_ratio: Minimum blob area as ratio of total image size

        Returns:
            List of [x1, y1, x2, y2] bounding boxes
        """
        binary = (soft_mask > 0.5).astype(np.uint8)
        h, w = binary.shape
        total_area = h * w

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        bboxes = []
        for i in range(1, num_components):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area / total_area < min_area_ratio:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            # Add padding (10% of bbox size)
            pad_x = int(bw * 0.1)
            pad_y = int(bh * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)

            bboxes.append([x1, y1, x2, y2])

        logger.info(f"Extracted {len(bboxes)} bounding boxes from BiRefNet mask")
        return bboxes

    def refine(self, image_rgb: np.ndarray, soft_mask: np.ndarray) -> tuple:
        """
        Run SAM box-prompted segmentation on COMPLEX scenes.

        Args:
            image_rgb: Input image (H, W, 3) uint8 RGB
            soft_mask: BiRefNet's soft probability mask [H, W]

        Returns:
            (refined_mask, latency_ms) where:
                refined_mask: Binary mask (H, W) float32 in [0.0, 1.0]
                latency_ms: float
        """
        start = time.time()

        # Step 1: Extract bounding boxes from BiRefNet mask
        bboxes = self._extract_bboxes(soft_mask)

        if not bboxes:
            # Fallback: use BiRefNet binary mask directly
            logger.warning("No bounding boxes extracted — using BiRefNet mask directly")
            binary = (soft_mask > 0.5).astype(np.float32)
            latency_ms = (time.time() - start) * 1000
            return binary, latency_ms

        # Step 2-3: SAM predict per-box and union all masks
        sam_mask = self.sam.predict_boxes(image_rgb, bboxes)

        # Step 4: Failsafe — union with BiRefNet binary mask
        birefnet_binary = soft_mask > 0.5
        final_mask = (sam_mask | birefnet_binary).astype(np.float32)

        latency_ms = (time.time() - start) * 1000
        logger.info(f"Stage 3b (SAM Segmentation): {latency_ms:.0f}ms ({len(bboxes)} boxes)")

        return final_mask, latency_ms
