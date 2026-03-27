"""
SAM Segmenter — Box-prompted, click-prompted, and auto-segmentation
with embedding caching for efficient multi-prediction.
"""
import numpy as np
import torch
import logging
from segment_anything import SamPredictor, SamAutomaticMaskGenerator

from bg_models.sam.sam_loader import load_sam_model
from bg_models.sam.config import (
    DEVICE, POINTS_PER_SIDE, PRED_IOU_THRESH,
    STABILITY_SCORE_THRESH, MIN_MASK_REGION_AREA, POINTS_PER_BATCH,
)

logger = logging.getLogger(__name__)


class SamSegmenter:
    """
    SAM segmentation with 3 modes: box, click, auto.
    Uses embedding caching — heavy ViT encoder runs once (~120ms),
    subsequent predictions reuse the cached embedding (<30ms each).
    """

    def __init__(self):
        self.sam_model = None
        self.predictor = None
        self.auto_generator = None
        self._loaded = False
        self._current_embedding_hash = None

    def load(self):
        """Load SAM model and create predictor."""
        self.sam_model = load_sam_model()
        self.predictor = SamPredictor(self.sam_model)
        self._loaded = True
        logger.info("SamSegmenter ready.")

    def ensure_loaded(self):
        if not self._loaded:
            self.load()

    def set_image(self, image_rgb: np.ndarray):
        """
        Set the image and compute embeddings (heavy operation, ~120ms).
        Cached — only recomputes if the image changes.
        """
        self.ensure_loaded()
        img_hash = hash(image_rgb.tobytes())
        if img_hash != self._current_embedding_hash:
            self.predictor.set_image(image_rgb)
            self._current_embedding_hash = img_hash
            logger.debug("SAM embeddings computed for new image.")

    def predict_box(self, image_rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Predict mask from a bounding box prompt.

        Args:
            image_rgb: Input image (H, W, 3) uint8 RGB
            box: Bounding box [x1, y1, x2, y2]

        Returns:
            Binary mask (H, W) as bool
        """
        self.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=True,
        )
        # Return the highest-scoring mask
        best_idx = np.argmax(scores)
        return masks[best_idx]

    def predict_click(self, image_rgb: np.ndarray, point: np.ndarray, label: int = 1) -> np.ndarray:
        """
        Predict mask from a click point prompt.

        Args:
            image_rgb: Input image (H, W, 3) uint8 RGB
            point: Click point [x, y]
            label: 1 = foreground, 0 = background

        Returns:
            Binary mask (H, W) as bool
        """
        self.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([label]),
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        return masks[best_idx]

    def predict_auto(self, image_rgb: np.ndarray) -> list:
        """
        Automatic mask generation — finds all objects in the image.

        Args:
            image_rgb: Input image (H, W, 3) uint8 RGB

        Returns:
            List of mask dicts with 'segmentation', 'area', 'bbox', etc.
        """
        self.ensure_loaded()
        if self.auto_generator is None:
            self.auto_generator = SamAutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=POINTS_PER_SIDE,
                pred_iou_thresh=PRED_IOU_THRESH,
                stability_score_thresh=STABILITY_SCORE_THRESH,
                min_mask_region_area=MIN_MASK_REGION_AREA,
                points_per_batch=POINTS_PER_BATCH,
            )
        return self.auto_generator.generate(image_rgb)

    def predict_boxes(self, image_rgb: np.ndarray, boxes: list) -> np.ndarray:
        """
        Predict and union masks from multiple bounding boxes.

        Args:
            image_rgb: Input image (H, W, 3) uint8 RGB
            boxes: List of [x1, y1, x2, y2] bounding boxes

        Returns:
            Union binary mask (H, W) as bool
        """
        self.set_image(image_rgb)
        h, w = image_rgb.shape[:2]
        union_mask = np.zeros((h, w), dtype=bool)

        for box in boxes:
            mask = self.predict_box(image_rgb, np.array(box))
            union_mask = union_mask | mask

        return union_mask
