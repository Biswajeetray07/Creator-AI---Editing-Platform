import os
import torch
import numpy as np
import cv2
from typing import List, Tuple
from mobile_sam import sam_model_registry, SamPredictor

class MobileSamSegmenter:
    """
    MobileSAM instance segmentation.
    Fast alternative to standard SAM for Object Segmentation.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[MobileSAM] CRITICAL: Weights not found at {model_path}. "
                f"Download from: https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            )

        sam = sam_model_registry["vit_t"](checkpoint=model_path)
        sam.to(device=self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        print(f"[MobileSAM] Loaded weights from {model_path} onto {self.device}.")

    def __call__(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Takes an RGB image and a list of bounding boxes, returns a binary mask.
        Args:
            image: RGB image (H, W, 3).
            boxes: List of bounding boxes [x1, y1, x2, y2].
        Returns:
            Binary mask (H, W) where 255 indicates the object.
        """
        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)

        if not boxes:
            return final_mask

        self.predictor.set_image(image)

        input_boxes = torch.tensor(boxes, device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Combine all instance masks into one binary mask
        for i in range(masks.shape[0]):
            instance_mask = masks[i, 0].cpu().numpy()
            final_mask[instance_mask > 0] = 255

        return final_mask
