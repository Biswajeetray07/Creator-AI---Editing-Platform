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

        # If the file doesn't exist OR it's a 130-byte Git LFS pointer, download it
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1024 * 1024:
            print(f"[MobileSAM] Weights not found at {model_path}. Downloading...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                torch.hub.download_url_to_file(
                    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
                    model_path,
                    progress=True
                )
            except Exception as e:
                print(f"[MobileSAM] Download failed: {e}")
                raise RuntimeError(f"Failed to auto-download MobileSAM weights: {e}")
            print("[MobileSAM] Download complete.")

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
