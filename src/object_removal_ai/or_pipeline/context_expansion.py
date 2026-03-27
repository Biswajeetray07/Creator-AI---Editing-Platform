import numpy as np
import cv2
from typing import Tuple

class ContextExpander:
    """
    Expands the bounding box of an object to capture surrounding global context.
    This gives inpainting models like LaMa context to reconstruct geometry and textures correctly.
    """
    def __init__(self, expansion_factor: float = 1.5, min_padding: int = 30):
        self.expansion_factor = expansion_factor
        self.min_padding = min_padding

    def expand_box(
        self, 
        box: Tuple[int, int, int, int], 
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Args:
            box: (x1, y1, x2, y2)
            image_shape: (H, W)
        Returns:
            Expanded box (new_x1, new_y1, new_x2, new_y2) clamped to image boundaries.
        """
        x1, y1, x2, y2 = box
        img_h, img_w = image_shape

        width = x2 - x1
        height = y2 - y1

        # Calculate padding needed to reach expansion factor, or minimum padding
        pad_x = max(int((width * self.expansion_factor - width) / 2), self.min_padding)
        pad_y = max(int((height * self.expansion_factor - height) / 2), self.min_padding)

        new_x1 = max(0, x1 - pad_x)
        new_y1 = max(0, y1 - pad_y)
        new_x2 = min(img_w, x2 + pad_x)
        new_y2 = min(img_h, y2 + pad_y)

        return (new_x1, new_y1, new_x2, new_y2)

    def crop_context(self, image: np.ndarray, mask: np.ndarray, box: Tuple[int, int, int, int]):
        """
        Crops the image and mask to the expanded bounding box for focused processing.
        Returns the cropped image, cropped mask, and the expanded box used.
        """
        h, w = image.shape[:2]
        expanded_box = self.expand_box(box, (h, w))
        ex1, ey1, ex2, ey2 = expanded_box
        
        crop_img = image[ey1:ey2, ex1:ex2]
        crop_mask = mask[ey1:ey2, ex1:ex2]
        
        return crop_img, crop_mask, expanded_box
