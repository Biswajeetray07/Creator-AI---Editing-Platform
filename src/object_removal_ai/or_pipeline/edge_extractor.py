import cv2
import numpy as np


class EdgeExtractor:
    """
    Extracts structural edge maps from the image (excluding the masked area).
    Supports Canny and optionally HED (Holistically-Nested Edge Detection).
    These maps guide the inpainting model to reconstruct geometric structures
    like walls, roads, and table edges correctly.
    """

    def __init__(self, canny_low: int = 50, canny_high: int = 150, use_hed: bool = False):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.use_hed = use_hed

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Args:
            image: RGB image (H, W, 3).
            mask:  Binary mask (H, W), 255 = object to remove.
        Returns:
            Edge map (H, W), uint8, edges = 255.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # Zero-out edges inside the mask so only background structure remains
        edges[mask > 127] = 0

        return edges
