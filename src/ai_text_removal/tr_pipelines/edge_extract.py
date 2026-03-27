import cv2
import numpy as np


class EdgeExtractor:
    """
    Extracts structural edge information from the image using Canny edge detection.
    The edge map is masked to remove text edges, leaving only the surrounding
    structure to guide inpainting reconstruction.
    
    This implements the "Structure-Guided Inpainting" technique used in
    research models like EdgeConnect and MTRNet++.
    """

    def __init__(self, canny_low: int = 50, canny_high: int = 150):
        self.canny_low = canny_low
        self.canny_high = canny_high

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract edges from the image and zero-out edges inside the text mask.

        Args:
            image (np.ndarray): RGB image (H, W, 3).
            mask  (np.ndarray): Binary mask where 255 = text region.

        Returns:
            np.ndarray: Edge map (H, W), uint8, with text-region edges removed.
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # Ensure mask is binary (0 or 255) and 2D
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        mask_binary = (mask > 127).astype(np.uint8)

        # Zero out edges that fall inside the text mask.
        # We only want structural edges from the *surrounding* background
        # so the inpainter knows how to continue lines/textures.
        structure_map = edges * (1 - mask_binary)

        # Optionally dilate the surviving edges slightly so the inpainter
        # has a stronger structural signal at mask boundaries.
        kernel = np.ones((3, 3), np.uint8)
        structure_map = cv2.dilate(structure_map, kernel, iterations=1)

        return structure_map
