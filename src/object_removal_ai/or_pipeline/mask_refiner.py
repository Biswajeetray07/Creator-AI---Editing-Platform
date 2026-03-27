import cv2
import numpy as np


class MaskRefiner:
    """
    Refines raw segmentation masks using morphological operations.
    Ensures smooth edges and slightly expanded coverage to prevent
    leftover object outline artifacts after inpainting.
    """

    def __init__(
        self,
        dilate_kernel: int = 15,
        dilate_iter: int = 10,  # Increased for massive context padding
        close_kernel: int = 11,
        blur_kernel: int = 31,  # Large blur for soft alpha mask
    ):
        self.dilate_kernel = dilate_kernel
        self.dilate_iter = dilate_iter
        self.close_kernel = close_kernel
        self.blur_kernel = blur_kernel

    def __call__(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            mask: Binary mask (H, W), 255 = object.
        Returns:
            Tuple of:
            - binary_mask: Expanded hard mask for LaMa inpainting
            - alpha_mask: Soft blurred mask (0-1 float32) for edge blending
        """
        # 1. Morphological closing — fill tiny holes inside the mask
        close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel)
        )
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

        # 2. Dilation — expand mask substantially for context padding
        dilate_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.dilate_kernel, self.dilate_kernel)
        )
        expanded_mask = cv2.dilate(closed, dilate_k, iterations=self.dilate_iter)

        # Ensure binary hard mask for LaMa
        _, binary_mask = cv2.threshold(expanded_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 3. Soft Alpha Mask — generated from the expanded mask for seamless edge blending
        if self.blur_kernel > 0:
            ksize = self.blur_kernel | 1  # ensure odd
            soft_blurred = cv2.GaussianBlur(expanded_mask, (ksize, ksize), 0)
            alpha_mask = soft_blurred.astype(np.float32) / 255.0
        else:
            alpha_mask = binary_mask.astype(np.float32) / 255.0

        return binary_mask, alpha_mask
