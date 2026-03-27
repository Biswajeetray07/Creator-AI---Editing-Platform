import numpy as np
from tr_utils.mask_utils import refine_mask

class MaskRefiner:
    """
    Pipeline step to refine a continuous/binary mask.
    """
    def __init__(self, dilate_kernel: int = 5, dilate_iter: int = 2, blur_kernel: int = 3):
        self.dilate_kernel = dilate_kernel
        self.dilate_iter = dilate_iter
        self.blur_kernel = blur_kernel

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """
        Refines the mask.
        Args:
            mask (np.ndarray): The binary mask from SAM.
        Returns:
            np.ndarray: Refined mask.
        """
        return refine_mask(
            mask, 
            dilate_kernel=self.dilate_kernel,
            dilate_iter=self.dilate_iter,
            blur_kernel=self.blur_kernel
        )
