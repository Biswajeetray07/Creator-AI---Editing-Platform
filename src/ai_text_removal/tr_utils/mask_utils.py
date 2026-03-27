import cv2
import numpy as np

def refine_mask(mask: np.ndarray, dilate_kernel: int = 5, dilate_iter: int = 2, blur_kernel: int = 3) -> np.ndarray:
    """
    Refine a binary mask through morphological operations and smoothing.
    
    Args:
        mask (np.ndarray): Binary mask, where 255 is the region to remove.
        dilate_kernel (int): Kernel size for dilation.
        dilate_iter (int): Number of dilation iterations.
        blur_kernel (int): Kernel size for Gaussian blur smoothing.
        
    Returns:
        np.ndarray: The refined mask.
    """
    # Ensure mask is 2D
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
        
    # Ensure background is 0 and text is 255
    # (If the mask from upstream happens to be boolean, convert it)
    if mask.dtype == bool:
        mask = (mask * 255).astype(np.uint8)
        
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    
    # 1. Dilation: Expands the mask slightly to ensure text edges are covered completely
    if dilate_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        
    # 2. Closing: Fills in small holes inside the text mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Smoothing: Gaussian blur to soften the hard edges of the mask
    # This creates a soft feathered edge, which is critical for seamless inpainting
    if blur_kernel > 0:
        # Must be an odd number
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        
    # We deliberately DO NOT threshold back to binary.
    # Returning a soft mask (values from 0 to 255) ensures gradual blending
    # at the boundary between inpainted and original pixels.
    
    return mask
