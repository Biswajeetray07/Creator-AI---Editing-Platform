"""
Debug visualization utilities — overlay mask, show trimap.
"""
import numpy as np
import cv2


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 color: tuple = (0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a mask on the image with a colored tint.

    Args:
        image_rgb: Input image (H, W, 3) uint8
        mask: Mask [H, W] in [0.0, 1.0] or bool
        color: RGB color for the overlay
        alpha: Transparency of the overlay

    Returns:
        Image with mask overlay (H, W, 3) uint8
    """
    overlay = image_rgb.copy()
    mask_bool = mask > 0.5 if mask.dtype in [np.float32, np.float64] else mask.astype(bool)

    colored = np.zeros_like(image_rgb)
    colored[:] = color

    overlay[mask_bool] = cv2.addWeighted(
        overlay[mask_bool], 1 - alpha,
        colored[mask_bool], alpha, 0,
    )
    return overlay


def visualize_trimap(trimap: np.ndarray) -> np.ndarray:
    """
    Visualize a trimap as a colored image.
    FG = Green, BG = Red, Unknown = Yellow

    Args:
        trimap: Trimap (H, W) uint8 with values {0, 128, 255}

    Returns:
        Colored visualization (H, W, 3) uint8
    """
    vis = np.zeros((*trimap.shape, 3), dtype=np.uint8)
    vis[trimap == 0] = [200, 50, 50]      # BG → Red
    vis[trimap == 128] = [255, 255, 50]    # Unknown → Yellow
    vis[trimap == 255] = [50, 200, 50]     # FG → Green
    return vis


def show_alpha_checkerboard(image_rgba: np.ndarray, check_size: int = 16) -> np.ndarray:
    """
    Create a checkerboard background for RGBA preview.

    Args:
        image_rgba: RGBA image (H, W, 4) uint8
        check_size: Size of checkerboard squares

    Returns:
        RGB preview image (H, W, 3) uint8
    """
    h, w = image_rgba.shape[:2]

    # Create checkerboard
    checker = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, check_size):
        for x in range(0, w, check_size):
            if ((y // check_size) + (x // check_size)) % 2 == 0:
                checker[y:y+check_size, x:x+check_size] = [240, 240, 240]
            else:
                checker[y:y+check_size, x:x+check_size] = [200, 200, 200]

    # Composite
    alpha = image_rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = image_rgba[:, :, :3].astype(np.float32)
    result = rgb * alpha + checker.astype(np.float32) * (1.0 - alpha)

    return result.astype(np.uint8)
