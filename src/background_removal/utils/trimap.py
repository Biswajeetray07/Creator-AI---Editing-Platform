"""
Trimap generation — erode/dilate to create FG/BG/Unknown zones.
"""
import numpy as np
import cv2


def generate_trimap(mask: np.ndarray, erode_size: int = 10, dilate_size: int = 20) -> np.ndarray:
    """
    Generate a trimap from a binary/soft mask.

    Trimap values:
        0   = Definite Background
        128 = Unknown (edge zone)
        255 = Definite Foreground

    Args:
        mask: Input mask [H, W] in [0.0, 1.0] or uint8
        erode_size: Kernel size for erosion (shrink foreground → definite FG)
        dilate_size: Kernel size for dilation (expand foreground → include edges)

    Returns:
        Trimap (H, W) uint8 with values {0, 128, 255}
    """
    # Convert to binary uint8
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        binary = (mask > 0.5).astype(np.uint8) * 255
    else:
        binary = mask.copy()

    # Erode → definite foreground
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    fg = cv2.erode(binary, erode_kernel)

    # Dilate → extend beyond edges
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    expanded = cv2.dilate(binary, dilate_kernel)

    # Build trimap
    trimap = np.zeros_like(binary, dtype=np.uint8)
    trimap[expanded > 0] = 128    # Unknown (expanded region)
    trimap[fg > 0] = 255          # Definite foreground (eroded inner)

    return trimap


def soft_trimap_from_confidence(confidence_map: np.ndarray,
                                 fg_thresh: float = 0.92,
                                 bg_thresh: float = 0.08) -> dict:
    """
    Build a soft trimap directly from BiRefNet's confidence map.
    No morphological operations — pure threshold-based.

    Returns:
        dict with 'foreground', 'background', 'uncertain' boolean arrays
    """
    return {
        "foreground": confidence_map >= fg_thresh,
        "background": confidence_map <= bg_thresh,
        "uncertain": (confidence_map > bg_thresh) & (confidence_map < fg_thresh),
    }
