"""
Mask utilities — morphological refinement, hole fill, island removal.
"""
import numpy as np
import cv2


def morphological_refine(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """Apply morphological closing then opening to clean a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return cleaned.astype(np.float32)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask using flood fill."""
    binary = (mask > 0.5).astype(np.uint8)
    h, w = binary.shape
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    filled = binary | (~flood.astype(bool)).astype(np.uint8)
    return filled.astype(np.float32)


def remove_small_components(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    binary = (mask > 0.5).astype(np.uint8)
    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cleaned = np.zeros_like(binary)
    for i in range(1, num_components):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    return cleaned.astype(np.float32)


def smooth_edges(mask: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """Smooth mask edges with Gaussian blur."""
    smoothed = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
    return smoothed
