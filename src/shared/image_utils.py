"""
Shared Image Utilities for Creator AI

Consolidates common image operations used across all pipelines:
- Format conversion (RGB/BGR, uint8/float32)
- Safe resizing with aspect ratio preservation
- Image validation and normalization
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is RGB uint8 with 3 channels."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    elif image.dtype in (np.float32, np.float64):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float image [0,1] to uint8 [0,255]."""
    if image.dtype in (np.float32, np.float64):
        return np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def ensure_float32(image: np.ndarray) -> np.ndarray:
    """Convert uint8 image [0,255] to float32 [0,1]."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def safe_resize(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """
    Resize image to target_size (width, height) safely.
    
    Uses INTER_AREA for downscaling and INTER_LANCZOS4 for upscaling
    to avoid artifacts.
    """
    h, w = image.shape[:2]
    tw, th = target_size
    
    if tw < w or th < h:
        interpolation = cv2.INTER_AREA
    
    return cv2.resize(image, (tw, th), interpolation=interpolation)


def resize_max_dim(
    image: np.ndarray,
    max_dim: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize image so its largest dimension equals max_dim, preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def pad_to_multiple(
    image: np.ndarray,
    multiple: int = 8,
    pad_value: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image so dimensions are multiples of `multiple`.
    
    Returns:
        (padded_image, (pad_h, pad_w)) — padding amounts for unpadding later
    """
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)
    
    padded = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w,
        cv2.BORDER_REFLECT_101,
    )
    return padded, (pad_h, pad_w)


def unpad(image: np.ndarray, padding: Tuple[int, int]) -> np.ndarray:
    """Remove padding added by pad_to_multiple."""
    pad_h, pad_w = padding
    if pad_h == 0 and pad_w == 0:
        return image
    h, w = image.shape[:2]
    return image[:h - pad_h, :w - pad_w]
