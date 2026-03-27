"""
Unified image utilities for the AI Image Dashboard.
"""
import cv2
import numpy as np


def bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded file bytes to an RGB numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode uploaded image.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    """Convert an RGB numpy array to PNG bytes for Streamlit download."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("Failed to encode image to PNG.")
    return buffer.tobytes()


def resize_safe(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize an image so the longest side is at most max_size, preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w = max(8, int(w * scale) // 8 * 8)
    new_h = max(8, int(h * scale) // 8 * 8)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
