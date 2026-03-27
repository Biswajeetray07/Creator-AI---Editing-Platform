"""
Image I/O utilities for the background removal pipeline.
Supports HEIC, PNG, JPG, WEBP input formats.
"""
import numpy as np
import cv2
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """
    Load an image from file path. Supports HEIC, PNG, JPG, WEBP.

    Returns:
        RGB numpy array (H, W, 3) uint8
    """
    # Try HEIC support
    if path.lower().endswith((".heic", ".heif")):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            raise ImportError("Install pillow-heif for HEIC support: pip install pillow-heif")

    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load an image from bytes. Returns RGB numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image bytes.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgba_png(image_rgba: np.ndarray, path: str):
    """Save an RGBA image as PNG with transparency."""
    img = Image.fromarray(image_rgba, "RGBA")
    img.save(path, "PNG")
    logger.info(f"Saved RGBA PNG: {path}")


def resize_for_display(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize an image for display purposes."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
