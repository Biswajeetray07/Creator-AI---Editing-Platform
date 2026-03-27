import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as RGB numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, path: str) -> None:
    """Save an RGB numpy array to disk."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    print(f"  → Saved: {path}")


def resize_max(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image so that the longest side is at most max_size."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
