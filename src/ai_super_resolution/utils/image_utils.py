import cv2
import numpy as np
import os
from PIL import Image

def load_image(image_path):
    """Load image and return as RGB numpy array."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(image_rgb, output_path):
    """Save RGB numpy array as image."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_bgr)

def resize_image(image, max_size=1024):
    """Resize image preserving aspect ratio if it exceeds max_size."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
        
    scale = max_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
