import cv2
import numpy as np
import os
from PIL import Image

def load_image(image_path):
    """Loads an image via OpenCV and converts it to RGB."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_image(image_rgb, output_path):
    """Saves an RGB NumPy array as an image."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)
    
def bytes_to_image(image_bytes):
    """Converts uploaded bytes (Streamlit) to a NumPy RGB array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
