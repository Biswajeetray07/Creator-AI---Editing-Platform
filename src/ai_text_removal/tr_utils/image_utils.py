import cv2
import numpy as np
import torch
from PIL import Image

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from path using OpenCV.
    Converts from BGR (OpenCV default) to RGB.
    Returns: numpy array of shape (H, W, 3) in RGB format.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resizes an image while preserving the aspect ratio,
    ensuring the longest side is at most `max_size`.
    """
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        # Ensure dimensions are multiples of 8 for model compatibility
        new_w = new_w - (new_w % 8)
        new_h = new_h - (new_h % 8)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def normalize_to_tensor(image: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Normalizes an image [0, 255] to [0.0, 1.0] and converts to a PyTorch Tensor
    of shape (1, C, H, W).
    """
    img_tensor = torch.from_numpy(image).float() / 255.0
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # (H, W, C) -> (1, C, H, W)
    elif len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
        
    return img_tensor.to(device)

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch Tensor (1, C, H, W) back to a numpy array [0, 255] in (H, W, C).
    """
    img = tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def save_image(image: np.ndarray, output_path: str):
    """
    Saves an RGB numpy array to disk.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)
