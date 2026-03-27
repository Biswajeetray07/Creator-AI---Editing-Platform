import cv2
import torch
import numpy as np

def preprocess_image(image_np, max_size=512):
    """
    Resizes image to heavily restrict max dimension to `max_size` (maintain aspect ratio)
    and converts it to a normalized PyTorch tensor.
    """
    h, w = image_np.shape[:2]
    
    # Calculate resize ratio if exceeding max size
    if max(h, w) > max_size:
        ratio = max_size / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # Models often require padding or dimensions divisible by 8 or 16
        # To keep it simple and safe for Restormer, we pad or ensure it's mod 8
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # Just ensure dimensions are multiples of 8 for Restormer
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    image_float = image_np.astype(np.float32) / 255.0
    
    # Convert to Tensor, shape: (1, C, H, W)
    image_tensor = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor
