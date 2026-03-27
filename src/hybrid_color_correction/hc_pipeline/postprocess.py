import torch
import numpy as np

def postprocess_tensor(tensor):
    """
    Converts a normalized PyTorch tensor back to a uint8 NumPy array.
    Applies clipping to [0, 1] before conversion to [0, 255].
    """
    # Remove batch dimension: (1, C, H, W) -> (C, H, W)
    tensor = tensor.squeeze(0).cpu().float()
    
    # Clip values to valid range
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    # Convert to NumPy (H, W, C)
    img_np = tensor.permute(1, 2, 0).numpy()
    
    # Scale to 0-255 uint8
    img_uint8 = (img_np * 255.0).round().astype(np.uint8)
    
    return img_uint8
