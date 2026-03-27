import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
try:
    import lpips
except ImportError:
    lpips = None

# Initialize LPIPS model lazily if installed
_lpips_vgg = None

def get_psnr(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise Ratio.
    Higher is better.
    """
    return compute_psnr(image_true, image_test, data_range=255)

def get_ssim(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """
    Computes Structural Similarity Index.
    Higher (closer to 1) is better.
    """
    # Using smaller win_size for smaller images if necessary, 
    # but 7 is standard for skimage SSIM with multichannel
    min_dim = min(image_true.shape[0], image_true.shape[1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
        
    return compute_ssim(image_true, image_test, channel_axis=-1, data_range=255, win_size=win_size)

def get_lpips(image_true: torch.Tensor, image_test: torch.Tensor, device: str = "cpu") -> float:
    """
    Computes Learned Perceptual Image Patch Similarity (LPIPS).
    Lower is better.
    
    Requires inputs to be [1, C, H, W] Tensors in range [-1, 1].
    (If passed as [0, 1] normalized, we convert it here).
    """
    global _lpips_vgg
    if lpips is None:
        print("LPIPS package not installed. Returning 0.0.")
        return 0.0
        
    if _lpips_vgg is None:
        _lpips_vgg = lpips.LPIPS(net='vgg').to(device)
        
    # Ensure inputs are normalized [-1, 1] for LPIPS as expected by the library
    # Assuming input is [0.0, 1.0]
    if image_true.max() <= 1.0:
        img_true = image_true * 2.0 - 1.0
        img_test = image_test * 2.0 - 1.0
    else:
        # Assuming input is [0, 255]
        img_true = (image_true / 255.0) * 2.0 - 1.0
        img_test = (image_test / 255.0) * 2.0 - 1.0
        
    with torch.no_grad():
        score = _lpips_vgg(img_true.to(device), img_test.to(device))
        
    return score.item()
