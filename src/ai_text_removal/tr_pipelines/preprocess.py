import numpy as np
from tr_utils.image_utils import resize_image, normalize_to_tensor
import torch

class ImagePreprocessor:
    """
    Preprocess image step: Resizing and potentially normalizing
    for downstream models.
    """
    def __init__(self, max_size: int = 1024, device: str = "cuda"):
        self.max_size = max_size
        self.device = device

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """
        Returns:
            - resized_image (np.ndarray): resized to max_size, maintaining aspect ratio.
            - tensor_image (torch.Tensor): normalized and moved to device.
        """
        resized_img = resize_image(image, max_size=self.max_size)
        tensor_img = normalize_to_tensor(resized_img, device=self.device)
        return resized_img, tensor_img
