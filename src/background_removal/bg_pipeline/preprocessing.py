"""
Stage 1 — Preprocessing
Scale, pad, normalize images for BiRefNet inference.
"""
import torch
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE = 1024


def preprocess_for_birefnet(image, target_size: int = TARGET_SIZE) -> tuple:
    """
    Preprocess an image for BiRefNet inference.

    Steps:
        1. Aspect-ratio scaling (longest side → 1024px, LANCZOS downscale)
        2. ImageNet normalization
        3. HWC → CHW conversion + unsqueeze to [1, 3, H, W]
        4. Zero-padding to exactly 1024×1024

    Args:
        image: PIL Image or numpy array (H, W, 3) RGB uint8
        target_size: Target size for padding (default 1024)

    Returns:
        (tensor, metadata_dict) where:
            tensor: torch.Tensor [1, 3, target_size, target_size]
            metadata: dict with 'original_size', 'scaled_size' for un-padding later
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()

    orig_h, orig_w = img_np.shape[:2]

    # Step 1: Aspect-ratio scaling (longest side → target_size)
    max_side = max(orig_h, orig_w)
    if max_side > target_size:
        scale = target_size / max_side
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_np)
        pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
        img_np = np.array(pil_img)
    else:
        new_h, new_w = orig_h, orig_w

    # Step 2: ImageNet normalization
    img_float = img_np.astype(np.float32) / 255.0
    img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD

    # Step 3: HWC → CHW
    img_chw = img_norm.transpose(2, 0, 1)  # [3, H, W]

    # Step 4: Zero-padding to target_size × target_size
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    if pad_h > 0 or pad_w > 0:
        img_chw = np.pad(
            img_chw,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )

    tensor = torch.from_numpy(img_chw).unsqueeze(0).float()  # [1, 3, 1024, 1024]

    metadata = {
        "original_size": (orig_h, orig_w),
        "scaled_size": (new_h, new_w),
        "target_size": target_size,
        "pad_h": pad_h,
        "pad_w": pad_w,
    }

    return tensor, metadata
