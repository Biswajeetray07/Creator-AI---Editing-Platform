"""
SAM Model Loader — Registry-based loading with FP16 support for CUDA.
"""
import torch
import logging
from segment_anything import sam_model_registry

from bg_models.sam.config import MODEL_TYPE, CHECKPOINT_PATH, DEVICE, USE_FP16

logger = logging.getLogger(__name__)


def load_sam_model():
    """
    Load SAM model from checkpoint using the registry.

    Returns:
        SAM model loaded on the appropriate device.
    """
    import os
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"SAM checkpoint not found at: {CHECKPOINT_PATH}\n"
            f"Download sam_vit_b_01ec64.pth from https://github.com/facebookresearch/segment-anything "
            f"and place it in the weights/ directory."
        )

    logger.info(f"Loading SAM ({MODEL_TYPE}) from {CHECKPOINT_PATH} on {DEVICE}")

    # MPS workaround: force float32 default dtype (float64 crashes MPS)
    if DEVICE.type == "mps":
        torch.set_default_dtype(torch.float32)

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    # CUDA optimization: half-precision
    if USE_FP16 and DEVICE.type == "cuda":
        sam = sam.half()
        logger.info("SAM loaded with FP16 half-precision (CUDA)")

    sam.eval()
    logger.info(f"SAM model loaded successfully on {DEVICE}")
    return sam
