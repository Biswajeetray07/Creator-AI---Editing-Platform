"""
MODNet Weight Loader — Handles DataParallel prefix stripping.
"""
import torch
import logging
import os
from collections import OrderedDict

from bg_models.modnet.arch.modnet import MODNet
from bg_models.modnet.config import CHECKPOINT_PATH, DEVICE

logger = logging.getLogger(__name__)


def load_modnet_model() -> MODNet:
    """
    Load MODNet model from checkpoint.
    Handles DataParallel prefix stripping (weights saved with nn.DataParallel
    have 'module.' prefix on all keys).

    Returns:
        MODNet model loaded on the appropriate device.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"MODNet checkpoint not found at: {CHECKPOINT_PATH}\n"
            f"Download modnet_photographic_portrait_matting.ckpt from "
            f"https://github.com/ZHKKKe/MODNet and place it in the weights/ directory."
        )

    logger.info(f"Loading MODNet from {CHECKPOINT_PATH} on {DEVICE}")

    # Create model
    modnet = MODNet(in_channels=3, hr_channels=32, backbone_pretrained=False)

    # Load weights
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    # Strip 'module.' prefix from DataParallel-saved weights
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    modnet.load_state_dict(new_state_dict, strict=False)
    modnet.to(DEVICE)
    modnet.eval()

    logger.info(f"MODNet loaded successfully on {DEVICE}")
    return modnet
