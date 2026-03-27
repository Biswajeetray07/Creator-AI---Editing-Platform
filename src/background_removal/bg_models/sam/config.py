"""
SAM (Segment Anything Model) — Device/path configuration.
"""
import os
import torch
import yaml
import logging

logger = logging.getLogger(__name__)

# ── Device Detection ─────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# ── Config from YAML ─────────────────────────────────────────
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "configs")
_SAM_YAML = os.path.join(_CONFIG_DIR, "sam.yaml")

if os.path.exists(_SAM_YAML):
    with open(_SAM_YAML, "r") as f:
        _cfg = yaml.safe_load(f)
else:
    _cfg = {}

MODEL_TYPE = _cfg.get("model", {}).get("type", "vit_b")
CHECKPOINT_NAME = _cfg.get("model", {}).get("checkpoint", "sam_vit_b_01ec64.pth")

# Weight search paths (project weights/ directory)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, CHECKPOINT_NAME)

# Per-device tuning
_device_key = DEVICE.type
_device_cfg = _cfg.get(_device_key, {})
POINTS_PER_BATCH = _device_cfg.get("points_per_batch", 16)
USE_FP16 = _device_cfg.get("use_fp16", False)

# Auto-segmentation parameters
_auto_cfg = _cfg.get("auto", {})
POINTS_PER_SIDE = _auto_cfg.get("points_per_side", 32)
PRED_IOU_THRESH = _auto_cfg.get("pred_iou_thresh", 0.86)
STABILITY_SCORE_THRESH = _auto_cfg.get("stability_score_thresh", 0.92)
MIN_MASK_REGION_AREA = _auto_cfg.get("min_mask_region_area", 1000)
