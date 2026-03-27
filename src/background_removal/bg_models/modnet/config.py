"""
MODNet — Device/path configuration.
"""
import os
import torch
import yaml

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
_MODNET_YAML = os.path.join(_CONFIG_DIR, "modnet.yaml")

if os.path.exists(_MODNET_YAML):
    with open(_MODNET_YAML, "r") as f:
        _cfg = yaml.safe_load(f)
else:
    _cfg = {}

CHECKPOINT_NAME = _cfg.get("model", {}).get("checkpoint", "modnet_photographic_portrait_matting.ckpt")

# Weight paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, CHECKPOINT_NAME)

# Preprocessing parameters
_preprocess = _cfg.get("preprocessing", {})
REF_SIZE = _preprocess.get("ref_size", 512)
MAX_INFERENCE_SIZE = _preprocess.get("max_inference_size", 2048)
NORMALIZE_MEAN = _preprocess.get("normalize_mean", 0.5)
NORMALIZE_STD = _preprocess.get("normalize_std", 0.5)

# Matting thresholds
_matting = _cfg.get("matting", {})
FOREGROUND_THRESHOLD = _matting.get("foreground_threshold", 0.92)
BACKGROUND_THRESHOLD = _matting.get("background_threshold", 0.08)
