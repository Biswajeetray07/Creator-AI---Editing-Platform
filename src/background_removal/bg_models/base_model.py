"""
Abstract base class for all AI models in the background removal pipeline.
Provides a consistent load + predict interface.
"""
from abc import ABC, abstractmethod
import torch
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BaseModel(ABC):
    """Abstract base for pipeline AI models."""

    def __init__(self):
        self.device = get_device()
        self.model = None
        self._loaded = False

    @abstractmethod
    def load(self):
        """Load model weights into memory."""
        ...

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Run inference on the input."""
        ...

    def ensure_loaded(self):
        """Lazily load model on first use."""
        if not self._loaded:
            logger.info(f"Loading {self.__class__.__name__} on {self.device}...")
            self.load()
            self._loaded = True
            logger.info(f"{self.__class__.__name__} loaded successfully.")
