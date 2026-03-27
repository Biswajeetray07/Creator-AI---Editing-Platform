"""
BiRefNet — 2024 SOTA Salient Object Detection Model
Loaded from HuggingFace: ZhengPeng7/BiRefNet

Uses the OFFICIAL inference pipeline exactly as documented:
  1. Resize to 1024x1024 (no padding)
  2. ToTensor + ImageNet normalize
  3. Forward pass → outputs[-1].sigmoid()
  4. Resize mask back to original size
"""
import torch
import numpy as np
import logging
from PIL import Image
from torchvision import transforms

from bg_models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BiRefNetModel(BaseModel):
    """BiRefNet Salient Object Detection — finds the main subject(s) in an image."""

    def __init__(self):
        super().__init__()
        self._transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load(self):
        """Load BiRefNet from HuggingFace (~800MB, auto-downloaded on first use)."""
        from transformers import AutoModelForImageSegmentation

        model_id = "ZhengPeng7/BiRefNet"
        logger.info(f"Downloading/loading BiRefNet from HuggingFace: {model_id}")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Use half precision on CUDA for speed
        if self.device.type == "cuda":
            self.model.half()

        self._loaded = True

    def predict(self, tensor_input: torch.Tensor, original_size: tuple) -> np.ndarray:
        """
        Run BiRefNet inference on a preprocessed tensor.

        Args:
            tensor_input: Preprocessed tensor [1, 3, 1024, 1024]
            original_size: (W, H) of the original image (PIL convention)

        Returns:
            Soft probability map as numpy array [H, W] in [0.0, 1.0]
        """
        self.ensure_loaded()

        with torch.no_grad():
            # Match model dtype (float16 on CUDA, float32 on CPU)
            dtype = next(self.model.parameters()).dtype
            tensor_input = tensor_input.to(device=self.device, dtype=dtype)

            preds = self.model(tensor_input)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()

            # Convert to PIL and resize to original image dimensions
            pred_pil = transforms.ToPILImage()(pred)
            mask_pil = pred_pil.resize(original_size, Image.BILINEAR)

            return np.array(mask_pil).astype(np.float32) / 255.0

    def predict_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convenience method: run BiRefNet on a PIL image directly.
        This is the OFFICIAL way to use BiRefNet.

        Args:
            pil_image: Input PIL Image (RGB)

        Returns:
            Soft probability map [H, W] in [0.0, 1.0]
        """
        self.ensure_loaded()
        original_size = pil_image.size  # (W, H)

        tensor_input = self._transform(pil_image).unsqueeze(0)
        return self.predict(tensor_input, original_size)
