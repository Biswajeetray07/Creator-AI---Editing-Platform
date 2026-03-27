import torch
import cv2
import numpy as np
import os

class MidasDepthEstimator:
    """
    MiDaS Depth Estimation model to understand the structure of the scene.
    Provides a depth map which can be used to guide structure-aware inpainting.
    """
    def __init__(self, model_type="MiDaS_v21_small", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        try:
            # We load from PyTorch Hub for ease of use in this pipeline.
            # Local weights could be loaded if `model_path` is passed.
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if "small" in self.model_type:
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.default_transform
                
            print(f"[MiDaS] Loaded model {model_type} on {self.device}.")
        except Exception as e:
            print(f"[MiDaS] Failed to load model: {e}")
            self.model = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB image.
        Args:
            image: RGB image (H, W, 3).
        Returns:
            Depth map normalized 0-255 (H, W).
        """
        if self.model is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # Apply MiDaS transforms
        input_batch = self.transform(image).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize the prediction to match the original image resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Normalize to 0-255 for standard imaging
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return depth_map
