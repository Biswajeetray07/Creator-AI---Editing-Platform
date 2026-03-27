"""
LaMa Inpainting — Pure structural reconstruction.

Returns raw LaMa output for the FULL image.
Compositing is handled by the pipeline orchestrator.
"""
import os
import cv2
import numpy as np
import torch


class LaMaInpainter:
    """
    Big-LaMa inpainting model. Reconstructs structure in masked regions.
    Returns the raw LaMa output — compositing is done externally.
    """

    def __init__(self, model_path: str, std_model_path: str = None, device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = None

        def load_model(path, name):
            if path and os.path.exists(path):
                try:
                    m = torch.jit.load(path, map_location=self.device)
                    m.eval()
                    print(f"[Inpaint] ✅ Loaded {name} on {self.device}")
                    return m
                except Exception as e:
                    print(f"[Inpaint] ⚠ Failed to load {name}: {e}")
            return None

        self.model = load_model(model_path, "Big-LaMa")
        if self.model is None and std_model_path:
            self.model = load_model(std_model_path, "Standard LaMa")

        if self.model is None:
            print("[Inpaint] ⚠ No LaMa model. Using OpenCV fallback.")

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint masked regions.
        
        Args:
            image: RGB uint8 (H, W, 3)
            mask:  Binary uint8 (H, W), 255 = fill
        Returns:
            Full inpainted image RGB uint8 (H, W, 3)
        """
        if self.model is None:
            return self._cv2_inpaint(image, mask)
        return self._lama_forward(image, mask)

    def _lama_forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run LaMa at native resolution with proper padding."""
        h, w = image.shape[:2]

        # Pad to multiples of 8 (FFC requirement)
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8

        if pad_h > 0 or pad_w > 0:
            img_p = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            msk_p = np.pad(mask,  ((0, pad_h), (0, pad_w)),          mode='constant', constant_values=0)
        else:
            img_p = image
            msk_p = mask

        _, msk_p = cv2.threshold(msk_p, 127, 255, cv2.THRESH_BINARY)

        img_t = (torch.from_numpy(img_p.astype(np.float32))
                 .permute(2, 0, 1).unsqueeze(0) / 255.0).to(self.device)
        msk_t = (torch.from_numpy(msk_p.astype(np.float32))
                 .unsqueeze(0).unsqueeze(0) / 255.0).to(self.device)

        with torch.no_grad():
            out_t = self.model(img_t, msk_t)

        out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return out[:h, :w]

    def _cv2_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, m = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        result = cv2.inpaint(bgr, m, inpaintRadius=5, flags=cv2.INPAINT_NS)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
