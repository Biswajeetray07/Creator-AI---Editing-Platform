import os
import cv2
import numpy as np
import torch


class LaMaInpainter:
    """
    LaMa (Large Mask Inpainting) with multi-scale and structure-guided support.
    Uses Fast Fourier Convolutions for resolution-robust inpainting.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = None

        if not os.path.exists(model_path):
            print(f"[LaMa] WARNING: Weights not found at {model_path}.")
            print(f"[LaMa] ⚠️  Will use OpenCV TELEA fallback — quality will be lower.")
            return

        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            # Note: LaMa uses FFT convolutions which require float32 (cuFFT doesn't support fp16)
            print(f"[LaMa] Loaded weights from {model_path} onto {self.device}.")
        except Exception as e:
            print(f"[LaMa] Failed to load: {e}. Using OpenCV TELEA fallback.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        edge_map: np.ndarray = None,
        depth_map: np.ndarray = None,
        multi_scale: bool = True,
        scales: list = None,
    ) -> np.ndarray:
        """
        Structure-guided, multi-scale inpainting.

        Args:
            image      : RGB image  (H, W, 3), uint8
            mask       : Binary mask (H, W), 255 = region to fill
            edge_map   : Optional edge structure map
            depth_map  : Optional depth map (from MiDaS)
            multi_scale: Whether to run multi-scale fusion
            scales     : Resolutions for multi-scale (default [256, 512])
        Returns:
            Inpainted RGB image (H, W, 3), uint8
        """
        if scales is None:
            scales = [256, 512]

        if multi_scale:
            return self._multi_scale_inpaint(image, mask, edge_map, scales)
        return self._single_inpaint(image, mask, edge_map)

    # ------------------------------------------------------------------
    # Single-resolution inpainting
    # ------------------------------------------------------------------
    def _single_inpaint(
        self, image: np.ndarray, mask: np.ndarray, edge_map: np.ndarray = None
    ) -> np.ndarray:

        if self.model is None:
            return self._cv2_fallback(image, mask, edge_map)

        h, w = image.shape[:2]

        def pad32(s):
            return (32 - (s % 32)) % 32

        pad_h, pad_w = pad32(h), pad32(w)

        img_padded = (
            np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            if (pad_h or pad_w)
            else image
        )
        mask_padded = (
            np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
            if (pad_h or pad_w)
            else mask
        )

        img_t = (
            torch.from_numpy(img_padded)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            / 255.0
        )
        mask_t = (
            torch.from_numpy(mask_padded)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
            / 255.0
        )

        with torch.no_grad():
            out = self.model(img_t, mask_t)

        out_np = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)

        return out_np[:h, :w] if (pad_h or pad_w) else out_np

    # ------------------------------------------------------------------
    # Multi-scale inpainting with Gaussian-weighted fusion
    # ------------------------------------------------------------------
    def _multi_scale_inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        edge_map: np.ndarray = None,
        scales: list = None,
    ) -> np.ndarray:
        if scales is None:
            scales = [256, 512]

        h, w = image.shape[:2]
        accumulator = np.zeros_like(image, dtype=np.float64)
        weight_sum = 0.0
        all_scales = scales + [max(h, w)]

        for i, target_size in enumerate(all_scales):
            sf = target_size / max(h, w)
            if sf >= 1.0:
                s_img, s_mask, s_edge = image, mask, edge_map
            else:
                nw = max(8, int(w * sf) // 8 * 8)
                nh = max(8, int(h * sf) // 8 * 8)
                s_img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
                s_mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
                s_edge = (
                    cv2.resize(edge_map, (nw, nh), interpolation=cv2.INTER_NEAREST)
                    if edge_map is not None
                    else None
                )

            result = self._single_inpaint(s_img, s_mask, s_edge)
            if result.shape[:2] != (h, w):
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)

            # Higher weight for higher resolution passes (more detail)
            weight = 2.0 ** i
            accumulator += result.astype(np.float64) * weight
            weight_sum += weight

        blended = (accumulator / weight_sum).astype(np.uint8)

        # Only replace pixels within the mask
        mask_3d = np.expand_dims((mask > 127).astype(np.uint8), axis=2)
        return (image * (1 - mask_3d) + blended * mask_3d).astype(np.uint8)

    # ------------------------------------------------------------------
    # OpenCV fallback (improved: TELEA + larger radius)
    # ------------------------------------------------------------------
    def _cv2_fallback(
        self, image: np.ndarray, mask: np.ndarray, edge_map: np.ndarray = None
    ) -> np.ndarray:
        print("[LaMa] ⚠️  Using OpenCV TELEA fallback (lower quality).")
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if edge_map is not None:
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            border = cv2.subtract(dilated, mask)
            border_edges = cv2.bitwise_and(edge_map, edge_map, mask=border)
            edge_3d = cv2.cvtColor(border_edges, cv2.COLOR_GRAY2BGR)
            hint_mask = border_edges > 0
            img_bgr[hint_mask] = (
                img_bgr[hint_mask].astype(np.float32) * 0.5
                + edge_3d[hint_mask].astype(np.float32) * 0.5
            ).astype(np.uint8)

        # TELEA is generally better than Navier-Stokes for larger regions
        inpainted = cv2.inpaint(img_bgr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
