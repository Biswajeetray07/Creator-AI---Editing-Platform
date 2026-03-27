"""
Module 11 — Color Preservation (Histogram Matching)

Prevents GAN-induced color shifts by matching the output's
color distribution to the original input in LAB color space.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger("creator_ai.color_matcher")



class ColorMatcher:
    """
    Matches the color distribution of a GAN-upscaled image to the
    original input to preserve the photographer's intended color grading.
    
    Works in LAB color space to independently correct luminance and chrominance,
    preventing the common GAN issue where warm tones shift cold or vice versa.
    """

    def __init__(self, method: str = "histogram"):
        """
        Args:
            method: 'histogram' for full histogram matching,
                    'mean_std' for faster mean/std transfer
        """
        self.method = method

    def __call__(self, output: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Match colors of output to reference.
        
        Args:
            output: GAN-upscaled image (H1, W1, 3) RGB uint8
            reference: Original input image (H2, W2, 3) RGB uint8
            
        Returns:
            Color-corrected output (H1, W1, 3) RGB uint8
        """
        if self.method == "histogram":
            result = self._histogram_match_lab(output, reference)
        else:
            result = self._mean_std_transfer(output, reference)

        logger.info(f"[ColorMatcher] Applied {self.method} color matching in LAB space")
        return result

    def _histogram_match_lab(self, output: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Per-channel histogram matching in LAB color space.
        
        LAB separates luminance (L) from color (A, B), so we can
        correct color drift without destroying the AI's enhanced luminance detail.
        """
        # Convert to LAB
        out_lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Match each channel independently
        for ch in range(3):
            out_lab[:, :, ch] = self._match_channel(out_lab[:, :, ch], ref_lab[:, :, ch])

        # Convert back to RGB
        out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)

    def _match_channel(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Match the histogram of a single channel using CDF mapping.
        """
        src_flat = source.flatten()
        ref_flat = reference.flatten()

        # Compute CDFs
        src_values, src_idx, src_counts = np.unique(src_flat, return_inverse=True, return_counts=True)
        ref_values, ref_counts = np.unique(ref_flat, return_counts=True)

        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]

        ref_cdf = np.cumsum(ref_counts).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        # Map source values to reference values via CDF interpolation
        mapped_values = np.interp(src_cdf, ref_cdf, ref_values)
        return mapped_values[src_idx].reshape(source.shape)

    def _mean_std_transfer(self, output: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Faster alternative: transfer mean and std from reference to output.
        """
        out_lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)

        for ch in range(3):
            out_mean, out_std = out_lab[:, :, ch].mean(), out_lab[:, :, ch].std()
            ref_mean, ref_std = ref_lab[:, :, ch].mean(), ref_lab[:, :, ch].std()

            if out_std > 1e-6:
                out_lab[:, :, ch] = (out_lab[:, :, ch] - out_mean) * (ref_std / out_std) + ref_mean

        out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
