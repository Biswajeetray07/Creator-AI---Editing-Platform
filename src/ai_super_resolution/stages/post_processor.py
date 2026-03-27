"""
Module 12 — Adaptive Post-Processing

Final polishing stage with edge-aware smoothing and controlled sharpening.
Replaces the old destructive cv2.addWeighted approach.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger("creator_ai.post_processor")



class PostProcessor:
    """
    Applies final post-processing to the upscaled image.
    
    Features:
        - Edge-aware bilateral smoothing to remove GAN micro-artifacts
        - Adaptive unsharp mask with safe strength limits  
        - No destructive filters (strength is capped)
    """

    def __init__(self, sharpen: bool = True, sharpen_strength: float = 0.15,
                 edge_aware_smooth: bool = True):
        self.sharpen = sharpen
        self.sharpen_strength = min(sharpen_strength, 0.4)  # Hard cap
        self.edge_aware_smooth = edge_aware_smooth

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply post-processing pipeline.
        
        Args:
            image: RGB uint8 (H, W, 3)
            
        Returns:
            Post-processed image (H, W, 3) uint8
        """
        result = image.copy()

        # 1. Edge-aware smoothing (bilateral filter)
        if self.edge_aware_smooth:
            result = self._edge_aware_smooth(result)

        # 2. Adaptive sharpening
        if self.sharpen:
            result = self._adaptive_sharpen(result)

        return result

    def _edge_aware_smooth(self, image: np.ndarray) -> np.ndarray:
        """
        Bilateral filter that smooths flat areas (removing GAN checkerboard
        artifacts) while preserving sharp edges.
        
        Parameters are deliberately conservative to avoid over-smoothing.
        """
        # d=5: small neighborhood, sigmaColor=30: moderate color sensitivity
        # sigmaSpace=30: moderate spatial sensitivity
        smoothed = cv2.bilateralFilter(image, d=5, sigmaColor=30, sigmaSpace=30)
        logger.info(f"[PostProcessor] Applied edge-aware bilateral smoothing")
        return smoothed

    def _adaptive_sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive unsharp mask that sharpens edges without amplifying noise.
        
        Uses a larger Gaussian sigma (3.0) to target structural edges
        rather than pixel-level noise.
        """
        # Gaussian blur with sigma=3.0 for structural sharpening
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)

        # Unsharp mask: result = original + strength * (original - blurred)
        sharpened = cv2.addWeighted(
            image, 1.0 + self.sharpen_strength,
            gaussian, -self.sharpen_strength,
            0
        )

        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        logger.info(f"[PostProcessor] Applied adaptive sharpening (strength={self.sharpen_strength})")
        return result
