"""
Module 4 — Noise & Artifact Removal (Learned Denoiser)

Uses an AI model rather than classic OpenCV filters to clean
input images before upscaling, preventing noise amplification
without destroying fine texture.
"""
import cv2
import torch
import numpy as np
import logging

logger = logging.getLogger("creator_ai.denoiser")


# For Phase 2, we implement a fallback to a smarter OpenCV structural 
# filter, and structure the wrapper to allow plugging in DnCNN/RIDNet later.

class Denoiser:
    """
    Intelligent denoising stage.
    
    Currently implements Edge-Preserving Filter (Domain Transform)
    as a much higher quality alternative to fastNlMeansDenoising Colored.
    It removes compression artifacts and noise while better preserving
    sharp structural edges.
    """

    def __init__(self, strength: int = 5, use_learned_model: bool = False):
        self.strength = strength
        self.use_learned = use_learned_model
        
        if self.use_learned:
            logger.info("[Denoiser] Note: Learned models (DnCNN/RIDNet) not yet downloaded. Using EPF fallback.")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise the image.
        
        Args:
            image: RGB uint8 (H, W, 3)
            
        Returns:
            Denoised RGB uint8 (H, W, 3)
        """
        # We use Domain Transform Edge Preserving Filter:
        # sigma_s controls spatial smoothing
        # sigma_r controls color/range smoothing (how much edges stop smoothing)
        
        # Scale parameters based on requested strength 1-10
        sigma_s = min(200, self.strength * 10)
        sigma_r = min(1.0, self.strength * 0.05)
        
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Normconv is faster than Recursive and yields similar results
        denoised_bgr = cv2.edgePreservingFilter(
            bgr, 
            flags=cv2.RECURS_FILTER, 
            sigma_s=sigma_s, 
            sigma_r=sigma_r
        )
        
        logger.info(f"[Denoiser] Applied Edge Preserving Denoising (strength={self.strength})")
        return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
