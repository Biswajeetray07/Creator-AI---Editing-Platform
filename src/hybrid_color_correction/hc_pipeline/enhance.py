import torch
import gc
import torchvision.transforms.functional as F_t
from hc_models.zero_dce import load_zero_dce_model
from hc_models.restormer import load_restormer_model

class HybridEnhancer:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"[Pipeline] Initializing Hybrid Enhancer on {self.device}...")
        
        # Load models
        self.zero_dce = load_zero_dce_model(device=self.device)
        self.restormer = load_restormer_model(device=self.device)

    def detect_line_art(self, img_tensor, variance_threshold=0.05):
        """
        Detect if image is grayscale/low color variance by computing std across RGB channels.
        """
        std_across_channels = img_tensor.std(dim=1)  # shape (1, H, W)
        mean_variance = std_across_channels.mean().item()
        
        is_line_art = mean_variance < variance_threshold
        print(f"[Pipeline] Color variance: {mean_variance:.4f} | Line-art detected: {is_line_art}")
        return is_line_art

    def histogram_stretch(self, img_tensor):
        """
        Normalize contrast using min-max scaling to stretch entire dynamic range.
        Removes gray backgrounds natively.
        """
        img_min = img_tensor.min()
        img_max = img_tensor.max()
        if img_max > img_min:
            stretched = (img_tensor - img_min) / ((img_max - img_min) + 1e-6)
        else:
            stretched = img_tensor
        return stretched

    def enhance_zero_dce(self, img_tensor):
        """Zero-DCE++ for Exposure and Color Balance"""
        return self.zero_dce(img_tensor)

    def refine_restormer(self, img_tensor):
        """Restormer for Detail enhancement and Denoising"""
        # Save VRAM by using mixed precision on GPU
        with torch.amp.autocast(device_type="cuda", enabled=(self.device == "cuda")):
            refined = self.restormer(img_tensor)
        return refined

    def sharpen_edges(self, img_tensor, alpha=0.3):
        """
        Enhance line clarity using Unsharp Mask (image + alpha * (image - blur(image)))
        """
        blurred = F_t.gaussian_blur(img_tensor, kernel_size=[3, 3], sigma=[1.0, 1.0])
        sharpened = img_tensor + alpha * (img_tensor - blurred)
        return torch.clamp(sharpened, 0.0, 1.0)

    def correct_black_white(self, img_tensor):
        """
        Force true whites and true blacks by clipping very bright and very dark pixels.
        """
        corrected = (img_tensor - 0.02) / (0.98 - 0.02)
        return torch.clamp(corrected, 0.0, 1.0)

    @torch.no_grad()
    def enhance(self, img_tensor, use_restormer=True):
        """
        Runs the advanced hybrid color correction pipeline.
        Args:
            img_tensor: Tensor of shape (1, C, H, W) normalized [0,1]
            use_restormer: Boolean to toggle Restormer refinement
        Returns:
            Enhanced tensor of same shape
        """
        img_tensor = img_tensor.to(self.device).float()
        
        # 1. Line-Art Detection (SMART FEATURE)
        is_line_art = self.detect_line_art(img_tensor)
        
        # Determine adaptive strengths
        alpha_sharp = 0.5 if is_line_art else 0.2
        
        # 2. Histogram Stretching (MANDATORY)
        img_tensor = self.histogram_stretch(img_tensor)
            
        # 3. Zero-DCE++ (Exposure and Color Balance)
        enhanced_tensor = self.enhance_zero_dce(img_tensor)
        
        # If line art, reduce color enhancement impact by blending with original after stretch
        if is_line_art:
            enhanced_tensor = 0.7 * enhanced_tensor + 0.3 * img_tensor

        # 4. Restormer (Detail enhancement, Denoising)
        if use_restormer:
            refined_tensor = self.refine_restormer(enhanced_tensor).float()
        else:
            refined_tensor = enhanced_tensor.float()
            
        # 5. Edge Sharpening (CRITICAL)
        sharpened_tensor = self.sharpen_edges(refined_tensor, alpha=alpha_sharp)
        
        # 6. Black & White Point Correction (VERY IMPORTANT)
        final_tensor = self.correct_black_white(sharpened_tensor)
        
        # Free memory aggressively inside process loop
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        return final_tensor

    def unload(self):
        """Frees GPU memory explicitly if needed."""
        del self.zero_dce
        del self.restormer
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
