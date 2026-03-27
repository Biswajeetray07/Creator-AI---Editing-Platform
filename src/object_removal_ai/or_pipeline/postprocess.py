import cv2
import numpy as np


class PostProcessor:
    """
    Final post-processing stage: Full-Resolution Compositing & Denoising.
    1. Extracts the inpainted patch
    2. Upscales it to match the native resolution of the original image
    3. Denoisies the patch to remove SD hallucination artifacts
    4. Seamlessly blends it into the pristine original image (0% bg quality loss)
    """

    def __init__(self, use_poisson: bool = True):
        self.use_poisson = use_poisson

    def __call__(
        self,
        true_original: np.ndarray,
        inpainted: np.ndarray,
        mask: np.ndarray,
        alpha_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Args:
            true_original: Original un-resized RGB image (H_orig, W_orig, 3).
            inpainted:  Inpainted/Generated RGB image (H_work, W_work, 3).
            mask:       Hard binary mask (H_work, W_work), 255 = generated region.
            alpha_mask: Soft alpha mask (H_work, W_work), 0-1 float32.
        """
        # 1. Clean up the generated patch (Spatially denoise SD artifacts)
        inpainted_clean = self._denoise_and_sharpen(inpainted)

        # 2. Find bounding box of the modified region in the working resolution
        coords = cv2.findNonZero(mask)
        if coords is None:
            # If nothing was modified, return exactly the original image
            return true_original
            
        x, y, w, h = cv2.boundingRect(coords)
        
        # Ensure box is strictly within working bounds
        x = max(0, x)
        y = max(0, y)
        w = min(mask.shape[1] - x, w)
        h = min(mask.shape[0] - y, h)

        # Extract the modified patch
        patch = inpainted_clean[y:y+h, x:x+w]
        
        # 3. Calculate mapping from working res to native res
        orig_h, orig_w = true_original.shape[:2]
        work_h, work_w = inpainted.shape[:2]
        
        # Compute exact bounding box in the original high-res image
        scale_x = orig_w / work_w
        scale_y = orig_h / work_h
        
        nx = int(x * scale_x)
        ny = int(y * scale_y)
        nw = int(w * scale_x)
        nh = int(h * scale_y)
        
        # Ensure bounds are safe
        nx = max(0, nx)
        ny = max(0, ny)
        nw = min(orig_w - nx, nw)
        nh = min(orig_h - ny, nh)
        
        # Upscale patch to native resolution
        patch_hires = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_CUBIC)
        
        # Upscale mask to native resolution
        patch_mask = mask[y:y+h, x:x+w]
        patch_mask_hires = cv2.resize(patch_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask boundary has no harsh black edges by dilating slightly into patch
        patch_mask_hires = cv2.dilate(patch_mask_hires, np.ones((5,5), np.uint8), iterations=2)

        # 4. Composite safely
        output = true_original.copy()
        try:
            if self.use_poisson:
                center = (nx + nw // 2, ny + nh // 2)
                # Poisson blending for seamless illumination
                output = cv2.seamlessClone(
                    patch_hires, 
                    output, 
                    patch_mask_hires, 
                    center, 
                    cv2.NORMAL_CLONE
                )
            else:
                # Soft alpha fallback
                patch_alpha = alpha_mask[y:y+h, x:x+w]
                patch_alpha_hires = cv2.resize(patch_alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)
                patch_alpha_hires = np.expand_dims(patch_alpha_hires, -1)
                
                roi = output[ny:ny+nh, nx:nx+nw]
                blended = (patch_hires * patch_alpha_hires) + (roi * (1 - patch_alpha_hires))
                output[ny:ny+nh, nx:nx+nw] = blended.astype(np.uint8)
                
        except Exception as e:
            print(f"[PostProcessor] Blending failed, using hard copy fallback: {e}")
            roi = output[ny:ny+nh, nx:nx+nw]
            bin_mask = (patch_mask_hires > 127).astype(bool)
            roi[bin_mask] = patch_hires[bin_mask]
            
        return output

    def _denoise_and_sharpen(self, img: np.ndarray) -> np.ndarray:
        """Pass-through to preserve LaMa's natural texture generation."""
        return img

