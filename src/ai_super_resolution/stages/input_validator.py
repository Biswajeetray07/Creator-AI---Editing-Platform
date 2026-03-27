"""
Module 1 -- Input Validation

Validates and normalizes input images for safe GPU processing.
Detects resolution, color space, bit depth.

SMART RESIZE: Instead of blindly downscaling large images, computes
the optimal working resolution based on the target output scale
and GPU capacity. This prevents the catastrophic case where a
high-res input gets downscaled then upscaled back to a SMALLER size.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger("creator_ai.input_validator")



class InputValidator:
    """
    Validates input images and normalizes them for the SR pipeline.

    Key intelligence:
        - If input is already large AND sharp, recommends scale=2 instead of 4
        - Computes working_resolution so that output >= original resolution
        - Preserves the original image for final compositing
    """

    def __init__(self, image_size_limit: int = 4096):
        self.image_size_limit = image_size_limit

    def __call__(self, image: np.ndarray, target_scale: int = 4) -> dict:
        """
        Validate and normalize the input image.

        Args:
            image:        Input image (H, W, 3), RGB, uint8
            target_scale: Requested upscale factor (2, 4, 8)

        Returns:
            dict with keys:
                'image':            working image (may be resized)
                'original_image':   untouched original (full-res)
                'original_shape':   (H, W) of original
                'was_resized':      bool
                'effective_scale':  actual scale to use (may differ from target)
                'info':             detected properties dict
        """
        h, w = image.shape[:2]
        original_shape = (h, w)
        was_resized = False
        original_image = image.copy()

        # Detect properties
        info = {
            "resolution": f"{w}x{h}",
            "megapixels": round((w * h) / 1e6, 2),
            "channels": image.shape[2] if image.ndim == 3 else 1,
            "dtype": str(image.dtype),
            "bit_depth": 8 if image.dtype == np.uint8 else (16 if image.dtype == np.uint16 else 32),
        }

        logger.info(f"[InputValidator] Resolution: {info['resolution']} ({info['megapixels']}MP)")
        logger.info(f"[InputValidator] Channels: {info['channels']}, Bit Depth: {info['bit_depth']}")

        # Ensure 3-channel RGB
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Convert to uint8 if needed
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # ── Smart Scale Computation ──
        # The output should be: working_dim * scale
        # We want: output_dim >= original_dim (never produce a smaller image!)
        # So: working_dim >= original_dim / scale
        #
        # GPU limit means: working_dim <= image_size_limit
        # Therefore: effective_scale = ceil(original_max / image_size_limit)
        #            but at least target_scale, at most 4

        effective_scale = target_scale
        max_dim = max(h, w)

        # If the image is already large, compute the working resolution
        # so that output >= original
        max_working = self.image_size_limit  # GPU-safe working resolution
        target_output = max_dim * target_scale  # What user wants

        if max_dim > max_working:
            # Image is too large to process at native res
            # Downscale to max_working, but ensure output >= original
            # output = max_working * effective_scale >= max_dim
            # effective_scale >= max_dim / max_working
            needed_scale = max_dim / max_working
            if needed_scale <= 2:
                effective_scale = 2
            elif needed_scale <= 4:
                effective_scale = 4
            else:
                # Even 4x won't recover original size, limit working to max_dim/4
                max_working = max(512, max_dim // 4)
                effective_scale = 4

            scale_f = max_working / max_dim
            new_w = int(w * scale_f)
            new_h = int(h * scale_f)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            was_resized = True

            output_dim = max(new_w, new_h) * effective_scale
            logger.info(f"[InputValidator] Smart resize: {w}x{h} -> {new_w}x{new_h} (working)")
            logger.info(f"[InputValidator] Output will be: ~{new_w*effective_scale}x{new_h*effective_scale} "
                  f"(scale={effective_scale}x)")
            if output_dim < max_dim:
                logger.info(f"[InputValidator] WARNING: Output ({output_dim}px) < Original ({max_dim}px)")
                logger.info(f"[InputValidator]   Consider using --scale {int(np.ceil(max_dim / max_working) * 2)}")
        else:
            logger.info(f"[InputValidator] Image fits in GPU memory. Using {effective_scale}x directly.")

        return {
            "image": image,
            "original_image": original_image,
            "original_shape": original_shape,
            "was_resized": was_resized,
            "effective_scale": effective_scale,
            "info": info,
        }
