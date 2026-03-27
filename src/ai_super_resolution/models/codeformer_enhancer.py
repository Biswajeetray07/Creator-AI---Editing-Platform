"""
CodeFormer Face Restoration Module

Replaces GFPGAN with CodeFormer for superior identity-preserving face restoration.
Uses facexlib for face detection/alignment and CodeFormer for restoration.

Auto-downloads weights on first use.
"""
import os
import cv2
import torch
import numpy as np
import logging

logger = logging.getLogger("creator_ai.codeformer_enhancer")

try:
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    HAS_FACEXLIB = True
except ImportError:
    HAS_FACEXLIB = False

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


def _download_codeformer_weights(weights_dir: str) -> str:
    """Download CodeFormer weights from HuggingFace if not present."""
    weight_path = os.path.join(weights_dir, "codeformer.pth")
    if os.path.exists(weight_path):
        return weight_path

    if not HAS_HF_HUB:
        logger.info("[CodeFormer] huggingface_hub not installed. Cannot download weights.")
        return None

    logger.info("[CodeFormer] Downloading codeformer.pth from HuggingFace...")
    try:
        downloaded = hf_hub_download(
            repo_id="sczhou/CodeFormer",
            filename="CodeFormer/codeformer.pth",
            local_dir=weights_dir,
            local_dir_use_symlinks=False,
        )
        # Move to expected location if nested
        if downloaded != weight_path and os.path.exists(downloaded):
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            os.rename(downloaded, weight_path)
        logger.info(f"[CodeFormer] Saved weights to {weight_path}")
        return weight_path
    except Exception as e:
        logger.info(f"[CodeFormer] HuggingFace download failed: {e}")
        # Fallback to direct download
        try:
            from urllib.request import urlretrieve
            url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
            logger.info(f"[CodeFormer] Trying direct download from GitHub releases...")
            urlretrieve(url, weight_path)
            logger.info(f"[CodeFormer] Saved weights to {weight_path}")
            return weight_path
        except Exception as e2:
            logger.info(f"[CodeFormer] Direct download also failed: {e2}")
            return None


class CodeFormerEnhancer:
    """
    CodeFormer-based face restoration.

    Better identity preservation than GFPGAN due to codebook-based
    discrete representation learning. Supports fidelity_weight to
    balance quality vs. identity.
    """

    def __init__(self, weights_dir: str, fidelity_weight: float = 0.7,
                 face_weight: float = 0.7, upscale: int = 4, device: str = "cuda"):
        """
        Args:
            weights_dir: Directory containing/to-download codeformer.pth
            fidelity_weight: 0.0 = max quality, 1.0 = max fidelity (identity)
            face_weight: Blending weight when pasting face back (0-1)
            upscale: Upscale factor for face helper
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fidelity_weight = fidelity_weight
        self.face_weight = face_weight
        self.upscale = upscale
        self.model = None
        self.face_helper = None

        if not HAS_FACEXLIB:
            logger.info("[CodeFormer] ⚠ facexlib not installed. Face enhancement disabled.")
            return

        # Download weights if needed
        weight_path = _download_codeformer_weights(weights_dir)
        if weight_path is None or not os.path.exists(weight_path):
            logger.info("[CodeFormer] ⚠ Weights not available. Face enhancement disabled.")
            return

        try:
            # Patch collections for basicsr in Python 3.10+ (Iterable was removed)
            import collections
            import collections.abc
            for attr in ['Iterable', 'Mapping', 'MutableSet', 'MutableMapping', 'Sequence']:
                if not hasattr(collections, attr):
                    setattr(collections, attr, getattr(collections.abc, attr))

            # Add local BasicSR + CodeFormer repos to sys.path
            import sys
            basicsr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "libs", "BasicSR"))
            codeformer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "libs", "CodeFormer"))

            # CRITICAL: Flush any pip-installed basicsr from the module cache.
            # The pip version (1.4.2) doesn't have codeformer_arch.py.
            # We must force Python to re-import from our local libs/ copy.
            mods_to_remove = [k for k in sys.modules if k == "basicsr" or k.startswith("basicsr.")]
            for mod in mods_to_remove:
                del sys.modules[mod]

            # Now inject our local paths BEFORE pip site-packages
            if basicsr_path not in sys.path:
                sys.path.insert(0, basicsr_path)
            if codeformer_path not in sys.path:
                sys.path.insert(1, codeformer_path)

            import basicsr.archs.codeformer_arch  # This registers CodeFormer in ARCH_REGISTRY
            from basicsr.utils.registry import ARCH_REGISTRY


            # Load CodeFormer model
            self.model = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512, codebook_size=1024, n_head=8,
                n_layers=9, connect_list=["32", "64", "128", "256"],
            ).to(self.device)

            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(
                ckpt.get("params_ema", ckpt.get("params", ckpt)),
                strict=True
            )
            self.model.eval()

            # Init face helper for detection + alignment + paste-back
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,  # Image is already upscaled by RealESRGAN
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=self.device,
            )

            logger.info(f"[CodeFormer] ✅ Loaded on {self.device} "
                  f"(fidelity={fidelity_weight}, face_weight={face_weight})")
        except Exception as e:
            logger.warning(f"[CodeFormer] ⚠ Failed to load: {e}. Face enhancement disabled.")
            self.model = None
            self.face_helper = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.face_helper is not None

    @torch.no_grad()
    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Detect faces, restore with CodeFormer, and paste back.

        Args:
            image_bgr: BGR uint8 (H, W, 3)

        Returns:
            Enhanced BGR image with restored faces
        """
        if not self.available:
            logger.info("[CodeFormer] Not available, returning input unchanged.")
            return image_bgr

        try:
            self.face_helper.clean_all()
            self.face_helper.read_image(image_bgr)

            # Detect and align faces
            num_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=5
            )
            self.face_helper.align_warp_face()

            if num_faces == 0:
                logger.info("[CodeFormer] No faces detected. Returning input unchanged.")
                return image_bgr

            logger.info(f"[CodeFormer] Processing {num_faces} face(s)...")

            # Restore each face
            for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                # Preprocess: BGR -> RGB, HWC -> BCHW, normalize to [-1, 1]
                face_t = torch.from_numpy(
                    cropped_face.astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(self.device)

                # Run CodeFormer
                with torch.amp.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
                    output = self.model(face_t, w=self.fidelity_weight, adain=True)[0]

                # Postprocess: BCHW -> HWC, denormalize
                restored_face = output.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                restored_face = (restored_face * 255.0).round().astype(np.uint8)

                self.face_helper.add_restored_face(restored_face)

            # Paste faces back using face_weight for blending
            self.face_helper.get_inverse_affine(None)

            # Use bg_upsampler=None since image is already upscaled by RealESRGAN
            restored_img = self.face_helper.paste_faces_to_input_image(
                save_path=None,
                upsample_img=image_bgr
            )

            logger.info(f"[CodeFormer] ✅ Restored {num_faces} face(s).")
            return restored_img

        except Exception as e:
            logger.info(f"[CodeFormer] Error during enhancement: {e}. Returning input unchanged.")
            return image_bgr

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.face_helper is not None:
            del self.face_helper
            self.face_helper = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[CodeFormer] Unloaded from GPU.")
