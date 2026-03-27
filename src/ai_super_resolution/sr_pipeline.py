"""
Smart Adaptive AI Super-Resolution Pipeline

Dynamically selects processing modules based on image quality analysis.
Three performance modes: Fast, Balanced, HD.

Pipeline flow:
  Input → Validation → Quality Analysis → Conditional Routing
  → Tile Gen → RealESRGAN → CodeFormer(auto) → SwinIR(conditional)
  → Diffusion(conditional) → Tile Fusion → Color Match → Sharpen → Output
"""
import os
import gc
import cv2
import torch
import numpy as np
import time

# Stage imports
from stages.input_validator import InputValidator
from stages.denoiser import Denoiser
from stages.tile_engine import TileEngine
from stages.color_matcher import ColorMatcher
from stages.post_processor import PostProcessor
from stages.quality_analyzer import QualityAnalyzer

# Model imports
from models.upsampler import Upsampler
from models.codeformer_enhancer import CodeFormerEnhancer
from models.swinir_refiner import SwinIRRefiner
from models.diffusion_refiner import DiffusionTextureRefiner
import logging

logger = logging.getLogger("creator_ai.sr_pipeline")



class SuperResolutionPipeline:
    """
    Smart adaptive super-resolution pipeline.

    Modes:
      ⚡ fast      — RealESRGAN + CodeFormer(auto) + Sharpening  (~15s)
      ⚖️ balanced  — + SwinIR if detail_score is low             (~45s)
      🎨 hd        — + SwinIR + Diffusion if image is degraded   (~3min)
    """

    def __init__(self, config: dict):
        self.config = config
        self.modules = config.get("modules", {})

        device_str = config["pipeline"]["device"]
        self.device = device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu"
        self.weights_dir = config["paths"]["weights_dir"]

        logger.info("=" * 60)
        logger.info("  Smart Adaptive Super-Resolution — Initializing")
        logger.info("=" * 60)

        self._init_stages()
        self._init_models()
        self._print_status()
        logger.info("=" * 60)

    # ──────────────────────────────────────────────────────────
    #  Initialization
    # ──────────────────────────────────────────────────────────

    def _init_stages(self):
        m = self.modules
        size_limit = self.config["pipeline"].get("image_size_limit", 4096)

        self.input_validator = InputValidator(image_size_limit=size_limit)

        # Quality Analyzer (always on)
        qa_cfg = m.get("quality_analysis", {})
        self.quality_analyzer = QualityAnalyzer(qa_cfg)

        # Denoiser
        d_cfg = m.get("denoiser", {})
        self.denoiser = Denoiser(
            strength=d_cfg.get("strength", 5),
            use_learned_model=d_cfg.get("use_learned_model", False),
        ) if d_cfg.get("enabled", False) else None

        # Tile Engine
        tile_cfg = m.get("tile_engine", {})
        self.tile_engine = TileEngine(
            tile_size=tile_cfg.get("tile_size", 512),
            overlap=tile_cfg.get("overlap", 64),
        ) if tile_cfg.get("enabled", True) else None

        # Color Matcher
        color_cfg = m.get("color_matching", {})
        self.color_matcher = ColorMatcher(
            method=color_cfg.get("method", "histogram")
        ) if color_cfg.get("enabled", False) else None

        # Post Processor
        pp_cfg = m.get("post_processing", {})
        self.post_processor = PostProcessor(
            sharpen=pp_cfg.get("sharpen", True),
            sharpen_strength=pp_cfg.get("sharpen_strength", 0.3),
            edge_aware_smooth=pp_cfg.get("edge_aware_smooth", True),
        ) if pp_cfg.get("enabled", True) else None

    def _init_models(self):
        m = self.modules

        # RealESRGAN (always loaded)
        up_cfg = m.get("upscaling", {})
        internal_tile = 0 if self.tile_engine else 512
        self.upsampler = Upsampler(
            weights_dir=self.weights_dir,
            model_name=up_cfg.get("model_name", "RealESRGAN_x4plus"),
            tile_size=internal_tile,
            device=self.device,
        )

        # CodeFormer (loaded eagerly — used in all modes when face detected)
        face_cfg = m.get("face_enhancement", {})
        self.codeformer = None
        if face_cfg.get("enabled", True):
            self.codeformer = CodeFormerEnhancer(
                weights_dir=self.weights_dir,
                fidelity_weight=face_cfg.get("fidelity_weight", 0.7),
                face_weight=face_cfg.get("face_weight", 0.7),
                upscale=1,  # Image is already upscaled by RealESRGAN
                device=self.device,
            )

        # SwinIR (loaded eagerly — small model)
        swinir_cfg = m.get("swinir_refinement", {})
        self.swinir = None
        if swinir_cfg.get("enabled", True):
            self.swinir = SwinIRRefiner(
                weights_dir=self.weights_dir,
                model_name=swinir_cfg.get("model_name", "SwinIR_large_x4"),
                tile_size=swinir_cfg.get("tile_size", 512),
                device=self.device,
            )

        # Diffusion (lazy-loaded — biggest model)
        diff_cfg = m.get("diffusion_refinement", {})
        self.diffusion_refiner = None
        if diff_cfg.get("enabled", True):
            self.diffusion_refiner = DiffusionTextureRefiner(
                weights_dir=self.weights_dir,
                num_inference_steps=diff_cfg.get("num_inference_steps", 20),
                guidance_scale=diff_cfg.get("guidance_scale", 4.0),
                device=self.device,
            )

    def _print_status(self):
        status = {
            "Input Validation": "✅",
            "Quality Analyzer": "✅",
            "Denoising": "✅" if self.denoiser else "⬜ (auto)",
            "Tile Engine": "✅" if self.tile_engine else "⬜",
            "RealESRGAN Upscaler": "✅" if self.upsampler.upsampler else "⚠",
            "CodeFormer Face": "✅" if (self.codeformer and self.codeformer.available) else "⬜",
            "SwinIR Refiner": "✅" if (self.swinir and self.swinir.available) else "⬜",
            "SD Texture": "✅" if (self.diffusion_refiner and self.diffusion_refiner.available) else "⬜ (lazy)",
            "Color Matching": "✅" if self.color_matcher else "⬜",
            "Post Processing": "✅" if self.post_processor else "⬜",
        }
        logger.info("\n  Module Status:")
        for name, state in status.items():
            logger.info(f"    {state} {name}")
        

    def _free_vram(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ──────────────────────────────────────────────────────────
    #  Main Run
    # ──────────────────────────────────────────────────────────

    def run(self, image_rgb: np.ndarray, scale: int = None,
            mode: str = "fast") -> np.ndarray:
        """
        Run the smart adaptive super-resolution pipeline.

        Args:
            image_rgb: Input RGB uint8 (H, W, 3)
            scale: Upscale factor (2 or 4)
            mode: "fast", "balanced", or "hd"
        """
        if scale is None:
            scale = self.modules.get("upscaling", {}).get("default_scale", 4)

        mode = mode.lower()
        mode_labels = {"fast": "⚡ Fast", "balanced": "⚖️ Balanced", "hd": "🎨 HD"}
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Mode: {mode_labels.get(mode, mode)}")
        logger.info(f"{'=' * 60}")

        total_start = time.time()

        # ══════════════════════════════════════════════════════
        # Stage 1: Input Validation
        # ══════════════════════════════════════════════════════
        logger.info("\n── Stage 1: Input Validation ──")
        validated = self.input_validator(image_rgb, target_scale=scale)
        image = validated["image"]
        effective_scale = validated["effective_scale"]
        reference_image = image.copy()

        if effective_scale != scale:
            logger.info(f"[Pipeline] Adjusted scale: {scale}x → {effective_scale}x")
            scale = effective_scale

        # ══════════════════════════════════════════════════════
        # Stage 2: Quality Analysis + Routing
        # ══════════════════════════════════════════════════════
        logger.info("\n── Stage 2: Quality Analysis ──")
        metrics = self.quality_analyzer.analyze(image)
        routing = self.quality_analyzer.get_routing(metrics, mode)

        # Print routing decision
        active = [k for k, v in routing.items() if v]
        skipped = [k for k, v in routing.items() if not v]
        logger.info(f"[Routing] Active:  {', '.join(active) if active else 'base only'}")
        logger.info(f"[Routing] Skipped: {', '.join(skipped) if skipped else 'none'}")

        # ══════════════════════════════════════════════════════
        # Stage 3: Conditional Pre-Denoising
        # ══════════════════════════════════════════════════════
        if routing["denoise"]:
            logger.info("\n── Stage 3: Pre-Denoising (auto-triggered) ──")
            if self.denoiser:
                image = self.denoiser(image)
            else:
                strength = self.modules.get("denoiser", {}).get("strength", 5)
                image = cv2.fastNlMeansDenoisingColored(
                    image, None, strength, strength, 7, 21
                )
                logger.info(f"[Denoiser] Applied cv2 denoising (strength={strength})")

        # ══════════════════════════════════════════════════════
        # Stage 4-5: Tile → RealESRGAN → Fuse
        # ══════════════════════════════════════════════════════
        logger.info(f"\n── Stage 4-5: RealESRGAN Upscale ({scale}x) ──")
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.tile_engine:
            result_bgr = self._tiled_upscale(img_bgr, scale)
        else:
            result_bgr = self.upsampler.enhance(img_bgr, outscale=scale)

        self._free_vram()

        # ══════════════════════════════════════════════════════
        # Stage 6: CodeFormer Face Restoration (auto)
        # ══════════════════════════════════════════════════════
        if routing["face_restore"]:
            logger.info("\n── Stage 6: CodeFormer Face Restoration ──")
            if self.codeformer and self.codeformer.available:
                result_bgr = self.codeformer.enhance(result_bgr)
            else:
                logger.info("[Pipeline] CodeFormer not available. Skipping.")
            self._free_vram()
        else:
            logger.info("\n── Stage 6: Face Restoration — SKIPPED (no face detected) ──")

        # ══════════════════════════════════════════════════════
        # Stage 7: SwinIR Detail Refinement (conditional)
        # ══════════════════════════════════════════════════════
        if routing["swinir"]:
            logger.info("\n── Stage 7: SwinIR Detail Refinement ──")
            if self.swinir and self.swinir.available:
                result_bgr = self.swinir.refine(result_bgr)
                self.swinir.unload()
                self._free_vram()
            else:
                logger.info("[Pipeline] SwinIR not available. Skipping.")
        else:
            reason = "mode=fast" if mode == "fast" else f"detail={metrics['detail_score']:.2f} (sufficient)"
            logger.info(f"\n── Stage 7: SwinIR — SKIPPED ({reason}) ──")
            # Unload to free VRAM for later stages
            if self.swinir:
                self.swinir.unload()

        # ══════════════════════════════════════════════════════
        # Stage 8: Diffusion Texture Enhancement (conditional)
        # ══════════════════════════════════════════════════════
        if routing["diffusion"]:
            logger.info("\n── Stage 8: Diffusion Texture Enhancement ──")
            if self.diffusion_refiner and self.diffusion_refiner.available:
                result_bgr = self.diffusion_refiner.refine(
                    result_bgr, prompt="high quality, detailed, sharp"
                )
                self.diffusion_refiner.unload()
                self._free_vram()
            else:
                logger.info("[Pipeline] Diffusion refiner not available. Skipping.")
        else:
            reason = "mode≠hd" if mode != "hd" else f"blur={metrics['blur_score']:.2f} (acceptable)"
            logger.info(f"\n── Stage 8: Diffusion — SKIPPED ({reason}) ──")
            if self.diffusion_refiner:
                self.diffusion_refiner.unload()
                self._free_vram()

        # Convert to RGB for remaining stages
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        # ══════════════════════════════════════════════════════
        # Stage 9: Histogram Color Matching
        # ══════════════════════════════════════════════════════
        if self.color_matcher:
            logger.info("\n── Stage 9: Color Preservation ──")
            result = self.color_matcher(result, reference_image)

        # ══════════════════════════════════════════════════════
        # Stage 10: Adaptive Post-Processing
        # ══════════════════════════════════════════════════════
        if self.post_processor:
            logger.info("\n── Stage 10: Adaptive Sharpening ──")
            result = self.post_processor(result)

        elapsed = time.time() - total_start
        h, w = result.shape[:2]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  ✅ Pipeline complete: {w}×{h} in {elapsed:.2f}s  [{mode_labels.get(mode, mode)}]")
        logger.info(f"{'=' * 60}")

        self._free_vram()
        return result

    # ──────────────────────────────────────────────────────────
    #  Tiled Upscaling
    # ──────────────────────────────────────────────────────────

    def _tiled_upscale(self, image_bgr: np.ndarray, scale: int) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        tiles = self.tile_engine.split(image_bgr)

        processed_tiles = []
        for i, t in enumerate(tiles):
            logger.info(f"        Tile {i + 1}/{len(tiles)}")
            tile_bgr = t["tile"]
            upscaled_tile = self.upsampler.enhance(tile_bgr, outscale=scale)

            processed_tiles.append({
                "tile": upscaled_tile,
                "x": t["x"] * scale,
                "y": t["y"] * scale,
                "w": upscaled_tile.shape[1],
                "h": upscaled_tile.shape[0],
            })
        

        output_shape = (h * scale, w * scale, 3)
        fused = self.tile_engine.fuse(processed_tiles, output_shape)
        return fused
