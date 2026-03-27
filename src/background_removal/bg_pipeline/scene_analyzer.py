"""
Stage 3 — Adaptive Scene Routing
Zero-cost router (<5ms) that classifies scenes as SIMPLE or COMPLEX
using 4 heuristics and a voting system.

SIMPLE (portraits, clean backgrounds): bypass SAM entirely
COMPLEX (products, multi-subject, cluttered): use SAM segmentation
"""
import numpy as np
import cv2
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SceneType(Enum):
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"


class SceneAnalyzer:
    """
    Stage 3: Adaptive scene router using 4 heuristics.
    
    Heuristics:
        H1: Multi-subject detection (>1 blob with area > 2% of image)
        H2: Background edge density (>12% Canny edge density in BG)
        H3: Confidence spread (σ(soft_mask) < 0.28)
        H4: Subject size ratio (FG < 15% of image)
    
    Decision: ≥ 2 votes → COMPLEX, otherwise → SIMPLE
    """

    # Tunable thresholds
    MULTI_SUBJECT_AREA_THRESH = 0.02    # H1: min 2% of image per blob
    EDGE_DENSITY_THRESH = 0.12          # H2: 12% Canny edge density
    CONFIDENCE_SPREAD_THRESH = 0.28     # H3: σ threshold
    SUBJECT_SIZE_THRESH = 0.15          # H4: FG < 15% of image
    VOTE_THRESHOLD = 2                  # ≥ 2 votes → COMPLEX

    def analyze(self, soft_mask: np.ndarray, image_rgb: np.ndarray = None) -> tuple:
        """
        Classify the scene as SIMPLE or COMPLEX.

        Args:
            soft_mask: Soft probability map [H, W] in [0.0, 1.0] from BiRefNet
            image_rgb: Optional original image for edge density analysis

        Returns:
            (scene_type, vote_details, latency_ms) where:
                scene_type: SceneType.SIMPLE or SceneType.COMPLEX
                vote_details: dict with individual heuristic results
                latency_ms: float
        """
        start = time.time()
        votes = 0
        details = {}

        # Binary mask for spatial analysis
        binary_mask = (soft_mask > 0.5).astype(np.uint8)
        h, w = soft_mask.shape
        total_pixels = h * w

        # ── H1: Multi-subject detection ──────────────────────
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        # Count blobs with area > 2% of image (exclude background label 0)
        large_blobs = 0
        for i in range(1, num_components):
            blob_area = stats[i, cv2.CC_STAT_AREA]
            if blob_area / total_pixels > self.MULTI_SUBJECT_AREA_THRESH:
                large_blobs += 1

        h1_complex = large_blobs > 1
        if h1_complex:
            votes += 1
        details["H1_multi_subject"] = {
            "large_blobs": large_blobs,
            "complex": h1_complex,
        }

        # ── H2: Background edge density ─────────────────────
        if image_rgb is not None:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            bg_mask = (1 - binary_mask).astype(bool)
            bg_pixels = bg_mask.sum()
            if bg_pixels > 0:
                bg_edge_density = edges[bg_mask].sum() / (255.0 * bg_pixels)
            else:
                bg_edge_density = 0.0
        else:
            bg_edge_density = 0.0

        h2_complex = bg_edge_density > self.EDGE_DENSITY_THRESH
        if h2_complex:
            votes += 1
        details["H2_edge_density"] = {
            "density": float(bg_edge_density),
            "complex": h2_complex,
        }

        # ── H3: Confidence spread ────────────────────────────
        confidence_std = float(np.std(soft_mask))
        h3_complex = confidence_std < self.CONFIDENCE_SPREAD_THRESH
        if h3_complex:
            votes += 1
        details["H3_confidence_spread"] = {
            "std": confidence_std,
            "complex": h3_complex,
        }

        # ── H4: Subject size ratio ───────────────────────────
        fg_ratio = float(binary_mask.sum()) / total_pixels
        h4_complex = fg_ratio < self.SUBJECT_SIZE_THRESH
        if h4_complex:
            votes += 1
        details["H4_subject_size"] = {
            "fg_ratio": fg_ratio,
            "complex": h4_complex,
        }

        # ── Decision ─────────────────────────────────────────
        scene_type = SceneType.COMPLEX if votes >= self.VOTE_THRESHOLD else SceneType.SIMPLE
        latency_ms = (time.time() - start) * 1000

        logger.info(
            f"Stage 3 (Scene Analyzer): {scene_type.value} "
            f"({votes}/{4} votes, {latency_ms:.1f}ms)"
        )

        return scene_type, details, latency_ms
