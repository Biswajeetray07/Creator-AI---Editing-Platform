"""
Module 5/6 — Tile Generation & Fusion Engine

Splits large images into overlapping tiles for GPU-safe processing,
then fuses them back using Gaussian-weighted blending to eliminate seams.
"""
import cv2
import numpy as np
import math
import logging

logger = logging.getLogger("creator_ai.tile_engine")



class TileEngine:
    """
    Handles tile splitting and Gaussian-weighted fusion for seamless 
    large-image super-resolution processing.
    
    The overlap region uses a raised-cosine (Gaussian-like) window
    so that tile boundaries are invisible in the final output.
    """

    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap

    def split(self, image: np.ndarray) -> list:
        """
        Split image into overlapping tiles.
        
        Args:
            image: (H, W, 3) uint8 RGB
            
        Returns:
            List of dicts, each with:
                'tile': cropped tile array
                'x': left coordinate in original image
                'y': top coordinate in original image
                'w': tile width
                'h': tile height
        """
        h, w = image.shape[:2]
        step = self.tile_size - self.overlap
        tiles = []

        # Calculate grid positions
        y_positions = list(range(0, h, step))
        x_positions = list(range(0, w, step))

        # Ensure last tile covers the edge
        if y_positions[-1] + self.tile_size < h:
            y_positions.append(h - self.tile_size)
        if x_positions[-1] + self.tile_size < w:
            x_positions.append(w - self.tile_size)

        for y in y_positions:
            for x in x_positions:
                # Clamp to image boundaries
                y1 = max(0, y)
                x1 = max(0, x)
                y2 = min(h, y1 + self.tile_size)
                x2 = min(w, x1 + self.tile_size)

                tile = image[y1:y2, x1:x2].copy()
                tiles.append({
                    "tile": tile,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                })

        logger.info(f"[TileEngine] Split {w}x{h} image → {len(tiles)} tiles "
              f"(size={self.tile_size}, overlap={self.overlap})")
        return tiles

    def fuse(self, tiles: list, output_shape: tuple) -> np.ndarray:
        """
        Fuse processed tiles back into a single image using Gaussian blending.
        
        Args:
            tiles: List of dicts from split(), but with 'tile' replaced by processed result.
                   The tiles may have been upscaled, so their dimensions may differ.
            output_shape: (H, W, 3) — the expected final image dimensions.
            
        Returns:
            Fused image (H, W, 3) uint8
        """
        out_h, out_w = output_shape[:2]
        accumulator = np.zeros((out_h, out_w, 3), dtype=np.float64)
        weight_map = np.zeros((out_h, out_w), dtype=np.float64)

        for t in tiles:
            tile = t["tile"]
            x, y = t["x"], t["y"]
            th, tw = tile.shape[:2]

            # Ensure tile fits in output
            th = min(th, out_h - y)
            tw = min(tw, out_w - x)
            tile = tile[:th, :tw]

            # Create 2D Gaussian weight window
            w_window = self._gaussian_window(tw, th)

            # Accumulate weighted tile
            accumulator[y:y+th, x:x+tw] += tile.astype(np.float64) * w_window[:, :, np.newaxis]
            weight_map[y:y+th, x:x+tw] += w_window

        # Normalize by weight
        weight_map = np.maximum(weight_map, 1e-8)  # avoid division by zero
        result = accumulator / weight_map[:, :, np.newaxis]

        logger.info(f"[TileEngine] Fused {len(tiles)} tiles → {out_w}x{out_h} image")
        return np.clip(result, 0, 255).astype(np.uint8)

    def _gaussian_window(self, width: int, height: int) -> np.ndarray:
        """
        Create a 2D Gaussian-like blending window using raised cosine.
        
        The window is 1.0 in the center and tapers smoothly to ~0 at
        the edges over the overlap region. This ensures seamless blending.
        """
        def _1d_window(size: int) -> np.ndarray:
            if size <= 1:
                return np.ones(size)
            # Raised cosine window (Hann window)
            x = np.linspace(0, 1, size)
            window = np.ones(size)
            
            fade = min(self.overlap, size // 2)
            if fade > 0:
                # Left taper
                window[:fade] = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))
                # Right taper
                window[-fade:] = 0.5 * (1 - np.cos(np.pi * np.arange(fade, 0, -1) / fade))
            
            return window

        w_x = _1d_window(width)
        w_y = _1d_window(height)
        return np.outer(w_y, w_x)
