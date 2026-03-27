"""
AI Super-Resolution — CLI Entry Point

Usage:
    python inference.py -i photo.jpg --scale 4 --face-enhance
    python inference.py -i photo.jpg --scale 2 --no-denoise
"""
import os
import sys
import argparse
import yaml
import time

# Add current dir to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils.image_utils import load_image, save_image
from sr_pipeline import SuperResolutionPipeline


def main():
    parser = argparse.ArgumentParser(description="Production AI Super-Resolution CLI")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path")
    parser.add_argument("--scale", "-s", type=int, choices=[2, 4, 8], default=4, help="Upscale factor")
    parser.add_argument("--config", "-c", type=str,
                        default=os.path.join(script_dir, "configs", "config.yaml"),
                        help="Path to config file")
    parser.add_argument("--no-denoise", action="store_true", help="Skip pre-restoration denoising")
    parser.add_argument("--face-enhance", action="store_true", help="Enable GFPGAN face reconstruction")
    args = parser.parse_args()

    # ── Load config ──
    if not os.path.exists(args.config):
        print(f"Error: Config not found at {args.config}")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve weights dir to absolute path
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    config["paths"]["weights_dir"] = os.path.join(project_root, config["paths"]["weights_dir"])

    # ── Initialize Pipeline ──
    sr_pipeline = SuperResolutionPipeline(config)

    # ── Load Image ──
    print(f"\nLoading: {args.input}")
    try:
        image = load_image(args.input)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # ── Run Pipeline ──
    upscaled = sr_pipeline.run(
        image,
        scale=args.scale,
        denoise=not args.no_denoise,
        face_enhance=args.face_enhance,
    )

    # ── Save Output ──
    out_dir = os.path.join(project_root, config["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    if args.output:
        out_path = args.output
    else:
        name, ext = os.path.splitext(os.path.basename(args.input))
        out_path = os.path.join(out_dir, f"{name}_x{args.scale}{ext}")

    save_image(upscaled, out_path)
    print(f"\n  → Saved to: {out_path}")


if __name__ == "__main__":
    main()
