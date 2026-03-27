import os
import argparse
from glob import glob
from utils.image_utils import load_image, save_image
from pipeline.preprocess import preprocess_image
from pipeline.enhance import HybridEnhancer
from pipeline.postprocess import postprocess_tensor

def process_image(enhancer, input_path, output_path, use_restormer):
    print(f"Processing: {input_path}")
    
    # 1. Load and Preprocess (Max 512px)
    img_np = load_image(input_path)
    img_tensor = preprocess_image(img_np, max_size=512)
    
    # 2. Enhance
    output_tensor = enhancer.enhance(img_tensor, use_restormer=use_restormer)
    
    # 3. Postprocess and Save
    out_np = postprocess_tensor(output_tensor)
    save_image(out_np, output_path)
    print(f"Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Hybrid Auto Color Correction Pipeline")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("-o", "--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--disable-restormer", action="store_true", help="Disable Restormer refinement (faster, less VRAM)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output, exist_ok=True)
    use_restormer = not args.disable_restormer
    
    # Load Enhancer
    enhancer = HybridEnhancer(device=args.device)

    if os.path.isfile(args.input):
        filename = os.path.basename(args.input)
        out_path = os.path.join(args.output, f"enhanced_{filename}")
        process_image(enhancer, args.input, out_path, use_restormer)
    elif os.path.isdir(args.input):
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_files = []
        for ext in extensions:
            image_files.extend(glob(os.path.join(args.input, ext)))
            image_files.extend(glob(os.path.join(args.input, ext.upper())))
            
        print(f"Found {len(image_files)} images in directory.")
        for img_path in image_files:
            filename = os.path.basename(img_path)
            out_path = os.path.join(args.output, f"enhanced_{filename}")
            try:
                process_image(enhancer, img_path, out_path, use_restormer)
            except Exception as e:
                print(f"Failed processing {img_path}: {e}")
    else:
        print(f"Input path {args.input} does not exist.")

if __name__ == "__main__":
    main()
