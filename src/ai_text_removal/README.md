# AI Text Removal — Production Pipeline

A high-performance text removal system that combines structural reconstruction (LaMa) with generative refinement (Stable Diffusion) for studio-quality results.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install diffusers transformers accelerate
```

### 2. Required Weights
The pipeline looks for weights in the `weights/` directory:
- `weights/yolov8n.pt` (Detection)
- `weights/sam_vit_l_0b3195.pth` (Segmentation)
- `weights/big-lama.pt` (Inpainting structure)
- Stable Diffusion weights are automatically downloaded on first run via `diffusers`.

### 3. Usage
Run on your image (replace `your_image.jpg` with your actual filename):
```bash
python inference.py --input your_image.jpg
```

> [!TIP]
> **Processing Speed:** By default, Stable Diffusion is **disabled** to make processing instant (~2-5 seconds). To enable high-resolution generative refinement, change `strength` to `0.75` in `configs/config.yaml`.

**Advanced Options:**
- `--save-artifacts`: Save intermediate masks and edge maps (useful for debugging).
- `--config configs/config.yaml`: Use a custom configuration.

## 🛠️ Architecture
1. **Detection (CRAFT/YOLO)**: Locates text boxes.
2. **Segmentation (SAM)**: Creates precise pixel-level masks.
3. **Edge Extraction (Canny)**: Guides the inpainter with background structural lines.
4. **Multi-Scale Inpainting (LaMa)**: Refills background at 256/512/768px resolutions.
5. **Generative Refinement (SD)**: Adds photorealistic textures to the inpainted area.
6. **Poisson Blending**: Seamlessly merges the new patch with zero visible seams.

---
*Optimized for RTX GPU (4GB+ VRAM recommended).*
