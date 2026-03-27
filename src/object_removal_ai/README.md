# AI Object Removal Pipeline

Production-quality object removal system powered by **GroundingDINO** (zero-shot detection), **MobileSAM** (segmentation), and **LaMa** (inpainting).

## Features

- 🎯 **Text-guided removal** — Describe what to remove ("person", "car", "deer")
- 🖌️ **Mask-based removal** — Provide your own binary mask
- 🌐 **FastAPI service** — REST API for web integration
- ⚡ **GPU accelerated** — CUDA inference with GroundingDINO + LaMa
- 🐳 **Docker ready** — One-command deployment

## Quick Start

### CLI — Remove by text prompt
```bash
python object_removal_ai/main_pipeline.py -i photo.jpg -p "person" --save-artifacts
```

### CLI — Remove by mask
```bash
python object_removal_ai/main_pipeline.py -i photo.jpg -m mask.png --save-artifacts
```

### FastAPI Server
```bash
pip install fastapi uvicorn python-multipart
uvicorn object_removal_ai.api.main_api:app --host 0.0.0.0 --port 8000
```

Then send requests:
```bash
# Remove by prompt
curl -X POST http://localhost:8000/remove-object \
  -F "image=@photo.jpg" \
  -F "prompt=person" \
  -o cleaned.jpg

# Remove by mask
curl -X POST http://localhost:8000/remove-object \
  -F "image=@photo.jpg" \
  -F "mask=@mask.png" \
  -o cleaned.jpg
```

### Docker
```bash
docker build -t object-removal .
docker run --gpus all -p 8000:8000 object-removal
```

## Architecture

```
INPUT IMAGE → GroundingDINO (detect) → MobileSAM (segment) → Mask Refinement → LaMa (inpaint) → Post-Processing → OUTPUT
```

| Stage | Model | Purpose |
|-------|-------|---------|
| Detection | GroundingDINO / YOLOv8 | Find objects by text prompt |
| Segmentation | MobileSAM | Pixel-precise object masks |
| Mask Refinement | OpenCV morphology | Clean edges, fill holes |
| Inpainting | LaMa | Fill removed regions naturally |
| Post-Processing | Poisson + Color Match | Seamless blending |

## Required Weights

Place these in `../weights/` (relative to `object_removal_ai/`):

| File | Model | Download |
|------|-------|----------|
| `big-lama.pt` | LaMa inpainter | [HuggingFace](https://huggingface.co/smartywu/big-lama) |
| `mobilesam.pt` | MobileSAM | [GitHub](https://github.com/ChaoningZhang/MobileSAM) |
| `yolov8n.pt` | YOLOv8 fallback | Auto-downloaded by ultralytics |

GroundingDINO is loaded from HuggingFace Hub automatically.

## Testing

```bash
python tests/test_pipeline.py
```

## Project Structure

```
object_removal_ai/
├── api/
│   └── main_api.py          # FastAPI service
├── configs/
│   └── model_config.yaml    # Model paths + parameters
├── models/
│   ├── groundingdino_detector.py
│   ├── sam_segmenter.py
│   ├── lama_inpainter.py
│   └── yolo_detector.py
├── pipeline/
│   ├── mask_refiner.py
│   ├── edge_extractor.py
│   ├── context_expansion.py
│   └── postprocess.py
├── utils/
│   └── image_utils.py
└── main_pipeline.py          # CLI + ObjectRemovalPipeline class
```
