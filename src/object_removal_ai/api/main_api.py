"""
FastAPI service for AI Object Removal.

Usage:
    uvicorn object_removal_ai.api.main_api:app --host 0.0.0.0 --port 8000
    
Endpoints:
    POST /remove-object  — Remove objects from images
    POST /remove-text    — Remove text from images
    GET  /health         — Health check
"""

import io
import os
import sys
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

# Ensure both 'object_removal_ai' and 'src' are in sys.path
obj_rem_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.dirname(obj_rem_dir)
if obj_rem_dir not in sys.path:
    sys.path.insert(0, obj_rem_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from main_pipeline import ObjectRemovalPipeline
try:
    from ai_text_removal.pipeline import TextRemovalPipeline
except ImportError:
    TextRemovalPipeline = None

try:
    from ai_super_resolution.pipeline import SuperResolutionPipeline
except ImportError:
    SuperResolutionPipeline = None

# ─── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Object Removal API",
    description="Production-quality object removal powered by GroundingDINO + MobileSAM + LaMa",
    version="2.0.0",
)

# ─── Lazy-load pipeline on first request ──────────────────────────────
_pipeline = None

def get_pipeline() -> ObjectRemovalPipeline:
    global _pipeline
    if _pipeline is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "model_config.yaml"
        )
        _pipeline = ObjectRemovalPipeline(config_path=config_path)
    return _pipeline

_text_pipeline = None

def get_text_pipeline() -> TextRemovalPipeline:
    global _text_pipeline
    if _text_pipeline is None and TextRemovalPipeline is not None:
        _text_pipeline = TextRemovalPipeline()
    return _text_pipeline

_sr_pipeline = None

def get_sr_pipeline() -> SuperResolutionPipeline:
    global _sr_pipeline
    if _sr_pipeline is None and SuperResolutionPipeline is not None:
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "ai_super_resolution", "configs", "config.yaml"
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        _sr_pipeline = SuperResolutionPipeline(config)
    return _sr_pipeline


# ─── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": _pipeline is not None}


@app.post("/remove-object")
async def remove_object(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    mask: UploadFile = File(None, description="Optional binary mask (white=remove, black=keep)"),
    prompt: str = Form(None, description="Text prompt describing what to remove (e.g. 'person', 'deer')"),
):
    """
    Remove objects from an image.
    
    Provide either:
    - `prompt` for automatic detection (e.g. "person", "deer", "car")
    - `mask` for manual selection (binary mask, white=remove)
    - Both `prompt` and `mask` (mask takes priority)
    """
    if not prompt and mask is None:
        raise HTTPException(400, "Either 'prompt' or 'mask' must be provided.")

    # Read image
    img_bytes = await image.read()
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Invalid image file.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Read mask if provided
    user_mask = None
    if mask is not None:
        mask_bytes = await mask.read()
        mask_np = np.frombuffer(mask_bytes, dtype=np.uint8)
        user_mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
        if user_mask is None:
            raise HTTPException(400, "Invalid mask file.")

    # Run pipeline
    pipeline = get_pipeline()
    try:
        result = pipeline.run(img_rgb, prompt=prompt, mask=user_mask)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {str(e)}")

    # Encode result as JPEG
    result_bgr = cv2.cvtColor(result["result"], cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=cleaned.jpg"}
    )


@app.post("/remove-text")
async def remove_text(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    enable_diffusion: bool = Form(True, description="Enable Stable Diffusion refinement (slower but better quality)"),
):
    """
    Automatically detect and remove all text from an image.
    Uses an ensemble of CRAFT (EasyOCR) and structural inpainting.
    """
    if TextRemovalPipeline is None:
        raise HTTPException(502, "Text removal module not correctly installed/configured.")

    # Read image
    img_bytes = await image.read()
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Invalid image file.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run pipeline
    pipeline = get_text_pipeline()
    if pipeline is None:
        raise HTTPException(500, "Failed to initialize Text Removal Pipeline.")

    try:
        result = pipeline.run(img_rgb, enable_diffusion=enable_diffusion)
    except Exception as e:
        raise HTTPException(500, f"Text removal error: {str(e)}")

    # Encode result as JPEG
    result_bgr = cv2.cvtColor(result["result"], cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=text_cleaned.jpg"}
    )


@app.post("/upscale")
async def upscale(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    scale: int = Form(4, description="Upscale factor (2, 4, 8)"),
    denoise: bool = Form(True, description="Enable pre-restoration denoising"),
):
    """
    Production-grade AI Image Upscaling.
    Powered by RealESRGAN + Tiled Inference + Pre-restoration.
    """
    if SuperResolutionPipeline is None:
        raise HTTPException(502, "Super-Resolution module not correctly installed.")

    # Read image
    img_bytes = await image.read()
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Invalid image file.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run pipeline
    pipeline = get_sr_pipeline()
    if pipeline is None:
        raise HTTPException(500, "Failed to initialize Super-Resolution Pipeline.")

    try:
        result = pipeline.run(img_rgb, scale=scale, denoise=denoise)
    except Exception as e:
        raise HTTPException(500, f"Upscaling error: {str(e)}")

    # Encode result as JPEG
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=upscaled.jpg"}
    )
