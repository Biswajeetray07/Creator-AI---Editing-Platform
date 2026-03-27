import os
import torch
import numpy as np
import cv2
from typing import List, Tuple
from PIL import Image

class GroundingDINODetector:
    """
    Open-vocabulary object detector using GroundingDINO.
    Allows detecting objects using zero-shot text prompts instead of fixed classes.
    """
    def __init__(self, config_path: str, model_path: str, device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.processor = None
        self.model = None
        
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            model_id = "IDEA-Research/grounding-dino-tiny"
            print(f"[GroundingDINO] Loading HF model {model_id} on {self.device}...")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            print("[GroundingDINO] Successfully loaded via transformers.")
        except Exception as e:
            print(f"[GroundingDINO] Initialization failed: {e}")
            raise  # Don't silently fail — let the caller handle fallback

    def __call__(self, image: np.ndarray, text_prompt: str, box_threshold: float = 0.3, **kwargs) -> List[Tuple[int, int, int, int]]:
        """
        Args:
            image: RGB image (H, W, 3)
            text_prompt: What object to find (e.g., "person. dog. car.")
            box_threshold: Minimum confidence for detection
        Returns:
            List of bounding boxes [x1, y1, x2, y2].
        """
        if self.model is None or not text_prompt:
            raise RuntimeError("[GroundingDINO] Model not loaded or prompt is empty.")
        
        # GroundingDINO requires "." separator in prompt
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."
            
        pil_img = Image.fromarray(image)
        
        inputs = self.processor(images=pil_img, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        h, w = image.shape[:2]
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            target_sizes=[(h, w)]
        )
        
        boxes_list = []
        if len(results) > 0:
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]
            
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                print(f"[GroundingDINO] Detected '{label}' (conf={score:.2f}) at [{x1},{y1},{x2},{y2}]")
                boxes_list.append((x1, y1, x2, y2))

        return boxes_list
