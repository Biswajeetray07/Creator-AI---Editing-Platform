import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple

# COCO class names for fuzzy matching
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Fuzzy synonym map: user query -> COCO classes to match
SYNONYM_MAP = {
    'deer': ['horse', 'cow', 'sheep', 'giraffe'],
    'animal': ['dog', 'cat', 'horse', 'cow', 'sheep', 'bird', 'bear', 'elephant', 'zebra', 'giraffe'],
    'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'airplane', 'boat'],
    'furniture': ['chair', 'couch', 'bed', 'dining table'],
    'human': ['person'],
    'people': ['person'],
    'man': ['person'],
    'woman': ['person'],
    'child': ['person'],
    'kid': ['person'],
    'food': ['banana', 'apple', 'sandwich', 'orange', 'pizza', 'donut', 'cake', 'hot dog'],
    'plant': ['potted plant'],
    'tree': ['potted plant'],
    'phone': ['cell phone'],
    'computer': ['laptop'],
    'monitor': ['tv'],
    'screen': ['tv', 'laptop'],
    'sofa': ['couch'],
    'table': ['dining table'],
}

class YoloDetector:
    """
    Standard object detector using YOLOv8 with fuzzy label matching.
    Supports synonym-based matching so "deer" can match nearby COCO classes.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = YOLO(model_path).to(self.device)
        print(f"[YOLOv8] Loaded weights from {model_path} onto {self.device}.")

    def _get_matching_classes(self, text_prompt: str) -> set:
        """Find all COCO class IDs that match the user's text prompt."""
        prompt_lower = text_prompt.lower().strip().rstrip(".")
        matching_labels = set()
        
        # Direct match: check if any COCO class contains the prompt or vice versa
        for coco_cls in COCO_CLASSES:
            if prompt_lower in coco_cls or coco_cls in prompt_lower:
                matching_labels.add(coco_cls)
        
        # Synonym match: check each word in the prompt against synonym map
        for word in prompt_lower.split():
            word = word.strip()
            if word in SYNONYM_MAP:
                matching_labels.update(SYNONYM_MAP[word])
        
        # If nothing matched, accept ALL detections as a last resort
        if not matching_labels:
            print(f"[YOLOv8] WARNING: No COCO class matches '{text_prompt}'. Accepting all detections.")
            matching_labels = set(COCO_CLASSES)
        else:
            print(f"[YOLOv8] Matching '{text_prompt}' → classes: {matching_labels}")
        
        return matching_labels

    def __call__(self, image: np.ndarray, text_prompt: str, box_threshold: float = 0.3, **kwargs) -> List[Tuple[int, int, int, int]]:
        """
        Args:
            image: RGB image (H, W, 3)
            text_prompt: Object name (e.g., "person", "deer", "white deer")
            box_threshold: Minimum confidence
        Returns:
            List of bounding boxes [x1, y1, x2, y2].
        """
        matching_labels = self._get_matching_classes(text_prompt)
        
        results = self.model(image, conf=box_threshold, verbose=False)
        
        boxes_list = []
        if len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                if label in matching_labels:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    print(f"[YOLOv8] Detected '{label}' (conf={conf:.2f}) at [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
                    boxes_list.append((int(x1), int(y1), int(x2), int(y2)))
        
        return boxes_list
