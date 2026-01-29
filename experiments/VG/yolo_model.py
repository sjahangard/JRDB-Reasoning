"""
YOLO-World model implementation for visual grounding.
"""

import torch
from PIL import Image
from ultralytics import YOLO

from .base_model import BaseVisualGroundingModel
from utils import validate_box
from config import YOLO_MODEL_PATH


class YOLOWorldModel(BaseVisualGroundingModel):
    """YOLO-World open-vocabulary object detection model."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "YOLO-World"
        self.model_path = YOLO_MODEL_PATH
    
    def load(self):
        """Load the YOLO-World model."""
        print(f"\n--- Loading {self.model_name} Model ---")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.model_path).to(self.device)
        print(f"--- {self.model_name} Model loaded on {self.device} ---")
        return self
    
    def predict(self, image: Image.Image, prompt: str, img_width: int, img_height: int) -> list:
        """
        Run YOLO-World inference.
        
        Args:
            image: PIL Image object
            prompt: Text description (used as class name)
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            List of predicted bounding boxes in [x1, y1, x2, y2] format
        """
        
        self.model.set_classes([prompt])
        
        results = self.model(image, verbose=False)
        
        pred_boxes = []
        if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                pred_boxes.append(validate_box(box.tolist(), img_width, img_height))
        
        return pred_boxes