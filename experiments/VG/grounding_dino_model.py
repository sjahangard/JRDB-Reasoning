"""
Grounding DINO model implementation for visual grounding.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from .base_model import BaseVisualGroundingModel
from utils import validate_box
from config import GROUNDING_DINO_MODEL_ID, GD_BOX_THRESHOLD, GD_TEXT_THRESHOLD


class GroundingDINOModel(BaseVisualGroundingModel):
    """Grounding DINO zero-shot object detection model."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "GroundingDINO"
        self.model_id = GROUNDING_DINO_MODEL_ID
        self.box_threshold = GD_BOX_THRESHOLD
        self.text_threshold = GD_TEXT_THRESHOLD
    
    def load(self):
        """Load the Grounding DINO model and processor."""
        print(f"\n--- Loading {self.model_name} Model ---")
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        
        print(f"--- {self.model_name} Model loaded on {self.device} ---")
        return self
    
    def _preprocess_prompt(self, prompt: str) -> str:
        """
        Preprocess the prompt for Grounding DINO.
        
        Args:
            prompt: Raw text prompt
        
        Returns:
            Processed prompt suitable for Grounding DINO
        """
        # Clean and format the prompt
        clean_prompt = prompt.replace("Find ", "")
        clean_prompt = clean_prompt.replace(" persons", " person")
        clean_prompt = clean_prompt.lower() + " ."
        return clean_prompt
    
    def predict(self, image: Image.Image, prompt: str, img_width: int, img_height: int) -> list:
        """
        Run Grounding DINO inference.
        
        Args:
            image: PIL Image object
            prompt: Text description of object(s) to detect
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            List of predicted bounding boxes in [x1, y1, x2, y2] format
        """
        # Preprocess prompt
        processed_prompt = self._preprocess_prompt(prompt)
        
        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=processed_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(img_height, img_width)]
        )
        
        
        pred_boxes = [
            validate_box(box.tolist(), img_width, img_height) 
            for box in results[0]['boxes']
        ]
        
        return pred_boxes