"""
Florence-2 model implementation for visual grounding.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from .base_model import BaseVisualGroundingModel
from utils import validate_box
from config import FLORENCE_MODEL_ID


class Florence2Model(BaseVisualGroundingModel):
    """Microsoft Florence-2 vision-language model for grounding."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "Florence-2"
        self.model_id = FLORENCE_MODEL_ID
        self.task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    
    def load(self):
        """Load the Florence-2 model and processor."""
        print(f"\n--- Loading {self.model_name} Model ---")
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype
        ).eval().to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        print(f"--- {self.model_name} Model loaded on {self.device} ---")
        return self
    
    def predict(self, image: Image.Image, prompt: str, img_width: int, img_height: int) -> list:
        """
        Run Florence-2 grounding inference.
        
        Args:
            image: PIL Image object
            prompt: Text description of object(s) to detect
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            List of predicted bounding boxes in [x1, y1, x2, y2] format
        """
        
        full_prompt = self.task_prompt + prompt
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        
        if self.device.type == "cuda":
            inputs = {
                k: v.to(dtype=self.dtype) if v.is_floating_point() else v 
                for k, v in inputs.items()
            }
        
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]
        
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=(img_width, img_height)
        )
        
        # Extract and validate boxes
        pred_data = parsed_answer.get(self.task_prompt, {})
        pred_boxes = [
            validate_box(box, img_width, img_height) 
            for box in pred_data.get('bboxes', [])
        ]
        
        return pred_boxes