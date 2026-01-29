"""
Qwen-VL model implementation for visual grounding.
"""

import re
import ast
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .base_model import BaseVisualGroundingModel
from config import QWEN_MODEL_PATH


class QwenVLModel(BaseVisualGroundingModel):
    """Qwen2.5-VL vision-language model for visual grounding."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "Qwen-VL"
        self.model_path = QWEN_MODEL_PATH
        self.system_prompt = (
            "You are a precise object detection system. "
            "Respond ONLY with a valid JSON array of bounding boxes "
            "in the format: [{'bbox_2d': [x1,y1,x2,y2]}]"
        )
    
    def load(self):
        """Load the Qwen-VL model and processor."""
        print(f"\n--- Loading {self.model_name} Model ---")
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        self.device = self.model.device
        print(f"--- {self.model_name} Model loaded on {self.device} ---")
        return self
    
    def _parse_response(
        self, 
        response: str, 
        model_w: int, 
        model_h: int, 
        orig_w: int, 
        orig_h: int
    ) -> list:
        """
        Parse Qwen-VL response and convert coordinates.
        
        Args:
            response: Model response string
            model_w: Model input width
            model_h: Model input height
            orig_w: Original image width
            orig_h: Original image height
        
        Returns:
            List of bounding boxes in [x1, y1, x2, y2] format
        """
        try:
            
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if not match:
                return []
            
            
            data = ast.literal_eval(match.group(0))
            
            
            boxes = []
            for item in data:
                if 'bbox_2d' in item:
                    x1, y1, x2, y2 = item['bbox_2d']
                    # Scale from model coordinates to original image coordinates
                    x1 = x1 / model_w * orig_w
                    x2 = x2 / model_w * orig_w
                    y1 = y1 / model_h * orig_h
                    y2 = y2 / model_h * orig_h
                    boxes.append([x1, y1, x2, y2])
            
            return boxes
            
        except Exception:
            return []
    
    def predict(self, image: Image.Image, prompt: str, img_width: int, img_height: int) -> list:
        """
        Run Qwen-VL inference.
        
        Args:
            image: PIL Image object
            prompt: Text description of object(s) to detect
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            List of predicted bounding boxes in [x1, y1, x2, y2] format
        """
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}. Respond ONLY with a valid JSON array using bbox_2d."
                    },
                    {"type": "image"}
                ]
            }
        ]
        
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        
        image_grid_thw = inputs['image_grid_thw'][0]
        model_input_h = image_grid_thw[1].item() * 14
        model_input_w = image_grid_thw[2].item() * 14
        
       
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
        
        
        response = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        
        
        return self._parse_response(
            response, 
            model_input_w, 
            model_input_h, 
            img_width, 
            img_height
        )