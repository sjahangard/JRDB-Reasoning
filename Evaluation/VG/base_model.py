"""
Base class for all visual grounding models.
"""

from abc import ABC, abstractmethod
from PIL import Image


class BaseVisualGroundingModel(ABC):
    """Abstract base class for visual grounding models."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.model_name = "BaseModel"
    
    @abstractmethod
    def load(self):
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def predict(self, image: Image.Image, prompt: str, img_width: int, img_height: int) -> list:
        """
        Run inference on an image with a text prompt.
        
        Args:
            image: PIL Image object
            prompt: Text description of object(s) to detect
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            List of predicted bounding boxes in [x1, y1, x2, y2] format
        """
        pass
    
    def __repr__(self):
        return f"{self.model_name}(device={self.device})"