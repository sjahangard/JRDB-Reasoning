"""
Configuration settings for Visual Grounding Benchmark.
"""

import os

# --- Dataset Paths ---
IMAGES_ROOT = '/path/to/your/images/'
JSON_PATH = '/path/to/your/annotations.json'

# --- Evaluation Parameters ---
IOU_THRESHOLD = 0.5
MODELS_TO_RUN = ["YOLO-World", "Florence-2", "GroundingDINO", "Qwen-VL"]

# --- Model-specific Parameters ---
# GroundingDINO confidence thresholds
GD_BOX_THRESHOLD = 0.3
GD_TEXT_THRESHOLD = 0.25

# --- Model IDs ---
YOLO_MODEL_PATH = "yolov8l-worldv2.pt"
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
QWEN_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"