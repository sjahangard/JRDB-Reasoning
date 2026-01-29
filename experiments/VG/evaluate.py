"""
Main evaluation script for Visual Grounding Benchmark.

This script evaluates multiple visual grounding models on a custom dataset,
computing metrics including Mean Average IoU, Precision, Recall, and F1 score.
"""

import os
import json
import time
import torch
import traceback
from PIL import Image
from collections import defaultdict

from config import (
    IMAGES_ROOT, JSON_PATH, IOU_THRESHOLD, MODELS_TO_RUN
)
from utils import (
    calculate_metrics_for_boxes,
    calculate_precision_recall_f1,
    validate_box,
    xywh_to_xyxy
)
from models import (
    YOLOWorldModel,
    Florence2Model,
    GroundingDINOModel,
    QwenVLModel
)


def get_model_instance(model_name: str):
    """
    Factory function to create model instances.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Model instance or None if not recognized
    """
    model_map = {
        "YOLO-World": YOLOWorldModel,
        "Florence-2": Florence2Model,
        "GroundingDINO": GroundingDINOModel,
        "Qwen-VL": QwenVLModel
    }
    
    if model_name in model_map:
        return model_map[model_name]()
    return None


def load_dataset(json_path: str) -> list:
    """
    Load the evaluation dataset from JSON.
    
    Args:
        json_path: Path to the JSON file
    
    Returns:
        List of dataset entries
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} entries from '{json_path}'")
    return dataset


def process_entry(
    entry: dict,
    models: dict,
    images_root: str,
    global_stats: dict,
    per_image_iou_scores: dict
) -> bool:
    """
    Process a single dataset entry through all models.
    
    Args:
        entry: Dataset entry dictionary
        models: Dictionary of loaded model instances
        images_root: Root path to images
        global_stats: Dictionary to accumulate statistics
        per_image_iou_scores: Dictionary to accumulate IoU scores
    
    Returns:
        True if entry was processed successfully, False otherwise
    """
    # Extract image path
    raw_seq_name = entry['data']['seq']
    seq_folder_name = raw_seq_name.replace('.json', '')
    
    if 'slot1' not in entry['labels'] or not entry['labels']['slot1'].get('image_ids'):
        return False
    
    img_filename = entry['labels']['slot1']['image_ids'][0]
    img_path = os.path.join(images_root, seq_folder_name, img_filename)
    
    if not os.path.exists(img_path):
        return False
    
    image_key = f"{seq_folder_name}/{img_filename}"
    
    try:
        
        image_pil = Image.open(img_path).convert("RGB")
        img_width, img_height = image_pil.size
        
        def evaluate_question(q_text: str, gt_boxes_xywh: list, q_type_key: str):
            """Evaluate a single question across all models."""
            # Convert GT boxes from xywh to xyxy format
            gt_boxes_xyxy = [
                validate_box(xywh_to_xyxy(box), img_width, img_height)
                for box in gt_boxes_xywh
            ]
            
            for model_name, model in models.items():
                try:
                    
                    pred_boxes_xyxy = model.predict(
                        image_pil, q_text, img_width, img_height
                    )
                    
                    
                    sum_iou_tp, tp, fp, fn = calculate_metrics_for_boxes(
                        gt_boxes_xyxy, pred_boxes_xyxy
                    )
                    
                    
                    global_stats[model_name][q_type_key]['tp'] += tp
                    global_stats[model_name][q_type_key]['fp'] += fp
                    global_stats[model_name][q_type_key]['fn'] += fn                 
                    total_gt_boxes = len(gt_boxes_xyxy)
                    avg_iou = (sum_iou_tp / total_gt_boxes) if total_gt_boxes > 0 else 0.0
                    per_image_iou_scores[model_name][q_type_key].append(avg_iou)
                    
                except Exception as e:
                    print(f"❌ Error during {model_name} inference on {image_key}: {e}")
                    traceback.print_exc()
        
        
        main_q_text = entry['question']['question']
        gt_main_q_xywh = entry['labels']['slot1'].get('final_bbox', [])
        evaluate_question(main_q_text, gt_main_q_xywh, 'main_question')
        
        sub_questions = entry['question'].get('sub_questions', {})
        slot_label = entry['labels']['slot1']
        
        for sub_q_key, sub_q_text in sub_questions.items():
            gt_sub_q_xywh = []
            if sub_q_key in slot_label and isinstance(slot_label[sub_q_key], dict):
                gt_sub_q_xywh = slot_label[sub_q_key].get('target_bbox', [])
            evaluate_question(sub_q_text, gt_sub_q_xywh, sub_q_key)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing entry {image_key}: {e}")
        traceback.print_exc()
        return False


def print_results(
    model_name: str,
    global_stats: dict,
    per_image_iou_scores: dict,
    iou_threshold: float
):
    """
    Print evaluation results for a model.
    
    Args:
        model_name: Name of the model
        global_stats: Dictionary of accumulated statistics
        per_image_iou_scores: Dictionary of IoU scores
        iou_threshold: IoU threshold used for evaluation
    """
    print(f"\n\n{'='*60}")
    print(f"📊 FINAL RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"IoU Threshold for TP: {iou_threshold}")
    
    stats_by_type = global_stats[model_name]
    iou_by_type = per_image_iou_scores[model_name]
    all_q_types = sorted(set(stats_by_type.keys()) | set(iou_by_type.keys()))
    
    if not all_q_types:
        print(f"  No data was processed for model {model_name}.")
        return
    
    for q_type in all_q_types:
        stats = stats_by_type.get(q_type, {'tp': 0, 'fp': 0, 'fn': 0})
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision, recall, f1 = calculate_precision_recall_f1(tp, fp, fn)
        
        iou_list = iou_by_type.get(q_type, [])
        mean_avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
        
        print(f"\n--- Question Type: {q_type} ---")
        print(f"  Mean Average IoU:  {mean_avg_iou:.4f}")
        print(f"  Precision:         {precision:.4f}")
        print(f"  Recall:            {recall:.4f}")
        print(f"  F1 Score:          {f1:.4f}")
        print(f"  TP: {tp} | FP: {fp} | FN: {fn}")


def main():
    """Main evaluation workflow."""
    print("="*60)
    print("Visual Grounding Benchmark Evaluation")
    print("="*60)
    
    
    print("\n--- Loading Models ---")
    models = {}
    for model_name in MODELS_TO_RUN:
        model = get_model_instance(model_name)
        if model:
            model.load()
            models[model_name] = model
    

    print("\n--- Loading Dataset ---")
    dataset = load_dataset(JSON_PATH)
    
    
    global_stats = {
        model_name: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        for model_name in MODELS_TO_RUN
    }
    per_image_iou_scores = {
        model_name: defaultdict(list)
        for model_name in MODELS_TO_RUN
    }
    
    
    print(f"\n--- Starting Evaluation on {len(dataset)} entries ---")
    processed_count = 0
    
    for idx, entry in enumerate(dataset):
        start_time = time.time()
        
        success = process_entry(
            entry, models, IMAGES_ROOT,
            global_stats, per_image_iou_scores
        )
        
        if success:
            processed_count += 1
            elapsed = time.time() - start_time
            print(f"✅ [{idx+1}/{len(dataset)}] Processed in {elapsed:.2f}s")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Complete: {processed_count}/{len(dataset)} entries processed")
    print(f"{'='*60}")
    
    for model_name in MODELS_TO_RUN:
        print_results(
            model_name,
            global_stats,
            per_image_iou_scores,
            IOU_THRESHOLD
        )
    
    print("\n--- Script Finished ---")


if __name__ == "__main__":
    main()