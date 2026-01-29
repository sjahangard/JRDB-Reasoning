"""
Utility functions for metric calculation and box operations.
"""

from config import IOU_THRESHOLD


def compute_iou(box1: list, box2: list) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First box in [x1, y1, x2, y2] format
        box2: Second box in [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def validate_box(box: list, img_width: int, img_height: int) -> list:
    """
    Clip bounding box coordinates to image boundaries.
    
    Args:
        box: Bounding box in [x1, y1, x2, y2] format
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Clipped bounding box
    """
    x1, y1, x2, y2 = box
    return [
        max(0, min(float(x1), float(img_width - 1))),
        max(0, min(float(y1), float(img_height - 1))),
        max(0, min(float(x2), float(img_width - 1))),
        max(0, min(float(y2), float(img_height - 1)))
    ]


def calculate_metrics_for_boxes(
    gt_boxes_xyxy: list, 
    pred_boxes_xyxy: list,
    iou_threshold: float = IOU_THRESHOLD
) -> tuple:
    """
    Calculate detection metrics using greedy matching.
    
    For each ground truth box, finds the best matching prediction (highest IoU).
    A match is considered a True Positive if IoU >= threshold.
    
    Args:
        gt_boxes_xyxy: List of ground truth boxes in [x1, y1, x2, y2] format
        pred_boxes_xyxy: List of predicted boxes in [x1, y1, x2, y2] format
        iou_threshold: Minimum IoU for a valid match
    
    Returns:
        Tuple of (sum_iou_for_tp, true_positives, false_positives, false_negatives)
    """
    tp, fp, fn = 0, 0, 0
    sum_iou_for_tp = 0.0
    matched_pred_indices = set()

    
    if not gt_boxes_xyxy:
        return 0.0, 0, len(pred_boxes_xyxy), 0
    
    if not pred_boxes_xyxy:
        return 0.0, 0, 0, len(gt_boxes_xyxy)

    
    for gt_box in gt_boxes_xyxy:
        best_iou = 0.0
        best_pred_idx = -1
        
        for pred_idx, pred_box in enumerate(pred_boxes_xyxy):
            if pred_idx in matched_pred_indices:
                continue
                
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx

        
        if best_iou >= iou_threshold and best_pred_idx != -1:
            tp += 1
            sum_iou_for_tp += best_iou
            matched_pred_indices.add(best_pred_idx)

    
    fp = len(pred_boxes_xyxy) - len(matched_pred_indices)
    fn = len(gt_boxes_xyxy) - tp
    
    return sum_iou_for_tp, tp, fp, fn


def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> tuple:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        tp: Number of true positives
        fp: Number of false positives
        fn: Number of false negatives
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def xywh_to_xyxy(box: list) -> list:
    """
    Convert bounding box from [x, y, w, h] to [x1, y1, x2, y2] format.
    
    Args:
        box: Bounding box in [x, y, width, height] format
    
    Returns:
        Bounding box in [x1, y1, x2, y2] format
    """
    x, y, w, h = box
    return [x, y, x + w, y + h]