import os
import json
import torch
from PIL import Image, ImageDraw
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# =========================
# Config
# =========================
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

# New JSON format, e.g.:
# labels -> slot1 -> image_ids / final_bbox / age,race,... -> target_bbox
JSON_PATH = os.path.expanduser(
    "~/PycharmProjects/pythonProject_visulizastion/Rewrite_Engine/Engine_Clean2/out_engine/"
    "VG_T=1_S=2_human_image_train_2samples_gates-ai-lab-2019-02-08_0.json_2026-05-05.json"
)

# Change this if your images are somewhere else.
IMAGES_ROOT = os.path.expanduser("~/JRDB-Reasoning/Test_images/image_stitched")

IOU_THRESHOLD = 0.5
RESIZE_FACTOR = 0.5
VISUALIZE = False
VIS_DIR = "visualizations"

print("Model:", MODEL_PATH)
print("JSON:", JSON_PATH)
print("Images root:", IMAGES_ROOT)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
).eval()


# =========================
# Box / metric utilities
# =========================
def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0)
    box2_area = max(box2[2] - box2[0], 0) * max(box2[3] - box2[1], 0)
    denom = box1_area + box2_area - inter_area
    return inter_area / denom if denom > 0 else 0.0


def validate_box(box, img_width, img_height):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(x1, img_width)))
    y1 = int(max(0, min(y1, img_height)))
    x2 = int(max(0, min(x2, img_width)))
    y2 = int(max(0, min(y2, img_height)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def xywh_to_xyxy(box, img_width, img_height):
    """The new JRDB-Reasoning JSON stores GT as [x, y, w, h]."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x, y, w, h = box
    return validate_box([x, y, x + w, y + h], img_width, img_height)


def calculate_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    matches = []
    remaining_preds = pred_boxes.copy()

    for gt in gt_boxes:
        best_iou = 0.0
        best_pred = None
        for pred in remaining_preds:
            iou = compute_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred
        if best_pred is not None and best_iou >= iou_threshold:
            matches.append(best_iou)
            remaining_preds.remove(best_pred)

    tp = len(matches)
    fp = len(remaining_preds)
    fn = len(gt_boxes) - tp
    return sum(matches), tp, fp, fn


def calculate_final_metrics(stats):
    precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
    recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = stats["sum_iou"] / stats["tp"] if stats["tp"] > 0 else 0
    return precision, recall, f1, avg_iou


# =========================
# JSON parsing for new format
# =========================
def strip_json_suffix(seq):
    # New data['seq'] is like gates-ai-lab-2019-02-08_0.json.
    # Image folders are often stored without the .json suffix.
    return seq[:-5] if isinstance(seq, str) and seq.endswith(".json") else seq


def get_slot(entry, slot_name="slot1"):
    labels = entry.get("labels", {})
    slot = labels.get(slot_name)
    if slot is None:
        # Fallback: take the first labels slot if slot1 is absent.
        for value in labels.values():
            if isinstance(value, dict) and "image_ids" in value:
                return value
    return slot if isinstance(slot, dict) else {}


def choose_gt_boxes(slot, question_dict):
    """
    For the main VG question, use slot['final_bbox'].
    If you later want to evaluate a sub-question, set QUESTION_KEY below and this
    function will use slot[QUESTION_KEY]['target_bbox'].
    """
    QUESTION_KEY = None  # e.g. "age", "age, race", "human"; None = final_bbox

    if QUESTION_KEY and isinstance(slot.get(QUESTION_KEY), dict):
        return slot[QUESTION_KEY].get("target_bbox", [])

    # This is the correct target for the full question in the new JSON.
    return slot.get("final_bbox", [])


def resolve_image_path(seq, img_name, images_root):
    seq_no_json = strip_json_suffix(seq)

    candidates = [
        os.path.join(images_root, seq, img_name),
        os.path.join(images_root, seq_no_json, img_name),
        os.path.join(images_root, img_name),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return path

    # Return the most likely path for a helpful missing-image message.
    return candidates[1]


def extract_entry_fields(entry):
    data = entry.get("data", {})
    question_obj = entry.get("question", {})
    slot = get_slot(entry, "slot1")

    seq = data.get("seq", "")
    image_ids = slot.get("image_ids") or question_obj.get("timestamps", {}).get("slot1", {}).get("image_ids")
    img_name = image_ids[0] if isinstance(image_ids, list) and image_ids else None

    question = question_obj.get("question", "") if isinstance(question_obj, dict) else str(question_obj)
    raw_gt_boxes = choose_gt_boxes(slot, question_obj)

    img_path = resolve_image_path(seq, img_name, IMAGES_ROOT) if img_name else None
    return img_path, question, raw_gt_boxes


# =========================
# Model output parsing
# =========================
def parse_json_from_text(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def parse_model_output(output_text, orig_width, orig_height, resize_factor):
    try:
        json_str = parse_json_from_text(output_text)
        data = json.loads(json_str)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        boxes = []
        for item in data:
            if not isinstance(item, dict):
                continue
            coords = item.get("bbox_2d") or item.get("bbox") or item.get("box")
            if not isinstance(coords, list) or len(coords) != 4:
                continue
            if not all(isinstance(c, (int, float)) for c in coords):
                continue

            # The model sees the resized image, so scale predictions back.
            scaled = [c / resize_factor for c in coords]
            boxes.append(validate_box(scaled, orig_width, orig_height))
        return boxes
    except Exception as e:
        print(f"Parsing failed: {e}")
        print("Raw output was:", output_text)
        return []


# =========================
# Inference / visualization
# =========================
def run_inference(image, question, resize_factor):
    orig_width, orig_height = image.size
    resized_image = image.resize((int(orig_width * resize_factor), int(orig_height * resize_factor)))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise visual grounding system. "
                "Return ONLY valid JSON. No markdown, no explanation. "
                "Format: [{\"bbox_2d\": [x1, y1, x2, y2]}]."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"image": resized_image},
            ],
        },
    ]

    inputs = processor(
        text=[processor.apply_chat_template(messages, tokenize=False)],
        images=[resized_image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )

    response = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    print("Raw Model Response:", response)
    return parse_model_output(response, orig_width, orig_height, resize_factor)


def visualize_results(image, gt_boxes, pred_boxes, output_path):
    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        draw.rectangle(box, outline="green", width=3)
    for box in pred_boxes:
        draw.rectangle(box, outline="red", width=2)
    image.save(output_path)


# =========================
# Main
# =========================
with open(JSON_PATH, "r") as f:
    dataset = json.load(f)

if isinstance(dataset, dict):
    dataset = dataset.get("data") or dataset.get("samples") or dataset.get("annotations") or dataset.get("items") or []

if not isinstance(dataset, list) or len(dataset) == 0:
    raise ValueError(f"JSON file is empty or invalid: {JSON_PATH}")

os.makedirs(VIS_DIR, exist_ok=True)

global_stats = {"tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0}
per_image_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0})
per_image_iou_scores = []
processed = 0

for idx, entry in enumerate(dataset):
    img_path = None
    try:
        img_path, question, raw_gt_boxes = extract_entry_fields(entry)

        if not img_path or not os.path.exists(img_path):
            print(f"[{idx}] Missing image: {img_path}")
            continue
        if not question:
            print(f"[{idx}] Missing question")
            continue
        if not raw_gt_boxes:
            print(f"[{idx}] Missing GT boxes: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # New JSON GT boxes are [x, y, w, h].
        gt_boxes = [xywh_to_xyxy(b, width, height) for b in raw_gt_boxes]
        gt_boxes = [b for b in gt_boxes if b is not None]

        if len(gt_boxes) == 0:
            print(f"[{idx}] No valid GT boxes after conversion: {img_path}")
            continue

        print(f"\n[{idx}] Image: {img_path}")
        print(f"Question: {question}")
        print(f"GT boxes xyxy: {gt_boxes}")

        pred_boxes = run_inference(image, question, RESIZE_FACTOR)
        print(f"Pred boxes xyxy: {pred_boxes}")

        sum_iou, tp, fp, fn = calculate_metrics(gt_boxes.copy(), pred_boxes.copy(), IOU_THRESHOLD)

        global_stats["tp"] += tp
        global_stats["fp"] += fp
        global_stats["fn"] += fn
        global_stats["sum_iou"] += sum_iou

        per_image_stats[img_path].update({"tp": tp, "fp": fp, "fn": fn, "sum_iou": sum_iou})
        per_image_iou_scores.append(sum_iou / tp if tp > 0 else 0)
        processed += 1

        if VISUALIZE:
            vis_path = os.path.join(VIS_DIR, f"result_{idx + 1}.jpg")
            visualize_results(image.copy(), gt_boxes, pred_boxes, vis_path)
            print(f"Visualization saved to: {vis_path}")

        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error processing entry {idx}, image={img_path}: {e}")
        continue

print("\n=== Per-image Results ===")
for img_path, stats in per_image_stats.items():
    p, r, f1, iou = calculate_final_metrics(stats)
    print(f"\nImage: {img_path}")
    print(f"  Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}, IoU: {iou:.2f}")
    print(f"  TP: {stats['tp']}, FP: {stats['fp']}, FN: {stats['fn']}")

print("\n=== Global Results ===")
g_precision, g_recall, g_f1, g_avg_iou = calculate_final_metrics(global_stats)
mean_iou = sum(per_image_iou_scores) / len(per_image_iou_scores) if per_image_iou_scores else 0
print(f"Processed:  {processed}/{len(dataset)}")
print(f"Precision:  {g_precision:.4f}")
print(f"Recall:     {g_recall:.4f}")
print(f"F1 Score:   {g_f1:.4f}")
print(f"Avg IoU:    {g_avg_iou:.4f}")
print(f"Mean IoU:   {mean_iou:.4f}")
print(f"TP: {global_stats['tp']}, FP: {global_stats['fp']}, FN: {global_stats['fn']}")
print("Model:", MODEL_PATH)
print("JSON:", JSON_PATH)
