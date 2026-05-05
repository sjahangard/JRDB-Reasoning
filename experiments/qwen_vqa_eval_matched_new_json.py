import os
import json
import torch
import re
import logging
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------
# Paths
# -------------------------
JSON_PATH = "/home/simin/SSD/simin/JRDB-Reasoning/test-jrdb-reasoning/Output_eingine_single_file/VQA_T=1_S=2_human_image_train_2samples_bytes-cafe-2019-02-07_0.json_2026-05-05.json"
IMAGE_ROOT = "/home/simin/SSD/simin/JRDB-Reasoning/JRDB_Social/Dataset/Test_images/image_stitched"

# Keep outputs somewhere writable, not in a protected working directory.
OUTPUT_DIR = os.path.expanduser("~/jrdb_vqa_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(OUTPUT_DIR, "qwen_vqa_results_new_json.json")


class VQAEvaluator:
    def __init__(self, json_data, model, processor):
        self.json_data = json_data
        self.model = model
        self.processor = processor
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        required_fields = ["question", "labels", "data"]
        for field in required_fields:
            if field not in self.json_data:
                raise ValueError(f"Missing required field: {field}")

        self.true_answer, self.answer_type = self._parse_answer()
        self.options = self.json_data.get("question", {}).get("options", [])
        self.option_type = self._parse_option_type()

    def _normalize_text(self, x):
        return str(x).lower().strip()

    def _parse_answer(self):
        """
        New JSON format:
        labels -> slot1 -> final_answer

        Old format fallback:
        labels -> answer
        """
        labels = self.json_data.get("labels", {})

        answer = None
        if "slot1" in labels and isinstance(labels["slot1"], dict):
            answer = labels["slot1"].get("final_answer")
        elif "answer" in labels:
            answer = labels.get("answer")

        if answer is None:
            raise ValueError("Answer field is missing. Expected labels.slot1.final_answer or labels.answer.")

        if isinstance(answer, (int, float)):
            return answer, "numeric"
        if isinstance(answer, str):
            return self._normalize_text(answer), "text"
        if isinstance(answer, list):
            if all(isinstance(item, (int, float)) for item in answer):
                return answer, "numeric_list"
            return [self._normalize_text(a) for a in answer], "text_list"

        raise ValueError(f"Unsupported answer type: {type(answer)}")

    def _parse_option_type(self):
        if not self.options:
            return None
        if all(isinstance(opt, (int, float)) for opt in self.options):
            return "numeric"
        if all(isinstance(opt, str) for opt in self.options):
            return "text"
        return "mixed"

    def _get_seq_name(self):
        """
        New JSON format has seq inside data.seq, e.g.:
        data.seq = 'gates-basement-elevators-2019-01-17_1.json'
        Image folders may be with or without '.json', so load_images tries both.
        """
        seq = self.json_data.get("data", {}).get("seq") or self.json_data.get("seq")
        if not seq:
            raise ValueError("Missing sequence name. Expected data.seq or seq.")
        return seq

    def load_images(self):
        timestamps = self.json_data.get("question", {}).get("timestamps", {}).get("slot1", {})
        image_ids = timestamps.get("image_ids", [])
        if not image_ids:
            raise ValueError("No image IDs found in question.timestamps.slot1.image_ids.")

        seq = self._get_seq_name()
        seq_candidates = [seq]
        if seq.endswith(".json"):
            seq_candidates.append(seq[:-5])

        image_paths = []
        for img_id in image_ids:
            found = False
            for seq_name in seq_candidates:
                img_path = os.path.join(IMAGE_ROOT, seq_name, img_id)
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    found = True
                    break
            if not found:
                logging.warning(f"Image not found for {img_id}. Tried seq candidates: {seq_candidates}")

        if not image_paths:
            raise ValueError(f"No valid images found under IMAGE_ROOT={IMAGE_ROOT}")
        return image_paths

    def _generate_prompt(self, image_paths):
        base_question = self.json_data.get("question", {}).get("question", "")
        if not base_question:
            raise ValueError("Question text is missing.")

        if self.options:
            options_str = ", ".join(map(str, self.options))
            base_question += (
                f"\nChoose only from these options where possible: {options_str}. "
                "Return only a comma-separated list of the selected options."
            )

        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": f"file://{path}"} for path in image_paths],
                {"type": "text", "text": base_question},
            ],
        }]
        return messages

    def predict(self, messages):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

        input_length = inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, input_length:]

        decoded_output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded_output[0].strip() if decoded_output else ""

    def _extract_prediction(self, raw_answer):
        answer_text = self._normalize_text(raw_answer)

        if self.answer_type in ["numeric", "numeric_list"]:
            numbers = re.findall(r"-?\d+\.?\d*", answer_text)
            if not numbers:
                return None
            return [float(n) if "." in n else int(n) for n in numbers]

        cleaned = re.sub(r"[^\w\s,\-]", "", answer_text).strip()
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]

        # Match model output back to available options when options exist.
        if self.options:
            norm_options = {self._normalize_text(opt): opt for opt in self.options}
            matched = []
            for part in parts:
                if part in norm_options:
                    matched.append(self._normalize_text(norm_options[part]))
                else:
                    # substring fallback, useful when model returns short text around option
                    for opt_norm in norm_options:
                        if opt_norm in part or part in opt_norm:
                            matched.append(opt_norm)
                            break
            if matched:
                return sorted(set(matched))

        if len(parts) > 1:
            return parts
        return cleaned

    def evaluate(self, prediction):
        results = {
            "true_answer": self.true_answer,
            "predicted": prediction,
            "exact_match": 0,
            "option_match": None,
            "option_accuracy": 0,
        }

        if not prediction:
            return results

        true_set = set(self.true_answer) if isinstance(self.true_answer, list) else {self.true_answer}
        pred_set = set(prediction) if isinstance(prediction, list) else {prediction}

        results["exact_match"] = int(pred_set == true_set)

        if self.options:
            option_set = {self._normalize_text(opt) for opt in self.options}
            matched_options = sorted(pred_set.intersection(option_set))
            results["option_match"] = matched_options

            # Multi-label option accuracy: all predicted options must be in true labels.
            # Empty match remains 0.
            results["option_accuracy"] = int(bool(matched_options) and set(matched_options).issubset(true_set))

        return results

    def run_evaluation(self):
        try:
            image_paths = self.load_images()
            messages = self._generate_prompt(image_paths)
            raw_answer = self.predict(messages)
            prediction = self._extract_prediction(raw_answer)
            results = self.evaluate(prediction)
            results["raw_answer"] = raw_answer
            results["image_paths"] = image_paths
            return results
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return {
                "true_answer": self.true_answer,
                "predicted": None,
                "raw_answer": None,
                "exact_match": 0,
                "option_match": None,
                "option_accuracy": 0,
                "error": str(e),
            }


def initialize_model():
    logging.info("Initializing model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    return model, processor


if __name__ == "__main__":
    try:
        model, processor = initialize_model()

        with open(JSON_PATH, "r") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            dataset = [dataset]

        all_results = []
        for idx, entry in enumerate(dataset):
            print(f"\nProcessing entry {idx + 1}/{len(dataset)}")
            evaluator = VQAEvaluator(entry, model, processor)
            results = evaluator.run_evaluation()

            item = {
                "index": idx,
                "seq": entry.get("data", {}).get("seq", entry.get("seq")),
                "question": entry.get("question", {}).get("question"),
                **results,
            }
            all_results.append(item)

            print(f"Question: {item['question']}")
            print(f"True Answer: {results['true_answer']}")
            print(f"Raw Answer: {results.get('raw_answer')}")
            print(f"Predicted: {results['predicted']}")
            print(f"Exact Match: {results['exact_match']}")
            print(f"Matched Options: {results['option_match']}")
            print(f"Option Accuracy: {results['option_accuracy']}")
            if "error" in results:
                print(f"Error: {results['error']}")
            print("-----------------------------")

        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results to: {RESULTS_PATH}")

    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        raise
