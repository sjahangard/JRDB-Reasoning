# Visual Grounding Benchmark

A unified evaluation framework for visual grounding models on custom datasets.

## Supported Models

| Model | Type | Paper/Source |
|-------|------|--------------|
| YOLO-World | Open-vocabulary Detection | [GitHub](https://github.com/AILab-CVC/YOLO-World) |
| Florence-2 | Vision-Language Model | [HuggingFace](https://huggingface.co/microsoft/Florence-2-base) |
| Grounding DINO | Zero-shot Detection | [GitHub](https://github.com/IDEA-Research/GroundingDINO) |
| Qwen2.5-VL | Vision-Language Model | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/visual-grounding-benchmark.git
cd visual-grounding-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt