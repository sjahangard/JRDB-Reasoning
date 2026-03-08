# JRDB-Reasoning: A Difficulty-Graded Benchmark for Visual Reasoning in Robotics

<div align="center">
  <img src="assets/teaser1_new.svg" width="900">
  <p><em>JRDB-Reasoning benchmark overview and task construction pipeline.</em></p>
</div>

<div align="center">
    <a href="https://github.com/sjahangard/JRDB-Reasoning/issues">
        <img src="https://img.shields.io/github/issues/sjahangard/JRDB-Reasoning?style=flat-square">
    </a>
    <a href="https://github.com/sjahangard/JRDB-Reasoning/network/members">
        <img src="https://img.shields.io/github/forks/sjahangard/JRDB-Reasoning?style=flat-square">
    </a>
    <a href="https://github.com/sjahangard/JRDB-Reasoning/stargazers">
        <img src="https://img.shields.io/github/stars/sjahangard/JRDB-Reasoning?style=flat-square">
    </a>
    <a href="https://github.com/sjahangard/JRDB-Reasoning/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/sjahangard/JRDB-Reasoning?style=flat-square">
    </a>
    <a href="https://arxiv.org/abs/2508.10287">
        <img src="https://img.shields.io/badge/arXiv-2508.10287-b31b1b.svg?style=flat-square">
    </a>
</div>

**This repo is the official implementation for the paper [JRDB-Reasoning: A Difficulty-Graded Benchmark for Visual Reasoning in Robotics](https://arxiv.org/pdf/2508.10287) in AAAI 2026.**

## Overview
JRDB-Reasoning is a difficulty-graded benchmark designed to evaluate visual reasoning
capabilities in real-world robotic perception systems. Built on top of the JRDB dataset,
the benchmark introduces structured spatial-temporal reasoning tasks for
Visual Grounding (VG) and Visual Question Answering (VQA), with explicit control over
spatial and temporal complexity.

The benchmark aims to facilitate systematic evaluation of reasoning-based perception models in robotics, beyond pure detection and tracking.

## Repository Structure

The repository is organized as follows:

```text
JRDB-Reasoning/
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ configs/
│  └─ config.yaml
│
├─ src/
│  ├─ main.py
│  │
│  ├─ graph/
│  │  ├─ __init__.py 
│  │  └─ graph.py
│  │
│  ├─ tasks/
│  │  ├─ __init__.py
│  │  ├─ VG_task.py
│  │  └─ VQA_task.py
│  │
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ Classes.py
│  │  └─ Category.py
│  │
│  └─ utils/
│     ├─ __init__.py
│     └─ global_functions.py
│
├─ experiments/
│  ├─ MM-DINO.py
│  └─ sub_question_work_flow_GroundingDINO.py
│
└─ assets/
   ├─ A_GAE.png
   └─ teaser1_new.svg
```

## Setup 
## Requirements 
Python 3.8+ is recommended. Install dependencies using: pip install -r requirements.txt 
### Download the Dataset
The JRDB dataset is not included in this repository.  
Please download it from the official website:

👉 https://jrdb.erc.monash.edu/

Access requires registration and approval.

After downloading the dataset, update the local dataset path in `config.local.yaml` (or modify `data_root` in `config.example.yaml`) before running the code.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{jahangard2025jrdbreasoning,
  title   = {JRDB-Reasoning: A Difficulty-Graded Benchmark for Visual Reasoning in Robotics},
  author  = {Jahangard, Simindokht and Mohammadi, Mehrzad and Shen, Yi and Cai, Zhixi and Rezatofighi, Hamid},
  journal = {arXiv preprint arXiv:2508.10287},
  year    = {2026}
}

