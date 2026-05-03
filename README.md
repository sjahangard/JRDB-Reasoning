# 🧠 JRDB-Reasoning: A Difficulty-Graded Benchmark for Visual Reasoning in Robotics

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

---

## 🚀 Overview
JRDB-Reasoning is a difficulty-graded benchmark designed to evaluate visual reasoning capabilities in real-world robotic perception systems. Built on top of the JRDB dataset, the benchmark introduces structured spatial-temporal reasoning tasks for **Visual Grounding (VG)** and **Visual Question Answering (VQA)**, with explicit control over spatial and temporal complexity.

The benchmark aims to facilitate systematic evaluation of reasoning-based perception models in robotics, going beyond traditional detection and tracking.

---

## ⚙️ Setup
We recommend using a Conda environment with Python 3.10.

```bash
conda create -n stsg_env python=3.10 -y
conda activate stsg_env

pip install --upgrade pip
pip install -r requirements.txt

```
## Quick Start
The generated scene graphs are stored in the `seq_graphs` folder. Please update the required settings in `config.yaml`; the corresponding JSON files will be generated in the `out_engine` folder.

```bash
git clone https://github.com/sjahangard/JRDB-Reasoning.git
cd JRDB-Reasoning

python Main.py

```

## Download the Dataset
The whole JRDB dataset is not included in this repository.  
Please download it from the official website:

👉 https://jrdb.erc.monash.edu/

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{jahangard2026jrdb,
  title={JRDB-Reasoning: A Difficulty-Graded Benchmark for Visual Reasoning in Robotics},
  author={Jahangard, Simindokht and Mohammadi, Mehrzad and Shen, Yi and Cai, Zhixi and Rezatofighi, Hamid},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={7},
  pages={5276--5286},
  year={2026}
}
