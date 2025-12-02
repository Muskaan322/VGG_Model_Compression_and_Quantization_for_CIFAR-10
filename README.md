  # VGG Model Compression & Quantization for CIFAR-10

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12-red)
![License](https://img.shields.io/badge/License-MIT-green)

Efficient VGG models using **compression** and **quantization** techniques for faster inference and lower memory usage on CIFAR-10 dataset.

---

## Table of Contents
- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [Contributing](#contributing)

---

## Overview
VGG models are accurate but resource-heavy. This project reduces their size and inference time using **pruning**, **weight optimization**, and **quantization**, making them suitable for deployment on edge devices while maintaining high CIFAR-10 accuracy.

---

## Folder Structure
```text
VGG_Model_Compression_and_Quantization/
│
├── data/                  # CIFAR-10 dataset or download scripts
├── models/                # Saved original & compressed models
├── scripts/               # Training, evaluation, and quantization scripts
├── notebooks/             # Optional Jupyter notebooks for analysis
├── results/               # Plots, GIFs, performance metrics
├── requirements.txt       # Required Python packages
└── README.md
# Clone the repository
git clone https://github.com/Muskaan322/VGG_Model_Compression_and_Quantization_for_CIFAR-10.git

# Navigate to project
cd VGG_Model_Compression_and_Quantization_for_CIFAR-10

# Install dependencies
pip install -r requirements.txt
python scripts/train.py --epochs 50 --quantize True
python scripts/evaluate.py --model_path models/compressed_vgg.pth

