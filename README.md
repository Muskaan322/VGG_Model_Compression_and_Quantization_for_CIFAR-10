# VGG Model Compression and Quantization for CIFAR-10

![Project Banner](results/banner.png)  <!-- Optional banner image -->

## Overview
This project implements **model compression and quantization** techniques on a VGG network trained on the CIFAR-10 dataset.  
The goal is to **reduce model size, speed up inference**, and maintain high accuracy.  
Techniques applied include **pruning, post-training quantization, and fine-tuning**.

---

## Table of Contents
- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Methodology](#methodology)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Plots & Visuals](#plots--visuals)
- [References](#references)
- [Author](#author)

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

---

## Methodology
1. **Model Selection**  
   Start with a pre-trained VGG model for CIFAR-10 classification.  

2. **Pruning (Compression)**  
   Reduce redundant weights in the network without significantly affecting accuracy.  

3. **Quantization**  
   Convert model weights from `float32` to `int8` to decrease model size and inference time.  

4. **Fine-tuning**  
   Train the compressed and quantized models to recover any lost accuracy.  

5. **Evaluation & Comparison**  
   Analyze **accuracy, model size, and inference time** for original, pruned, and quantized models.

---

## Dependencies
- Python >= 3.9  
- PyTorch >= 2.1  
- torchvision  
- numpy  
- matplotlib  

Install dependencies using:
```bash
pip install -r requirements.txt
## Results

### Model Performance Comparison
| Model Version  | Accuracy | Model Size (MB) | Inference Time (ms) |
|----------------|---------|----------------|-------------------|
| Original VGG   | 93%     | 200            | 25                |
| Pruned VGG     | 92%     | 120            | 18                |
| Quantized VGG  | 91%     | 50             | 10                |

---

### Accuracy & Loss Curves
![Accuracy & Loss](results/accuracy_loss_plot.png)  
*Figure 1: Training and validation accuracy & loss curves for different models.*

---

### Model Size Comparison
![Model Size](results/model_size_plot.png)  
*Figure 2: Comparison of model sizes before and after compression/quantization.*

---

### Inference Time Comparison
![Inference Time](results/inference_time_plot.png)  
*Figure 3: Inference time comparison for original, pruned, and quantized models.*

---

### Notes
- Accuracy is measured on the CIFAR-10 test dataset.  
- Model size is calculated after saving the model in `.pth` format.  
- Inference time is measured per image on CPU.
