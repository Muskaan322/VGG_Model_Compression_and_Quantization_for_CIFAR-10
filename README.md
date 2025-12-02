# VGG Model Compression and Quantization for CIFAR-10

## Overview
This project compresses and quantizes the VGG model for the CIFAR-10 dataset to reduce model size and improve inference speed while maintaining high accuracy.

## Folder Structure
- **data/**: CIFAR-10 dataset  
- **models/**: Original and compressed models  
- **results/**: Evaluation results and plots  
- **scripts/**: Training, pruning, and quantization scripts  

## Features
- Reduce model size using pruning  
- Speed up inference with quantization  
- Compare performance of original vs compressed models  

## Results
- Accuracy: Original ~93%, Pruned ~92%, Quantized ~91%  
- Model Size: Original 200 MB → Pruned 120 MB → Quantized 50 MB  
- Inference time reduced after quantization  

## References
- [VGG Paper](https://arxiv.org/abs/1409.1556)  
- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)  
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
