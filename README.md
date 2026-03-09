# Bone Marrow Cytomorphology Classification with Swin Transformer

This repository contains the implementation of a high-performance pipeline for the automatic multiclass classification of bone marrow cells using the **Swin Transformer** architecture.

## 📌 Overview
Manual classification of bone marrow cells is a fundamental but time-consuming task subject to inter-observer variability. This project leverages the hierarchical self-attention mechanisms of the Swin Transformer to capture both global cellular context and fine-grained morphological features, achieving state-of-the-art results on the Bone-Marrow-Cytomorphology dataset.



## 🚀 Key Features
* **Architecture:** Swin Transformer Tiny, demonstrating superior performance over traditional CNNs.
* **Imbalance Handling:** Utilizes **Focal Loss** ($\gamma=2.0, \alpha=1.0$) and a **WeightedRandomSampler** to mitigate the impact of the dataset's highly skewed class distribution.
* **Data Augmentation:** Includes geometric transformations (flips, rotations, affine) and a custom **color stain augmentation** module to handle variability in staining protocols.
* **Efficiency:** Reaches peak performance (88.35% F1-score) in only 20 training epochs, offering a computationally efficient alternative to complex Siamese networks.

## 📊 Performance
The Swin Transformer model achieved a **weighted F1-score of 88.35%**. 

| Class Examples | Accuracy |
| :--- | :---: |
| Eosinophil | >90% |
| Erythroblast | >90% |
| Plasma cell | >90% |



*A detailed per-class analysis via a normalized confusion matrix is included to highlight the impact of morphological similarity among rare cell types.*

## 🛠️ Methodology & Hyperparameters
* **Dataset:** Bone-Marrow-Cytomorphology (21 classes, highly imbalanced).
* **Input Resolution:** $224 \times 224$ pixels (Center Crop).
* **Optimizer:** AdamW ($LR=1 \times 10^{-4}$, $Weight Decay=1 \times 10^{-5}$).
* **Scheduler:** StepLR (multiplicative factor of 0.8 per epoch).
* **Framework:** PyTorch 2.6.0 with CUDA 12.4 support.
* **Hardware:** Single NVIDIA Tesla P100 GPU (16 GB RAM).

## 🧪 Ablation Study
Our experiments confirm the critical role of the attention mechanism:
* **Full Model:** 88.35% F1-score.
* **Without Attention:** 78.87% (-9.48% drop), proving that self-attention is vital for distinguishing subtle morphological features.

---
*Based on the research: "A Comparative Study of Convolutional and Transformer Models for Bone Marrow Cytomorphology Classification".*
