# Integrating Facial Feature Recognition and Person Verification  
### Hybrid Deep Learning System using CelebA & LFW (PyTorch)

## Project Overview

This project presents a **hybrid deep learning system** that integrates **facial feature recognition** and **person verification** into a unified identity management pipeline. The system combines a **multi-label facial attribute classifier** trained on the **CelebA dataset** with a **Siamese network–based person verification model** trained on the **Labeled Faces in the Wild (LFW)** dataset.

The goal is to move beyond isolated facial recognition tasks and demonstrate how **feature extraction and identity verification can be jointly leveraged** for more robust, scalable, and real-world identity verification applications such as medical systems, access control, and security.

---

## Key Contributions

- Designed and trained **two complementary deep learning models**:
  - Facial attribute recognition (CelebA)
  - Person verification using similarity learning (LFW)
- Integrated both models into a **single end-to-end identity verification pipeline**
- Conducted **systematic experimentation** on regularization, mini-batching, batch normalization, and optimizers
- Achieved **significant improvements in accuracy, convergence speed, and training efficiency**
- Demonstrated practical applicability to **secure identity management systems**

---

## System Architecture

### 1️⃣ Facial Feature Recognition (CelebA)

**Purpose:** Extract detailed facial attributes to build a rich feature representation.

**Model Architecture**
- Backbone: **Pretrained ResNet-18**
- Fully Connected Head:
  - 512 → 256 (ReLU)
  - Dropout: 0.6
  - 256 → 40 (facial attributes)
- Loss Function: Binary Cross-Entropy with Logits
- Optimizer: Adam (lr = 0.0001, weight decay = 1e-5)

**Attributes Predicted**
Examples include:
- Gender
- Age (Young)
- Hair color
- Facial structure features
- Appearance-related traits

---

### 2️⃣ Person Verification (LFW – Siamese Network)

**Purpose:** Verify whether two facial images belong to the same person.

**Model Architecture**
- Shared convolutional base with 4 convolution layers
- Final embedding size: 4096
- Distance metric: Euclidean distance
- Loss Function: Contrastive Loss (margin = 1.0)
- Optimizer: Adam (lr = 0.0005)

**Learning Objective**
- Minimize distance for same-person image pairs
- Maximize distance for different-person image pairs

---

### 3️⃣ Integrated Pipeline

1. Input face image
2. CelebA model extracts detailed facial attributes
3. Feature representation passed into Siamese network
4. Embeddings compared against stored identities
5. Identity verified using Euclidean distance thresholds

This integration allows the system to **combine descriptive facial attributes with similarity-based verification**, making it more robust than either approach alone. :contentReference[oaicite:0]{index=0}

---

## Results & Performance

| Configuration | CelebA Accuracy | LFW Accuracy | Training Time |
|--------------|----------------|--------------|---------------|
| Baseline | 84% | 78% | High |
| + Dropout | 89% | 80% | Moderate |
| + Mini-Batching | 89% | 80% | Low |
| + Batch Normalization | 90% | 81% | Low |
| + Adam Optimizer | **90%** | **81%** | **Lowest** |

**Key Observations**
- Dropout significantly reduced overfitting for CelebA
- Mini-batching stabilized gradients and reduced training time
- Batch normalization improved convergence speed
- Adam optimizer provided the best overall performance

Detailed experimental analysis and visualizations are documented in the project report and analysis files. :contentReference[oaicite:1]{index=1}

---

## Technical Stack

- **Language:** Python  
- **Framework:** PyTorch  
- **Libraries:** torchvision, pandas, matplotlib  
- **Datasets:** CelebA, LFW  
- **Model Types:** CNN, ResNet-18, Siamese Network  

---

## How to Run

1. Open the Jupyter notebooks:
   - `cnnCelebA.ipynb`
   - `cnnLFW.ipynb`
2. Ensure datasets are correctly downloaded and paths updated
3. Run training and evaluation cells sequentially
4. Use provided examples to test verification scenarios

---

## What This Project Demonstrates

- Strong understanding of **deep learning for computer vision**
- Practical use of **transfer learning**
- Experience with **metric learning and Siamese networks**
- Ability to design and evaluate **end-to-end AI systems**
- Applied experimentation and performance optimization

---

## Academic Context

Developed as part of an AI / Machine Learning coursework submission.  
The project emphasizes **real-world applicability**, **rigorous experimentation**, and **system-level thinking** rather than isolated model training.

---

## Notes

This repository is intended for **educational and portfolio demonstration purposes** and showcases applied deep learning skills in facial analysis and identity verification.
