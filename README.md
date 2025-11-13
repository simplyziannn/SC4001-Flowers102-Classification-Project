# SC4001 – Flowers102 Classification Project
Complete Baseline, Augmentation, Tuning, Ensemble, and Few-Shot Pipeline

This repository contains the full workflow for the SC4001 Neural Networks and Deep Learning project.  
The goal is to classify the Oxford Flowers-102 dataset using both supervised transfer learning and metric-learning approaches.

The project is organised into the following notebooks:

1. project_full_pipeline (Baseline+fewshot).ipynb  
2. project_full_pipeline (hyperparameter finetuning).ipynb  
3. project_full_pipeline (ensemble).ipynb  
4. project_ful_pipeline(Visualization).ipynb  
5. flowers_common.py (shared utilities)

---

## 1. Baseline and Few-Shot Models
Baseline (no augmentation): ResNet-50 with a cosine classifier, achieving 78.78%.  
Augmented baseline: Adds cropping, flipping, colour jitter, and rotation, improving accuracy to 90.32%.

Few-Shot Models:
- Siamese Network (contrastive loss)  
  - 1-shot: 96.66%  
  - 5-shot: 98.34%
- Triplet Network (triplet loss, margin 0.8)  
  - 1-shot: 96.72%  
  - 5-shot: 98.26%

Few-shot methods outperform all supervised models except the largest ensemble.

---

## 2. Hyperparameter Tuning
Performed using:
- Greedy coordinate search  
- Optuna TPE joint search  

Optuna produced the best supervised single-model result at 91.97%.

---

## 3. K-Fold Ensembles
A complete ensemble study was run with k = 1 to 15.  
Final predictions were obtained via softmax probability averaging.

The best supervised model is the 12-fold large-train Optuna ensemble (96.89%).

---

## 4. Visualization Notebook
Contains plots and summaries for:
- Training and validation curves  
- Confusion matrices  
- Hyperparameter search results  
- Ensemble accuracy trends  

---

## Repository Structure

```
├── flowers_common.py
├── project_full_pipeline (Baseline+fewshot).ipynb
├── project_full_pipeline (hyperparameter finetuning).ipynb
├── project_full_pipeline (ensemble).ipynb
├── project_ful_pipeline(Visualization).ipynb
├── ckpt/
├── data/
├── utils/
└── README.md
```
---

## Requirements
- Python 3.10+  
- PyTorch  
- torchvision  
- numpy  
- matplotlib  
- tqdm  
- optuna  
