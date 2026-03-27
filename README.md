# Augmentation Scheduling on CIFAR-10 — Baseline vs Scheduled RandAugment

Research project conducted at **POSTECH (GSAI)**, 2025.

## Problem

Strong data augmentation (RandAugment, ColorJitter, Random Erasing) is widely used to improve model robustness, but applying it from the start of training can slow optimization and increase compute cost. This raises a practical scheduling question: when should strong augmentation be introduced during training?

## Approach

We compare three augmentation strategies on CIFAR-10 with ResNet-18:

1. **Baseline-only** — RandomCrop + HorizontalFlip throughout training
2. **Strong-only** — adds RandAugment (N=2, M=9), ColorJitter, and Random Erasing throughout
3. **Staged schedules** — start with baseline, switch to strong augmentation either at a fixed epoch (3, 6, 9, 12) or conditionally when validation accuracy plateaus

Evaluation covers clean accuracy, rotation robustness, Gaussian noise robustness, calibration (ECE, NLL), confidence statistics, and wall-clock training time.

## Key Results

| Strategy | Score | Time (s) |
|---|---|---|
| Baseline-only | 147.01 | 1049 |
| Strong-only | 148.64 | 1759 (+67.6%) |
| Staged conditional | 150.13 | 1040 |
| Staged fixed (epoch 12) | **151.53** | 1190 |

- Strong-only training is **Pareto-dominated** by staged schedules: more compute for marginally better score
- Staged conditional switching achieves near-baseline training time with +3.11 score gain over baseline
- Fixed-switch schedules are non-monotonic — switching at epoch 6 or 9 underperforms both earlier and later switches
- Strong augmentation primarily improves rotation robustness (+8.77 pp RotMean) at the cost of clean accuracy (−1.70 pp)

## Repository Structure

```
├── CIFAR_10.ipynb   # Full training and evaluation pipeline
└── README.md
```

## How to Use

Open in **Google Colab** with GPU runtime (T4 is sufficient).

The notebook is self-contained:
1. Run all cells in order
2. CIFAR-10 is downloaded automatically (~170 MB)
3. All strategies are trained sequentially and results are aggregated in a summary table

Training all strategies takes approximately 2–3 hours on a T4 GPU.

Key hyperparameters are defined at the top of the notebook and can be adjusted:
- `MAX_EPOCHS = 40`
- `RAND_AUG_N`, `RAND_AUG_M` — RandAugment parameters
- `patience`, `min_delta` — plateau detection for conditional switching

## Stack

Python · PyTorch · torchvision · NumPy · Pandas · Matplotlib

## Report

*Preprint — link to be added.*

## Context

Student research project, POSTECH Graduate School of AI (GSAI), 2025.  
Authors: Zachari Arnaud, Noé Stefani.
