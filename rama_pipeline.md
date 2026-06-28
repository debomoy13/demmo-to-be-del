# STELSION V3 Hybrid Exoplanet Detection Architecture

## Overview

STELSION V3 is a hybrid deep learning pipeline for exoplanet detection from Kepler light curves. The system combines astronomical preprocessing with multi-branch neural networks to improve robustness against noise and false positives.

---

# Complete Pipeline

```text
Raw Kepler/TESS Light Curve (.npz)
            │
            ▼
────────────────────────────────────
1. Adaptive Preprocessing
────────────────────────────────────
• NaN Interpolation
• 3σ Outlier Clipping
• Noise Estimation
• Adaptive Filtering
    - Savitzky-Golay
    - Wavelet Denoising
    - Median Filtering
• Z-score Normalization

            │
            ▼
────────────────────────────────────
2. Transit Candidate Generation
────────────────────────────────────
Box Least Squares (BLS)

Outputs:
• Period
• Transit Duration
• Transit Depth
• Epoch (t0)

            │
            ▼
────────────────────────────────────
3. Phase Folding
────────────────────────────────────

Generates three representations:

• Global View (2000 samples)
• Local Transit View (200 samples)
• Orbit Matrix (2D representation)

            │
            ▼
────────────────────────────────────
4. Hybrid Neural Network
────────────────────────────────────

Branch A
---------
Global View
→ InceptionTime Blocks
→ Residual Connections

Branch B
---------
Local View
→ 1D CNN
→ Residual CNN Layers

Branch C
---------
Orbit Matrix
→ 2D CNN
→ Minor-Axis Attention
→ Spatial Attention

            │
            ▼
────────────────────────────────────
5. Feature Fusion
────────────────────────────────────

Concatenate

↓

Dense Layer

↓

Dropout

↓

Dense Layer

↓

Sigmoid Output

            │
            ▼
────────────────────────────────────
6. Training Optimization
────────────────────────────────────

• Focal Loss / BCE
• Dynamic Class Weights
• Data Augmentation
• Early Stopping
• ReduceLROnPlateau
• F1 Checkpoint Callback
• Stratified 5-Fold Cross Validation
• Threshold Optimization

            │
            ▼
────────────────────────────────────
7. Inference
────────────────────────────────────

Prediction Probability

↓

Optimal Threshold

↓

Astronomical Heuristics

• Transit Depth Check
• Physical Plausibility
• Reliability Score
• Confidence Score

↓

Final Classification

Planet Candidate / Non-Planet

---

# Model Components

## Input

- Raw Flux
- Observation Time

---

## Preprocessing

- NaN interpolation
- Sigma clipping
- Adaptive denoising
- Z-score normalization

---

## Feature Extraction

- Box Least Squares
- Phase Folding
- Global View
- Local View
- Orbit Matrix

---

## Neural Architecture

- InceptionTime
- 1D CNN
- 2D CNN
- Minor-Axis Attention
- Feature Fusion

---

## Training Strategy

- Dynamic Class Weights
- Focal Loss
- Learning Rate Scheduling
- Cross Validation
- Threshold Calibration

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve
- False Positive Rate
- False Negative Rate

---

## Current Experimental Results

Dataset Size: 932 Samples

Training:
- 757

Validation:
- 175

Optimal Threshold:
- 0.37

Validation Accuracy:
- 80.57%

Precision:
- 76.92%

Recall:
- 72.46%

F1 Score:
- 74.63%

ROC-AUC:
- 83.88%

---

# Future Improvements

- Larger Kepler + TESS datasets
- Hard Negative Mining
- Metadata Fusion (Stellar Radius, Temperature, log g)
- Self-Supervised Pretraining
- Ensemble Learning
- Bayesian Uncertainty Estimation
- Real-time deployment for astronomical surveys
```
