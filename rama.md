Yes. For a technical round, judges usually ask about **architecture, hyperparameters, training, preprocessing, evaluation, and deployment**. Below is a concise "cheat sheet" you can keep open.

---

# STELSION V3 Model Specification

## Model Type

```
Hybrid Deep Learning Network
```

Combines:

* InceptionTime
* 1D CNN
* 2D CNN
* Minor-Axis Attention
* Feature Fusion

---

# Input

```
Raw Kepler Light Curve (.npz)
```

Contains

```
time[]
flux[]
```

---

# Input Representations

| Branch       | Shape     | Purpose                   |
| ------------ | --------- | ------------------------- |
| Global View  | 2000      | Entire orbital behaviour  |
| Local View   | 200       | Transit morphology        |
| Orbit Matrix | 2D Matrix | Periodic spatial patterns |

---

# Preprocessing

Order:

```
NaN Interpolation
↓

3σ Outlier Clipping
↓

Noise Estimation

↓

Adaptive Filtering

↓

Detrending

↓

Z-score Normalization

↓

BLS

↓

Phase Folding
```

---

# Filters Used

Depending on noise

Low Noise

```
Savitzky-Golay
```

Medium

```
Wavelet
```

High

```
Wavelet + Median
```

---

# Period Detection

Algorithm

```
Box Least Squares (BLS)
```

Outputs

* Period
* Duration
* Epoch
* Depth

---

# Neural Network

## Branch 1

```
InceptionTime
```

Purpose

Multiple kernel sizes capture

* short transit
* medium transit
* long transit

---

## Branch 2

```
Residual 1D CNN
```

Purpose

Learn local transit profile.

---

## Branch 3

```
2D CNN

+

Minor Axis Attention
```

Purpose

Detect repeated orbital bands.

---

# Attention

Used

```
Minor Axis Attention
```

Learns

* orbital periodicity
* important regions
* suppresses noise

---

# Fusion

```
Concatenate

↓

Dense

↓

Dropout

↓

Dense

↓

Sigmoid
```

---

# Activation

Hidden

```
ReLU
```

Output

```
Sigmoid
```

---

# Loss

Configurable

```
Binary Cross Entropy

or

Focal Loss
```

---

# Optimizer

```
Adam
```

---

# Learning Rate

```
0.0003
```

---

# Batch Size

(check config)

Probably

```
8

or

16
```

---

# Epochs

Maximum

```
100
```

Stopped by

```
EarlyStopping
```

---

# Best Epoch

```
80
```

---

# Regularization

* Dropout
* Layer Normalization
* EarlyStopping
* ReduceLROnPlateau

---

# Dataset

```
932 samples
```

Train

```
757
```

Validation

```
175
```

---

# Class Distribution

Positive

```
390
```

Negative

```
542
```

---

# Augmentation

Training only

* Time Shift
* Gaussian Noise
* Amplitude Scaling

---

# Cross Validation

```
5 Fold Stratified
```

---

# Threshold

Default ML

```
0.5
```

Our model

```
0.37
```

Chosen by maximizing Validation F1.

---

# Metrics

Accuracy

```
80.57%
```

Precision

```
76.92%
```

Recall

```
72.46%
```

F1

```
74.63%
```

ROC AUC

```
83.88%
```

---

# Parameters

```
172,627
```

---

# Training Time

```
1448 sec

≈24 minutes
```

---

# Inference

```
13.5 ms/sample
```

---

# Output

Model predicts

```
Probability

↓

Threshold

↓

Astronomical Checks

↓

Planet Candidate
```

---

# Explainability

Generated

* Confusion Matrix
* ROC Curve
* Precision Recall Curve
* Probability Histogram

---

# Why not only CNN?

Answer:

> A single CNN learns local transit features but struggles with multi-scale temporal patterns. InceptionTime captures multiple transit durations, while the 2D branch and Minor-Axis Attention learn periodic orbital structures that are difficult for a plain CNN to model.

---

# Why BLS before Deep Learning?

Answer:

> BLS estimates the orbital period and folds the light curve so repeated transits align. This reduces the learning burden on the neural network and improves robustness to noise.

---

# Why 3 Branches?

Answer:

> Each branch learns complementary information:
>
> * Global branch captures long-term orbital context.
> * Local branch focuses on transit shape.
> * 2D branch captures periodic spatial patterns across multiple orbits.

---

# Why Threshold = 0.37?

Answer:

> Instead of assuming 0.5, we optimized the threshold on the validation set to maximize the F1 score, improving the balance between precision and recall.

---

# Biggest Limitation

Be honest if asked:

> The current model is trained primarily on Kepler data. Increasing the diversity of training data, incorporating more confirmed planets and false positives, and validating across additional missions such as TESS would further improve robustness and generalization.

---

## 5 Questions I would almost expect

1. **Why did you use BLS before deep learning?**
2. **Why InceptionTime instead of a plain CNN?**
3. **What is Minor-Axis Attention doing?**
4. **How do you reduce false positives?**
5. **Why did you choose F1 score instead of only reporting accuracy?**

If you can answer those confidently, you'll be well prepared for most technical discussions around this model.
