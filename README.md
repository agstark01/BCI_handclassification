# 🧠 Brain-Computer Interface (BCI) — EEG Motor Imagery Classification

> Real-time EEG signal decoding using **FBCSP + LDA** with windowed epoch prediction and temporal smoothing.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Core Concepts](#core-concepts)
  - [BCI — Brain-Computer Interface](#1-bci--brain-computer-interface)
  - [EEG & Motor Imagery](#2-eeg--motor-imagery)
  - [Frequency Bands](#3-frequency-bands)
  - [Epochs — 500ms Windows](#4-epochs--500ms-windows)
  - [BPS — Bits Per Second](#5-bps--bits-per-second)
  - [CSP — Common Spatial Patterns](#6-csp--common-spatial-patterns)
  - [FBCSP — Filter Bank CSP](#7-fbcsp--filter-bank-csp)
  - [LDA — Linear Discriminant Analysis](#8-lda--linear-discriminant-analysis)
  - [Smoothing Function](#9-smoothing-function)
- [Pipeline Architecture](#pipeline-architecture)
- [Mathematics Deep Dive](#mathematics-deep-dive)
- [Results & Performance](#results--performance)
- [Getting Started](#getting-started)
- [References](#references)

---

## Project Overview

This project implements a **Motor Imagery BCI system** that classifies imagined movements (e.g., left hand vs. right hand vs. feet vs. rest) directly from raw EEG signals in real time.

The pipeline:

```
Raw EEG → Bandpass Filter Banks → CSP Spatial Filtering → Feature Extraction → LDA Classifier → Smoothed Prediction
```

**Key design choices:**
- **500ms non-overlapping epochs** for prediction windows
- **Filter Bank CSP (FBCSP)** for multi-band spatial feature extraction
- **LDA** for fast, robust linear classification
- **Step-wise smoothing** so each 500ms window holds its predicted label for the entire duration

---

## Core Concepts

### 1. BCI — Brain-Computer Interface

A BCI creates a direct communication pathway between the brain and an external device **without** using muscles or peripheral nerves.

```
Brain Activity (EEG)
       │
       ▼
  Signal Processing
       │
       ▼
  Classification
       │
       ▼
  Control Command  ──► Device / Computer / Prosthetic
```

**Types of BCI signals used:**
| Signal | Source | Invasiveness |
|--------|--------|-------------|
| EEG | Scalp electrodes | Non-invasive |
| ECoG | Brain surface | Semi-invasive |
| LFP / Spike | Implanted electrodes | Invasive |

This project uses **non-invasive EEG**.

---

### 2. EEG & Motor Imagery

**EEG (Electroencephalography)** records electrical potential differences on the scalp caused by synchronized neural activity.

**Motor Imagery (MI)** is the mental simulation of movement **without** actual muscle activation. When a subject *imagines* moving their left hand, the contralateral (right) motor cortex shows:

- **Event-Related Desynchronization (ERD):** decrease in mu (8–12 Hz) and beta (13–30 Hz) power
- **Event-Related Synchronization (ERS):** increase in power on the ipsilateral side

This contralateral ERD pattern is the core discriminative signal we exploit.

```
Imagined Left Hand Movement:
  Right hemisphere  →  ERD (↓ power)
  Left hemisphere   →  ERS (↑ power)
```

---

### 3. Frequency Bands

EEG rhythms relevant to motor imagery:

| Band | Frequency (Hz) | Role in MI |
|------|---------------|------------|
| Delta | 0.5 – 4 | Not typically used |
| Theta | 4 – 8 | Cognitive load |
| **Mu / Alpha** | **8 – 12** | **Primary MI marker (ERD/ERS)** |
| **Beta** | **13 – 30** | **Secondary MI marker** |
| Gamma | 30 – 100 | High-level processing |

FBCSP exploits **multiple bands simultaneously** rather than picking one.

---

### 4. Epochs — 500ms Windows

An **epoch** is a fixed-length time segment cut from the continuous EEG stream.

```
Continuous EEG stream:
──────────────────────────────────────────────────────►  time

  |← 500ms →|← 500ms →|← 500ms →|← 500ms →|← 500ms →|
      Epoch 1    Epoch 2    Epoch 3    Epoch 4    Epoch 5
        ↓           ↓          ↓          ↓          ↓
    Predict 1   Predict 2  Predict 1  Predict 1  Predict 2
```

**Why 500ms?**

- Short enough for **responsive, near-real-time** control (~2 decisions per second)
- Long enough to capture **multiple cycles** of mu (8 Hz → one cycle = 125ms; 500ms = ~4 cycles) and beta rhythms
- Balances temporal resolution vs. spectral reliability

**Samples per epoch** (depends on sampling rate):

```
If sampling rate Fs = 250 Hz:
  Samples per epoch = Fs × T = 250 × 0.5 = 125 samples

If Fs = 500 Hz:
  Samples per epoch = 500 × 0.5 = 250 samples
```

Each epoch is a matrix of shape:

```
X_epoch ∈ ℝ^(C × T)

where:
  C = number of EEG channels (e.g., 22)
  T = number of time samples per epoch (e.g., 125 or 250)
```

---

### 5. BPS — Bits Per Second

**BPS (Bits Per Second)** is the standard information-theoretic metric to evaluate BCI performance. It quantifies how much information is successfully communicated per unit time.

#### ITR Formula (Nykopp / Wolpaw)

```
B = log₂(N) + P·log₂(P) + (1−P)·log₂((1−P)/(N−1))
```

Where:
- `B` = information transfer rate in **bits per trial**
- `N` = number of possible classes (e.g., N=4 for left, right, feet, rest)
- `P` = classification accuracy (0 to 1)

Then convert to **bits per minute** or **bits per second**:

```
ITR (bits/min) = B × (60 / T_trial)

ITR (bits/sec) = B × (1 / T_trial)
```

Where `T_trial` is the total trial duration in seconds (including feedback, pause, etc.)

#### Example

```
N = 4 classes
P = 0.80 accuracy
T_trial = 0.5s (our 500ms epoch)

B = log₂(4) + 0.8·log₂(0.8) + 0.2·log₂(0.2/3)
  = 2 + 0.8·(−0.322) + 0.2·(−3.907)
  = 2 − 0.258 − 0.781
  ≈ 0.961 bits/trial

ITR = 0.961 × (1/0.5) ≈ 1.92 bits/second
    = 0.961 × 120 ≈ 115 bits/minute
```

Higher accuracy AND faster epochs → higher BPS. The 500ms window is chosen to maximize this trade-off.

---

### 6. CSP — Common Spatial Patterns

CSP finds **spatial filters** (electrode weights) that **maximize variance for one class while minimizing it for another**.

#### Objective

Given two classes with covariance matrices `Σ₁` and `Σ₂`, find filter matrix `W` such that:

```
W = argmax  (wᵀ Σ₁ w) / (wᵀ Σ₂ w)
```

#### Algorithm

**Step 1:** Compute normalized covariance matrices for each class.

```
For each trial X ∈ ℝ^(C×T):

  R = (X · Xᵀ) / trace(X · Xᵀ)

Average across all trials of class k:  R̄_k = mean(R_trials_k)
```

**Step 2:** Solve the generalized eigenvalue problem:

```
Σ₁ · W = λ · Σ₂ · W

Equivalent to:  (Σ₁ + Σ₂)⁻¹ · Σ₁ · W = λ · W
```

**Step 3:** Select the `m` eigenvectors with the **largest** and **smallest** eigenvalues (both ends capture discriminative spatial patterns for each class).

```
W_selected = [w₁, w₂, ..., w_m, w_(C-m+1), ..., w_C]   (shape: C × 2m)
```

**Step 4:** Apply spatial filter to raw EEG:

```
Z = Wᵀ · X       Z ∈ ℝ^(2m × T)
```

**Step 5:** Extract log-variance features:

```
f_i = log( var(Z_i) )    for i = 1, ..., 2m

feature vector f ∈ ℝ^(2m)
```

The log-variance is approximately **Gaussian distributed**, which plays well with LDA.

---

### 7. FBCSP — Filter Bank CSP

CSP applied to a **single frequency band** can miss discriminative information in other bands. **FBCSP** applies CSP to **multiple bandpass-filtered versions** of the EEG signal and concatenates features.

#### Filter Bank Design

```
Band 1:   4  –  8 Hz   (Theta)
Band 2:   8  – 12 Hz   (Mu / Alpha)
Band 3:  12  – 16 Hz   (Low Beta)
Band 4:  16  – 20 Hz   (Mid Beta)
Band 5:  20  – 24 Hz   (High Beta)
Band 6:  24  – 28 Hz   (High Beta)
Band 7:  28  – 32 Hz   (Low Gamma)
... (configurable)
```

Each band uses a **Chebyshev Type II** or **Butterworth** bandpass filter.

#### FBCSP Pipeline

```
X_raw ∈ ℝ^(C×T)
        │
        ├──► BPF(4–8Hz)   ──► CSP ──► log-var features f₁ ∈ ℝ^(2m)
        │
        ├──► BPF(8–12Hz)  ──► CSP ──► log-var features f₂ ∈ ℝ^(2m)
        │
        ├──► BPF(12–16Hz) ──► CSP ──► log-var features f₃ ∈ ℝ^(2m)
        │
        ├──► BPF(16–20Hz) ──► CSP ──► log-var features f₄ ∈ ℝ^(2m)
        │
        └──► ...
             │
             ▼
     Concatenate all f_i:
     f_total = [f₁ | f₂ | f₃ | ... | f_K]   ∈ ℝ^(K×2m)
             │
             ▼
     Feature Selection (MIBIF / mutual information)
             │
             ▼
     Selected features  ──►  LDA Classifier
```

**K** = number of filter bank bands  
**2m** = number of CSP filters per band (typically m=2 → 4 features per band)

#### Feature Selection with MIBIF

To avoid curse of dimensionality, we select the most discriminative features using **Mutual Information Based Individual Feature** (MIBIF) ranking:

```
MI(f_i ; y) = Σ_x Σ_y  p(f_i, y) · log[ p(f_i, y) / (p(f_i) · p(y)) ]
```

Top features ranked by MI score are passed to the classifier.

---

### 8. LDA — Linear Discriminant Analysis

**LDA** finds a linear decision boundary that **maximally separates** class means relative to within-class scatter.

#### Binary LDA

For two classes with:
- Class means: `μ₁`, `μ₂`  
- Shared within-class scatter matrix: `S_W`

The **optimal projection direction** is:

```
w* = S_W⁻¹ · (μ₁ − μ₂)
```

**Decision rule:**

```
y_hat = argmin_k  (x − μ_k)ᵀ · S_W⁻¹ · (x − μ_k)  −  2·log(π_k)

where π_k = prior probability of class k
```

#### Multi-class LDA (OVR or direct)

For K classes, LDA computes:

**Between-class scatter:**
```
S_B = Σ_k  n_k · (μ_k − μ)·(μ_k − μ)ᵀ
```

**Within-class scatter:**
```
S_W = Σ_k  Σ_{x∈class_k}  (x − μ_k)·(x − μ_k)ᵀ
```

**Generalized eigenproblem:**
```
S_B · v = λ · S_W · v
```

Solve for top `K−1` eigenvectors → project data → classify by nearest class centroid.

#### Why LDA for BCI?

| Property | Benefit |
|----------|---------|
| Linear boundary | Fast inference, real-time capable |
| Closed-form solution | No iterative training, deterministic |
| Naturally handles covariance | CSP log-var features are near-Gaussian |
| Low sample requirement | EEG datasets are often small |
| Interpretable | Weights map back to scalp topography |

---

### 9. Smoothing Function

#### The Problem

Raw 500ms epoch predictions can be noisy — a single corrupted epoch might flip the predicted class momentarily, causing jitter in control output.

#### Our Smoothing Approach: Step-Hold

The smoothing function assigns the **predicted label for an entire 500ms window** — the prediction does not change mid-window. This is a **zero-order hold (step function)**:

```
Epoch 1 (0–500ms)   → predict class 1 → output = 1 for ALL of 0–500ms
Epoch 2 (500–1000ms)→ predict class 2 → output = 2 for ALL of 500–1000ms
Epoch 3 (1000–1500ms)→ predict class 1 → output = 1 for ALL of 1000–1500ms
```

Visually:

```
Raw prediction signal:
  1 1 1 2 1 1 2 2 2 1 ...  (sample-level, noisy)

After step-hold smoothing (500ms window):

Output:
  ┌────────┐        ┌────────┐        ┌────────┐
  │  1     │        │  1     │        │  1     │
  │        └────────┘        │        │        └──...
  │           2              │  2     │
  └──────────────────────────┘────────┘

  |← 500ms →|← 500ms →|← 500ms →|
```

**Mathematically:**

```
Let y_n = predicted class for epoch n
Let t ∈ [n·T, (n+1)·T)  where T = 0.5s

Output(t) = y_n    for all t in that epoch window
```

#### Extended Smoothing: Majority Vote over N epochs

For additional stability, a **majority vote** over a sliding window of N consecutive epochs can be applied:

```
y_smoothed(n) = mode( y_{n-N+1}, y_{n-N+2}, ..., y_n )
```

Example with N=3:
```
Epoch predictions:  [1, 1, 2, 1, 1, 1, 2, 2, 2]
Majority vote (N=3):[ -, -, 1, 1, 1, 1, 2, 2, 2]
```

This removes single-epoch "glitches" while adding only N×500ms = 1.5s of latency.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FULL BCI PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │   EEG Amp    │  ← Physical electrode cap on subject's scalp
     └──────┬───────┘
            │  Raw EEG  (C channels × continuous samples)
            ▼
     ┌──────────────┐
     │  Epoching    │  ← Slice into non-overlapping 500ms windows
     └──────┬───────┘       X_epoch ∈ ℝ^(C × T)
            │
            ▼
     ┌──────────────┐
     │  Filter Bank │  ← Apply K bandpass filters (Butterworth/Cheby2)
     └──────┬───────┘       X_k ∈ ℝ^(C × T)  for k = 1..K
            │
            ▼
     ┌──────────────┐
     │  CSP per     │  ← Apply trained spatial filters W_k
     │  frequency   │       Z_k = W_kᵀ · X_k
     │  band        │       f_k = log(var(Z_k))  ∈ ℝ^(2m)
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  Feature     │  ← Concatenate: f = [f₁|f₂|...|f_K]  ∈ ℝ^(K·2m)
     │  Concat &    │    Select top features by MIBIF ranking
     │  Selection   │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │     LDA      │  ← Project to discriminant subspace
     │  Classifier  │    Classify by nearest centroid
     └──────┬───────┘
            │  y_raw ∈ {1,2,...,K_classes}
            ▼
     ┌──────────────┐
     │  Smoothing   │  ← Step-hold: output y_raw for entire 500ms window
     │  Function    │    Optional: majority vote over N windows
     └──────┬───────┘
            │  y_smooth ∈ {1,2,...,K_classes}
            ▼
     ┌──────────────┐
     │  Control     │  ← Map class to device command
     │  Output      │    (cursor move, prosthetic, communication)
     └──────────────┘
```

---

## Mathematics Deep Dive

### End-to-End Feature Extraction

Given raw epoch `X ∈ ℝ^(C×T)` and filter bank with K bands:

```
For each band k = 1, ..., K:

  1. Bandpass filter:
     X_k = BPF_k(X)          ← zero-phase filter, X_k ∈ ℝ^(C×T)

  2. Apply CSP spatial filter (learned during training):
     Z_k = W_kᵀ · X_k        ← Z_k ∈ ℝ^(2m × T)

  3. Log-variance features:
     f_k[i] = log( (1/T) · Σ_t (Z_k[i,t])² )    i = 1,...,2m

     (assuming zero-mean after bandpass filtering)

  4. Resulting feature vector per band:
     f_k ∈ ℝ^(2m)

Total feature vector:
  f = [f_1ᵀ | f_2ᵀ | ... | f_Kᵀ]ᵀ  ∈ ℝ^(K·2m)
```

### LDA Inference (Real-Time)

```
Given new feature vector f ∈ ℝ^d (after selection):

  Score_k(f) = fᵀ · S_W⁻¹ · μ_k  −  (1/2) · μ_kᵀ · S_W⁻¹ · μ_k  +  log(π_k)

  ŷ = argmax_k  Score_k(f)
```

This is computed in **O(d·K)** time — negligible latency for real-time use.

### Smoothing: Zero-Order Hold

```
Prediction at sample t:

  ŷ_smooth(t) = ŷ_epoch( ⌊t / T_epoch⌋ )

where T_epoch = 500ms = 0.5s,  ⌊·⌋ = floor function

Example:
  t = 0.73s  →  epoch index = ⌊0.73/0.5⌋ = ⌊1.46⌋ = 1
               → use prediction from epoch 1
```

---

## Results & Performance

| Metric | Value |
|--------|-------|
| Epoch length | 500 ms |
| Decision rate | 2 decisions/second |
| Classification method | FBCSP + LDA |
| Smoothing | Step-hold per epoch |
| Evaluation | Session-to-session cross-validation |

### BPS Calculation Summary

```
At P = 0.75, N = 4 classes, T = 0.5s epoch:

B = log₂(4) + 0.75·log₂(0.75) + 0.25·log₂(0.25/3)
  = 2 − 0.311 − 0.528
  = 1.161 bits/trial

ITR = 1.161 / 0.5 = 2.32 bits/second
    = 1.161 × 120 = 139.3 bits/minute
```

---

---

## References

1. **Ang, K.K. et al. (2008)** — *Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface.* IJCNN.
2. **Blankertz, B. et al. (2008)** — *Optimizing Spatial Filters for Robust EEG Single-Trial Analysis.* IEEE Signal Processing Magazine.
3. **Wolpaw, J.R. et al. (2000)** — *Brain-Computer Interface Technology: A Review of the First International Meeting.* IEEE TNSRE.
4. **Nykopp, T. (2001)** — *Statistical Modelling Issues for the Adaptive Brain Interface.* HUT Thesis.
5. **Fisher, R.A. (1936)** — *The Use of Multiple Measurements in Taxonomic Problems.* Annals of Eugenics.
6. **Pfurtscheller, G. & Lopes da Silva, F.H. (1999)** — *Event-related EEG/MEG synchronization and desynchronization.* Clinical Neurophysiology.

---

<div align="center">

**Built for real-time motor imagery decoding · FBCSP + LDA · 500ms epochs**

</div>
