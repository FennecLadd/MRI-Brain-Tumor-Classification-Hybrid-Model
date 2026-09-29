# Adaptive Brain MRI Classification using CNN-18 and Liquid Neural Networks

## Overview

This project presents a deep learning pipeline for automated classification of brain MRI images into six broad neurological disease categories.

The system uses a **custom CNN-18-inspired convolutional backbone** for spatial feature extraction followed by a sequence of **LNN-inspired residual nonlinear layers** for deeper feature transformation and a final six-class classifier.

The original dataset contains **38 disease-related categories**, which are grouped into six broader classes:

* Normal
* Tumor
* Stroke
* Infection
* Degenerative
* Structural

The project additionally addresses **class imbalance** using class-dependent weighting and **focal loss**, with performance evaluated using accuracy, precision, recall, F1-score, and confusion-matrix analysis.

> **Important architectural note:** The implementation in this repository uses an LNN-inspired residual layer:
>
> `x + tanh(Wx + b)`
>
> rather than a full canonical Liquid Time-Constant Network (LTC). The README therefore distinguishes the implemented architecture from the formal LTC/Liquid Neural Network formulation.

---

# 1. Problem Statement

Brain MRI interpretation involves identifying structural or pathological patterns from medical images. Manual analysis requires specialist expertise and can be time-consuming, while automated computer-aided systems can potentially assist with large-scale image analysis.

The objective of this project is to build a deep learning system that learns visual representations from brain MRI images and classifies them into six broad categories.

The task can be represented as:

$$
f_\theta(X) \rightarrow Y
$$

where:

* \(X\) = input MRI image
* \(f_\theta\) = trained neural network
* \(\theta\) = learnable model parameters
* \(Y\) = predicted disease category

The final model performs **six-class image classification**.

---

# 2. Dataset

The project uses the **NINS brain MRI dataset**.

The dataset contains 38 original disease-related categories. Instead of training a 38-class classifier, the categories are mapped into six broader super-classes.

### Final Classes

| Class ID | Class        |
| -------: | ------------ |
|        0 | Normal       |
|        1 | Tumor        |
|        2 | Stroke       |
|        3 | Infection    |
|        4 | Degenerative |
|        5 | Structural   |

### Why 38 Classes Were Grouped into 6?

The grouping was used to:

1. Reduce the complexity of the classification problem.
2. Combine related disease categories.
3. Improve the number of samples available per broad category.
4. Reduce the difficulty caused by highly imbalanced fine-grained classes.
5. Focus the model on broader MRI pattern classification rather than highly specific diagnosis.

The model should therefore be described as a **six-class MRI categorization system**, rather than as a system that independently diagnoses 38 specific neurological diseases.

---

# 3. Overall Pipeline

```text
                 NINS MRI DATASET
                        │
                        ▼
              38 Original Categories
                        │
                        ▼
                Class Consolidation
                        │
                        ▼
                  6 Super Classes
                        │
                        ▼
              Train / Validation / Test
                   70% / 15% / 15%
                        │
                        ▼
               Image Preprocessing
                        │
              ┌─────────┴─────────┐
              │                   │
          Training            Validation/Test
              │                   │
       Augmentation              │
              │                   │
              └─────────┬─────────┘
                        ▼
                Grayscale MRI
                 1 × 224 × 224
                        │
                        ▼
               CNN-18 Backbone
                        │
                        ▼
                  512 Features
                        │
                        ▼
                 Linear 512→256
                        │
                        ▼
                  Dropout 0.3
                        │
                        ▼
                   LNN Layer 1
                        │
                        ▼
                   LNN Layer 2
                        │
                        ▼
                   LNN Layer 3
                        │
                        ▼
                 Linear 256→6
                        │
                        ▼
                   6 Logits
                        │
                        ▼
              Class Prediction
```

---

# 4. Data Preprocessing

Each MRI image undergoes the following preprocessing.

## 4.1 Grayscale Conversion

Images are converted to grayscale.

```python
image.convert("L")
```

This changes the input from three RGB channels to one intensity channel.

Therefore:

```text
RGB:
224 × 224 × 3

        ↓

Grayscale:
224 × 224 × 1
```

MRI images are naturally represented primarily through intensity information, so a grayscale representation reduces the input dimensionality.

---

# 5. Image Resizing

Every image is resized to:

```text
224 × 224
```

The final CNN input therefore becomes:

```text
1 × 224 × 224
```

Using a fixed resolution provides a consistent input shape for the convolutional network.

---

# 6. Data Augmentation

Training images are augmented using:

* Random horizontal flipping
* Random rotation up to approximately ±10°

Conceptually:

```text
Original MRI
     │
     ├── Original
     ├── Horizontally flipped
     └── Slightly rotated
```

The purpose is to expose the model to small variations and reduce overfitting.

### Important medical-imaging consideration

Horizontal flipping should not automatically be assumed to be clinically valid for every MRI task because left-right anatomical information may be important.

For a production medical system, augmentation strategies should therefore be validated against the clinical task and, ideally, with domain experts.

---

# 7. Normalization

Pixel values are converted to tensors and normalized using:

$$
x'=\frac{x-\mu}{\sigma}
$$

with:

$$
\mu=0.5
$$

and:

$$
\sigma=0.5
$$

This approximately maps the normalized input from:

$$
[0,1]\rightarrow[-1,1]
$$

Normalization helps provide a consistent numerical scale for optimization.

---

# 8. Dataset Split

The final pipeline uses:

```text
70% → Training
15% → Validation
15% → Testing
```

### Training Set

Used to learn model parameters.

### Validation Set

Used to monitor generalization and compare experimental configurations.

### Test Set

Used to evaluate the final model.

A rigorous experiment should avoid using test performance to select the final model.

### Medical-AI Consideration

If multiple MRI images originate from the same patient, an even stronger evaluation strategy is **patient-level splitting**, where all images belonging to one patient remain in only one of train, validation, or test.

This helps prevent patient-level information leakage.

---

# 9. Final Model Architecture

The final model consists of:

```text
Input
1 × 224 × 224

        ↓

CNN-18-inspired backbone

        ↓

7 × 7 × 512

        ↓

Global Average Pooling

        ↓

512

        ↓

Linear
512 → 256

        ↓

Dropout
p = 0.3

        ↓

LNN-inspired Layer
256 → 256

        ↓

LNN-inspired Layer
256 → 256

        ↓

LNN-inspired Layer
256 → 256

        ↓

Linear
256 → 6

        ↓

6 class logits
```

---

# 10. CNN-18 Backbone

The convolutional feature extractor follows the general depth structure of an 18-layer ResNet-style network, but the implementation does **not** contain the canonical ResNet residual skip connections.

Therefore, the most accurate description is:

> **Custom CNN-18 backbone inspired by the ResNet-18 architecture, without residual skip connections.**

It should not be described simply as a standard ResNet-18.

---

# 11. CNN Architecture

The backbone consists of:

```text
Input
1 × 224 × 224
      │
      ▼
7×7 Conv
1 → 64
Stride = 2
      │
      ▼
BatchNorm
      │
      ▼
ReLU
      │
      ▼
3×3 MaxPool
Stride = 2
      │
      ▼
56 × 56 × 64
      │
      ▼
Layer 1
2 CNN Blocks
      │
      ▼
56 × 56 × 64
      │
      ▼
Layer 2
2 CNN Blocks
      │
      ▼
28 × 28 × 128
      │
      ▼
Layer 3
2 CNN Blocks
      │
      ▼
14 × 14 × 256
      │
      ▼
Layer 4
2 CNN Blocks
      │
      ▼
7 × 7 × 512
      │
      ▼
Global Average Pooling
      │
      ▼
512
```

---

# 12. CNN Block

Each CNN block consists of two convolutional transformations:

```text
Input
  │
  ▼
3×3 Convolution
  │
  ▼
Batch Normalization
  │
  ▼
ReLU
  │
  ▼
3×3 Convolution
  │
  ▼
Batch Normalization
  │
  ▼
ReLU
  │
  ▼
Output
```

The architecture contains two such blocks at each of four stages.

---

# 13. Feature Map Progression

The spatial and channel dimensions progress approximately as follows:

| Stage                  | Feature Map     |
| ---------------------- | --------------- |
| Input                  | `1 × 224 × 224` |
| Stem                   | `64 × 56 × 56`  |
| Layer 1                | `64 × 56 × 56`  |
| Layer 2                | `128 × 28 × 28` |
| Layer 3                | `256 × 14 × 14` |
| Layer 4                | `512 × 7 × 7`   |
| Global Average Pooling | `512 × 1 × 1`   |
| Flatten                | `512`           |

The CNN therefore converts a high-dimensional MRI image into a compact **512-dimensional learned representation**.

---

# 14. Global Average Pooling

The final convolutional representation is:

$$
7\times7\times512
$$

Global Average Pooling converts this into:

$$
1\times1\times512
$$

which becomes:

$$
512
$$

after flattening.

Instead of flattening:

$$
7\times7\times512=25088
$$

features, global average pooling produces only:

$$
512
$$

features.

This significantly reduces the number of parameters in the classifier.

---

# 15. Feature Projection

The 512-dimensional CNN representation is passed through:

$$
512\rightarrow256
$$

using a fully connected layer.

```text
CNN output
512
 │
 ▼
Linear
512 → 256
 │
 ▼
256-dimensional representation
```

A dropout layer with:

$$
p=0.3
$$

is then applied.

The purpose is to reduce overfitting by randomly dropping a fraction of activations during training.

---

# 16. Liquid Neural Networks

## What is a Liquid Neural Network?

Liquid Neural Networks (LNNs) are a family of neural architectures based on **continuous-time dynamical systems**.

A particularly important formulation is the **Liquid Time-Constant Network (LTC)** introduced by Hasani et al.

LTCs model hidden-state evolution using differential equations and adaptive time constants. Their dynamics are influenced by the hidden state and input, producing time-varying or "liquid" temporal behavior.

A simplified conceptual representation is:

$$
\frac{dh(t)}{dt}
=
\frac{-h(t)+f(h(t),x(t))}
{\tau(h(t),x(t))}
$$

where:

* \(h(t)\) = hidden state
* \(x(t)\) = input
* \(f\) = learned nonlinear transformation
* \(\tau\) = effective time constant

The key characteristic is that the state evolves continuously rather than simply being updated using a fixed discrete recurrence.

---

# 17. Why Are They Called "Liquid"?

The term "liquid" refers to the fact that the network's dynamics can adapt according to the input and current state.

Instead of having a completely fixed temporal behavior:

```text
Input → Fixed update → Hidden state
```

a liquid network can be viewed as:

```text
Input
  │
  ▼
Adaptive dynamical system
  │
  ▼
Continuously evolving hidden state
```

The effective time constants can vary with the network state and input.

This makes liquid networks particularly interesting for:

* time-series prediction
* robotics
* autonomous systems
* sensor processing
* physical systems
* continuous-time signals
* resource-constrained applications

The original LTC work specifically formulates liquid networks as time-continuous recurrent neural networks with varying time constants.

---

# 18. Canonical LTC Formulation

A Liquid Time-Constant Network is based on continuous-time differential equations.

Conceptually:

$$
\tau_i(x,h)
\frac{dh_i}{dt}
=
-h_i+F_i(x,h)
$$

or:

$$
\frac{dh_i}{dt}
=
\frac{-h_i+F_i(x,h)}
{\tau_i(x,h)}
$$

Here:

* \(h_i\) represents the state of neuron \(i\)
* \(x\) represents external input
* \(F_i\) represents nonlinear synaptic interactions
* \(\tau_i\) is an adaptive time constant

Unlike a conventional feed-forward layer, the network therefore describes **how its internal state evolves with time**.

The original LTC formulation uses nonlinear interlinked gates to modulate first-order dynamical systems and produces outputs through numerical differential-equation solving.

---

# 19. LNN vs Conventional RNN

A standard RNN can be expressed as:

$$
h_t=f(h_{t-1},x_t)
$$

The state changes at discrete time steps.

An LTC/LNN instead models:

$$
\frac{dh(t)}{dt}=f(h(t),x(t),t)
$$

Therefore:

| RNN                      | LTC / Liquid Network |
| ------------------------ | -------------------- |
| Discrete-time recurrence |                      |
