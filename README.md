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

| RNN                                  | LTC / Liquid Network                |
| ------------------------------------ | ----------------------------------- |
| Discrete-time recurrence             | Continuous-time dynamics            |
| \(h_t=f(h_{t-1},x_t)\)               | \(dh/dt=f(h,x,t)\)                  |
| Fixed discrete update                | Dynamical state evolution           |
| Natural for discrete sequences       | Natural for continuous-time signals |
| No explicit time constant adaptation | Adaptive/liquid time constants      |

---

# 20. LNN vs LSTM

LSTM uses explicit gates:

```text
Forget Gate
Input Gate
Output Gate
```

to control information flow through a cell state.

An LTC instead models the hidden state using continuous-time differential equations with adaptive time constants.

### LSTM

$$
c_t=f(c_{t-1},x_t)
$$

### LTC

$$
\frac{dh(t)}{dt}
=
F(h(t),x(t),\tau(t))
$$

Therefore, LSTM and LNN are not simply "better" and "worse" versions of one another.

They use different mathematical approaches to state evolution.

---

# 21. LNN vs Transformer

Transformers rely primarily on:

> **self-attention**

to model relationships between elements in a sequence.

A liquid network instead models:

> **continuous-time state dynamics**

Therefore:

```text
Transformer
    ↓
Attention-based sequence modeling


LNN / LTC
    ↓
Continuous-time dynamical modeling
```

Transformers are highly effective for many large-scale sequence problems, whereas liquid networks are particularly interesting when continuous-time dynamics, temporal sparsity, low-latency computation, or resource efficiency are important.

---

# 22. Canonical LTC vs This Project's LNN Layer

This distinction is important.

The final implementation in this project uses a simplified LNN-inspired layer:

```python
class LNNLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return x + torch.tanh(self.fc(x))
```

Mathematically:

$$
h_{out}
=
h+\tanh(Wh+b)
$$

This provides:

* nonlinear transformation
* residual connection
* feature refinement
* stable information flow

However, it does **not explicitly contain**:

* continuous-time state \(h(t)\)
* \(dh/dt\)
* adaptive time constant \(\tau\)
* an ODE solver
* explicit recurrent temporal state

Therefore, this implementation should technically be described as:

> **LNN-inspired residual nonlinear layers**

rather than a full canonical LTC.

---

# 23. Why Include LNN-Inspired Layers?

The CNN extracts spatial information from the MRI:

```text
MRI
 ↓
edges
 ↓
textures
 ↓
local structures
 ↓
high-level spatial features
```

The resulting 256-dimensional representation is then passed through three nonlinear residual transformations.

Conceptually:

```text
MRI
 ↓
CNN
 ↓
Spatial representation
 ↓
LNN-inspired transformations
 ↓
Refined representation
 ↓
Classifier
```

This creates a hybrid CNN + liquid-inspired architecture.

For a truly canonical LNN application, the strongest motivation would be when temporal information is available, such as:

* longitudinal MRI sequences
* dynamic imaging
* physiological time-series
* patient monitoring signals

An individual static MRI is not inherently a temporal sequence, so the liquid component in this implementation should be viewed as an experimental architectural component rather than a claim that the image itself contains temporal dynamics.

---

# 24. Three LNN Layers

The final model contains:

```text
256
 ↓
LNN Layer 1
 ↓
256
 ↓
LNN Layer 2
 ↓
256
 ↓
LNN Layer 3
 ↓
256
```

Each layer computes:

$$
h_{k+1}
=
h_k+\tanh(W_kh_k+b_k)
$$

Therefore:

$$
h_1=h_0+\tanh(W_1h_0+b_1)
$$

$$
h_2=h_1+\tanh(W_2h_1+b_2)
$$

$$
h_3=h_2+\tanh(W_3h_2+b_3)
$$

The final representation \(h_3\) is passed to the classifier.

---

# 25. Residual Connection

The LNN-inspired layer contains:

$$
x+\tanh(Wx+b)
$$

The addition of \(x\) is a residual connection.

Instead of learning a complete transformation:

$$
y=F(x)
$$

the layer learns:

$$
y=x+F(x)
$$

This allows the layer to preserve information from the previous representation while learning an additional nonlinear correction.

---

# 26. Final Classification Layer

The final layer is:

$$
256\rightarrow6
$$

It produces six logits:

```text
z0 → Normal
z1 → Tumor
z2 → Stroke
z3 → Infection
z4 → Degenerative
z5 → Structural
```

The predicted class is:

$$
\hat y=\arg\max_i(z_i)
$$

The logits are used directly by the classification loss during training.

---

# 27. Class Imbalance

The dataset is not uniformly distributed across the six categories.

A model trained only with ordinary loss can become biased toward classes with more samples.

For example:

```text
Majority class
████████████████████

Minority class
████
```

A model could achieve high overall accuracy while performing poorly on minority categories.

Therefore, class imbalance is explicitly addressed.

---

# 28. Class Weighting

The final implementation derives class weights from the training-set frequencies.

The general strategy is:

$$
w_c\propto\frac{1}{\sqrt{N_c}}
$$

where:

* \(N_c\) = number of training samples in class \(c\)
* \(w_c\) = class weight

Thus:

```text
more samples
     ↓
lower weight

fewer samples
     ↓
higher weight
```

This makes errors on underrepresented classes contribute more strongly to training.

---

# 29. Focal Loss

The final model uses weighted focal loss.

The standard focal-loss formulation is:

$$
FL(p_t)
=
-\alpha_t(1-p_t)^\gamma\log(p_t)
$$

where:

* \(p_t\) = predicted probability for the correct class
* \(\alpha_t\) = class-specific weighting
* \(\gamma\) = focusing parameter

The project uses:

$$
\gamma=2
$$

---

# 30. Why Focal Loss?

Cross entropy can allow easy examples to dominate the training objective when there are many of them.

Focal loss introduces:

$$
(1-p_t)^\gamma
$$

For an easy example:

$$
p_t\approx1
$$

therefore:

$$
(1-p_t)^\gamma\approx0
$$

Its contribution is reduced.

For a difficult example:

$$
p_t\ll1
$$

the contribution remains relatively large.

Therefore focal loss emphasizes:

> **hard-to-classify examples**

while class weighting emphasizes:

> **underrepresented classes**

Combining both provides a strategy for dealing with the dataset's class imbalance.

---

# 31. Optimization

The final model uses the **Adam optimizer**.

```text
Optimizer: Adam
Learning rate: 3 × 10⁻⁵
Weight decay: 1 × 10⁻⁴
```

Adam maintains moving estimates of gradients and squared gradients and adapts the parameter update accordingly.

Conceptually:

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
$$

where:

* \(\eta\) = learning rate
* \(\hat m_t\) = estimated first moment
* \(\hat v_t\) = estimated second moment

---

# 32. Regularization

The project uses several regularization mechanisms.

### Dropout

$$
p=0.3
$$

Randomly removes activations during training.

### Weight decay

$$
10^{-4}
$$

Discourages excessively large weights.

### Data augmentation

Introduces controlled variations of training images.

Together these techniques aim to reduce overfitting.

---

# 33. Training Configuration

The final experiment uses approximately:

| Parameter                |       Value |
| ------------------------ | ----------: |
| Input size               | `224 × 224` |
| Channels                 |           1 |
| Batch size               |          16 |
| Epochs                   |         100 |
| Optimizer                |        Adam |
| Learning rate            |      `3e-5` |
| Weight decay             |      `1e-4` |
| Dropout                  |       `0.3` |
| Focal-loss gamma         |       `2.0` |
| Number of output classes |           6 |

Fixed random seeds are also used for reproducibility.

---

# 34. Training Process

For every batch:

```text
Input MRI
    ↓
Forward Pass
    ↓
CNN Feature Extraction
    ↓
LNN-inspired Layers
    ↓
6 Logits
    ↓
Weighted Focal Loss
    ↓
Backpropagation
    ↓
Adam Update
```

The PyTorch training sequence follows the standard pattern:

```python
optimizer.zero_grad()

output = model(images)

loss = criterion(output, labels)

loss.backward()

optimizer.step()
```

---

# 35. Evaluation Metrics

Accuracy is not sufficient for an imbalanced medical-image classification task.

The project therefore evaluates:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* TP
* FP
* FN
* TN

---

# 36. Accuracy

$$
Accuracy=
\frac{TP+TN}{TP+TN+FP+FN}
$$

Accuracy measures the overall proportion of correct predictions.

However, it can be misleading when classes are imbalanced.

---

# 37. Precision

$$
Precision=
\frac{TP}{TP+FP}
$$

It answers:

> Of the samples predicted as a particular class, how many actually belong to that class?

---

# 38. Recall

$$
Recall=
\frac{TP}{TP+FN}
$$

It answers:

> Of all samples that actually belong to a class, how many did the model identify?

Recall is particularly important in medical screening contexts because false negatives may be consequential.

---

# 39. F1 Score

$$
F1=
2\frac{Precision\times Recall}
{Precision+Recall}
$$

F1 balances precision and recall.

---

# 40. Confusion Matrix

A confusion matrix shows the relationship between actual and predicted classes.

Example:

```text
                    Predicted
             N   T   S   I   D   St
Actual N
       T
       S
       I
       D
       St
```

The diagonal represents correct predictions.

Off-diagonal entries represent misclassifications.

For example:

```text
Actual Infection
        ↓
Predicted Stroke
```

indicates that the model confused Infection with Stroke.

---

# 41. Why Confusion Matrix Matters

A model may have good overall accuracy while failing badly on one specific class.

The confusion matrix helps identify:

* which classes are confused
* minority-class weaknesses
* systematic errors
* potential dataset ambiguity
* classes requiring additional data

This is particularly important for medical classification.

---

# 42. Overfitting

A major concern in deep learning is overfitting.

If:

```text
Training accuracy ↑↑
Validation accuracy ↑ but much lower
```

the model may be learning training-specific patterns rather than generalizable representations.

The project uses:

* augmentation
* dropout
* weight decay
* validation monitoring
* class-aware loss

to help control overfitting.

---

# 43. Important Experimental Limitation

For rigorous evaluation, the test set should ideally be used only for final evaluation.

If test performance is monitored repeatedly during training or used to make architectural decisions, it can indirectly influence experimentation.

A cleaner workflow is:

```text
Train
 ↓
Validation
 ↓
Select final model
 ↓
Evaluate ONCE on test set
```

This is a methodological improvement for future versions.

---

# 44. Important Medical-AI Limitation: Patient Leakage

Medical datasets can contain multiple images from the same patient.

If images from one patient appear in both training and testing:

```text
Patient A
 ├── MRI 1 → Training
 └── MRI 2 → Testing
```

the model may learn patient-specific characteristics.

This can make test performance appear better than true generalization.

A better strategy is:

```text
Patient A → Train only

Patient B → Validation only

Patient C → Test only
```

when patient identifiers are available.

---

# 45. Canonical LTC as a Future Extension

A natural extension of this project would be to replace the simplified LNN-inspired layers with an actual **Liquid Time-Constant Network**.

The canonical architecture would introduce continuous-time hidden-state dynamics such as:

$$
\frac{dh}{dt}
=
F(h,x,\tau)
$$

with adaptive time constants.

The resulting architecture could be:

```text
MRI / MRI sequence
       ↓
CNN feature extractor
       ↓
Temporal feature sequence
       ↓
Canonical LTC
       ↓
Continuous-time hidden state
       ↓
Classifier
```

This would be particularly meaningful if the input consisted of:

* multiple MRI scans over time
* temporal imaging sequences
* physiological signals associated with MRI
* longitudinal patient observations

The original LTC work establishes this continuous-time dynamical formulation, while later CfC work derives efficient closed-form approximations to LTC dynamics to reduce dependence on numerical differential-equation solvers.

---

# 46. LTC vs CfC

Two important developments in liquid neural networks are:

### Liquid Time-Constant Network (LTC)

Uses continuous-time differential equations and adaptive time constants.

```text
Input
 ↓
Liquid dynamics
 ↓
ODE
 ↓
Numerical solver
 ↓
Hidden state
```

### Closed-form Continuous-time Network (CfC)

CfC was developed to approximate LTC-style continuous-time dynamics in a closed form, reducing the computational burden associated with numerical ODE solving. The authors report substantial speed advantages over ODE-based counterparts.

Conceptually:

```text
LTC
 ↓
Continuous dynamics
 ↓
Numerical ODE solving

CfC
 ↓
Continuous-time formulation
 ↓
Closed-form approximation
```

---

# 47. Why a Canonical LTC Would Be Interesting Here

The current project primarily deals with static MRI images.

Therefore, the strongest justification for canonical LTC would be to extend the project from:

```text
Single MRI
```

to:

```text
MRI sequence / longitudinal patient data
```

For example:

```text
MRI at t1
      ↓
MRI at t2
      ↓
MRI at t3
      ↓
CNN feature extraction
      ↓
Temporal sequence
      ↓
LTC
      ↓
Disease progression representation
```

The CNN would learn spatial information while the LTC would model how the representation changes over time.

This would create a more natural **spatio-temporal CNN + LTC architecture**.

---

# 48. Project Strengths

### 1. Hybrid architecture

Combines convolutional spatial feature extraction with LNN-inspired nonlinear processing.

### 2. Class imbalance handling

Uses class-aware weighting and focal loss.

### 3. Detailed evaluation

Uses class-wise metrics and confusion matrices rather than relying only on accuracy.

### 4. Reproducibility

Uses fixed random seeds.

### 5. Medical-AI awareness

The project explicitly considers imbalance, generalization, and class-wise performance.

---

# 49. Limitations

The current implementation has several limitations:

1. The LNN component is LNN-inspired rather than canonical LTC.
2. The current input is a static MRI image rather than a temporal sequence.
3. Random image-level splitting can potentially cause patient-level leakage if multiple images belong to the same patient.
4. The test set should ideally be reserved for one final evaluation.
5. The six broad classes simplify the original 38-class problem.
6. Performance on a single dataset does not establish clinical generalization.
7. Horizontal flipping should be validated for anatomical appropriateness.
8. Clinical deployment would require external validation and substantially more rigorous testing.

---

# 50. Future Improvements

## Model Improvements

* Implement a canonical LTC layer.
* Experiment with CfC.
* Compare CNN + LTC against CNN + LSTM.
* Compare against modern CNN architectures.
* Compare against vision transformers.
* Investigate pretrained transfer learning.
* Perform systematic hyperparameter optimization.

## Data Improvements

* Increase dataset size.
* Use patient-level splitting.
* Use external datasets for validation.
* Investigate domain shift across hospitals/scanners.
* Validate augmentation strategies with domain experts.

## Evaluation Improvements

* Report macro-F1.
* Report class-wise sensitivity and specificity.
* Evaluate calibration.
* Perform external validation.
* Perform statistical comparison between models.
* Use confidence intervals where appropriate.

## Explainability

Potential future methods include:

* Grad-CAM
* Integrated Gradients
* Saliency maps
* Feature visualization

These can help investigate whether the model is focusing on medically relevant regions rather than artifacts.

---

# 51. Research-Oriented Extension

A more complete future architecture could be:

```text
              MRI Sequence
                   │
        ┌──────────┴──────────┐
        │                     │
       MRI₁                  MRI₂ ... MRIₙ
        │                     │
        ▼                     ▼
     CNN Encoder          CNN Encoder
        │                     │
        └──────────┬──────────┘
                   ▼
          Temporal Feature
              Sequence
                   │
                   ▼
          Canonical LTC / CfC
                   │
                   ▼
       Continuous-Time State
                   │
                   ▼
             Classifier
                   │
                   ▼
          Disease Category
```

This would make the use of a genuine liquid neural network much more theoretically motivated.

---

 Disclaimer

This project is a research/academic machine-learning prototype and is **not a clinical diagnostic system**.

Predictions from the model should not be used as a substitute for professional medical diagnosis.

Clinical deployment would require appropriate clinical validation, independent testing, regulatory review, data governance, interpretability assessment, and clinician oversight.

---

# 58. References

### Liquid Time-Constant Networks

Hasani, R., Lechner, M., Amini, A., Rus, D., Grosu, R. et al.

**Liquid Time-constant Networks.**

The original work introduces continuous-time recurrent neural networks whose dynamics involve varying, state-coupled time constants.

### Closed-form Continuous-time Neural Models

Hasani et al.

**Closed-form Continuous-time Neural Models.**

This work develops closed-form approximations of LTC dynamics, reducing reliance on computationally expensive numerical differential-equation solvers.

---

Technologies Used

```text
Python
PyTorch
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
PIL / Pillow
Jupyter Notebook
Google Colab
```

---

 Keywords

```text
Brain MRI Classification
Medical Image Classification
Deep Learning
Computer Vision
CNN
CNN-18
ResNet-inspired CNN
Liquid Neural Networks
Liquid Time-Constant Networks
LTC
CfC
Focal Loss
Class Imbalance
PyTorch
Medical AI
Neuroimaging
Image Classification
```
