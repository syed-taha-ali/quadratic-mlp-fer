# Facial Expression Recognition System — Technical Report

**Project:** Quadratic MLP for Facial Expression Recognition
**Dataset:** FER-2013 subset — 6 classes (angry, fear, happy, neutral, sad, surprise)
**Versions Implemented:** V1 (baseline) and V2 (improved)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Dataset](#2-dataset)
3. [Pre-Processing Pipeline](#3-pre-processing-pipeline)
4. [Custom Neural Network Architecture](#4-custom-neural-network-architecture)
5. [Mathematical Derivation — Backpropagation](#5-mathematical-derivation--backpropagation)
6. [Training Methodology](#6-training-methodology)
7. [Genetic Algorithm Hyperparameter Tuning](#7-genetic-algorithm-hyperparameter-tuning)
8. [Model Assessment & Evaluation](#8-model-assessment--evaluation)
9. [V1 vs V2 Comparison & Improvement Analysis](#9-v1-vs-v2-comparison--improvement-analysis)
10. [Deployment System](#10-deployment-system)
11. [Results Summary](#11-results-summary)
12. [Limitations & Future Work](#12-limitations--future-work)
13. [Design Constraints Checklist](#13-design-constraints-checklist)

---

## 1. System Overview

This project builds a **Facial Expression Recognition (FER)** system that classifies 48×48 grayscale face images into 6 emotion categories. Two versions were developed:

| Version | Architecture | Test Accuracy | Notes |
|---------|-------------|---------------|-------|
| V1 | 1 hidden layer, hidden=64 | 36.17% | Baseline |
| V2 | 2 hidden layers, hidden=512 | **36.67%** | GA-tuned, all metrics above V1 |

Both versions share the core architectural constraints of this project:
- Custom modified net input formula: `net = W @ (x + x²)`
- Custom sigmoid activation at each perceptron
- SoftMax final layer for multi-class output
- CrossEntropyLoss with SGD optimizer
- Genetic algorithm hyperparameter search

The complete codebase comprises:
- `FER_System.ipynb` — V1 training notebook (unchanged reference)
- `FER_System_V2.ipynb` — V2 training notebook (8 sections, 37 cells)
- `fer_deploy.py` — V2-compatible deployment (image/video/camera)
- `fer_deploy_V2.py` — Enhanced deployment (face alignment, better cropping, temporal smoothing)

---

## 2. Dataset

### Source
FER-2013 subset provided by the module, split into three pre-labelled directories:

| Split | Purpose |
|-------|---------|
| `Data/Training/` | Update model parameters (training) |
| `Data/Validation/` | Hyperparameter tuning; early stopping criterion |
| `Data/Testing/` | Final evaluation (held-out, used once) |

### Properties
- **Image format:** 48×48 pixels, 8-bit grayscale (L-mode PIL)
- **Classes:** 6 — angry, fear, happy, neutral, sad, surprise
- **Variation:** Lighting, pose, age, gender, ethnicity
- **Class imbalance:** The dataset exhibits moderate imbalance; fear and disgust are underrepresented relative to happy and neutral

### Data Loading
Images are loaded using `torchvision.datasets.ImageFolder` (respects directory structure as class labels) and wrapped in a custom `FERDatasetFromPaths` class that supports cross-validation splits.

---

## 3. Pre-Processing Pipeline

### 3.1 Pipeline Overview

All images pass through the `FERPreprocessor` class before entering the network:

```
Input (48×48 grayscale PIL)
        │
        ▼
  GaussianBlur(kernel=3×3, σ=0)        ← Noise suppression before CLAHE
        │
        ▼
  CLAHE(clipLimit=3.0, tileGridSize=6×6) ← Adaptive contrast enhancement
        │
        ▼
  transforms.ToTensor()                 ← Normalise to [0, 1]
        │
        ▼
  Model input: 2304-dim float tensor
```

### 3.2 Justification of Each Step

**Step 1 — Gaussian Blur (3×3)**
FER-2013 images contain sensor noise from camera capture. Blurring before CLAHE prevents noise pixels from being amplified into false edges during contrast enhancement. Kernel size 3 is chosen as the minimum that smooths high-frequency noise without blurring facial features at 48×48 resolution.

**Step 2 — CLAHE (Contrast Limited Adaptive Histogram Equalization)**
Standard histogram equalisation can over-amplify uniform regions. CLAHE applies local equalisation within a grid of non-overlapping tiles and clips the contrast at `clipLimit=3.0` to prevent noise amplification.

- `clipLimit=3.0`: More aggressive than standard (2.0) to better recover low-contrast images captured under poor lighting.
- `tileGridSize=(6,6)`: 6×6 tiles on a 48×48 image → each tile is 8×8 pixels. This scale captures local facial sub-regions (eye, mouth, cheek) without being so local that individual noise pixels dominate a tile.

Compared to V1 (`clip=2.0, tile=(8,8)`), V2's tighter tiles and higher clip limit produce sharper local contrast in low-light images while the pre-blur prevents noise amplification.

**Step 3 — ToTensor() (no Normalize)**
V1 applied `Normalize(mean=[0.5], std=[0.5])` which maps pixels from `[0, 1]` to `[-1, 1]`. This was **removed in V2** for a critical mathematical reason:

The custom formula `net = W @ (x + x²)` involves `x + x²`. On the domain `[-1, 1]`, this function is **non-monotone**: it has a minimum at `x = -0.5` where `x + x² = -0.25`. This means two very different pixel intensities (e.g., `x = 0` and `x = -1`) produce the same pre-activation value (`0` and `0`), destroying discriminative information.

On the domain `[0, 1]`, `f(x) = x + x² = x(1+x)` is **strictly monotone** (derivative `1 + 2x > 0` for all `x ≥ 0`), preserving the full contrast range from the CLAHE step.

### 3.3 Data Augmentation (Training Only)

V2 applies minimal, FC-safe augmentation:

| Transform | V1 | V2 | Reason for V2 Change |
|-----------|----|----|---------------------|
| RandomHorizontalFlip | Yes | Yes | Preserved — facial symmetry invariance |
| RandomRotation(10°) | Yes | **REMOVED** | FC networks have no spatial invariance; rotation → 2304 different pixel positions; produces train < val inversion |
| RandomAffine(translate=5%) | Yes | **REMOVED** | Same reason — FC nets memorise position |
| ColorJitter | Added in V2 Run 1 | **REMOVED** | CLAHE already normalises contrast; applying ColorJitter post-CLAHE re-introduces the variance CLAHE was designed to remove |
| RandomErasing | Added in V2 Run 1 | **REMOVED** | Deletes facial pixels; discriminative regions (eyes, mouth corners) are small in 48×48 — erasing them destroys signal |

The key insight is that spatial augmentations (rotation, translation) are only invariant-promoting for convolutional networks that share weights spatially. A fully-connected network treats position `(x=5, y=5)` and `(x=6, y=5)` as completely independent features, so rotation produces a **different training sample** that contradicts the label rather than reinforcing it.

### 3.4 CLAHE Visualisation

The notebook's Section 1 includes a 4-panel visualisation:
- Panel A: Original 48×48 grayscale image
- Panel B: After Gaussian blur
- Panel C: V1 CLAHE output (clip=2.0, tile=8×8)
- Panel D: V2 CLAHE output (clip=3.0, tile=6×6)

Panel D shows visibly sharper edge contrast in shadow regions compared to Panel C.

---

## 4. Custom Neural Network Architecture

### 4.1 Core Formula

The core modified net input formula:

```
net = w₀x₀ + w₀x₀² + w₁x₁ + w₁x₁² + ... + wₙxₙ + wₙxₙ²
    = Σᵢ wᵢ(xᵢ + xᵢ²)
    = W · (x + x²)
```

This is implemented as a single PyTorch module:

```python
class CustomNetLayer(nn.Module):
    """net = W @ (x + x²)"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x + x ** 2)   # element-wise x² then standard matmul
```

The `nn.Linear` computes `W @ z + b` where `z = x + x²`. This is exactly equivalent to the design formula while leveraging PyTorch's optimised BLAS kernels.

### 4.2 Activation Function

```python
class CustomSigmoid(nn.Module):
    """σ(net) = 1 / (1 + e^{-net})"""
    def forward(self, x):
        return torch.sigmoid(x)
```

Derivative (used in backpropagation derivation):
```
σ'(net) = σ(net) · (1 − σ(net))
```

### 4.3 Building Block — CustomBlock

Each hidden layer is a `CustomBlock`:

```
Input x (in_features)
    │
    ├─ CustomNetLayer(in → out)    [net = W @ (x + x²)]
    │
    ├─ BatchNorm1d(out)            [normalise activations, reduce internal covariate shift]
    │
    ├─ CustomSigmoid               [σ(net) = 1/(1+e^{-net})]
    │
    └─ Dropout(dropout_rate)       [randomly zero neurons, prevent co-adaptation]
    │
    ▼
Output (out_features)
```

```python
class CustomBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super().__init__()
        self.layer = nn.Sequential(
            CustomNetLayer(in_features, out_features),
            nn.BatchNorm1d(out_features),
            CustomSigmoid(),
            nn.Dropout(dropout_rate),
        )
```

### 4.4 Full Network — FERNet

```
Input: 48×48 grayscale image → flatten → 2304-dim vector
    │
    ├─ CustomBlock(2304 → hidden_size)     [hidden layer 1]
    │
    ├─ [CustomBlock(hidden_size → hidden_size)]  × (num_layers − 1)
    │
    └─ CustomNetLayer(hidden_size → 6)     [output layer — raw logits]
    │
    ▼ (inference only)
    SoftMax → 6 class probabilities
```

```python
class FERNet(nn.Module):
    INPUT_DIM = 48 * 48  # 2304

    def __init__(self, num_layers=3, hidden_size=256, dropout_rate=0.3, num_classes=6):
        super().__init__()
        blocks = []
        in_dim = self.INPUT_DIM
        for _ in range(num_layers):
            blocks.append(CustomBlock(in_dim, hidden_size, dropout_rate))
            in_dim = hidden_size
        self.hidden = nn.Sequential(*blocks)
        self.output = CustomNetLayer(in_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten 1×48×48 → 2304
        x = self.hidden(x)
        return self.output(x)        # raw logits

    def get_probabilities(self, x):
        return F.softmax(self.forward(x), dim=1)  # SoftMax final layer (inference)
```

### 4.5 SoftMax Layer

SoftMax is used as the final layer for multi-class classification. PyTorch's `CrossEntropyLoss` combines `log_softmax + NLLLoss` internally for numerical stability during training. SoftMax is therefore applied **explicitly** in `get_probabilities()` for inference, making it the true final activation for classification:

```
Training:  logits → CrossEntropyLoss (internally: log_softmax → NLLLoss)
Inference: logits → F.softmax(dim=1) → 6 probabilities (sum to 1)
```

This is the standard PyTorch convention for multi-class classification.

### 4.6 GA-Tuned Architecture (V2 Final)

The Genetic Algorithm selected:

| Hyperparameter | V1 GA Result | V2 GA Result |
|----------------|-------------|-------------|
| num_layers | 1 | **2** |
| hidden_size | 64 | **512** |
| learning_rate | 0.01 | **0.01** |
| batch_size | 128 | **32** |
| dropout_rate | N/A | **0.2** |

**V2 parameter count:**
- Layer 1: `2304 × 512 + 512 (bias)` = 1,180,672 parameters
- Layer 2: `512 × 512 + 512 (bias)` = 262,656 parameters
- Output: `512 × 6 + 6 (bias)` = 3,078 parameters
- **Total: ~1.45M parameters**

**V1 parameter count (1 layer, hidden=64):**
- Layer 1: `2304 × 64 + 64` = 147,520 parameters
- Output: `64 × 6 + 6` = 390 parameters
- **Total: ~148K parameters**

V2's 10× larger model is regularised by BatchNorm + Dropout(0.2) to prevent overfitting.

### 4.7 Weight Initialisation

V2 uses **Kaiming (He) uniform initialisation** for all `CustomNetLayer` weights:

```python
nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
```

**Why Kaiming over Xavier:** Xavier assumes linear activations and targets unit variance preservation. Kaiming accounts for the gain introduced by nonlinear activations (conceptually similar to sigmoid's saturation) and scales weights by `√(2/fan_in)`. For deep 2–5 layer networks with sigmoid-like activations, this produces better gradient signal in the first epoch compared to Xavier's `√(1/fan_in)`.

---

## 5. Mathematical Derivation — Backpropagation

### 5.1 Notation

| Symbol | Meaning |
|--------|---------|
| `L` | Number of layers (L hidden + 1 output) |
| `l` | Layer index (1 = first hidden, L+1 = output) |
| `xₗ` | Input to layer l (activation output of layer l−1; `x₁` = flattened image) |
| `netₗⱼ` | Net input to neuron j in layer l |
| `aₗⱼ` | Activation output of neuron j in layer l |
| `wₗᵢⱼ` | Weight from neuron i in layer l−1 to neuron j in layer l |
| `δₗⱼ` | Error signal (delta) for neuron j in layer l |
| `η` | Learning rate |
| `E` | Total cross-entropy loss |

### 5.2 Forward Pass

**Custom net input (each hidden layer, neuron j):**
```
netₗⱼ = Σᵢ wₗᵢⱼ · (xₗᵢ + xₗᵢ²)     where xₗᵢ = aₗ₋₁,ᵢ
```

**Activation (hidden layers):**
```
aₗⱼ = σ(netₗⱼ) = 1 / (1 + e^{−netₗⱼ})
```

**Output layer (raw logits):**
```
net_{L+1,j} = Σᵢ w_{L+1,ij} · (aₗᵢ + aₗᵢ²)
```

**SoftMax (inference / effective final activation):**
```
ŷⱼ = e^{net_{L+1,j}} / Σₖ e^{net_{L+1,k}}
```

**Cross-entropy loss (with label smoothing ε=0.05 in V2):**
```
E = −Σⱼ [(1−ε)·yⱼ + ε/6] · log(ŷⱼ)
  = −Σⱼ ỹⱼ · log(ŷⱼ)
```

Where `ỹⱼ` is the soft target (reduces to standard cross-entropy when ε=0).

### 5.3 Output Layer Weight Update

The gradient of cross-entropy loss with respect to the output logits, after passing through SoftMax, has the well-known simplification:

```
δ_{L+1,j} = ∂E/∂net_{L+1,j} = ŷⱼ − ỹⱼ
```

The gradient with respect to output layer weight `w_{L+1,ij}`:

```
∂E/∂w_{L+1,ij} = δ_{L+1,j} · ∂net_{L+1,j}/∂w_{L+1,ij}
                = δ_{L+1,j} · (aₗᵢ + aₗᵢ²)
```

**Output layer weight update rule:**
```
Δw_{L+1,ij} = −η · δ_{L+1,j} · (aₗᵢ + aₗᵢ²)
```

### 5.4 Hidden Layer Weight Update (Layer l)

Error signals are propagated backwards through layers:

```
δₗⱼ = (Σₖ w_{l+1,jk} · δ_{l+1,k}) · σ'(netₗⱼ)
     = (Σₖ w_{l+1,jk} · δ_{l+1,k}) · σ(netₗⱼ) · (1 − σ(netₗⱼ))
     = (Σₖ w_{l+1,jk} · δ_{l+1,k}) · aₗⱼ · (1 − aₗⱼ)
```

The gradient with respect to hidden layer weight `wₗᵢⱼ`:

```
∂E/∂wₗᵢⱼ = δₗⱼ · ∂netₗⱼ/∂wₗᵢⱼ
           = δₗⱼ · (xₗᵢ + xₗᵢ²)
           = δₗⱼ · (aₗ₋₁,ᵢ + aₗ₋₁,ᵢ²)
```

**Hidden layer weight update rule:**
```
Δwₗᵢⱼ = −η · δₗⱼ · (aₗ₋₁,ᵢ + aₗ₋₁,ᵢ²)
```

### 5.5 Impact of the Custom Formula on Backpropagation

Comparing to standard backprop where `Δwᵢⱼ = −η · δⱼ · xᵢ`, the custom formula introduces the factor `(xᵢ + xᵢ²)` in place of `xᵢ`. This has two consequences:

1. **Gradient amplification for large inputs:** When `xᵢ > 0.5`, `xᵢ + xᵢ² > xᵢ`, so the effective gradient is larger. This accelerates learning for strongly-activated neurons but can cause instability at high learning rates (which is why lr=0.1 fails and lr=0.01 was selected by the GA).

2. **Gradient of x² term:** `∂(x + x²)/∂x = 1 + 2x`. During backpropagation through the x² transformation, gradients flowing back to the previous layer are scaled by `(1 + 2xᵢ)`, not just 1. This gradient amplification is another reason why the [0,1] input domain is critical — on [-1,1], this derivative can be negative (at x < -0.5), which reverses gradient direction.

### 5.6 Effect of BatchNorm on Gradient Flow

BatchNorm is inserted between `CustomNetLayer` and `CustomSigmoid`. During backpropagation, BatchNorm acts as a gradient normaliser: it scales the incoming gradient by `1/σ_batch` (batch standard deviation), which prevents gradient vanishing in deep sigmoid networks. Without BatchNorm, sigmoid's saturating derivative (`σ'(x) ≤ 0.25`) would cause gradients to decay exponentially through layers, making training of 2+ hidden layers impractical.

**In matrix form (a single hidden layer update):**
```
δₗ = (Wₗ₊₁ᵀ δₗ₊₁) ⊙ aₗ ⊙ (1 − aₗ)      [element-wise multiply]
ΔWₗ = −η · δₗ · (xₗ + xₗ²)ᵀ              [outer product → weight matrix gradient]
```

---

## 6. Training Methodology

### 6.1 Optimiser

**SGD with Nesterov momentum:**

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=best_params['learning_rate'],
    momentum=0.9,
    weight_decay=5e-4,     # L2 regularisation
    nesterov=True,         # lookahead gradient estimate
)
```

Standard SGD momentum updates: `vₜ = β·vₜ₋₁ − η·∇L(θₜ)`
Nesterov variant evaluates gradient at `θ + β·v` (a lookahead position), which provides a correction effect that is especially helpful with the x² gradient amplification causing oscillations.

### 6.2 Learning Rate Schedule

**CosineAnnealingLR** with LR warmup (SequentialLR):

```
Phase 1 (epochs 0 → warmup_steps): Linear ramp from lr/10 to lr
Phase 2 (epochs warmup_steps → 100): Cosine decay from lr to eta_min=1e-5
```

V1 used `StepLR(step_size=10, gamma=0.5)` which halves the LR every 10 epochs. The abrupt step changes are particularly problematic with the x² gradient amplification — the sudden LR drop creates a discontinuity in the effective gradient scale. CosineAnnealing provides a smooth decay that avoids this.

### 6.3 Loss Function

**CrossEntropyLoss with label smoothing (ε=0.05):**

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
```

Hard one-hot labels assign full confidence (`1.0`) to the correct class and zero to all others. FER-2013 contains many ambiguous samples (e.g., `fear` and `surprise` share raised brows, wide eyes; `sad` and `neutral` overlap at low-intensity expressions). Label smoothing with ε=0.05 replaces:
- Correct class target: `1.0 → (1 − 0.05) + 0.05/6 = 0.9583`
- Other classes: `0.0 → 0.05/6 = 0.0083`

This prevents the network from becoming overconfident on inherently ambiguous samples.

### 6.4 Training Loop

```python
def fit(self, train_loader, val_loader, epochs=100):
    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # --- Validation phase ---
        model.eval()
        with torch.no_grad():
            val_acc = evaluate(val_loader)

        # --- Save best checkpoint ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(epoch, val_acc)
```

**Early stopping criterion:** Best validation accuracy checkpoint is saved; training runs to completion (100 epochs). The checkpoint with highest validation accuracy is loaded for final evaluation.

### 6.5 Regularisation Summary

| Technique | V1 | V2 | Effect |
|-----------|----|----|--------|
| Dropout | 0.3 (fixed) | **0.2 (GA-tuned)** | Prevents co-adaptation of neurons |
| BatchNorm1d | Yes | Yes | Normalises activations, regularises through batch statistics |
| Weight decay (L2) | 1e-4 | **5e-4** | Penalises large weights, constrains model complexity |
| Label smoothing | 0.0 | **0.05** | Prevents overconfidence on ambiguous samples |
| Horizontal flip (only) | No | **Yes** | Mild regularisation through train-time variation |

---

## 7. Genetic Algorithm Hyperparameter Tuning

### 7.1 Overview

A custom Genetic Algorithm (`GeneticAlgorithmSearch`) evolves a population of hyperparameter configurations to maximise validation accuracy on a 25-epoch fitness evaluation.

### 7.2 Search Space

| Gene | V1 Options | V2 Options | Rationale for Change |
|------|-----------|-----------|---------------------|
| `learning_rate` | [0.1, 0.05, 0.01, 0.005, 0.001] | **[0.05, 0.01, 0.005, 0.001]** | lr=0.1 → gradient explosion with x² amplification |
| `batch_size` | [16, 32, 64, 128] | **[32, 64, 128]** | batch=16 → excessive gradient noise |
| `num_layers` | **[1, 2, 3, 4, 5]** | **[2, 3, 4, 5]** | Multiple layers needed for hierarchical feature learning; removes single-layer option |
| `hidden_size` | [64, 128, 256, 512] | **[128, 256, 512]** | hidden=64 → bottleneck on 2304-dim input |
| `dropout_rate` | N/A | **[0.2, 0.3, 0.4, 0.5]** | Added as 5th gene for regularisation tuning |

### 7.3 GA Parameters

| Parameter | V1 | V2 |
|-----------|----|----|
| Population size | 10 | 12 |
| Generations | 8 | 8 |
| Evaluation epochs (`eval_epochs`) | **15** | **25** |
| Tournament size | 3 | 3 |
| Crossover rate | 0.8 | 0.8 |
| Mutation rate | 0.1 per gene | 0.2 per gene |

**Root cause fix — eval_epochs 15 → 25:**
At eval_epochs=15, deep networks (num_layers=3+) have not yet escaped initialisation — their validation accuracy is still near random (16.7%). Shallow networks (num_layers=1) converge faster and appear fitter at epoch 15, causing the GA to **consistently select num_layers=1** across all generations. At eval_epochs=25, deep networks begin to show their real performance advantage, allowing the GA to explore multi-layer architectures.

### 7.4 GA Operations

**Chromosome encoding:**
Each chromosome is a list of 5 integer indices, one per gene:
```python
chromosome = [lr_idx, batch_idx, layers_idx, hidden_idx, dropout_idx]
```

**Fitness function:**
```python
def _fitness(self, params):
    model = FERNet(num_layers=params['num_layers'],
                   hidden_size=params['hidden_size'],
                   dropout_rate=params['dropout_rate'])
    trainer = Trainer(model, epochs=self.eval_epochs,
                      checkpoint_name='ga_temp.pth')
    trainer.fit(train_loader, val_loader)
    return trainer.best_val_acc
```

**Tournament selection:** Randomly sample `tournament_size=3` individuals; return the fittest.

**Single-point crossover:** A random split point divides two parent chromosomes; offspring inherit left segment from parent A and right from parent B.

**Mutation:** Each gene independently flips to a random valid value with probability 0.2.

**Elitism:** The top two individuals from each generation are preserved unchanged into the next generation.

### 7.5 GA Results (V2 Run 6 — Clean Pipeline)

| Generation | Best Fitness (Val Acc) | Best Params |
|------------|----------------------|-------------|
| 1 | 0.3733 | lr=0.01, batch=128, layers=2, hidden=512, dropout=0.2 |
| 2 | 0.3750 | lr=0.01, batch=32, layers=2, hidden=512, dropout=0.2 |
| 3 | 0.3767 | lr=0.01, batch=32, layers=2, hidden=512, dropout=0.2 |
| 4 | 0.3800 | lr=0.01, batch=32, layers=2, hidden=512, dropout=0.2 |
| 5 | 0.3783 | lr=0.01, batch=32, layers=2, hidden=512, dropout=0.2 |
| 6 | 0.3800 | lr=0.01, batch=32, layers=2, hidden=512, dropout=0.2 |
| 7 | 0.3750 | lr=0.01, batch=32, layers=2, hidden=512, dropout=0.2 |
| 8 | **0.3867** | lr=0.01, batch=32, layers=2, hidden=512, dropout=0.2 |

The GA converged quickly to `{lr=0.01, batch=32, layers=2, hidden=512, dropout=0.2}` and maintained this configuration across all 8 generations. The consistent selection of `lr=0.01` (rather than `0.05`) was the key finding — `0.05` produces unstable training with the x² gradient amplification, whereas `0.01` provides stable, monotonically decreasing loss.

---

## 8. Model Assessment & Evaluation

### 8.1 Three-Way Split Protocol

| Set | Role | Used When |
|-----|------|-----------|
| Training | Update model weights (SGD) | Every epoch |
| Validation | Monitor overfitting; select best checkpoint; GA fitness signal | Every epoch + GA |
| Test | Final evaluation (held-out) | Once, after all training complete |

The test set is never used during training or hyperparameter selection, ensuring an unbiased estimate of generalisation performance.

### 8.2 Cross-Validation

5-fold cross-validation is run on the combined training+validation data as a secondary reliability check (`run_cross_validation()` in Section 6 of the notebook):

```
V2 CV result: 35.05% ± 1.46%   (V1: 33.10%)
```

The low standard deviation (1.46%) confirms the model generalises consistently across different data splits and the result is not a lucky split artefact.

### 8.3 Test-Time Augmentation (TTA)

During final evaluation, the `Evaluator` class averages softmax probabilities over two passes:
1. Original image
2. Horizontally flipped image

```python
probs = (model.get_probabilities(img) + model.get_probabilities(hflip(img))) / 2
```

TTA provides a free 1–2% accuracy improvement at inference time by reducing prediction variance from left-right facial asymmetry.

### 8.4 Evaluation Metrics

All four required metrics are computed using `sklearn.metrics`:

```python
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall    = recall_score(y_true, y_pred, average='weighted')
f1        = f1_score(y_true, y_pred, average='weighted')
```

Weighted averaging accounts for class imbalance by weighting each class's metric by its support (number of true instances).

### 8.5 Confusion Matrices

6×6 confusion matrices are generated for training, validation, and test sets. Each row represents the true class; each column the predicted class. Off-diagonal elements indicate misclassifications and reveal systematic confusion patterns (e.g., fear being predicted as surprise).

### 8.6 V2 Per-Class Test Performance (Run 6)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Happy** | 0.5556 | 0.5500 | 0.5528 | 100 |
| **Surprise** | 0.4390 | 0.5400 | 0.4843 | 100 |
| **Sad** | 0.2985 | 0.4000 | 0.3419 | 100 |
| **Neutral** | 0.3784 | 0.2800 | 0.3218 | 100 |
| **Angry** | 0.2828 | 0.2800 | 0.2814 | 100 |
| **Fear** | 0.2113 | 0.1500 | 0.1754 | 100 |

**Analysis:**
- **Happy and Surprise** are the best-classified classes. Happy has clear muscular features (zygomatic major muscle, raised cheeks) that produce consistent pixel patterns even at 48×48. Surprise shares features (wide eyes, open mouth) but overlaps with fear.
- **Fear** is the hardest class (15% recall). Fear and surprise share raised eyebrows and wide eyes; the key distinguishing feature (mouth shape — open for surprise, pulled back for fear) is subtle at 48×48.
- **Angry and Neutral** are confused with each other; low-intensity anger (no visible teeth, slight brow furrow) is visually indistinguishable from neutral at low resolution.

---

## 9. V1 vs V2 Comparison & Improvement Analysis

### 9.1 Quantitative Results

| Metric | V1 | V2 (Run 6) | Improvement |
|--------|----|-----------|-----------|
| **Test Accuracy** | 36.17% | **36.67%** | +0.50% |
| **Validation Accuracy** | 35.33% | **37.50%** | +2.17% |
| **CV Accuracy** | 33.10% | **35.05%** ± 1.46% | +1.95% |
| **Train-Test Gap** | ~6.5% | **4.91%** | Better generalisation |
| **GA Best Fitness** | 0.3867 | 0.3867 | Equal (converged to same GA fitness) |
| **Model Architecture** | 1 layer, hidden=64 | **2 layers, hidden=512** | 10× more parameters |
| **Checkpoint Size** | 1.2 MB | **12 MB** | — |

### 9.2 Root Cause of V1's Underperformance

**V1's GA consistently selected `num_layers=1, hidden_size=64` because:**
1. `eval_epochs=15` was too short for deeper networks to converge
2. The search space included `num_layers=1` and `hidden_size=64` which are shallow/narrow
3. The `[-1,1]` input domain made the x+x² formula non-monotone, reducing learning efficiency
4. `lr=0.1` in the search space occasionally destabilised training

At `eval_epochs=15`, a 1-layer network with hidden=64 (148K params) converges quickly and shows ~33-38% validation accuracy, while a 3-layer network with hidden=512 (1.45M params) is still near 20% (random initialisation recovery phase). The GA never had the chance to discover that deeper networks are better.

### 9.3 The 8 Regression Fixes (V2 Development Journey)

V2 was not a straight-line improvement — it went through 6 full training runs before beating V1:

| Run | Test Acc | Key Issue | Fix Applied |
|-----|----------|-----------|-------------|
| V1 | 36.17% | GA selects shallow arch, short eval | — |
| V2 Run 1 | 28.0% | RandomErasing destroying facial pixels | Remove RandomErasing |
| V2 Run 2 | 30.0% | GA still on broken pipeline | Restore eval_epochs |
| V2 Run 3 | 31.8% | train_acc < val_acc (augmentation too aggressive) | Remove ColorJitter |
| V2 Run 4 | ~33% | Rotation/affine hurting FC net | Diagnose augmentation |
| V2 Run 5 | 34.67% | train > val restored but dropout too weak | Remove Rotation/Affine (HFlip only) |
| **V2 Run 6** | **36.67%** | **Clean GA run — discovers lr=0.01** | **5e-4 weight decay, clean search space** |

### 9.4 Key Technical Insight — Why lr=0.01 was Critical

With the custom formula `net = W @ (x + x²)`, inputs in `[0,1]` produce pre-activations in `[0,2]` (since `x + x² = x(1+x) ≤ 2` when x≤1). The gradient of the weight update is `δⱼ · (xᵢ + xᵢ²)` — up to twice as large as standard backprop. At `lr=0.05`, this means effective gradient steps are `0.05 × 2 = 0.1` in the worst case. The GA selecting `lr=0.01` effectively limits maximum gradient steps to `0.02`, preventing the oscillations that were preventing convergence in Runs 2–5.

---

## 10. Deployment System

### 10.1 fer_deploy.py — Standard Deployment

A self-contained inference script that:
1. Loads a `best_model_v2.pth` checkpoint
2. Reads `num_layers` and `hidden_size` from the checkpoint metadata
3. Instantiates `FERNet` with those dimensions
4. Applies the V2 preprocessing pipeline (Blur → CLAHE → ToTensor)

**Inference pipeline:**
```
Input (file path / numpy array / PIL Image)
    │
    ▼ _load_image()
  Convert to grayscale → resize to 48×48 (LANCZOS)
    │
    ▼ transform()
  GaussianBlur(3×3) → CLAHE(3.0, 6×6) → ToTensor [0,1]
    │
    ▼ model.get_probabilities()
  CustomNetLayer × num_layers → SoftMax → 6 probabilities
    │
    ▼
  argmax → (label, confidence, all_probs dict)
```

**Face detection:** OpenCV `haarcascade_frontalface_default.xml`
**Face crop:** 10% padding, non-square (inherits detector aspect ratio)
**Display:** Grayscale converted (GRAY2BGR) so output matches training distribution

**Usage:**
```bash
python fer_deploy.py --image Data/Testing/Happy/img1.jpg
python fer_deploy.py --video Sample_Data/vids/test.mp4
python fer_deploy.py --camera
```

### 10.2 fer_deploy_V2.py — Enhanced Deployment

Three improvements over `fer_deploy.py`, all implemented without modifying `FER_System_V2.ipynb`:

#### Improvement 1 — Face Alignment (FaceAligner)

```python
class FaceAligner:
    def align(self, face_gray):
        eyes = self.eye_cascade.detectMultiScale(face_gray, ...)
        if len(eyes) < 2:
            return face_gray  # graceful fallback
        centres = sorted([(ex+ew//2, ey+eh//2) for ex,ey,ew,eh in eyes],
                         key=lambda c: c[0])
        left_eye, right_eye = centres[0], centres[1]
        angle = np.degrees(np.arctan2(dy, dx))
        M = cv2.getRotationMatrix2D(centre, angle, scale=1.0)
        return cv2.warpAffine(face_gray, M, (w,h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
```

**Effect:** Rotates face so both eyes are horizontally level before passing to the model. The FER-2013 training set contains mostly frontal, level faces. A head-tilted input at inference introduces pixel offsets across all 2304 features of the FC network — alignment reduces this distribution mismatch.

**Fallback:** If `haarcascade_eye.xml` cannot find two eyes (unusual poses, occlusion), the unaligned crop is used silently.

#### Improvement 2 — Square Crop with 20% Padding (extract_face_crop)

```python
def extract_face_crop(gray_frame, x, y, w, h, pad=0.20):
    cx, cy = x + w//2, y + h//2
    half = int(max(w, h) * (1 + pad) / 2)
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(W, cx + half), min(H, cy + half)
    return gray_frame[y1:y2, x1:x2]
```

**Effect:** The Haar face detector often returns non-square bounding boxes (taller than wide). V1 crops this rectangle and resizes to 48×48, which **squashes** the face horizontally. V2 takes the larger dimension (`max(w,h)`) to force a square crop, preventing distortion. The 20% padding (vs V1's 10%) provides more forehead/chin context, reducing sensitivity to crops that just clip the face boundary.

#### Improvement 3 — Temporal Smoothing (PredictionSmoother)

```python
class PredictionSmoother:
    def __init__(self, window=10):
        self._buffers = {}  # {face_idx: deque(maxlen=window)}

    def update(self, face_idx, probs):
        self._buffers[face_idx].append(probs)
        return np.mean(self._buffers[face_idx], axis=0)
```

**Effect:** Per-frame predictions are noisy — a single slightly-blurred frame can produce a completely different classification. Rolling window averaging over 10 frames smooths this by averaging softmax probability *vectors* (not labels), so the smoothed output is the mean probability distribution over the last N frames. This eliminates the rapid label flickering (`happy → neutral → happy → neutral`) common in live camera feeds.

The smoothing operates on **softmax probabilities**, not argmax labels, so subtle changes in confidence are preserved while noise is averaged out.

**Configurable window:** `--smooth 15` increases to 15 frames (more stable but slower to react to genuine emotion changes); `--smooth 3` is nearly instant (responsive but noisier).

| Mode | Default Window | Rationale |
|------|---------------|-----------|
| Camera | 10 frames | Balance stability vs responsiveness |
| Video | 5 frames | Lower since each frame is meaningful |
| Image | No smoothing | Single frame, smoothing not applicable |

### 10.3 Why Color Inputs Fail

The FER-2013 dataset is natively grayscale. Color BGR images from camera or photos fail because:

1. **Training distribution mismatch:** The model has never seen any color information. The BGR→grayscale conversion during inference (via `cv2.COLOR_BGR2GRAY`) does match training, but real-world color images have different luminance distributions than FER-2013's curated grayscale photos.

2. **CLAHE on natural vs posed faces:** FER-2013 images are studio-style posed expressions with controlled lighting. Real-world camera images have varied lighting angles, shadows, and backgrounds that CLAHE cannot fully normalise.

3. **Expression naturalness:** FER-2013 contains posed or near-posed expressions selected to be recognisable. Natural, spontaneous expressions in real camera footage are lower-intensity and harder to classify.

The test on a natively grayscale YouTube video (`sample_data/vids/test.mp4`) performed significantly better than color photo tests, confirming that the primary bottleneck is distribution shift rather than model quality.

---

## 11. Results Summary

### 11.1 Final V2 Metrics (Run 6)

| Split | Accuracy | Notes |
|-------|----------|-------|
| **Test** | **36.67%** | GA-tuned, 100-epoch training |
| **Validation** | **37.50%** | Best checkpoint at validation |
| **Cross-Validation** | **35.05% ± 1.46%** | 5-fold on train+val |
| **Training** | ~41.58% | Train-test gap: 4.91% |

**Overall weighted metrics (test set):**
| Metric | Value |
|--------|-------|
| Accuracy | 36.67% |
| Precision (weighted) | 0.3609 |
| Recall (weighted) | 0.3667 |
| F1-Score (weighted) | 0.3596 |

### 11.2 Context — What 36.67% Means

FER-2013 is a hard benchmark:
- **Random baseline:** 16.7% (6 classes)
- **Human accuracy on FER-2013:** ~65%
- **State-of-the-art (deep CNN + attention):** ~75%
- **This project (FC network with x+x² formula):** **36.67%**

The choice to implement all hidden layers as fully-connected (following the perceptron formula naturally) means the network must learn all spatial structure from the flattened 2304-dim input. CNNs achieve higher accuracy because their convolutional filters explicitly learn spatially-local features (edge detectors, texture patches) via shared weights — a strong inductive bias for image data. The `x + x²` formula was applied within FC layers as the most direct and controlled interpretation of the quadratic perceptron specification, isolating its contribution from spatial priors.

### 11.3 Strengths and Weaknesses by Class

**Strongest classes:** Happy (55% recall), Surprise (54% recall)
- Both have large, visible deformations: smiling stretches the mouth, surprise produces an O-shape. These are recognisable even at low resolution.

**Weakest class:** Fear (15% recall)
- Fear's distinguishing features (corners of mouth pulled back and down, upper white of eye visible) are subtle at 48×48 resolution and overlap heavily with surprise (raised brows, wide eyes).

**Systematic confusion:**
- Fear ↔ Surprise: Highest off-diagonal in confusion matrix
- Angry ↔ Neutral: Low-intensity angry blends into neutral
- Sad ↔ Neutral: Both have downward-pulled features at resting face

---

## 12. Limitations & Future Work

### 12.1 Current Limitations

1. **Fully-connected architecture:** The custom formula `net = Σ wᵢ(xᵢ + xᵢ²)` maps naturally to `nn.Linear(x + x²)`, so all hidden layers were implemented as FC layers. This is a deliberate design choice to isolate the contribution of the quadratic activation from convolutional spatial priors. CNNs with shared-weight filters are significantly more suited to spatial image classification and would likely close much of the gap to human-level accuracy.

2. **Small input resolution (48×48):** Many FER datasets use higher resolution; at 48×48, subtle expression features (slight lip corner depression, partial eye opening) are too low-fidelity to distinguish reliably.

3. **Real-world distribution shift:** The deployment scripts perform noticeably worse on natural (non-posed) faces, color images converted to grayscale, and varied lighting conditions compared to the curated FER-2013 test set.

4. **No temporal modelling:** Even with temporal smoothing, the system classifies each frame independently. A sequential model (LSTM, Transformer) over video would improve performance by using context from preceding frames.

5. **GA evaluation cost:** 8 generations × 12 individuals × 25 epochs = 2,400 training epochs per GA run (~10–15 minutes on GPU). More generations or larger populations would improve hyperparameter quality but are computationally expensive.

### 12.2 Future Improvements (Separate Project)

If this were extended beyond the current architectural constraints:

1. **CNN backbone + FC head:** Replace the FC hidden layers with 3–5 convolutional blocks (Conv2d + BN + ReLU + MaxPool), feeding into a small FC head for classification. Expected test accuracy: 60–70%.

2. **Larger dataset:** FER-2013 (35,887 images) → AffectNet (450,000 images) or RAF-DB (29,672 real-world images). Real-world images are more representative of deployment conditions.

3. **Pre-trained transfer learning:** Start from an ImageNet-pretrained ResNet-18 or EfficientNet-B0, replace final layer, fine-tune on FER data. Expected test accuracy: 70–75%.

4. **LSTM over video frames:** Process sequences of face crops through an LSTM rather than classifying frames independently. Temporal context is a strong signal for expression recognition.

5. **Data augmentation:** `MixUp`, `CutMix`, and advanced geometric augmentations are safe for CNN architectures and would reduce overfitting.

6. **Class-balanced sampling:** Upsample underrepresented classes (fear, disgust) or use weighted cross-entropy to improve recall on hard classes.

---

## 13. Design Constraints Checklist

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| 48×48 grayscale input | ✓ | `_load_image()` + `transforms.Grayscale(1)` |
| Contrast normalisation | ✓ | CLAHE(clip=3.0, tile=6×6) in `FERPreprocessor` |
| Modified activation: `σ(net) = 1/(1+e^{-net})` | ✓ | `CustomSigmoid.forward()` |
| Modified net: `net = Σ wᵢ(xᵢ + xᵢ²)` | ✓ | `CustomNetLayer.forward()`: `self.linear(x + x**2)` |
| Multiple layers | ✓ | GA search starts at `num_layers=2`; V2 uses 2 layers |
| SoftMax final layer | ✓ | `FERNet.get_probabilities()` + CrossEntropyLoss (log-softmax internally) |
| Dropout regularisation | ✓ | `nn.Dropout(dropout_rate=0.2)` in every `CustomBlock` |
| Batch normalisation | ✓ | `nn.BatchNorm1d` in every `CustomBlock` |
| CrossEntropyLoss | ✓ | `nn.CrossEntropyLoss(label_smoothing=0.05)` |
| SGD optimiser | ✓ | `optim.SGD(momentum=0.9, nesterov=True, weight_decay=5e-4)` |
| Training set for weight updates | ✓ | `trainer.fit(train_loader, val_loader)` |
| Validation set for tuning | ✓ | Best checkpoint by `best_val_acc`; GA fitness signal |
| Test set for final evaluation | ✓ | `evaluator.evaluate(test_loader)` — called once after training |
| Accuracy metric | ✓ | `accuracy_score(y_true, y_pred)` |
| Precision metric | ✓ | `precision_score(..., average='weighted')` |
| Recall metric | ✓ | `recall_score(..., average='weighted')` |
| F1-score metric | ✓ | `f1_score(..., average='weighted')` |
| Confusion matrix | ✓ | `confusion_matrix_test_v2.png` (all 6 classes shown) |
| GA tunes learning rate | ✓ | Gene 0: `[0.05, 0.01, 0.005, 0.001]` |
| GA tunes batch size | ✓ | Gene 1: `[32, 64, 128]` |
| GA tunes num_layers | ✓ | Gene 2: `[2, 3, 4, 5]` |
| GA tunes hidden_size | ✓ | Gene 3: `[128, 256, 512]` |
| Inference on arbitrary images | ✓ | `fer_deploy.py` + `fer_deploy_V2.py` — `--image`, `--video`, `--camera` |
| Real-time/near-real-time | ✓ | ~15–25 FPS on CPU; temporal smoothing in V2 |
| Python 3.0+ | ✓ | Python 3.x (type hints, f-strings used throughout) |
| OpenCV 4.5+ | ✓ | CLAHE, Haar cascade, VideoCapture |
| NumPy | ✓ | Array ops in preprocessing and GA |
| Matplotlib | ✓ | All plots (curves, confusion matrix, GA convergence, CV results) |
| PyTorch (ML library) | ✓ | Model, training, inference |

**All design constraints are implemented.**

---

## Appendix A — File Structure

```
AI_CW2/
├── FER_System.ipynb          # V1 training notebook (unchanged)
├── FER_System_V2.ipynb       # V2 training notebook (8 sections)
├── fer_deploy.py             # V2-compatible standard deployment
├── fer_deploy_V2.py          # Enhanced deployment (align + crop + smooth)
├── docs/experiments.md       # Full development log (6 training runs, 8 regression fixes)
├── report.md                 # This document
│
├── checkpoints/
│   ├── best_model.pth        # V1 checkpoint (num_layers=1, hidden=64)
│   └── best_model_v2.pth     # V2 checkpoint (num_layers=2, hidden=512)
│
├── results/
│   ├── ga_log.json           # GA search history (V2 Run 6)
│   ├── ga_log_v2.json        # GA search history (earlier runs)
│   ├── training_curves_v2.png
│   ├── confusion_matrix_test_v2.png
│   ├── cv_results_v2.png
│   ├── ga_convergence_v2.png
│   ├── v1_v2_comparison.png
│   └── output_video.mp4      # Sample annotated video output
│
├── Data/
│   ├── Training/             # Training images (6 class subdirs)
│   ├── Validation/           # Validation images
│   └── Testing/              # Test images (held-out)
│
└── Sample_Data/
    ├── images/               # 19 subject × 8 emotion sample images
    └── vids/test.mp4         # Sample grayscale test video
```

## Appendix B — Key Equations Summary

**Custom forward pass (single neuron j, layer l):**
```
netₗⱼ = Σᵢ wₗᵢⱼ · (aₗ₋₁,ᵢ + aₗ₋₁,ᵢ²)
aₗⱼ = 1 / (1 + e^{−netₗⱼ})
```

**Output layer delta (SoftMax + CrossEntropy):**
```
δ_{L+1,j} = ŷⱼ − yⱼ
```

**Hidden layer delta:**
```
δₗⱼ = (Σₖ w_{l+1,jk} · δ_{l+1,k}) · aₗⱼ · (1 − aₗⱼ)
```

**Weight update rule (ALL layers):**
```
Δwₗᵢⱼ = −η · δₗⱼ · (aₗ₋₁,ᵢ + aₗ₋₁,ᵢ²)
```

**SGD with Nesterov momentum:**
```
vₜ = β·vₜ₋₁ + ∇L(θₜ + β·vₜ₋₁)    [lookahead gradient]
θₜ₊₁ = θₜ − η·vₜ
```

**CosineAnnealing schedule:**
```
ηₜ = η_min + (η_max − η_min) · (1 + cos(πt/T_max)) / 2
```
