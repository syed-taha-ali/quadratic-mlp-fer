# Quadratic MLP for Facial Expression Recognition

An exploration of **polynomial feature expansion in fully-connected neural networks** applied to facial expression classification. The core research question: can a quadratic activation formula learn discriminative facial features without convolutional spatial priors?

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/AI_CW2.git
cd AI_CW2
pip install -r requirements.txt

# 2. Place dataset (see Dataset Setup below)

# 3. Run V2 notebook end-to-end
jupyter notebook FER_System_V2.ipynb
# Run all cells. GA takes ~10–15 min on GPU, ~60–90 min on CPU.
# Skip GA by setting SKIP_GA = True in Cell 16 — hardcoded best params are provided.

# 4. Run live inference
python fer_deploy_V2.py --camera
```

---

## The Architecture

### Quadratic Net Input Formula

Each neuron computes a modified net input that combines linear and quadratic terms under a shared weight:

```
net = w₀x₀ + w₀x₀² + w₁x₁ + w₁x₁² + ... + wₙxₙ + wₙxₙ²
    = W · (x + x²)
```

Implemented as a single efficient PyTorch module:

```python
class CustomNetLayer(nn.Module):
    def forward(self, x):
        return self.linear(x + x ** 2)  # element-wise x², then W @ (x + x²)
```

This is a form of **polynomial feature expansion** — it implicitly augments each input with its square without doubling the parameter count. The same weight `wᵢ` governs both the linear and quadratic contribution of input `xᵢ`, coupling the two terms and constraining the search space.

### Why FC-only (no CNN)?

The architecture is intentionally fully-connected throughout. This isolates the contribution of the quadratic activation from convolutional spatial inductive bias, making it possible to study what the `x + x²` formula adds in a controlled setting. CNNs achieve higher accuracy on image tasks precisely because their shared-weight filters are a strong prior for spatially-local features — removing that prior reveals the raw expressive capacity of the quadratic activation.

### Activation & Output

- **Hidden layers:** `CustomSigmoid` — `σ(net) = 1 / (1 + e^{-net})`
- **Output:** SoftMax over 6 classes
- **Block structure:** `CustomNetLayer → BatchNorm1d → CustomSigmoid → Dropout`

---

## Dataset Setup

Download the dataset from [Kaggle — FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) and place it in the following structure:

```
AI_CW2/
└── Data/
    ├── Training/
    │   ├── Angry/
    │   ├── Fear/
    │   ├── Happy/
    │   ├── Neutral/
    │   ├── Sad/
    │   └── Surprise/
    ├── Validation/
    │   └── (same 6 subfolders)
    └── Testing/
        └── (same 6 subfolders)
```

The notebooks use `torchvision.datasets.ImageFolder`, so folder names must match the class names exactly (capital first letter).

## Dataset

FER-2013 subset — 48×48 grayscale face images across 6 expression classes:

| Class | Training | Validation | Test |
|-------|----------|------------|------|
| Angry | 1,000 | 100 | 100 |
| Fear | 1,000 | 100 | 100 |
| Happy | 1,000 | 100 | 100 |
| Neutral | 1,000 | 100 | 100 |
| Sad | 1,000 | 100 | 100 |
| Surprise | 1,000 | 100 | 100 |

---

## Preprocessing

Images pass through a **V2 CLAHE pipeline** before entering the network:

```
GaussianBlur(3×3)              — suppress sensor noise before contrast enhancement
CLAHE(clipLimit=3.0, tile=6×6) — local adaptive histogram equalisation
ToTensor() → [0, 1]            — no global normalisation (see below)
```

**Why no normalisation to `[-1, 1]`?**
The quadratic term `x + x²` is strictly monotone on `[0, 1]` (derivative `1 + 2x > 0`). On `[-1, 1]` it is non-monotone — two inputs with different signs can produce the same pre-activation value, destroying discriminative information. Keeping pixels in `[0, 1]` preserves the full contrast range from CLAHE.

---

## Hyperparameter Search

A custom **Genetic Algorithm** tunes 5 hyperparameters simultaneously:

| Gene | Search Space |
|------|-------------|
| Learning rate | [0.05, 0.01, 0.005, 0.001] |
| Batch size | [32, 64, 128] |
| Number of layers | [2, 3, 4, 5] |
| Hidden size | [128, 256, 512] |
| Dropout rate | [0.2, 0.3, 0.4, 0.5] |

**GA parameters:** population=12, generations=8, eval_epochs=25, tournament_k=3, mutation_rate=0.2, top-2 elitism.

GA is used rather than grid search because the non-standard activation creates a loss landscape where gradient-free population-based search explores architectural decisions (depth, width) more effectively than exhaustive enumeration.

**GA-discovered best config:** `lr=0.01, batch=32, layers=2, hidden=512, dropout=0.2`

---

## Results

| Version | Architecture | Test Acc | Val Acc | CV (5-fold) |
|---------|-------------|----------|---------|-------------|
| V1 (baseline) | 1 layer, hidden=64 | 36.17% | 35.33% | 33.10% |
| **V2 (GA-tuned)** | **2 layers, hidden=512** | **36.67%** | **37.50%** | **35.05% ± 1.46%** |

### Accuracy in Context

| System | FER-2013 Accuracy |
|--------|-----------------|
| Random baseline | 16.7% |
| **This project (Quadratic MLP)** | **36.67%** |
| Human accuracy | ~65% |
| CNN + attention (SOTA) | ~75% |

The gap to CNN-based systems is expected and intentional — the FC architecture has no spatial inductive bias. A CNN learns position-invariant edge and texture detectors via shared convolutional weights; this network must infer all spatial structure from the raw 2304-dim pixel vector. The ~36% result is above random by 2.2× and comparable to other MLP-only baselines on this dataset.

**Per-class performance (V2 test set):**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Happy | 0.5556 | 0.55 | 0.5528 |
| Surprise | 0.4390 | 0.54 | 0.4843 |
| Sad | 0.2985 | 0.40 | 0.3419 |
| Neutral | 0.3784 | 0.28 | 0.3218 |
| Angry | 0.2828 | 0.28 | 0.2814 |
| Fear | 0.2113 | 0.15 | 0.1754 |

Fear and Angry are hardest — both share visual features with other classes (Fear≈Surprise, Angry≈Neutral) that are indistinguishable at 48×48 without spatially-localised filters.

---

## Backpropagation with the Custom Formula

The quadratic term modifies the weight update rule. For a hidden layer weight `wᵢⱼ`:

```
Standard:   Δwᵢⱼ = −η · δⱼ · xᵢ
Quadratic:  Δwᵢⱼ = −η · δⱼ · (xᵢ + xᵢ²)
```

The factor `(xᵢ + xᵢ²)` amplifies gradient updates for strongly-activated inputs — larger steps where the network is already confident. This accelerates learning but requires a lower learning rate (`lr=0.01` vs typical `0.1`) and gradient clipping (`max_norm=1.0`) to maintain stability.

---

## Training

- **Optimiser:** SGD with Nesterov momentum (momentum=0.9, weight_decay=5e-4)
- **Schedule:** Linear LR warmup (5 epochs) → CosineAnnealingLR (eta_min=1e-5)
- **Loss:** CrossEntropyLoss with label smoothing (ε=0.05)
- **Epochs:** 100 (final training)
- **Regularisation:** BatchNorm1d + Dropout(0.2) + L2 weight decay

---

## Deployment

Two deployment scripts support image, video, and live camera inference:

```bash
# Standard (fer_deploy.py)
python fer_deploy.py --image path/to/face.jpg
python fer_deploy.py --video path/to/video.mp4
python fer_deploy.py --camera

# Enhanced (fer_deploy_V2.py) — face alignment + temporal smoothing
python fer_deploy_V2.py --camera
python fer_deploy_V2.py --camera --smooth 15
```

**Face detection:** OpenCV Haar Cascade (bundled with opencv-python, no extra installs).

**`fer_deploy_V2.py` additions:**
- **Face alignment** — detects eye centres, rotates crop to horizontal eye line before inference
- **Square crop with 20% padding** — prevents aspect-ratio distortion from non-square detector boxes
- **Temporal smoothing** — rolling average of softmax probability vectors over N frames, eliminates per-frame label flickering on live feeds

---

## Project Structure

```
AI_CW2/
├── FER_System.ipynb        — V1 notebook (baseline)
├── FER_System_V2.ipynb     — V2 notebook (GA-tuned, all improvements)
├── fer_deploy.py           — deployment script
├── fer_deploy_V2.py        — enhanced deployment script
├── requirements.txt        — Python dependencies
├── LICENSE                 — copyright notice
├── docs/
│   ├── plan.md             — implementation plan
│   ├── report.md           — full technical report
│   ├── experiments.md      — development log (6 training runs, root cause analysis)
│   └── code_review.md      — static code review (21 issues across 4 files)
├── checkpoints/            — saved model weights (.pth)
└── results/                — plots, confusion matrices, GA logs
```

---

## Reproducibility

All experiments use `seed=42` (Python, NumPy, PyTorch CPU/CUDA). Results were produced on an NVIDIA GPU; CPU runs may differ slightly due to non-deterministic operations in some PyTorch backends.

---

## Requirements

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch separately first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## License

Copyright © 2026 Syed Taha Ali. All rights reserved.

This source code is provided for viewing purposes only. See [LICENSE](LICENSE) for details.
