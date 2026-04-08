# Facial Expression Recognition System — Implementation Plan

## Context
A Facial Expression Recognition system built on a **quadratic MLP** — a fully-connected network using a modified net input formula (`net = W @ (x + x²)`) and genetic algorithm hyperparameter search. Dataset: 7,200 grayscale 48×48 images across 6 emotion classes (angry, fear, happy, neutral, sad, surprise) with pre-split Train/Val/Test folders.

Deadline: 09-04-2026. Deliverables: `FER_System.ipynb` (V1 baseline) + `FER_System_V2.ipynb` (improved) + max 6-page report.

---

## Deliverable Structure

Two notebooks — V1 baseline and V2 improved:

```
AI_CW2/
├── FER_System.ipynb        ← V1 baseline (1 layer, hidden=64, 36.17% test acc)
├── FER_System_V2.ipynb     ← V2 improved (2 layers, hidden=512, 36.67% test acc)
├── fer_deploy.py           ← deployment script (image / video / camera)
├── fer_deploy_V2.py        ← enhanced deployment (face alignment, temporal smoothing)
├── checkpoints/
│   ├── best_model.pth      ← V1 best checkpoint
│   └── best_model_v2.pth   ← V2 best checkpoint
└── results/                ← plots, confusion matrices, GA logs (_v2 suffix for V2 artefacts)
```

---

## Section-by-Section Plan

### Section 0 — Imports & Config
- Central `CONFIG` dict: `data_root`, `classes`, `img_size=48`, `seed=42`, `device`
- `set_seed(seed)` utility (Python, NumPy, PyTorch)
- `os.makedirs` for `checkpoints/` and `results/`

---

### Section 1 — Pre-Processing (10%)

**`FERPreprocessor`** (callable class using OpenCV CLAHE):
```python
class FERPreprocessor:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    def __call__(self, pil_image):
        img_np = np.array(pil_image, dtype=np.uint8)
        return Image.fromarray(self.clahe.apply(img_np))
```

**Transform chain** (`build_transforms(preprocessor, augment=False)`):
1. `transforms.Grayscale(1)` — ensure single channel
2. `transforms.Lambda(preprocessor)` — CLAHE
3. Training only: `RandomHorizontalFlip(0.5)`, `RandomRotation(10)`, `RandomAffine(translate=0.05)`
4. `transforms.ToTensor()` → `[0,1]`, shape `[1,48,48]`
5. `transforms.Normalize([0.5],[0.5])` → `[-1,1]`

**`build_dataloaders(config, batch_size, preprocessor)`**: wraps `torchvision.datasets.ImageFolder` for all three splits.

**Justification** (markdown in notebook): CLAHE performs local histogram equalization, enhancing discriminative facial regions (eyes, mouth) without over-amplifying noise in uniform areas that global HE would cause.

---

### Section 2 — Custom Neural Network (50%)

#### `CustomNetLayer` — core custom formula
The spec requires: `net = w0*x0 + w0*x0² + w1*x1 + w1*x1² + ... = W @ (x + x²)`
Same weight `w_i` is used for both linear and quadratic terms.

```python
class CustomNetLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x):
        return self.linear(x + x ** 2)   # element-wise x², then W @ (x + x²)
```

#### `CustomSigmoid`
```python
class CustomSigmoid(nn.Module):
    # sigma(net) = 1 / (1 + exp(-net))
    def forward(self, x):
        return torch.sigmoid(x)
```

#### `CustomBlock` — one hidden layer unit
```
CustomNetLayer → BatchNorm1d → CustomSigmoid → Dropout
```
BatchNorm before activation prevents sigmoid saturation from x² magnitude amplification.

#### `FERNet`
- Input: `[batch, 1, 48, 48]` → flatten to `[batch, 2304]`
- `num_layers` × `CustomBlock(in→hidden_size)`
- Final: `CustomNetLayer(hidden_size → 6)` returns raw logits
- `forward(x)`: raw logits (used with `CrossEntropyLoss`)
- `get_probabilities(x)`: applies `nn.Softmax(dim=1)` for inference only

#### `initialize_weights(model)`: Xavier uniform on all `nn.Linear` weights, zeros on biases.

**Critical:** Do NOT apply softmax before `nn.CrossEntropyLoss` — it applies log-softmax internally.

---

### Section 3 — Training Infrastructure

#### `Trainer` class
- `criterion = nn.CrossEntropyLoss()`
- `optimizer = optim.SGD(lr=lr, momentum=0.9, weight_decay=1e-4)`
- `scheduler = StepLR(step_size=10, gamma=0.5)`
- `train_one_epoch()`: includes `clip_grad_norm_(max_norm=1.0)` — necessary because x² terms amplify gradients
- `evaluate(loader)`: `model.eval()` + `torch.no_grad()`
- `fit(verbose)`: saves `best_model.pth` on val_acc improvement; records `history` dict
- `save_checkpoint / load_checkpoint`

---

### Section 4 — Genetic Algorithm (10%)

**Search space:**
```python
GA_SEARCH_SPACE = {
    "learning_rate": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005],
    "batch_size":    [16, 32, 64, 128],
    "num_layers":    [1, 2, 3, 4, 5],
    "hidden_size":   [64, 128, 256, 512],
}
```

**Chromosome:** list of 4 integers (indices into above lists).

#### `GeneticAlgorithmSearch` class
| Parameter | Value |
|-----------|-------|
| `population_size` | 12 |
| `num_generations` | 8 |
| `eval_epochs` | 15 |
| `mutation_rate` | 0.2 |

- `_fitness(chromosome)`: builds FERNet, trains for `eval_epochs`, returns best val_acc
- `_tournament_select(k=3)`: k-tournament selection
- `_crossover(p1, p2)`: single-point crossover (80% probability)
- `_mutate(chromosome)`: per-gene random replacement
- **Elitism**: top 2 individuals carried forward unchanged each generation
- `run()`: returns `best_params` dict; saves `ga_log` to `results/ga_log.json`

**Runtime estimate:** 12×8 = 96 training runs × 15 epochs each. CPU: ~45-60 min; GPU: ~5-10 min.

---

### Section 5 — Final Training & Evaluation

`train_final_model(best_params, config, preprocessor, full_epochs=50)`:
- Train with best GA hyperparameters for 50 full epochs
- Load `best_model.pth` after training (best val checkpoint)

#### `Evaluator` class
- `predict_all(loader)` → `(y_true, y_pred)` numpy arrays
- `compute_metrics(loader)` → accuracy + `sklearn.classification_report` (precision, recall, F1 per class + macro averages)
- `plot_confusion_matrix(loader)` → `results/confusion_matrix.png`
- `plot_training_history(history)` → `results/training_curves.png`

Evaluate on: training set, validation set, and **test set** (primary reported metric).

---

### Section 6 — Cross-Validation Assessment (10%)

5-fold `StratifiedKFold` on combined Training + Validation (6,600 images):
- Helper `FERDatasetFromPaths(paths, labels, transform)`: `Dataset` subclass for arbitrary path+label lists
- Helper `load_all_file_paths(data_root, splits)`: collects file paths and integer labels from folder structure
- Train `cv_epochs=30` per fold using `best_params`
- Report: mean ± std accuracy across 5 folds

---

### Section 7 — Inference Module

#### `FERInference` class
- `__init__(model_path, config, preprocessor, num_layers, hidden_size)`: loads checkpoint, `model.eval()`, `dropout_rate=0.0`
- `_load_image(input)`: accepts file path (str), numpy array, or PIL Image → grayscale PIL
- `predict(image_input)` → `(predicted_label: str, confidence: float, all_probs: dict)`
- `predict_batch(image_list)` → list of predictions

Demo cell: load 1 image per class from test set, display image grid with predicted labels and confidence bar charts.

---

## Data Flow

```
Image file
    ↓ transforms.Grayscale(1)          ensure single channel
    ↓ FERPreprocessor (CLAHE/OpenCV)   local contrast enhancement
    ↓ transforms.ToTensor()            PIL→Tensor [0,1], shape [1,48,48]
    ↓ transforms.Normalize([0.5],[0.5]) scale to [-1,1]
    ↓
FERNet.forward()
    x.view(batch, -1)                  flatten → [batch, 2304]
    CustomBlock × N:
        CustomNetLayer: W @ (x + x²)   core custom formula
        BatchNorm1d                     stabilize pre-activations
        CustomSigmoid: 1/(1+exp(-net))  custom activation
        Dropout                         regularization
    CustomNetLayer → logits [batch, 6]
    ↓
nn.CrossEntropyLoss (training)  |  nn.Softmax (inference)
```

---

## Critical Files

| Path | Purpose |
|------|---------|
| `AI_CW2/FER_System.ipynb` | Primary deliverable — all 7 sections |
| `AI_CW2/Data/Training/{class}/` | ImageFolder root for training (1,000/class) |
| `AI_CW2/Data/Validation/{class}/` | ImageFolder root for validation (100/class) |
| `AI_CW2/Data/Testing/{class}/` | ImageFolder root for test evaluation (100/class) |
| `AI_CW2/checkpoints/best_model.pth` | Best model checkpoint |
| `AI_CW2/results/` | Plots, confusion matrices, GA log JSON |

---

## Key Libraries

```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
import cv2, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import os, random, json
```

---

## Component Overview

| Component | Implementation Location |
|-----------|------------------------|
| Pre-Processing (CLAHE + justification) | Section 1 |
| Custom NN Design (`CustomNetLayer` + architecture) | Section 2 |
| Cross-validation | Section 6 |
| Results (metrics, confusion matrix) | Section 5 |
| Hyperparameter Tuning (GA) | Section 4 |
| Inference & Demo | Section 7 |

---

## Verification Checklist

1. **Data loading**: Print class counts; display 6 sample images post-CLAHE
2. **Custom layer**: Unit test — verify `CustomNetLayer([1,2,3]) == nn.Linear` applied to `[x+x²]`
3. **Training loop**: Confirm loss decreases over first 5 epochs
4. **GA**: Print fitness per generation; verify `best_params` decoding is correct
5. **Evaluation**: Check confusion matrix rows sum to 100 (100 test images per class)
6. **Inference**: Run on 6 test images; print predicted vs. ground truth

---

## Deployment — `fer_deploy.py`

### Context
Standalone Python script providing a live demo of the trained model with three input modes: static image, video file, and live webcam. Kept separate from the notebook because `cv2.imshow()` for real-time display requires a proper Python process. The existing `FERInference.predict()` already accepts numpy arrays (OpenCV frames), so the core inference pipeline is reused directly.

### File
`AI_CW2/fer_deploy.py`

---

### Key Design Decisions

1. **Face detection**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) — bundled with `opencv-python`, no extra installs
2. **Model classes**: Re-defined verbatim in `fer_deploy.py` (cannot import from `.ipynb`)
3. **Display**: `cv2.imshow()` for camera and video modes; `matplotlib` for image mode
4. **Interface**: `argparse` CLI — `--image PATH`, `--video PATH`, `--camera [INDEX]`
5. **Checkpoint loading**: reads `num_layers` and `hidden_size` from the checkpoint dict (already stored by `Trainer.save_checkpoint`)

---

### Face Detection + Inference Pipeline (per frame)

```
OpenCV frame (BGR)
    → cvtColor(BGR→GRAY)             for Haar detection
    → detectMultiScale()             → list of (x, y, w, h) bounding boxes
    → for each face:
        crop with 10% padding        prevent tight crops cutting expression edges
        → resize to 48×48            cv2.resize()
        → FERInference.predict()     CLAHE + normalize + model forward pass
        → draw overlay on frame      bounding box, label, confidence bar
```

If no face detected → overlay "No face detected", continue loop.

---

### Overlay (drawn with `cv2.putText` / `cv2.rectangle`)

- Green bounding box around detected face
- Emotion label + confidence % above box (e.g. `Happy 87.3%`)
- Small horizontal probability bar (6 bars, one per class) in top-left corner
- FPS counter in camera/video modes

---

### `fer_deploy.py` Structure

```python
# Imports: argparse, os, time, cv2, numpy, torch, torch.nn, PIL, torchvision

# Model class definitions (verbatim from notebook):
#   FERPreprocessor, CustomNetLayer, CustomSigmoid, CustomBlock, FERNet

# FERInference — same as Section 7, with resize(48,48) added before transform

class FaceDetector:
    # loads haarcascade_frontalface_default.xml from cv2.data.haarcascades
    def detect(self, gray_frame) -> list: ...   # returns [(x,y,w,h), ...]

def draw_overlay(frame, faces, results, fps=None): ...
    # bounding boxes, labels, confidence bars, FPS

def run_image(args, inference, detector): ...
    # load → detect → predict → save annotated → matplotlib display

def run_video(args, inference, detector): ...
    # VideoCapture(path) → frame loop → detect → predict → VideoWriter → imshow
    # saves to results/output_video.mp4 | 'q' to quit early

def run_camera(args, inference, detector): ...
    # VideoCapture(index) → real-time loop → detect → predict → imshow
    # 'q' to quit

def main():
    # argparse: --image, --video, --camera, --model
    # load checkpoint → read num_layers/hidden_size → build FERInference
    # dispatch to run_image / run_video / run_camera
```

---

### Usage (from `AI_CW2/` directory)

**Standard (`fer_deploy.py` — V2 model, V2 preprocessing):**
```bash
python fer_deploy.py --image Data/Testing/Happy/Happy.jpg
python fer_deploy.py --video path/to/video.mp4
python fer_deploy.py --camera          # default camera 0
python fer_deploy.py --camera 1        # specific camera index
```

**Enhanced (`fer_deploy_V2.py` — face alignment, square crop, temporal smoothing):**
```bash
python fer_deploy_V2.py --image Data/Testing/Happy/Happy.jpg
python fer_deploy_V2.py --camera
python fer_deploy_V2.py --camera --smooth 15   # wider smoothing window
```

---

### Critical Notes

- **Resize required**: cropped face ROIs are arbitrary size; `pil_img.resize((48, 48), Image.LANCZOS)` applied inside `predict()` before the transform
- **No extra installs**: uses only `torch`, `torchvision`, `opencv-python`, `Pillow`, `matplotlib` — all already installed
- **Run from AI_CW2/**: default `--model` path is `checkpoints/best_model_v2.pth` (relative)
- **V2 preprocessing**: both deploy scripts use V2 pipeline — GaussianBlur(3×3) → CLAHE(clip=3.0, tile=6×6) → ToTensor [0,1] (no Normalize)

---

### Deployment Verification

1. `python fer_deploy.py --image Data/Testing/Happy/Happy.jpg` → annotated image saved to `results/`, matplotlib window opens
2. `python fer_deploy.py --camera` → webcam opens, face box + emotion label overlaid live, 'q' closes
3. `python fer_deploy.py --video <any_mp4>` → processes frames, saves `results/output_video.mp4`
4. `python fer_deploy_V2.py --camera` → same as above with face alignment + temporal smoothing
