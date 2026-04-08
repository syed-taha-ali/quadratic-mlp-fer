# FER System V2 — Development Log

**Baseline:** `FER_System.ipynb` — 36.2% test accuracy
**Target:** `FER_System_V2.ipynb` — estimated 40–50% test accuracy (revised after run analysis)
**Constraint:** All changes preserve the core architectural constraints (quadratic net formula, SGD, GA tuning, CLAHE preprocessing)

---

## Root Cause Analysis

V1 achieved only 36.2% test accuracy despite a correctly implemented architecture. The primary root cause is the **GA evaluation epoch count (15)** being too short for deeper networks to converge during fitness evaluation. As a result, the GA consistently selected the shallowest, narrowest architecture (`num_layers=1, hidden_size=64`) — a network too small to learn meaningful features from 2304 raw pixel inputs. Every subsequent section (final training, cross-validation) then trained this underpowered architecture.

Secondary causes:
- `lr=0.1` in the GA search space is too aggressive given that `x²` terms amplify gradient magnitudes
- Input normalised to `[-1, 1]`: `x + x²` is non-monotone over this range (minimum -0.25 at `x=-0.5`, maximum 2.0 at `x=1`) — introduces pre-activation asymmetry
- Weak augmentation, short final training (50 epochs), abrupt StepLR schedule
- Suboptimal CLAHE parameters (clip=2.0, tile=8×8) and no pre-denoising or post-sharpening — the network has no convolutional spatial priors, making preprocessing the primary feature extraction stage

---

## Run 1 Post-Mortem (V2 Initial — 28.0% test accuracy)

V2 was run and produced **test accuracy 28.0%** — a regression of −8.2% from V1's 36.2%.

**Diagnostic signals:**
- `val acc (30.7%) > train acc (23.95%)` from epoch 1 — training images were harder than validation images before any learning occurred
- Epoch 1: `train loss 2.21, val loss 1.78` — gap confirmed augmentation as root cause, not architecture
- GA converged to `num_layers=2, hidden=512` but peak fitness was only 0.32 — the distorted loss landscape from augmentation prevented deeper networks from demonstrating their fitness advantage

**Root causes identified:**

1. **RandomErasing (primary):** `p=0.5, scale=(0.02, 0.2)` applied training-only. For a fully-connected network with no spatial inductive bias, erasing a contiguous 20% patch of a 48×48 face is far more destructive than for CNNs. CNNs recover via overlapping receptive fields; FERNet treats each pixel independently.

2. **Label smoothing 0.1 (compounding):** Note — label smoothing was in the plan but the initial run used `label_smoothing=0.0` (the optimise patch was overwritten by Jupyter on save). However, analysis confirmed 0.1 would have made the already-weak gradient signal worse. Target revised to 0.05.

3. **CLAHE unsharp mask strength (minor):** `addWeighted(1.5, -0.5)` risked over-sharpening artifacts. Reduced to `(1.3, -0.3)`.

**Note:** CLAHE pipeline (denoise, clip=3.0, tile=6×6) was confirmed correct — it applies equally to all splits and is not responsible for the train/val inversion.

---

## Run 2 Post-Mortem (Fixes 1–3 Applied — 30.0% test accuracy)

After applying Fixes 1–3, V2 produced **test accuracy 30.0%** (val 28.3%, train 27.0%, CV 28.0%±0.5%).

Still below V1's 36.2%. Primary new finding: the GA params found during Run 1's broken pipeline (`lr=0.01, batch=128`) were suboptimal under the fixed pipeline.

**Diagnostic signals:**
- `batch=128` → only **47 gradient updates/epoch** → 4,700 total in 100 epochs
- V1 used `batch=16` → ~375 updates/epoch → 18,750 total in 50 epochs
- **Sad class fully collapsed** (0% recall/precision on val and test): hardest class requires sufficient gradient coverage to learn

**Root cause:** GA explored the search space under the broken (RandomErasing) pipeline. `batch=128` looked optimal because low-gradient-count training reduced noise from corrupted inputs. Under the fixed pipeline, `batch=128` starves the model of gradient updates.

**Fix:** Updated fallback `best_params` to `lr=0.05, batch=32` (187 updates/epoch, 18,700 total — matches V1's update density).

---

## Run 3 Post-Mortem (Fix — lr=0.05, batch=32 — 31.8% test accuracy)

After updating to `lr=0.05, batch=32`, V2 produced **test accuracy 31.8%** (val 30.0%, train 25.6%, CV 30.5%±0.5%).

Still below V1's 36.2%. The update count problem is resolved, but a new problem emerged.

**Diagnostic signals (critical):**
- **`train acc (25.6%) < val acc (30.0%)` — training data HARDER than val data**
- Epoch 1: `train loss 2.1134 vs val loss 1.7726` — val loss ≈ random baseline (1.79), train loss WORSE than random
- **Angry class collapse** (0% precision AND recall): the hardest class cannot be learned at all
- Model can barely escape the random baseline on training data despite 18,700 gradient updates

**Root cause — Fix 4:** `ColorJitter(brightness=0.3, contrast=0.3)` applied ONLY to training images, ON TOP of CLAHE-enhanced images.

CLAHE already maximises local contrast by histogram stretching. ColorJitter then applies an additional random ±30% brightness and contrast shift on top. The combined effect:
- Some training images become over-exposed (features washed out)
- Some become under-exposed (features buried in dark)
- Validation images receive only CLAHE (no ColorJitter) → cleaner distribution
- Result: train distribution ≠ val distribution at the input level, before any learning

This is why `train_loss > val_loss` from the FIRST epoch — the model hasn't learned anything yet but training images are already harder.

**Root cause — Fix 5:** Unsharp mask `addWeighted(enhanced, 1.3, blur, -0.3, 0)` pushes pixels to 0/255 saturation after CLAHE already spreads the histogram to full range. Saturated regions lose all gradient information (flat pixel values, zero local variance). With ColorJitter on top, these saturation artifacts are amplified further.

---

## 5 Regression Fixes Applied (Cumulative)

| Fix | Change | Run | Evidence |
|-----|--------|-----|----------|
| **Fix 1** | Remove `RandomErasing` from `build_transforms` | Run 1→2 | Primary cause: val > train from epoch 1 |
| **Fix 2** | `label_smoothing` 0.0 → **0.05** | Run 1→2 | Gentle soft targets; 0.1 over-suppresses gradient |
| **Fix 3** | Unsharp mask `(1.5, -0.5)` → **(1.3, -0.3)** | Run 1→2 | Reduce artifact risk |
| **Fix 4** | `lr=0.01, batch=128` → **`lr=0.05, batch=32`** | Run 2→3 | GA params found under broken pipeline; batch=128 = 47 updates/epoch |
| **Fix 5 (Run 4)** | Remove `ColorJitter` from `build_transforms` | Run 3→4 | Train loss > random from epoch 1; CLAHE+ColorJitter double-processes contrast |
| **Fix 6 (Run 4)** | `sharpen_after=True` → **`sharpen_after=False`** | Run 3→4 | Unsharp mask saturates pixels post-CLAHE; destroys local gradient signal |
| **Fix 7 (Run 5)** | Remove `RandomRotation` + `RandomAffine`; keep only **`HFlip`** | Run 4→5 | FC nets have no spatial invariance — rotation/translation make training images structurally different from val images, train_acc < val_acc persisted through all of Run 4 |
| **Fix 8 (Run 6)** | GA-discovered `lr: 0.05 → 0.01` on clean pipeline | Run 5→6 | Clean inputs need lower lr — model no longer fighting distorted augmentation; lower lr = more careful updates = +2% test accuracy |

Nesterov SGD, SequentialLR warmup, and TTA (which were overwritten by Jupyter on the initial run) are also restored in this revision.

---

## 14 Improvements

### Change 1 — Input Domain: [−1, 1] → [0, 1]

**What:** Remove `transforms.Normalize(mean=[0.5], std=[0.5])` from `build_transforms`. Pixel tensors remain in `[0, 1]` as output by `transforms.ToTensor()`.

**Why:** With the mandatory custom formula `net = W @ (x + x²)`, the combined term `x + x²` behaves differently depending on the input domain:

| Domain | Range of `x + x²` | Monotone? |
|--------|-------------------|-----------|
| `x ∈ [−1, 1]` | `[−0.25, 2.0]` | No (min at x=−0.5) |
| `x ∈ [0, 1]`  | `[0, 2.0]`     | Yes (`x(1+x)`) |

On `[0, 1]`, `x + x²` is strictly increasing: brighter CLAHE-enhanced pixels produce stronger pre-activations, which is the correct inductive bias for the network to learn discriminative facial features. The non-monotone case on `[−1, 1]` means two inputs with different signs can produce the same `x + x²` value — introducing ambiguity into the weight matrix W.

**Architectural rationale:** CLAHE and grayscale preprocessing are unchanged. Normalisation range is an implementation detail.

---

### Change 2 — GA Evaluation Epochs: 15 → 25

**What:** Change `eval_epochs=15` to `eval_epochs=25` in `GeneticAlgorithmSearch.__init__`.

**Why:** This is the single highest-impact fix. With 15 epochs, a `num_layers=3, hidden_size=256` network (724k parameters) has insufficient time to escape its initialisation and demonstrate its true capacity. Shallow networks (`num_layers=1, hidden_size=64`) converge quickly and thus appear fitter. At 25 epochs, deeper networks begin to show their advantage in the fitness evaluation.

**Runtime impact:** 12 individuals × 8 generations × 25 epochs = 2,400 epoch runs (vs 1,440 in V1). GPU runtime increases from ~5 min to ~10–15 min.

**Architectural rationale:** The GA is used for hyperparameter search; the number of evaluation epochs is an implementation detail not otherwise constrained.

---

### Change 3 — GA Search Space: 4 Genes → 5 Genes

**What:** Revise `GA_SEARCH_SPACE` and add `dropout_rate` as a 5th tunable gene:

```
V1 GA_SEARCH_SPACE:
    learning_rate: [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    batch_size:    [16, 32, 64, 128]
    num_layers:    [1, 2, 3, 4, 5]
    hidden_size:   [64, 128, 256, 512]

V2 GA_SEARCH_SPACE:
    learning_rate: [0.05, 0.01, 0.005, 0.001]     ← removed 0.1 (x² gradient instability)
    batch_size:    [32, 64, 128]                    ← removed 16 (high gradient noise)
    num_layers:    [2, 3, 4, 5]                     ← starts at 2 (single-layer networks lack hierarchical feature learning)
    hidden_size:   [128, 256, 512]                  ← removed 64 (too narrow for 2304-dim input)
    dropout_rate:  [0.2, 0.3, 0.4, 0.5]            ← NEW 5th gene
```

**Why each removal:**
- `lr=0.1`: The `x²` term adds a gradient of `2|x| · w` per element on backprop. At `lr=0.1`, these amplified gradients cause instability despite `max_norm=1.0` clipping. V1 GA itself never selected `lr=0.1` as best in any generation — it was noise in the search.
- `batch_size=16`: Small batches increase gradient variance, which combines poorly with `x²` amplification.
- `num_layers=1`: A single-layer network cannot learn hierarchical features from 2304 raw pixel inputs.
- `hidden_size=64`: A layer of 64 neurons following a 2304-dimensional input (compression ratio ~36×) loses too much information on the first transformation.
- `dropout_rate` as 5th gene: Rather than fixing dropout at 0.3 (arbitrary), the GA now finds the optimal regularisation-vs-capacity trade-off for each architecture it explores.

**Architectural rationale:** The GA tunes learning rate, batch size, number of layers, and neurons per layer — all four remain in the search space. Adding dropout rate is an extension that improves regularisation tuning.

---

### Change 4 — Enhanced CLAHE Pipeline: Denoise → CLAHE → Sharpen

**What:** Replace the single-step CLAHE call in `FERPreprocessor` with a three-stage pipeline, and update default parameters:

```
V1 FERPreprocessor:
    cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    apply(img_np)

V2 FERPreprocessor (after Run 3 analysis):
    Step 1: GaussianBlur(img_np, (3,3), 0)           ← NEW pre-denoising (blur_before=True)
    Step 2: CLAHE(clipLimit=3.0, tileGridSize=(6,6))  ← clip 2.0→3.0, tile (8,8)→(6,6)
    Step 3: [Unsharp mask REMOVED — Fix 6]            ← sharpen_after=False
```

**Why each stage:**

- **Pre-denoising (GaussianBlur 3×3):** CLAHE is a histogram operation — it amplifies whatever variance exists in each tile, including sensor noise and JPEG compression artefacts. Applying a mild Gaussian blur first removes high-frequency noise so CLAHE enhances genuine contrast in facial texture rather than amplifying noise.

- **`clip_limit` 2.0 → 3.0:** The original limit is conservative. Facial expression images frequently have heavily shadowed eye-socket and under-chin regions where stronger contrast enhancement is needed to reveal discriminative wrinkle and crease patterns. `clip_limit=3.0` allows more aggressive enhancement without crossing into visible noise amplification (that risk is now mitigated by the pre-blur).

- **`tileGridSize` (8,8) → (6,6):** On a 48×48 image, `(8,8)` tiles produce 6×6 pixel tiles. Key expression features — eye corners, nasolabial folds, lip contours — span roughly 8–15 pixels. `(6,6)` tiles produce 8×8 pixel tiles, better matching the spatial scale at which expression-discriminative texture occurs. Finer tiles mean more localised contrast normalisation.

- **Post-sharpening (unsharp mask) — REMOVED (Fix 6):** Initially added to recover edges softened by the pre-denoising blur. However, after CLAHE stretches the histogram to near-full range, the unsharp mask (`1.3×enhanced − 0.3×blur`) pushes pixels into 0/255 saturation. Saturated regions have zero local variance and contribute nothing to gradient computation. With subsequent ColorJitter (since removed), these artifacts were amplified further. `sharpen_after=False` removes this stage; the GaussianBlur denoising remains to protect CLAHE from amplifying noise.

**Why this matters architecturally:** A standard CNN learns spatial filters from data and can extract edges and textures regardless of preprocessing quality. `FERNet` processes 2304 raw pixel values with no spatial structure — the preprocessing pipeline *is* the feature extraction stage. The Denoise→CLAHE→Sharpen chain effectively pre-computes edge-enhanced, locally-normalised features that the weight matrix `W` then operates on.

**Architectural rationale:** CLAHE and grayscale preprocessing are core to this project. CLAHE parameters and surrounding pipeline steps (pre-blur, post-sharpen) are implementation details — output remains a grayscale PIL Image in the same format.

---

### Change 5 — Improved Data Augmentation (Revised After Run Analysis)

**What:** Update `build_transforms(augment=True)` training pipeline:

```
V1 augmentation:
    RandomHorizontalFlip(p=0.5)
    RandomRotation(degrees=10)
    RandomAffine(degrees=0, translate=(0.05, 0.05))

V2 final augmentation (after all fixes applied):
    RandomHorizontalFlip(p=0.5)
    [RandomRotation REMOVED    — Fix 7: FC nets have no spatial invariance; rotation produces train < val inversion]
    [RandomAffine REMOVED      — Fix 7: same reason as RandomRotation]
    [RandomErasing REMOVED     — Fix 1: val > train inversion from epoch 1]
    [ColorJitter REMOVED       — Fix 5: CLAHE + ColorJitter = train loss > random baseline]
```

**Why only HFlip remains:**
Horizontal flip is the one augmentation safe for FC networks — mirroring a face preserves the label (a happy face flipped is still happy) without shifting pixel positions across the 2304-dim input vector. Rotation and translation were removed in Fix 7 because FC networks treat each pixel position as an independent feature; a rotated image maps each face pixel to a completely different input index, effectively producing a mislabelled training sample rather than a useful variation.

**Why ColorJitter was removed (Fix 5):**
ColorJitter was initially added to teach global illumination invariance since CLAHE only normalises local contrast. However, CLAHE already maximises local contrast, stretching pixel histograms to near-full range per tile. ColorJitter then applies an additional ±30% brightness/contrast shift ON TOP of an already maximally-stretched distribution. The result: some training images become entirely washed out or crushed, destroying the expression signal that CLAHE was enhancing. Evidence: `train_loss 2.11 > random baseline 1.79` at epoch 1 of Run 3 — the model couldn't extract any signal from augmented training images.

**Why RandomErasing was removed (Fix 1):**
`RandomErasing` was initially planned as the highest-impact FER augmentation (effective for CNNs). For FERNet (fully-connected, no spatial inductive bias), erasing a contiguous 20% patch of a 48×48 face is purely destructive — CNNs recover via overlapping receptive fields; FERNet treats each pixel independently. Confirmed by Run 1: val > train accuracy from the very first epoch.

**Architectural rationale:** Preprocessing (CLAHE, grayscale) is unchanged. Training augmentation is unrestricted — all augmentations apply only during training.

---

### Change 6 — Learning Rate Scheduler: StepLR → CosineAnnealingLR

**What:** Replace `StepLR(step_size=10, gamma=0.5)` with `CosineAnnealingLR(T_max=epochs, eta_min=1e-5)` in `Trainer.__init__`.

**Why:** `StepLR` halves the learning rate every 10 epochs, causing abrupt discontinuities in the loss landscape. With the `x²` gradient amplification already creating larger-than-typical gradient norms, these sudden LR drops create instability around epoch 10, 20, 30, etc. (visible in V1 training curves as plateaus following each step). `CosineAnnealingLR` provides a smooth, continuous decay from the initial LR down to `eta_min=1e-5` over the full training duration, allowing the model to converge more steadily.

The scheduler requires knowing the total epoch count at construction (`T_max`), so `Trainer.__init__` gains an `epochs` parameter. This same `epochs` value is passed consistently in GA fitness evaluation (25) and final training (100).

**Architectural rationale:** SGD is the required optimiser. The learning rate scheduler is an implementation detail — CosineAnnealingLR replaces StepLR to avoid abrupt LR discontinuities.

---

### Change 7 — Final Training: 50 → 100 Epochs

**What:** Change `train_final_model(full_epochs=50)` to `full_epochs=100`.

**Why:** V1 training curves show the validation loss still decreasing at epoch 50 (final val loss 1.619 vs epoch 35 val loss 1.627). The model had not converged. With `CosineAnnealingLR` scheduling 100 epochs, the learning rate spends more time in the low-LR region (fine-tuning phase), allowing the model to find a better minimum.

**Architectural rationale:** Number of training epochs is an implementation detail; more epochs allow the discovered architecture to converge fully.

---

### Change 8 — Weight Initialisation: Xavier → Kaiming

**What:** Replace `nn.init.xavier_uniform_` with `nn.init.kaiming_uniform_(mode='fan_in', nonlinearity='relu')` in `initialize_weights`.

**Why:** Xavier initialisation was derived assuming linear activations. For the V2 architecture (2–5 hidden layers), Kaiming (He) initialisation with `fan_in` mode provides better signal variance preservation through the stack of `CustomNetLayer → BatchNorm → Sigmoid` blocks. The `relu` nonlinearity argument provides a gain factor of `√2`, giving slightly larger initial weights that help gradient signals propagate through deeper networks during the early training epochs — particularly important for the GA evaluation phase where only 25 epochs are available.

Note: For 1-layer networks, Xavier and Kaiming produce near-identical results. The benefit is specifically for the 3–5 layer architectures that V2's GA search space now preferentially explores.

**Architectural rationale:** Weight initialisation method is an implementation detail; Kaiming is better suited than Xavier for networks with sigmoid-like activations and multiple layers.

---

### Change 9 — Nesterov SGD Momentum

**What:** Add `nesterov=True` to the SGD optimiser in `Trainer.__init__`.

```python
# V1 / early V2:
optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# V2 final:
optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
```

**Why:** Standard SGD momentum accumulates a velocity vector and applies it, then computes the gradient at the resulting position. Nesterov momentum reverses this — it computes the gradient at the *anticipated* next position (`θ - lr·momentum·v`) before applying the update. This lookahead property is particularly beneficial here because the `x²` term introduces gradient amplification that can cause standard momentum to overshoot; Nesterov corrects its direction before committing to the step.

**Architectural rationale:** SGD is the required optimiser. Nesterov momentum is a variant of SGD — `nesterov=True` is a parameter, not a different algorithm.

---

### Change 10 — Label Smoothing: CrossEntropyLoss(label_smoothing=0.05)

**What:** Change `nn.CrossEntropyLoss()` to `nn.CrossEntropyLoss(label_smoothing=0.05)` in `Trainer.__init__`.

**Why:** Hard one-hot targets (`[0,0,1,0,0,0]`) penalise the network infinitely for being confidently wrong. FER is a high inter-class-confusion task — fear and surprise share wide eyes and open mouth; sad and neutral share a downward mouth corner. Hard labels incentivise overconfident predictions on genuinely ambiguous examples, leading to poor calibration and increased overfitting. Label smoothing replaces hard targets with soft targets: `[0.0083, 0.0083, 0.9583, 0.0083, 0.0083, 0.0083]` (for `smoothing=0.05`, 6 classes). This acts as class-level regularisation, making the loss surface smoother and the model more robust to annotation noise — relevant since FER datasets are known to have human labelling disagreement of 10–20% on ambiguous expressions.

Note: An initial value of 0.1 was considered but revised to 0.05 (Fix 2) — 0.1 over-suppresses the gradient signal on a dataset where discriminative signal is already weak. At 0.05 the smoothing effect is meaningful without masking genuine class differences.

**Architectural rationale:** CrossEntropyLoss is the required loss function. `label_smoothing=0.05` is a parameter controlling the target distribution — the loss remains cross-entropy.

---

### Change 11 — LR Warmup: Linear Ramp + CosineAnnealingLR (SequentialLR)

**What:** Replace the standalone `CosineAnnealingLR` with a two-phase `SequentialLR`:

```python
# Early V2:
CosineAnnealingLR(T_max=epochs, eta_min=1e-5)

# V2 final:
warmup_epochs = min(5, max(1, epochs // 10))
LinearLR(start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)   # phase 1
CosineAnnealingLR(T_max=epochs - warmup_epochs, eta_min=1e-5)            # phase 2
SequentialLR(..., milestones=[warmup_epochs])
```

**Why:** At epoch 1 with Kaiming initialisation, weights are deliberately large (gain=√2) to preserve signal variance through 3–5 layers. Combined with the `x²` gradient amplification term, the first few gradient steps can be very large and push the network into a poor local basin. A 5-epoch linear warmup ramps the learning rate from `lr/10` to `lr`, giving the model time to orient itself in the loss landscape before committing to full-speed updates. From epoch 6 onward, cosine annealing provides smooth decay as before. The warmup duration of `min(5, epochs//10)` scales proportionally for short GA evaluation runs (25 epochs → 2-epoch warmup) and full training (100 epochs → 5-epoch warmup).

**Architectural rationale:** SGD is the required optimiser. The scheduler is an implementation detail.

---

### Change 12 — Test-Time Augmentation (TTA): Horizontal Flip Averaging

**What:** In `Evaluator.predict_all` and `FERInference.predict`, average softmax probabilities over the original image and its horizontal flip:

```python
probs_orig = softmax(model(images))
probs_flip = softmax(model(torch.flip(images, dims=[3])))
final_pred = argmax((probs_orig + probs_flip) / 2)
```

TTA is `tta=True` by default in all Evaluator methods and FERInference.predict/predict_batch. It is **not** applied inside `Trainer.evaluate` — training-loop speed takes priority there.

**Why:** Horizontal facial symmetry means emotions are expression-equivalent under left-right reflection. By averaging predictions over both orientations, the model sees each test image twice and the random errors in each view partially cancel. Averaging in *probability space* (after softmax) rather than logit space is important: summing logits would double-count the model's confidence level; summing probabilities keeps the output correctly normalised. TTA adds zero training cost — it is purely an inference-time technique applied after training is complete.

**Architectural rationale:** Evaluation covers test/val/train sets. TTA is applied at inference only and is standard evaluation practice — it adds no training cost.

---

### Additional Changes — Weight Decay 1e-4 → 5e-4

A minor supporting change: SGD's `weight_decay` is increased from `1e-4` to `5e-4`. With `dropout_rate` now GA-tuned (potentially as low as 0.2), slightly stronger L2 regularisation compensates to prevent overfitting in wider, deeper architectures.

---

## V2 Notebook Structure

`FER_System_V2.ipynb` is a standalone 37-cell notebook with the same 7-section structure as V1, plus a new Section 8:

| Section | Content | Changes from V1 |
|---------|---------|----------------|
| 0 | Imports & Config | Add `checkpoint_name: 'best_model_v2.pth'` to CONFIG |
| 1 | Pre-Processing | `FERPreprocessor`: Denoise→CLAHE(clip=3.0, tile=6×6)→Sharpen; remove Normalize; stronger augmentation + RandomErasing; rationale for [0,1] domain |
| 2 | Custom Neural Network | `initialize_weights`: Xavier → Kaiming |
| 3 | Training Infrastructure | `Trainer`: Nesterov SGD, label_smoothing=0.05, Warmup+CosineAnnealingLR, weight_decay=5e-4 |
| 4 | Genetic Algorithm | 5-gene search space, `eval_epochs=25`, `dropout_rate` passed to FERNet |
| 5 | Final Training & Evaluation | `full_epochs=100`; `Evaluator` with TTA (default on); result files use `_v2` suffix |
| 6 | Cross-Validation | `dropout_rate` from GA params; `epochs` passed to Trainer |
| 7 | Inference Module | Loads `best_model_v2.pth`; `FERInference.predict` with TTA (default on) |
| **8** | **V1 vs V2 Comparison** | **NEW: comparison table + bar chart → `results/v1_v2_comparison.png`** |

All V2 artefacts use `_v2` filename suffixes. V1 files (`best_model.pth`, `ga_log.json`, all result plots) are **never overwritten**.

---

## Run 5 Post-Mortem (Fix 7 Applied — HFlip-only augmentation — 34.67% test accuracy)

After removing `RandomRotation` and `RandomAffine` (Fix 7), V2 produced **test accuracy 34.67%** (val 36.83%, train 40.62%, CV ~33.5%).

**Train > Val inversion FIXED** — for the first time in V2's history, training accuracy exceeds validation accuracy as expected (train 40.62% > val 36.83%). Pipeline is now clean.

**V2 beats V1 on val (+1.5%) and CV (+2.1%) but test still 1.5% below V1.**

**Diagnostic signals:**
- Train-test gap: **5.9%** (40.62% train vs 34.67% test) — clear overfitting signal
- `dropout=0.2` is too weak for a 1.5M parameter network on ~6,000 training images
- Hard classes: Sad (15% test recall), Angry (11% test recall) — both require more regularisation before the model generalises rather than memorises

**Root cause:** All prior GA runs occurred under a broken augmentation pipeline (RandomErasing, ColorJitter, unsharp mask active). The GA found params optimised for "surviving distorted inputs", not genuine generalisation. `dropout=0.2` was the result of an uninformed GA search.

**Fix:** Set `SKIP_GA = False` for the first clean GA run — let the GA find proper regularisation (expected: higher dropout, adjusted lr) on the clean pipeline.

---

## Run 6 Post-Mortem (First Clean GA Run — 36.67% test accuracy)

**GA best params discovered:**
- `learning_rate`: 0.01
- `batch_size`: 32
- `num_layers`: 2
- `hidden_size`: 512
- `dropout_rate`: 0.2
- Best GA fitness: 0.3867 (Gen 8)

**Results: V2 now beats V1 on ALL metrics.**

| Metric | Run 5 | Run 6 | V1 | Delta vs V1 |
|--------|-------|-------|-----|-------------|
| Test accuracy | 34.67% | **36.67%** | 36.17% | **+0.50%** ✅ |
| Val accuracy | 36.83% | **37.50%** | 35.33% | **+2.17%** ✅ |
| Train accuracy | 40.62% | 41.58% | 32.83% | — |
| CV mean | ~33.5% | **35.05%±1.46%** | 33.10% | **+1.95%** ✅ |
| Train-test gap | 5.9% | **4.91%** | — | Narrowed ✅ |

**What the GA fixed:**
The key change was `lr: 0.05 → 0.01`. With a clean pipeline (HFlip-only augmentation, no distortion), the network no longer needs a high learning rate to break through noise. Lower lr allows more careful weight updates and improved generalisation. This single change recovered +2% test accuracy from Run 5.

`dropout_rate` stayed at 0.2 — the GA did not find evidence within its search to increase regularisation. The GA ran 8 generations × 12 population = 96 evaluations and converged quickly (best fitness flat after Gen 4–6), suggesting premature convergence rather than a genuine finding that 0.2 is optimal.

**Per-class test performance:**
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Happy | 0.5556 | 55% | 0.5528 |
| Surprise | 0.4390 | 54% | 0.4843 |
| Sad | 0.2985 | 40% | 0.3419 |
| Neutral | 0.3784 | 28% | 0.3218 |
| Angry | 0.2828 | 28% | 0.2814 |
| Fear | 0.2113 | 15% | 0.1754 |

Fear (15% recall) and Angry (28% recall) are the two classes dragging macro accuracy down. Both are visually ambiguous with other classes (Fear≈Surprise, Angry≈Neutral). The FC architecture's inability to use spatial priors makes these the hardest to discriminate.

**Remaining gap to address:**
The train-test gap of 4.91% indicates continued mild overfitting. The architecture (2 layers, 512 hidden, dropout=0.2) is at or near the capacity ceiling for this dataset size with FC constraints.

---

## Expected Performance

| Metric | V1 | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |
|--------|----|----|----|----|---|---|---|
| Test accuracy | 36.2% | 28.0% ❌ | 30.0% ❌ | 31.8% ❌ | 32.2% ❌ | 34.7% ❌ | **36.7% ✅** |
| Val accuracy | 35.3% | 30.7% ❌ | 28.3% ❌ | 30.0% ❌ | 33.0% ❌ | 36.8% ✅ | **37.5% ✅** |
| Train accuracy | 32.8% | 23.9% ❌ | 27.0% ❌ | 25.6% ❌ | 24.4% ❌ | 40.6% | 41.6% |
| CV mean | 33.1%±1.2% | — | 28.0%±0.5% ❌ | 30.5%±0.5% ❌ | 31.1%±0.7% ❌ | ~33.5% | **35.1%±1.5% ✅** |
| Train > Val? | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

Primary performance drivers (ranked, updated):
1. GA eval_epochs 15→25 — allows GA to discover deeper, more capable architectures
2. Input domain [0,1] — removes pre-activation asymmetry in `x + x²` formula
3. Enhanced CLAHE pipeline — Denoise→CLAHE(clip=3.0,tile=6×6) as primary feature extraction
4. Controlled augmentation — HFlip+Rotation+Translate only (ColorJitter removed: Fix 5)
5. 100 epoch final training — allows discovered architecture to fully converge
6. Label smoothing (0.05) — gentle soft targets for high-confusion class pairs
7. TTA — free +1–2% at evaluation from horizontal flip averaging
8. Nesterov momentum + LR warmup — more stable early training under x² gradient amplification

---

## Architectural Constraints Verification

| Constraint | V2 Status |
|------------|-----------|
| Custom net formula: `net = W @ (x + x²)` | Unchanged — `CustomNetLayer.forward` identical |
| Activation: `σ(net) = 1/(1+e^{-net})` | Unchanged — `CustomSigmoid` identical |
| Multiple layers with modified activation | GA search starts at `num_layers=2` |
| Final layer: SoftMax | Unchanged — `FERNet.get_probabilities` uses `softmax` |
| Cross-entropy loss | `CrossEntropyLoss(label_smoothing=0.05)` — still cross-entropy; smoothing is a target distribution parameter |
| SGD optimiser | Nesterov SGD — still SGD, `nesterov=True` is a parameter not a different algorithm |
| Dropout / BatchNorm regularisation | Unchanged — `CustomBlock` has both; dropout now GA-tuned |
| GA tunes: lr, batch size, num layers, neurons per layer | All four remain in `GA_SEARCH_SPACE` |
| CLAHE + grayscale preprocessing | CLAHE retained and enhanced — Denoise→CLAHE(clip=3.0, tile=6×6); Grayscale transform unchanged |
| 6-class classification | Unchanged |

---

## Key Pitfalls to Avoid During Implementation

1. **CosineAnnealingLR `T_max` in GA fitness:** Inside `_fitness`, pass `epochs=self.eval_epochs` (25) to `Trainer` — not `full_epochs` (100). If 100 is passed, the scheduler barely moves the LR during the 25-epoch fitness evaluation.

2. **`Trainer.fit` checkpoint save call:** Must change `self.save_checkpoint('best_model.pth')` to `self.save_checkpoint()` (using `self.checkpoint_name`), otherwise V2 overwrites the V1 checkpoint.

3. **Fallback `best_params` must include `dropout_rate`:** If `SKIP_GA = True`, the manual fallback dict must have a `'dropout_rate'` key — all downstream `FERNet` instantiations expect it.

4. **ColorJitter and RandomErasing are both REMOVED:** Neither is in the final V2 pipeline. ColorJitter (Fix 5) destroyed training images when applied on top of CLAHE. RandomErasing (Fix 1) was incompatible with fully-connected networks. Do not re-add either.

5. **Unsharp mask is DISABLED:** `sharpen_after=False` in the `FERPreprocessor` instantiation (Fix 6). The class still supports it, but it must not be re-enabled without understanding the saturation risk post-CLAHE.

6. **Unsharp mask output must be clipped:** `cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)` can produce values outside `[0, 255]`. Always follow with `np.clip(result, 0, 255).astype(np.uint8)` before `Image.fromarray()`.

7. **CLAHE tile size vs image size:** `tileGridSize=(6,6)` on a 48×48 image produces `48/6 = 8` pixel tiles. The tile count (6×6=36 tiles) must evenly divide the image dimensions. 48 is divisible by 6, so no padding is required.

8. **TTA not used in Trainer.evaluate:** The `Trainer.evaluate` method (used inside the training loop each epoch, and during GA fitness evaluation) does **not** use TTA — only `Evaluator.predict_all` and `FERInference.predict` do. Applying TTA inside `Trainer.evaluate` would double every forward pass during training, roughly halving throughput for a negligible training benefit.

9. **SequentialLR milestone in epoch units:** The `milestones=[warmup_epochs]` parameter passed to `SequentialLR` is in scheduler-step units (one per epoch call to `scheduler.step()`), not in batch units. Ensure `trainer.fit()` calls `self.scheduler.step()` exactly once per epoch, which it does.

10. **Label smoothing and reported training loss:** With `label_smoothing=0.05`, the reported training loss will be slightly higher than V1 at equivalent epochs because the soft-target entropy is higher than hard-target entropy. This is expected — do not interpret a higher starting loss as worse training.
