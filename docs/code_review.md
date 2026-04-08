# Code Review — Quadratic MLP Facial Expression Recognition

**Scope:** `FER_System.ipynb`, `FER_System_V2.ipynb`, `fer_deploy.py`, `fer_deploy_V2.py`
**Severity levels:** 🔴 Critical (correctness bug) · 🟡 Major (design/robustness) · 🔵 Minor (style/nit)

---

## Summary

The codebase is well-structured, clearly commented, and demonstrates strong understanding of the PyTorch training loop. Class boundaries are logical, type hints are present throughout, and the V1→V2 diff is clearly documented. The custom `net = W @ (x + x²)` implementation is correct and the unit test in Cell 11 is a professional touch.

**Issue counts:**

| Severity | V1 Notebook | V2 Notebook | fer_deploy.py | fer_deploy_V2.py |
|----------|-------------|-------------|---------------|------------------|
| 🔴 Critical | 1 | 0 | 0 | 0 |
| 🟡 Major | 3 | 3 | 3 | 1 |
| 🔵 Minor | 2 | 3 | 2 | 1 |

---

## FER_System.ipynb (V1)

### 🔴 CR-V1-01 — GA fitness evaluations overwrite `best_model.pth`

**Location:** `Trainer.fit()`, Cell 13, line:
```python
self.save_checkpoint('best_model.pth')
```

`_fitness()` creates a `Trainer` and calls `fit()`. Each of the 96 GA evaluations (12 individuals × 8 generations) that improves on its own `best_val_acc=0.0` will overwrite `best_model.pth`. By the time `train_final_model()` runs, `best_model.pth` contains the last GA candidate's best weights — not the final model.

This is saved by the fact that `train_final_model()` also calls `fit()`, which starts fresh with `best_val_acc=0.0` and will immediately overwrite the checkpoint on the first epoch it beats 0%. However, if the run is interrupted between GA completion and `train_final_model()`, the loaded checkpoint is a GA intermediate.

**Fix:** Pass `checkpoint_name` as a parameter to `Trainer.__init__` (as V2 does) and use a temporary name (e.g. `'ga_temp.pth'`) during GA fitness evaluation:
```python
# In _fitness():
trainer = Trainer(model, train_loader, val_loader,
                  lr=params['learning_rate'],
                  device=device,
                  checkpoint_dir=self.config['checkpoint_dir'],
                  checkpoint_name='ga_temp.pth')   # isolate from final model
```

---

### 🟡 CR-V1-02 — `dropout_rate` not drawn from `best_params` in final training and CV

**Location:** `train_final_model()` Cell 19 and `run_cross_validation()` Cell 26:
```python
model = FERNet(
    num_layers=best_params['num_layers'],
    hidden_size=best_params['hidden_size'],
    dropout_rate=0.3,    # hardcoded — not from best_params
)
```

V1's GA does not tune dropout, so `best_params` has no `'dropout_rate'` key — the hardcoding is intentional. However, the parameter is passed in alongside `best_params` with no comment explaining why it is not included. A reader will assume it was overlooked.

**Fix:** Add a comment clarifying the intent:
```python
dropout_rate=0.3,   # fixed — V1 GA does not tune dropout (see V2 for GA-tuned version)
```

---

### 🟡 CR-V1-03 — `'fold_accuracies' in dir()` is unreliable

**Location:** Cell 33 (final summary):
```python
if 'fold_accuracies' in dir():
```

`dir()` in a notebook returns a mix of local, global, and built-in names and its behaviour in cell scope is implementation-defined. The correct idiom is:
```python
if 'fold_accuracies' in globals():
```

---

### 🟡 CR-V1-04 — `torch.load` without `weights_only`

**Location:** `Trainer.load_checkpoint()`, Cell 13:
```python
ckpt = torch.load(path, map_location=self.device)
```

PyTorch ≥ 2.0 emits a `FutureWarning` for `torch.load` without `weights_only`. Loading pickled data from untrusted sources with `weights_only=False` (the default) allows arbitrary code execution.

**Fix:**
```python
ckpt = torch.load(path, map_location=self.device, weights_only=True)
```

Note: if the checkpoint contains non-tensor types (e.g. Python scalars like `best_val_acc`), use `weights_only=False` explicitly to suppress the warning and document the reason.

---

### 🔵 CR-V1-05 — `from collections import Counter` inside a cell body

**Location:** Cell 7:
```python
from collections import Counter
labels = [label for _, label in train_loader.dataset.samples]
```

Imports inside cell bodies are untidy and will cause a `NameError` if the cell is re-run after a kernel restart without re-running the imports cell. Move to Cell 2 (imports).

---

### 🔵 CR-V1-06 — `DataLoader` `pin_memory=False` on GPU device

**Location:** `build_dataloaders()`, Cell 6:
```python
train_loader = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=0, pin_memory=False)
```

When `CONFIG['device'] == 'cuda'`, `pin_memory=True` reduces host-to-device transfer latency. The value is hardcoded regardless of device. On CPU, `pin_memory=True` is silently ignored by PyTorch, so it is safe to always enable it:
```python
pin_memory = (config['device'] == 'cuda')
DataLoader(..., pin_memory=pin_memory)
```

---

## FER_System_V2.ipynb

### 🟡 CR-V2-02 — `FERPreprocessor` docstring describes disabled step as active

**Location:** Cell 5, class docstring:
```python
"""Callable pre-processor for FER images: Denoise → CLAHE → Sharpen.

    ...
    3. Unsharp mask (gentle): re-sharpens edges softened by step 1.
       Strength reduced from 1.5/-0.5 to 1.3/-0.3 to avoid artifact amplification.
"""
```

The default is `sharpen_after=False` (Fix 6) and the instantiation in Cell 7 explicitly passes `sharpen_after=False`. The docstring describes a three-stage pipeline that is disabled by default, which will mislead anyone reading the class definition.

**Fix:** Update the class docstring to reflect the actual default pipeline:
```
Pipeline (default, sharpen_after=False):
    1. GaussianBlur(3x3) — suppresses sensor noise before CLAHE amplification.
    2. CLAHE(clip=3.0, tile=6x6) — local adaptive contrast enhancement.

Optional (sharpen_after=True, disabled by default — risk of pixel saturation post-CLAHE):
    3. Unsharp mask(1.3, -0.3) — edge recovery; clip to [0,255] required.
```

---

### 🟡 CR-V2-03 — Duplicate `augment` parameter in `build_transforms` docstring

**Location:** Cell 6:
```python
"""
    ...
    augment:      If True, adds HFlip-only training augmentation
    augment:      If True, adds training-time augmentations
"""
```

Two `augment:` lines — a copy-paste artifact from when the docstring was updated. The second line also incorrectly implies broader augmentation (which was removed in Fix 7).

**Fix:** Remove the second line.

---

### 🟡 CR-V2-04 — `Trainer.load_checkpoint` / `save_checkpoint` asymmetry

**Location:** Cell 13:
```python
def save_checkpoint(self) -> None:          # uses self.checkpoint_name
    ...
def load_checkpoint(self, filename: str) -> None:   # requires explicit arg
    ...
```

`save_checkpoint` takes no argument and uses `self.checkpoint_name`. `load_checkpoint` requires an explicit filename string. This means callers must remember the checkpoint name to load, while save handles it internally. Either both should use `self.checkpoint_name`, or both should accept an explicit filename.

**Fix (preferred):**
```python
def load_checkpoint(self, filename: str = None) -> None:
    filename = filename or self.checkpoint_name
    ...
```

---

### 🔵 CR-V2-06 — Internal fix notation in `FERPreprocessor` comment

**Location:** Cell 5:
```python
# Fix 3: strength reduced 1.5/-0.5 -> 1.3/-0.3 to reduce artifact risk
```

Development iteration tags (`Fix 3`) are internal notes. They do not convey useful information to a reader and should be replaced with the reasoning:
```python
# Reduced from (1.5, -0.5) — stronger sharpening risks pixel saturation post-CLAHE
```

---

### 🔵 CR-V2-07 — `torch.load` without `weights_only` (same as V1)

**Location:** `Trainer.load_checkpoint()`, Cell 13. Same issue as CR-V1-04 above. Apply the same fix.

---

### 🔵 CR-V2-08 — `pin_memory=False` with CUDA (same as V1)

**Location:** `build_dataloaders()`, Cell 6. Same issue as CR-V1-06. Apply the same fix.

---

## fer_deploy.py

### 🟡 CR-D1-01 — `draw_overlay` FPS parameter is dead code

**Location:** `draw_overlay()` and `process_frame()`:
```python
def draw_overlay(frame, faces, results, fps=None):
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", ...)
```

`process_frame()` always calls `draw_overlay(frame, faces, results)` — without `fps`. FPS text is then separately drawn in `run_video()` and `run_camera()` via a direct `cv2.putText` call after `process_frame` returns. The `fps` parameter in `draw_overlay` is never exercised.

**Fix (option A):** Pass FPS into `process_frame` and forward it to `draw_overlay`:
```python
def process_frame(frame, inference, detector, fps=None):
    ...
    draw_overlay(frame, faces, results, fps=fps)
```

**Fix (option B):** Remove the `fps` parameter from `draw_overlay` and keep FPS drawing in the run functions (current behaviour, just remove dead code from `draw_overlay`).

---

### 🟡 CR-D1-02 — `VideoWriter` not validated before use

**Location:** `run_video()`:
```python
writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))
# No check here
while True:
    ...
    writer.write(annotated)
```

If the codec (`mp4v`) is unavailable on the system (common on some Windows configurations), `VideoWriter` silently creates an invalid writer. All subsequent `write()` calls are no-ops and the output file is empty.

**Fix:**
```python
writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))
if not writer.isOpened():
    print(f"[ERROR] Failed to open VideoWriter for {out_path} — codec may be unavailable")
    cap.release()
    return
```

---

### 🟡 CR-D1-03 — Unused import `matplotlib.patches`

**Location:** Line 24:
```python
import matplotlib.patches as mpatches
```

`mpatches` is never referenced anywhere in the file. This is a leftover from development.

**Fix:** Remove the import.

---

### 🔵 CR-D1-04 — `torch.load` without `weights_only`

**Location:** `FERInference.__init__`:
```python
ckpt = torch.load(model_path, map_location=self.device)
```
Same issue as CR-V1-04. Fix identically.

---

### 🔵 CR-D1-05 — Camera resolution set without checking success

**Location:** `run_camera()`:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

`cap.set()` returns a boolean indicating whether the property was accepted. If the camera does not support 640×480, it silently falls back to its native resolution. This is fine for most use cases, but can cause unexpected behaviour if downstream code assumes a specific size.

**Fix:** Log the actual resolution in use:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_w}x{actual_h}")
```

---

## fer_deploy_V2.py

### 🟡 CR-D2-01 — `extract_face_crop` may return non-square crop near frame edges

**Location:** `extract_face_crop()`:
```python
x1 = max(0, cx - half)
y1 = max(0, cy - half)
x2 = min(W, cx + half)
y2 = min(H, cy + half)
return gray_frame[y1:y2, x1:x2]
```

When the face is near a frame boundary, the clip to `[0, W]` or `[0, H]` makes the crop non-square (e.g. 80×96 instead of 96×96). This is then passed to `FaceAligner.align()` (which uses `cv2.warpAffine` with the crop's own dimensions) and then to `FERInference._load_image()` which resizes to 48×48. The resize of a non-square crop to a square target distorts the face geometry — the same distortion the square crop was designed to prevent.

**Fix:** Pad the shortfall with `cv2.copyMakeBorder` rather than clipping, to maintain square dimensions:
```python
crop = gray_frame[y1:y2, x1:x2]
# If clipped to boundary, pad with edge pixels to restore square shape
h_crop, w_crop = crop.shape
target = max(h_crop, w_crop)
pad_bottom = target - h_crop
pad_right  = target - w_crop
if pad_bottom > 0 or pad_right > 0:
    crop = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right,
                              cv2.BORDER_REPLICATE)
return crop
```

---

### 🔵 CR-D2-02 — `torch.load` without `weights_only`

**Location:** `FERInference.__init__`. Same issue as CR-V1-04. Fix identically.

---

## Cross-Cutting Observations

### ✅ Strengths

- **Type hints throughout** — all class methods in notebooks use `int`, `float`, `str`, `torch.Tensor`, `DataLoader` annotations. Improves readability and enables static analysis.
- **`CustomNetLayer` unit test** — Cell 11 in both notebooks verifies `W @ (x + x²)` numerically against a manual computation. This is the right approach for a non-standard layer.
- **V2 input range assertion** — Cell 16's runtime assertion that pixels are in `[0, 1]` catches preprocessing misconfiguration immediately rather than silently producing wrong results.
- **Gradient clipping** — `clip_grad_norm_(max_norm=1.0)` is correctly placed after `loss.backward()` and before `optimizer.step()` in both notebooks.
- **Seeding** — `set_seed()` covers Python, NumPy, PyTorch CPU, and CUDA. `torch.backends.cudnn.deterministic = True` is included. Correct and complete.
- **Elitism in GA** — deep-copying elite chromosomes before carrying them forward prevents mutation of the elite through the list reference. Correct use of `copy.deepcopy`.
- **`model.eval()` + `torch.no_grad()`** — consistently paired in all evaluation paths across notebooks and deploy scripts. No missing `no_grad` contexts found.
- **Checkpoint metadata** — `num_layers` and `hidden_size` saved alongside weights, allowing deployment scripts to reconstruct the exact model architecture without hardcoding.
- **`BORDER_REPLICATE` in `FaceAligner.warpAffine`** — correct choice for face rotation; avoids black border artifacts at rotation edges.
- **`FERInference` sets `dropout_rate=0.0` at inference** (V1 notebook) — explicitly disabling dropout in the inference object is cleaner than relying solely on `model.eval()`.

---

### 🟡 Cross-file — `torch.load` without `weights_only`

Appears in: `Trainer.load_checkpoint` (both notebooks), `FERInference.__init__` (both deploy scripts). Four locations total. All should add `weights_only=False` (explicit, to suppress the warning and document intent) since checkpoint dicts contain Python scalars alongside tensors.

```python
ckpt = torch.load(path, map_location=self.device, weights_only=False)
```

---

## Issue Index

| ID | File | Severity | Description |
|----|------|----------|-------------|
| CR-V1-01 | FER_System.ipynb | 🔴 | GA fitness evaluations overwrite `best_model.pth` |
| CR-V1-02 | FER_System.ipynb | 🟡 | `dropout_rate=0.3` hardcoded without explanatory comment |
| CR-V1-03 | FER_System.ipynb | 🟡 | `'fold_accuracies' in dir()` — should use `globals()` |
| CR-V1-04 | FER_System.ipynb | 🟡 | `torch.load` without `weights_only` |
| CR-V1-05 | FER_System.ipynb | 🔵 | `from collections import Counter` inside cell body |
| CR-V1-06 | FER_System.ipynb | 🔵 | `pin_memory=False` hardcoded regardless of device |
| CR-V2-02 | FER_System_V2.ipynb | 🟡 | `FERPreprocessor` docstring describes disabled sharpen step |
| CR-V2-03 | FER_System_V2.ipynb | 🟡 | Duplicate `augment:` parameter in `build_transforms` docstring |
| CR-V2-04 | FER_System_V2.ipynb | 🟡 | `load_checkpoint` / `save_checkpoint` interface asymmetry |
| CR-V2-06 | FER_System_V2.ipynb | 🔵 | Internal fix tag `# Fix 3:` in public docstring |
| CR-V2-07 | FER_System_V2.ipynb | 🔵 | `torch.load` without `weights_only` |
| CR-V2-08 | FER_System_V2.ipynb | 🔵 | `pin_memory=False` hardcoded regardless of device |
| CR-D1-01 | fer_deploy.py | 🟡 | `draw_overlay` FPS parameter never exercised |
| CR-D1-02 | fer_deploy.py | 🟡 | `VideoWriter` not validated before writing frames |
| CR-D1-03 | fer_deploy.py | 🟡 | Unused import `matplotlib.patches as mpatches` |
| CR-D1-04 | fer_deploy.py | 🔵 | `torch.load` without `weights_only` |
| CR-D1-05 | fer_deploy.py | 🔵 | Camera resolution set without logging actual value |
| CR-D2-01 | fer_deploy_V2.py | 🟡 | `extract_face_crop` returns non-square crop near frame edges |
| CR-D2-02 | fer_deploy_V2.py | 🔵 | `torch.load` without `weights_only` |
