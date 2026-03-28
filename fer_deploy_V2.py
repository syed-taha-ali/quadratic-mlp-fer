"""
fer_deploy_V2.py — Facial Expression Recognition deployment script V2.

Improvements over fer_deploy.py:
  1. Face alignment  — detects eyes (Haar cascade) and rotates face to horizontal
                       eye line before feeding the 48x48 crop to the model.
                       Falls back to unaligned crop if eye detection fails.
  2. Better cropping — square bounding box with 20% padding (was 10%), ensures
                       consistent face framing regardless of detector aspect ratio.
  3. Temporal smoothing — camera predictions averaged over a rolling window of N
                          frames, eliminating per-frame label flickering.

Usage (from AI_CW2/ directory):
    python fer_deploy_V2.py --image  Data/Testing/Happy/img1.jpg
    python fer_deploy_V2.py --video  path/to/video.mp4
    python fer_deploy_V2.py --camera
    python fer_deploy_V2.py --camera --smooth 15        # smoothing window (default 10)
    python fer_deploy_V2.py --model checkpoints/best_model_v2.pth --camera
"""

import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Model class definitions (verbatim from FER_System_V2.ipynb)
# ---------------------------------------------------------------------------

CLASSES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]


class FERPreprocessor:
    """V2 CLAHE pipeline: Denoise(3x3) -> CLAHE(clip=3.0, tile=6x6).
    Matches FER_System_V2.ipynb: blur_before=True, sharpen_after=False.
    """
    def __init__(self, clip_limit=3.0, tile_grid_size=(6, 6), blur_before=True):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.blur_before = blur_before

    def __call__(self, pil_image):
        img_np = np.array(pil_image, dtype=np.uint8)
        if self.blur_before:
            img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
        return Image.fromarray(self.clahe.apply(img_np))


class CustomNetLayer(nn.Module):
    """Core custom formula: net = W @ (x + x^2)"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x + x ** 2)


class CustomSigmoid(nn.Module):
    """sigma(net) = 1 / (1 + exp(-net))"""
    def forward(self, x):
        return torch.sigmoid(x)


class CustomBlock(nn.Module):
    """One hidden layer: CustomNetLayer -> BatchNorm1d -> CustomSigmoid -> Dropout"""
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super().__init__()
        self.layer = nn.Sequential(
            CustomNetLayer(in_features, out_features),
            nn.BatchNorm1d(out_features),
            CustomSigmoid(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.layer(x)


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
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        return self.output(x)

    def get_probabilities(self, x):
        return F.softmax(self.forward(x), dim=1)


# ---------------------------------------------------------------------------
# Improvement 1: Face aligner using eye detection
# ---------------------------------------------------------------------------

class FaceAligner:
    """Aligns a face crop by rotating it so both eyes are on a horizontal line.

    Uses OpenCV's bundled haarcascade_eye.xml to locate eye centres within
    the face bounding box.  If fewer than two eyes are detected the original
    (unaligned) crop is returned unchanged.
    """

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(cascade_path)
        if self.eye_cascade.empty():
            print("[WARNING] haarcascade_eye.xml not found — face alignment disabled.")
            self.enabled = False
        else:
            self.enabled = True

    def align(self, face_gray):
        """Return aligned (or original) grayscale face crop."""
        if not self.enabled:
            return face_gray

        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
        )

        if len(eyes) < 2:
            return face_gray  # fall back — eye detection failed

        # Sort eyes left-to-right by x centre
        centres = sorted(
            [(ex + ew // 2, ey + eh // 2) for ex, ey, ew, eh in eyes],
            key=lambda c: c[0],
        )
        left_eye, right_eye = centres[0], centres[1]

        # Angle between eye centres
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate around face centre
        h, w = face_gray.shape[:2]
        centre = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(centre, angle, scale=1.0)
        aligned = cv2.warpAffine(
            face_gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned


# ---------------------------------------------------------------------------
# Improvement 3: Temporal smoother for camera predictions
# ---------------------------------------------------------------------------

class PredictionSmoother:
    """Rolling average of softmax probability vectors over the last N frames.

    Reduces per-frame label flickering on live camera feed by averaging
    the model's softmax output across a sliding window.
    """

    def __init__(self, window=10):
        self.window = window
        # One deque per tracked face slot (index 0 = first detected face)
        self._buffers = {}

    def update(self, face_idx, probs):
        """Push a new probability vector; return the smoothed vector."""
        if face_idx not in self._buffers:
            self._buffers[face_idx] = deque(maxlen=self.window)
        self._buffers[face_idx].append(probs)
        return np.mean(self._buffers[face_idx], axis=0)

    def reset(self):
        self._buffers.clear()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class FERInference:
    """Load a trained FERNet checkpoint and run single-image or batch inference."""

    def __init__(self, model_path, num_layers, hidden_size, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = FERPreprocessor()   # V2: clip=3.0, tile=6x6, blur_before=True
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Lambda(self.preprocessor),
            transforms.ToTensor(),
            # V2: Normalize removed — pixels stay in [0,1] for x+x^2 monotonicity
        ])
        self.model = FERNet(num_layers=num_layers, hidden_size=hidden_size)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.classes = CLASSES

    def _load_image(self, input_img):
        """Accept file path, numpy array (BGR or gray), or PIL Image -> grayscale PIL 48x48."""
        if isinstance(input_img, str):
            pil = Image.open(input_img).convert("L")
        elif isinstance(input_img, np.ndarray):
            if input_img.ndim == 3:
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            pil = Image.fromarray(input_img)
        elif isinstance(input_img, Image.Image):
            pil = input_img.convert("L")
        else:
            raise ValueError(f"Unsupported image type: {type(input_img)}")
        return pil.resize((48, 48), Image.LANCZOS)

    def predict(self, image_input):
        """
        Returns:
            predicted_label (str), confidence (float 0-1), all_probs (dict label->prob)
        """
        pil = self._load_image(image_input)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.model.get_probabilities(tensor).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        all_probs = {cls: float(probs[i]) for i, cls in enumerate(self.classes)}
        return self.classes[idx], float(probs[idx]), all_probs

    def predict_probs(self, image_input):
        """Return raw softmax probability array (length 6) for smoothing."""
        pil = self._load_image(image_input)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.get_probabilities(tensor).squeeze(0).cpu().numpy()

    def predict_batch(self, image_list):
        return [self.predict(img) for img in image_list]


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

class FaceDetector:
    """OpenCV Haar Cascade frontal face detector."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    def detect(self, gray_frame):
        """Returns list of (x, y, w, h) bounding boxes."""
        faces = self.cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        return list(faces) if len(faces) > 0 else []


# ---------------------------------------------------------------------------
# Improvement 2: Better crop extraction
# ---------------------------------------------------------------------------

def extract_face_crop(gray_frame, x, y, w, h, pad=0.20):
    """Extract a square, padded, aligned face crop.

    Improvements over V1:
    - Square crop: takes the larger of (w, h) so the face is never stretched.
    - 20% padding (was 10%): gives more context around the face boundary,
      reducing sensitivity to slightly off-centre bounding boxes.
    - Clips to frame boundaries safely.
    """
    H, W = gray_frame.shape[:2]

    # Make bounding box square around the face centre
    cx, cy = x + w // 2, y + h // 2
    half = int(max(w, h) * (1 + pad) / 2)

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, cx + half)
    y2 = min(H, cy + half)

    return gray_frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "fear":     (180, 0,   180),
    "happy":    (0,   220, 0),
    "neutral":  (180, 180, 180),
    "sad":      (220, 100, 0),
    "surprise": (0,   180, 220),
}


def draw_overlay(frame, faces, results, fps=None):
    """Draw bounding boxes, smoothed labels, and confidence bars on frame."""
    h_frame, w_frame = frame.shape[:2]

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w_frame - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    for (x, y, w, h), (label, conf, _) in zip(faces, results):
        color = EMOTION_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label.capitalize()} {conf * 100:.1f}%"
        text_y = max(y - 10, 20)
        cv2.putText(frame, text, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    if results:
        all_probs = results[0][2]
        bar_x0, bar_y0 = 8, 12
        bar_w_max, bar_h, gap = 120, 12, 16
        cv2.rectangle(frame,
                      (bar_x0 - 4, bar_y0 - 4),
                      (bar_x0 + bar_w_max + 50, bar_y0 + len(CLASSES) * gap + 4),
                      (30, 30, 30), -1)
        for i, cls in enumerate(CLASSES):
            prob = all_probs.get(cls, 0.0)
            bw = int(prob * bar_w_max)
            by = bar_y0 + i * gap
            color = EMOTION_COLORS.get(cls, (200, 200, 200))
            cv2.rectangle(frame, (bar_x0, by), (bar_x0 + bw, by + bar_h), color, -1)
            cv2.putText(frame, f"{cls[:3]} {prob * 100:.0f}%",
                        (bar_x0 + bar_w_max + 4, by + bar_h - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def process_frame(frame, inference, detector, aligner, smoother=None,
                  face_idx_offset=0):
    """Detect faces, align, run inference, optionally smooth, return annotated frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert display frame to grayscale so output matches what the model sees
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    faces = detector.detect(gray)
    results = []

    if not faces:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2, cv2.LINE_AA)
        return frame, faces, results

    for i, (x, y, w, h) in enumerate(faces):
        # Improvement 2: square crop with 20% padding
        crop = extract_face_crop(gray, x, y, w, h, pad=0.20)

        # Improvement 1: align face via eye detection
        crop = aligner.align(crop)

        if smoother is not None:
            # Improvement 3: temporal smoothing — average over rolling window
            raw_probs = inference.predict_probs(crop)
            smoothed_probs = smoother.update(face_idx_offset + i, raw_probs)
            idx = int(np.argmax(smoothed_probs))
            label = CLASSES[idx]
            conf = float(smoothed_probs[idx])
            all_probs = {cls: float(smoothed_probs[j]) for j, cls in enumerate(CLASSES)}
        else:
            label, conf, all_probs = inference.predict(crop)

        results.append((label, conf, all_probs))

    draw_overlay(frame, faces, results)
    return frame, faces, results


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_image(args, inference, detector, aligner):
    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[ERROR] Could not read image: {args.image}")
        return

    annotated, faces, results = process_frame(
        frame.copy(), inference, detector, aligner,
        smoother=None,
    )

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "v2_annotated_" + os.path.basename(args.image))
    cv2.imwrite(out_path, annotated)
    print(f"Saved annotated image -> {out_path}")

    if results:
        for i, (label, conf, all_probs) in enumerate(results):
            print(f"  Face {i+1}: {label.capitalize()} ({conf*100:.1f}%)")
            for cls, p in sorted(all_probs.items(), key=lambda kv: -kv[1]):
                bar = "|" * int(p * 20)
                print(f"    {cls:<10} {bar:<20} {p*100:5.1f}%")
    else:
        print("  No face detected.")

    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, cmap="gray")
    ax.axis("off")
    title = results[0][0].capitalize() + f" {results[0][1]*100:.1f}%" if results else "No face"
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def run_video(args, inference, detector, aligner):
    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.video}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "v2_output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))

    # Use a smoother for video too — 5-frame window (less aggressive than camera)
    smoother = PredictionSmoother(window=5)
    frame_count = 0
    t0 = time.time()

    print("Processing video — press 'q' to quit early...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated, _, _ = process_frame(
            frame, inference, detector, aligner,
            smoother=smoother,
        )

        elapsed = time.time() - t0
        fps_display = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(annotated, f"FPS: {fps_display:.1f}",
                    (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated)
        cv2.imshow("FER V2 — Video", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved output video -> {out_path}  ({frame_count} frames)")


def run_camera(args, inference, detector, aligner):
    cam_index = args.camera if isinstance(args.camera, int) else 0
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {cam_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Improvement 3: temporal smoother — default window from --smooth arg
    smoother = PredictionSmoother(window=args.smooth)

    print(f"Camera started — smoothing window={args.smooth} — press 'q' to quit.")

    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame grab failed, retrying...")
            continue

        frame_count += 1
        annotated, _, _ = process_frame(
            frame, inference, detector, aligner,
            smoother=smoother,
        )

        elapsed = time.time() - t0
        fps_display = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(annotated, f"FPS: {fps_display:.1f}",
                    (annotated.shape[1] - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("FER V2 — Live Camera", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FER Deployment V2 — with face alignment, better cropping, temporal smoothing",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--image",     type=str,               help="Path to a static image file")
    parser.add_argument("--video",     type=str,               help="Path to a video file")
    parser.add_argument("--camera",    type=int, nargs="?", const=0,
                                                               help="Camera index (default 0)")
    parser.add_argument("--model",     type=str,
                        default="checkpoints/best_model_v2.pth",
                        help="Path to model checkpoint (default: checkpoints/best_model_v2.pth)")
    parser.add_argument("--smooth",    type=int,   default=10,
                        help="Temporal smoothing window in frames for camera/video (default 10)")
    args = parser.parse_args()

    modes = [m for m in (args.image, args.video, args.camera) if m is not None]
    if len(modes) == 0:
        parser.error("Specify one of --image, --video, or --camera")

    if not os.path.isfile(args.model):
        print(f"[ERROR] Checkpoint not found: {args.model}")
        print("  Train the model first by running FER_System_V2.ipynb through Section 5.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device)
    num_layers  = ckpt.get("num_layers",  2)
    hidden_size = ckpt.get("hidden_size", 512)
    val_acc     = ckpt.get("best_val_acc", None)

    print(f"Loaded checkpoint: {args.model}")
    print(f"  num_layers={num_layers}, hidden_size={hidden_size}"
          + (f", best_val_acc={val_acc:.3f}" if val_acc is not None else ""))
    print(f"  Device: {device}")

    inference = FERInference(args.model, num_layers=num_layers,
                             hidden_size=hidden_size, device=device)
    detector  = FaceDetector()
    aligner   = FaceAligner()

    if args.image is not None:
        run_image(args, inference, detector, aligner)
    elif args.video is not None:
        run_video(args, inference, detector, aligner)
    elif args.camera is not None:
        run_camera(args, inference, detector, aligner)


if __name__ == "__main__":
    main()
