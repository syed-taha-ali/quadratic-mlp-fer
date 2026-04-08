"""
fer_deploy.py — Facial Expression Recognition deployment script (V2).

Usage (from AI_CW2/ directory):
    python fer_deploy.py --image  Data/Testing/Happy/img1.jpg
    python fer_deploy.py --video  path/to/video.mp4
    python fer_deploy.py --camera          # default camera index 0
    python fer_deploy.py --camera 1        # specific camera index
    python fer_deploy.py --model  checkpoints/best_model_v2.pth --camera
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Model class definitions (verbatim from FER_System_V2.ipynb)
# ---------------------------------------------------------------------------

CLASSES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

class FERPreprocessor:
    """V2 CLAHE pipeline: Denoise(3×3) → CLAHE(clip=3.0, tile=6×6).
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
    """Core custom formula: net = W @ (x + x²)"""
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
# Inference
# ---------------------------------------------------------------------------

class FERInference:
    """Load a trained FERNet checkpoint and run single-image or batch inference."""

    def __init__(self, model_path, num_layers, hidden_size, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = FERPreprocessor()   # V2: clip=3.0, tile=6×6, blur_before=True
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Lambda(self.preprocessor),
            transforms.ToTensor(),
            # V2: Normalize removed — pixels stay in [0,1] for x+x² monotonicity
        ])
        self.model = FERNet(num_layers=num_layers, hidden_size=hidden_size)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.classes = CLASSES

    def _load_image(self, input_img):
        """Accept file path (str), numpy array (BGR or gray), or PIL Image -> grayscale PIL 48×48."""
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
# Overlay drawing
# ---------------------------------------------------------------------------

# Colour palette per emotion (BGR)
EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "fear":     (180, 0,   180),
    "happy":    (0,   220, 0),
    "neutral":  (180, 180, 180),
    "sad":      (220, 100, 0),
    "surprise": (0,   180, 220),
}


def draw_overlay(frame, faces, results, fps=None):
    """
    Draw bounding boxes, labels, confidence bars on frame (in-place).
    faces  : list of (x, y, w, h)
    results: list of (label, confidence, all_probs) aligned with faces
    fps    : float or None
    """
    h_frame, w_frame = frame.shape[:2]

    # --- FPS counter (top-right) ---
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w_frame - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # --- Per-face bounding box + label ---
    for (x, y, w, h), (label, conf, _) in zip(faces, results):
        color = EMOTION_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label.capitalize()} {conf * 100:.1f}%"
        text_y = max(y - 10, 20)
        cv2.putText(frame, text, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # --- Probability bars (top-left corner) for first face only ---
    if results:
        all_probs = results[0][2]
        bar_x0 = 8
        bar_y0 = 12
        bar_w_max = 120
        bar_h = 12
        gap = 16
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
# Per-frame processing helper
# ---------------------------------------------------------------------------

def process_frame(frame, inference, detector):
    """Detect faces, run inference, return (annotated_frame, faces, results)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert display frame to grayscale so output matches what the model sees.
    # GRAY2BGR keeps 3 channels so cv2 drawing functions (putText, rectangle) still work.
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    faces = detector.detect(gray)
    results = []

    if not faces:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2, cv2.LINE_AA)
        return frame, faces, results

    h_gray, w_gray = gray.shape
    for (x, y, w, h) in faces:
        # 10% padding around detected face
        pad_x = int(w * 0.10)
        pad_y = int(h * 0.10)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_gray, x + w + pad_x)
        y2 = min(h_gray, y + h + pad_y)
        face_crop = gray[y1:y2, x1:x2]
        label, conf, all_probs = inference.predict(face_crop)
        results.append((label, conf, all_probs))

    draw_overlay(frame, faces, results)
    return frame, faces, results


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_image(args, inference, detector):
    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[ERROR] Could not read image: {args.image}")
        return

    annotated, faces, results = process_frame(frame.copy(), inference, detector)

    # Save annotated image
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "annotated_" + os.path.basename(args.image))
    cv2.imwrite(out_path, annotated)
    print(f"Saved annotated image -> {out_path}")

    # Console summary
    if results:
        for i, (label, conf, all_probs) in enumerate(results):
            print(f"  Face {i+1}: {label.capitalize()} ({conf*100:.1f}%)")
            for cls, p in sorted(all_probs.items(), key=lambda kv: -kv[1]):
                bar = "|" * int(p * 20)
                print(f"    {cls:<10} {bar:<20} {p*100:5.1f}%")
    else:
        print("  No face detected.")

    # Matplotlib display
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.axis("off")
    title = results[0][0].capitalize() + f" {results[0][1]*100:.1f}%" if results else "No face"
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def run_video(args, inference, detector):
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
    out_path = os.path.join("results", "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))

    frame_count = 0
    t0 = time.time()
    fps_display = 0.0

    print("Processing video — press 'q' to quit early...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated, _, _ = process_frame(frame, inference, detector)

        # FPS overlay
        elapsed = time.time() - t0
        if elapsed > 0:
            fps_display = frame_count / elapsed
        cv2.putText(annotated, f"FPS: {fps_display:.1f}",
                    (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated)
        cv2.imshow("FER — Video", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved output video -> {out_path}  ({frame_count} frames)")


def run_camera(args, inference, detector):
    cam_index = args.camera if isinstance(args.camera, int) else 0
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {cam_index}")
        return

    # Prefer 640×480 for real-time performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera feed started — press 'q' to quit.")
    frame_count = 0
    t0 = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame grab failed, retrying...")
            continue

        frame_count += 1
        annotated, _, _ = process_frame(frame, inference, detector)

        elapsed = time.time() - t0
        if elapsed > 0:
            fps_display = frame_count / elapsed
        cv2.putText(annotated, f"FPS: {fps_display:.1f}",
                    (annotated.shape[1] - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("FER — Live Camera", annotated)
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
        description="Facial Expression Recognition — deployment script",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--image",  type=str,          help="Path to a static image file")
    parser.add_argument("--video",  type=str,          help="Path to a video file")
    parser.add_argument("--camera", type=int, nargs="?", const=0,
                        help="Camera index (default 0)")
    parser.add_argument("--model",  type=str,
                        default="checkpoints/best_model_v2.pth",
                        help="Path to model checkpoint (default: checkpoints/best_model_v2.pth)")
    args = parser.parse_args()

    # Validate exactly one mode selected
    modes = [m for m in (args.image, args.video, args.camera) if m is not None]
    if len(modes) == 0:
        parser.error("Specify one of --image, --video, or --camera")

    # Load checkpoint -> read num_layers / hidden_size
    if not os.path.isfile(args.model):
        print(f"[ERROR] Checkpoint not found: {args.model}")
        print("  Train the model first by running FER_System_V2.ipynb through Section 5.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device)
    num_layers  = ckpt.get("num_layers",  3)
    hidden_size = ckpt.get("hidden_size", 256)
    val_acc     = ckpt.get("best_val_acc", None)

    print(f"Loaded checkpoint: {args.model}")
    print(f"  num_layers={num_layers}, hidden_size={hidden_size}"
          + (f", best_val_acc={val_acc:.3f}" if val_acc is not None else ""))
    print(f"  Device: {device}")

    inference = FERInference(args.model, num_layers=num_layers,
                             hidden_size=hidden_size, device=device)
    detector  = FaceDetector()

    if args.image is not None:
        run_image(args, inference, detector)
    elif args.video is not None:
        run_video(args, inference, detector)
    elif args.camera is not None:
        run_camera(args, inference, detector)


if __name__ == "__main__":
    main()
