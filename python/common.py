"""Shared helpers for the Python scripts.

The preprocessing here mirrors src/preprocess.cpp step by step, so Python and C++
results can be compared exactly.
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
LABELS_PATH = DATA_DIR / "imagenet_labels.txt"
IMAGENETTE_DIR = DATA_DIR / "imagenette2-320"

MODEL_NAMES = ["resnet50", "mobilenet_v3_large"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(img_bgr: np.ndarray, resize_short: int = 256, crop: int = 224) -> np.ndarray:
    """BGR uint8 HWC image -> float32 CHW tensor, same steps as the C++ code."""
    h, w = img_bgr.shape[:2]
    # Resize so the shorter side == resize_short, keeping aspect ratio.
    # Long side is truncated with int(), the same rule torchvision uses.
    if h < w:
        new_h, new_w = resize_short, int(resize_short * w / h)
    else:
        new_h, new_w = int(resize_short * h / w), resize_short
    img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop. Python round() is round-half-to-even, same as torchvision's CenterCrop
    # (the C++ side uses std::nearbyint to match).
    top = int(round((new_h - crop) / 2.0))
    left = int(round((new_w - crop) / 2.0))
    img = img[top:top + crop, left:left + crop]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC -> CHW


def load_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return img


def sample_val_images(n: int = 20) -> list:
    """Deterministic sample: the first n/10 images (sorted) of each of the 10 val classes."""
    val_dir = IMAGENETTE_DIR / "val"
    if not val_dir.exists():
        raise SystemExit("Imagenette missing. Run: bash scripts/download_data.sh")
    class_dirs = sorted(p for p in val_dir.iterdir() if p.is_dir())
    per_class = max(1, n // len(class_dirs))
    images = []
    for d in class_dirs:
        images.extend(sorted(d.glob("*.JPEG"))[:per_class])
    return images[:n]
