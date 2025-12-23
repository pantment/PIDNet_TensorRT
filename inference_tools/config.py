"""Shared constants and helpers for TensorRT validation tools."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root_on_path() -> None:
    """Add the repository root to sys.path so `models` can be imported."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


TARGET_SIZE = (360, 640)  # (W, H)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
COLOR_MAP = [
    (0, 0, 0),  # background
    (0, 255, 0),  # foreground
]

NUM_CLASSES = 2
DEFAULT_DATASET_ROOT = PROJECT_ROOT / ""


def is_supported_image(path: Path) -> bool:
    """Return True when the path has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTS


__all__ = [
    "PROJECT_ROOT",
    "ensure_project_root_on_path",
    "TARGET_SIZE",
    "MEAN",
    "STD",
    "SUPPORTED_EXTS",
    "COLOR_MAP",
    "NUM_CLASSES",
    "DEFAULT_DATASET_ROOT",
    "is_supported_image",
]
