#!/usr/bin/env python3
"""
Compute mIoU for PIDNet binary segmentation models exported as
PyTorch checkpoints, ONNX, or TensorRT engines.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from config import DEFAULT_DATASET_ROOT, NUM_CLASSES, ensure_project_root_on_path

ensure_project_root_on_path()

from inference import PyTorchInferencer, ONNXInferencer, TensorRTInferencer  # noqa: E402


def resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def load_mask(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    return mask


def fast_hist(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (gt >= 0) & (gt < num_classes)
    hist = np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask],
        minlength=num_classes**2,
    )
    return hist.reshape(num_classes, num_classes)


def compute_iou(hist: np.ndarray) -> Tuple[np.ndarray, float]:
    intersection = np.diag(hist)
    ground_truth = hist.sum(axis=1)
    predicted = hist.sum(axis=0)
    union = ground_truth + predicted - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan, dtype=np.float64),
        where=union > 0,
    )
    miou = np.nanmean(iou)
    return iou, miou


def build_dataset(dataset_root: str) -> List[Tuple[str, str]]:
    root = Path(dataset_root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(f"Expected 'images' and 'masks' under {dataset_root}")

    pairs: List[Tuple[str, str]] = []
    for img_path in sorted(images_dir.glob("*.png")):
        mask_path = masks_dir / img_path.name
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path)))
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {dataset_root}")
    return pairs


def evaluate(
    name: str,
    inferencer: Callable[[np.ndarray], np.ndarray],
    dataset: Sequence[Tuple[str, str]],
) -> Dict[str, float]:
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for image_path, mask_path in dataset:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        gt = load_mask(mask_path)
        pred = inferencer(image)
        pred = resize_mask(pred, gt.shape[0], gt.shape[1])
        hist += fast_hist(gt, pred, NUM_CLASSES)

    iou, miou = compute_iou(hist)
    result = {
        "mIoU": miou,
        "IoU_background": iou[0],
        "IoU_foreground": iou[1],
    }
    print(f"\n{name} results on {len(dataset)} samples:")
    print(f"  mIoU:         {miou * 100:.2f}%")
    print(f"  IoU background: {iou[0] * 100:.2f}%")
    print(f"  IoU foreground: {iou[1] * 100:.2f}%")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PIDNet binary segmentation exports by computing mIoU.",
    )
    default_dataset = str(DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Root folder containing 'images' and 'masks' (default: %(default)s)",
    )
    parser.add_argument("--pt", type=str, help="Path to the .pt checkpoint")
    parser.add_argument("--onnx", type=str, help="Path to the ONNX model")
    parser.add_argument("--engine", type=str, help="Path to the TensorRT engine")
    args = parser.parse_args()

    if not (args.pt or args.onnx or args.engine):
        parser.error("Provide at least one of --pt, --onnx, or --engine")
    return args


def main():
    args = parse_args()
    dataset = build_dataset(args.dataset)

    if args.pt:
        inferencer = PyTorchInferencer(args.pt)
        evaluate("PyTorch", inferencer, dataset)
    if args.onnx:
        inferencer = ONNXInferencer(args.onnx)
        evaluate("ONNX", inferencer, dataset)
    if args.engine:
        inferencer = TensorRTInferencer(args.engine)
        evaluate("TensorRT", inferencer, dataset)


if __name__ == "__main__":
    main()
