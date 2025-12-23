#!/usr/bin/env python3
"""
Run PIDNet exports (PyTorch, ONNX, TensorRT) on a folder of images and
save colorized overlay predictions for visual inspection.
"""
import argparse
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import cv2
import numpy as np

from config import COLOR_MAP, SUPPORTED_EXTS, ensure_project_root_on_path

ensure_project_root_on_path()

from inference import PyTorchInferencer, ONNXInferencer, TensorRTInferencer  # noqa: E402



def colorize(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), np.uint8)
    out[mask == 1] = COLOR_MAP[1]
    return out


def resize_mask(mask, h0, w0):
    return cv2.resize(
        mask.astype(np.uint8),
        (w0, h0),
        interpolation=cv2.INTER_NEAREST,
    )


def make_overlay(orig_bgr, mask):
    color = colorize(mask)
    return cv2.addWeighted(orig_bgr, 0.6, color, 0.4, 0)


def collect_images(images_arg: str) -> List[Path]:
    path = Path(images_arg)
    if not path.exists():
        raise FileNotFoundError(f"{images_arg} does not exist")
    if path.is_file():
        return [path]
    image_paths = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_paths:
        raise RuntimeError(f"No images with extensions {SUPPORTED_EXTS} under {images_arg}")
    return image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate overlay visualizations for PIDNet exports.",
    )

    parser.add_argument(
        "--images",
        type=str,
        help="Path to an image file or folder of images (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory where overlay images are saved (default: %(default)s)",
    )
    parser.add_argument("--pt", type=str, help="Path to the .pt checkpoint")
    parser.add_argument("--onnx", type=str, help="Path to the ONNX model")
    parser.add_argument("--engine", type=str, help="Path to the TensorRT engine")
    args = parser.parse_args()
    if not (args.pt or args.onnx or args.engine):
        parser.error("Provide at least one of --pt, --onnx, or --engine")
    return args


def run_inference(image_paths: Sequence[Path], outputs: Path, runners: Dict[str, Callable[[np.ndarray], np.ndarray]]):
    outputs.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: unable to read {img_path}, skipping.")
            continue
        h0, w0 = image.shape[:2]
        for name, runner in runners.items():
            mask = runner(image)
            mask_full = resize_mask(mask, h0, w0)
            overlay = make_overlay(image, mask_full)
            out_path = outputs / f"{img_path.stem}_{name}.png"
            cv2.imwrite(str(out_path), overlay)
        print(f"Processed {img_path.name}")

    print(f"\nSaved overlays for {len(image_paths)} images to {outputs}")


def main():
    args = parse_args()
    image_paths = collect_images(args.images)
    runners: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    if args.pt:
        runners["pt"] = PyTorchInferencer(args.pt)
    if args.onnx:
        runners["onnx"] = ONNXInferencer(args.onnx)
    if args.engine:
        runners["trt"] = TensorRTInferencer(args.engine)

    run_inference(image_paths, Path(args.output), runners)


if __name__ == "__main__":
    main()
