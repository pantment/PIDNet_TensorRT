import argparse
from pathlib import Path

import torch
import torch.nn as nn

import _init_paths
from models.pidnet import get_pred_model  # make sure this import is correct


class ExportWrapper(nn.Module):
    """
    Thin wrapper that calls your custom forward_export(x) and returns only logits.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assumes PIDNet has a method: forward_export(self, x) -> logits
        return self.model.forward_export(x)


def build_export_model(
    weights_path: str,
    device: str = "cuda",
    num_classes: int = 2,
) -> nn.Module:
    """
    Build PIDNet, load checkpoint, wrap with ExportWrapper and return eval model.
    """
    # 1. Build base model (augment=False inside get_pred_model)
    model = get_pred_model(name="pidnet_s", num_classes=num_classes)
    model.to(device)

    # 2. Load checkpoint
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # PIDNet checkpoints often need strict=False (key mismatches / prefixes)
    model.load_state_dict(state, strict=False)
    print(f"[OK] Loaded weights from: {weights_path}")

    model.eval()

    # 3. Wrap for export
    export_model = ExportWrapper(model).eval().to(device)
    return export_model


def export_torchscript(
    export_model: nn.Module,
    device: str = "cuda",
    out_path: str = "pidnet_s_export.ts",
    img_height: int = 1024,
    img_width: int = 2048,
) -> None:
    """
    Export model to TorchScript using tracing with a dummy input.
    """
    example_inputs = torch.randn(1, 3, img_height, img_width, device=device)
    traced = torch.jit.trace(export_model, example_inputs)
    traced.save(out_path)
    print(f"[OK] TorchScript saved to: {out_path}")


def export_onnx(
    export_model: nn.Module,
    device: str = "cuda",
    out_path: str = "pidnet_s_export.onnx",
    img_height: int = 1024,
    img_width: int = 2048,
) -> None:
    """
    Export model to ONNX. This is usually what you feed into TensorRT.
    """
    example_inputs = torch.randn(1, 3, img_height, img_width, device=device)

    torch.onnx.export(
        export_model,
        example_inputs,
        out_path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,       # fixed size; easiest & fastest for TRT
        do_constant_folding=True,
    )
    print(f"[OK] ONNX saved to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="PIDNet export utility.")

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained PIDNet weights (.pt/.pth).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes for the exported model. Default: 19",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=1024,
        help="Input image height used for tracing/export. Default: 1024",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=2048,
        help="Input image width used for tracing/export. Default: 2048",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where exported files will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model + load weights + wrap
    export_model = build_export_model(
        weights_path=args.weights,
        device=device,
        num_classes=args.num_classes,
    )

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export TorchScript
    export_torchscript(
        export_model,
        device=device,
        out_path=str(output_dir / "pidnet_s_export.ts"),
        img_height=args.img_height,
        img_width=args.img_width,
    )

    # Export ONNX
    export_onnx(
        export_model,
        device=device,
        out_path=str(output_dir / "pidnet_s_export.onnx"),
        img_height=args.img_height,
        img_width=args.img_width,
    )
