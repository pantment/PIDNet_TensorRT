#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def extract_state_dict(obj):
    """
    Try to robustly extract a state_dict from common checkpoint formats.
    """
    # If it's already an nn.Module
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        return obj.state_dict()

    if not isinstance(obj, dict):
        raise ValueError("Checkpoint object is not a dict or nn.Module; cannot extract state_dict")

    # Common nesting patterns in training code
    candidate_keys = [
        "state_dict",
        "model",
        "net",
        "model_state",
        "model_state_dict",
        "ema",
        "ema_state_dict",
    ]

    for k in candidate_keys:
        if k in obj and isinstance(obj[k], dict):
            return obj[k]

    # If all values look like tensors, assume this is already a state_dict
    if all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj

    # Last resort: try to guess a dict-of-tensors
    tensor_subdicts = {k: v for k, v in obj.items() if isinstance(v, dict)}
    if tensor_subdicts:
        # If there is exactly one plausible candidate, use it
        if len(tensor_subdicts) == 1:
            return next(iter(tensor_subdicts.values()))

    raise ValueError("Could not find a state_dict-like mapping in checkpoint")


def load_state_dict(path: Path):
    print(f"Loading {path} ...")
    ckpt = torch.load(path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    print(f"  Loaded {len(sd)} parameters/buffers.")
    return sd


def compare_state_dicts(sd_a, sd_b, name_a="A", name_b="B", atol=1e-6, rtol=1e-5):
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    print("\n=== Key comparison ===")
    if only_in_a:
        print(f"Keys only in {name_a} ({len(only_in_a)}):")
        for k in only_in_a:
            print("  ", k)
    else:
        print(f"No keys unique to {name_a}")

    if only_in_b:
        print(f"\nKeys only in {name_b} ({len(only_in_b)}):")
        for k in only_in_b:
            print("  ", k)
    else:
        print(f"No keys unique to {name_b}")

    print(f"\nCommon keys: {len(common)}")

    # Compare shapes and values
    shape_mismatch = []
    large_diff = []

    print("\n=== Detailed per-parameter comparison (common keys) ===")
    for k in common:
        t_a = sd_a[k]
        t_b = sd_b[k]

        if t_a.shape != t_b.shape:
            shape_mismatch.append((k, t_a.shape, t_b.shape))
            print(f"[SHAPE MISMATCH] {k}: {t_a.shape} vs {t_b.shape}")
            continue

        # Cast to float for numeric comparison (handles Long / Int / etc.)
        if not (torch.is_floating_point(t_a) or torch.is_complex(t_a)):
            t_a_f = t_a.to(torch.float32)
            t_b_f = t_b.to(torch.float32)
        else:
            t_a_f = t_a
            t_b_f = t_b

        diff = (t_a_f - t_b_f).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        is_close = torch.allclose(t_a_f, t_b_f, atol=atol, rtol=rtol)

        if not is_close:
            large_diff.append((k, max_diff, mean_diff))

        print(
            f"{k}: shape={tuple(t_a.shape)}, "
            f"max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g}, "
            f"{'OK' if is_close else 'DIFF'}"
        )

    print("\n=== Summary ===")
    print(f"Total common keys: {len(common)}")
    print(f"Shape mismatches: {len(shape_mismatch)}")
    print(f"Parameters with numeric differences (beyond atol/rtol): {len(large_diff)}")

    if large_diff:
        print("\nTop differing parameters (sorted by max_diff):")
        for k, max_diff, mean_diff in sorted(large_diff, key=lambda x: -x[1])[:20]:
            print(f"  {k}: max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g}")


def main():
    parser = argparse.ArgumentParser(description="Compare two PyTorch checkpoints (.pt)")
    parser.add_argument("ckpt_a", type=str, help="Path to first checkpoint (.pt)")
    parser.add_argument("ckpt_b", type=str, help="Path to second checkpoint (.pt)")
    parser.add_argument("--name-a", type=str, default="A", help="Label for first checkpoint")
    parser.add_argument("--name-b", type=str, default="B", help="Label for second checkpoint")
    args = parser.parse_args()

    ckpt_a = Path(args.ckpt_a)
    ckpt_b = Path(args.ckpt_b)

    sd_a = load_state_dict(ckpt_a)
    sd_b = load_state_dict(ckpt_b)

    compare_state_dicts(sd_a, sd_b, name_a=args.name_a, name_b=args.name_b)


if __name__ == "__main__":
    main()
