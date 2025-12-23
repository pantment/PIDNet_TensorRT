#!/usr/bin/env python3
"""Inspect TensorRT engine IO tensor names and shapes."""
import argparse
from pathlib import Path

import tensorrt as trt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display TensorRT engine inputs/outputs.")
    parser.add_argument("engine", type=str, help="Path to engine file (.engine)")
    parser.add_argument("--verbose", action="store_true", help="Use verbose TensorRT logger")
    return parser.parse_args()


def main():
    args = parse_args()
    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)

    log_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    logger = trt.Logger(log_level)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    engine = runtime.deserialize_cuda_engine(engine_data)
    print(f"Engine: {engine_path}")
    print("num_io_tensors:", engine.num_io_tensors)

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        print(f"{i}: {name} | {mode.name} | shape={shape} | dtype={dtype}")


if __name__ == "__main__":
    main()
