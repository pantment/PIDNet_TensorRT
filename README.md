# PIDNet_TensorRT

This repository provides end-to-end workflows for optimizing PIDNet semantic segmentation models with TorchScript, ONNX, and TensorRT, plus utilities to validate and debug the exported artifacts.

## Highlights
- Reference scripts to export PIDNet checkpoints to TorchScript, ONNX, and TensorRT engines.
- Ready-to-run benchmarking (`models/speed/pidnet_speed.py`) and inference (`tools/inference.py`) utilities.
- TensorRT-specific validation helpers to compute mIoU, visualize overlays, and inspect engine metadata.
- Extra tooling (`tools/custom_export.py`, `tools/compare_checkpoints.py`) to customize export pipelines and compare checkpoints on disk.
- A dedicated `forward_export` path inside `models/pidnet.py` with branchless logic (no runtime `if` checks) so TensorRT can build faster, fully-static engines.

## Prerequisites
### Device: RTX 3050
* CUDA: 12.0 (driver: 525) 
* cuDNN: 8.9
* TensorRT: 8.6
* PyCUDA
  
### Device: NVIDIA Jetson Nano
* Jetpack: 4.6.2
* PyCUDA
  
## Usage
### 0. Setup
* Clone this repository and download the pretrained model from the official [PIDNet](https://github.com/XuJiacong/PIDNet/tree/main) repository.
* (Optional for validation) Prepare a dataset folder with the following layout so the TensorRT validation scripts can read pairs:
  ```
  dataset_root/
    images/*.png
    masks/*.png   # binary masks; >127 is treated as foreground
  ```

### 1. Export the model
You can use the original export helper or the new customizable pipeline:

**Quick export (`tools/export.py`):**

For TorchScript:
````bash
python tools/export.py --a pidnet-s --p ./pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt --f torchscript
````
For ONNX:
````bash
python tools/export.py --a pidnet-s --p ./pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt --f onnx
````
For TensorRT (using the above ONNX model):
```bash
trtexec --onnx=path/to/onnx/model --saveEngine=path/to/engine 
```
### 2. Inference
```bash
python tools/inference.py --f pytorch
```

### 3. Custom export helper
`tools/custom_export.py` builds PIDNet, loads your checkpoint, and writes TorchScript + ONNX files in one go:
```bash
python tools/custom_export.py \
  --weights ./pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt \
  --num-classes 2 \
  --img-height 1024 \
  --img-width 2048 \
  --output-dir ./exports
```
Adjust `--num-classes` and resolution if you trained PIDNet for a different setting.
Under the hood this script calls `PIDNet.forward_export`, a stripped-down forward function with the conditional heads removed (see `models/pidnet.py`). By eliminating `if self.augment` branches we ensure the traced graph stays static, which improves TensorRT engine build times and runtime latency.

### 4. Build TensorRT engines
Use the exported ONNX file with `trtexec` (or TensorRT Python APIs) to build a `.engine`:
```bash
trtexec --onnx=./exports/pidnet_s_export.onnx --saveEngine=./exports/pidnet_s_export.engine
```
`inference_tools/trt_inspect.py` can be used to verify the resulting engine IO details:
```bash
python inference_tools/trt_inspect.py ./exports/pidnet_s_export.engine
```

### 5. Validate TensorRT exports
The `inference_tools/` folder adds several validation workflows that run PyTorch, ONNX, and TensorRT variants side-by-side.

**mIoU computation on a labeled dataset**
```bash
python inference_tools/trt_validate_miou.py \
  --dataset /path/to/dataset_root \
  --pt ./pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt \
  --onnx ./exports/pidnet_s_export.onnx \
  --engine ./exports/pidnet_s_export.engine
```
This prints per-class IoU and overall mIoU for each export given a dataset folder containing matching `images` and `masks`.

**Visual overlay generation**
```bash
python inference_tools/trt_validate_all.py \
  --images ./samples \
  --output ./samples/overlays \
  --pt ./pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt \
  --onnx ./exports/pidnet_s_export.onnx \
  --engine ./exports/pidnet_s_export.engine
```
For every input image, the script resizes predictions back to the original resolution and saves blended overlays (`*_pt.png`, `*_onnx.png`, `*_trt.png`) for quick qualitative checks.

All validation utilities share normalization / resize parameters defined in `inference_tools/config.py`. Update `TARGET_SIZE`, `NUM_CLASSES`, or `DEFAULT_DATASET_ROOT` there if your dataset differs.

**Checkpoint comparison utility**
When you need to verify that two checkpoints are equivalent, run:
```bash
python tools/compare_checkpoints.py \
  ./pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt \
  ./exports/other_model.pt \
  --name-a baseline --name-b optimized
```
The script reports missing keys, tensor shape mismatches, and statistics about parameter differences.

### 6. Speed Measurement
* Measure the inference speed of PIDNet-S for Cityscapes:
````bash
python models/speed/pidnet_speed.py --f all
````
|             | FPS         | % increase |
| :---------- | :---------: |:---------: |
| PyTorch     | 24.72       | -          |
| TorchScript | 27.09       | 9.59       |
| ONNX (with TensorRT EP)   | 33.52       | 35.60      |
| TensorRT    | 32.93       | 33.21      |

speed test is performed on a single Nvidia GeForce RTX 3050 GPU

### Acknowledgement
1. [PIDNet](https://github.com/XuJiacong/PIDNet/tree/main)
