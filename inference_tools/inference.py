import cv2
import numpy as np
import onnxruntime as ort
import pycuda.autoinit  # noqa: F401  # initializes CUDA context for pycuda
import pycuda.driver as cuda
import tensorrt as trt
import torch

from config import MEAN, STD, TARGET_SIZE, ensure_project_root_on_path

ensure_project_root_on_path()

from models.pidnet import get_pred_model  # noqa: E402


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Resize + normalize image into NCHW float32 batch"""
    img = cv2.resize(image_bgr, TARGET_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img -= MEAN
    img /= STD
    img = img.transpose(2, 0, 1)[None]  # (1,3,H,W)
    return np.ascontiguousarray(img, dtype=np.float32)


class TensorRTInferencer:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self._d_input = None
        self._d_output = None
        self._out_shape = None

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        img = preprocess(image_bgr)
        self.context.set_input_shape(self.input_name, img.shape)

        if self._d_input is None:
            self._d_input = cuda.mem_alloc(img.nbytes)
            self._out_shape = tuple(self.context.get_tensor_shape(self.output_name))
            out_bytes = int(np.prod(self._out_shape) * 4)
            self._d_output = cuda.mem_alloc(out_bytes)

        cuda.memcpy_htod_async(self._d_input, img, self.stream)
        self.context.set_tensor_address(self.input_name, int(self._d_input))
        self.context.set_tensor_address(self.output_name, int(self._d_output))
        self.context.execute_async_v3(self.stream.handle)

        out = np.empty(self._out_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(out, self._d_output, self.stream)
        self.stream.synchronize()
        return out.argmax(1)[0]


class PyTorchInferencer:
    def __init__(self, checkpoint_path: str):
        model = get_pred_model("pidnet_s", num_classes=2)
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        self.model = model

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        img = preprocess(image_bgr)
        with torch.no_grad():
            out = self.model(torch.from_numpy(img))[0]
        return out.argmax(1)[0].cpu().numpy()


class ONNXInferencer:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        img = preprocess(image_bgr)
        out = self.session.run(None, {"input": img})[0]
        return out.argmax(1)[0]
