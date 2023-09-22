import time
import torch
import pathlib
import onnxruntime

from typing import Tuple
from utils.logging import logging

# Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Engine Class
class OnnxEngine:
    def __init__(
        self,
        onnx_model_path: str,
        image_resolution: Tuple[int, int]
    ):
        log.info(f'[ENGINE]: Started ONNX engine loading!')
        self.onnx_model_path: pathlib.Path = pathlib.Path(onnx_model_path)        
        self.image_resolution = image_resolution
        log.info("[ENGINE]: Onnx engine inference object initialized.")

    def warmup(self, warmup_iters: int) -> None:
        log.info(f'[ONNX]: Model warm up for {warmup_iters} iteration/s.')
        dummy_input = torch.rand((1, 3, self.patch_w, self.patch_h))

        for i in range(1, warmup_iters):
            self.predict_image(dummy_input, None, False)

    def predict_image(
        self,
        input_tensor: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, int]:
        """
        'input_tensor' expected to contain 4 dimensional data - (N, C, H, W)
        """
        input_numpy_array = input_tensor.numpy().astype("float32")
        options = onnxruntime.SessionOptions()
        #options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC

        providers = [
            ('CUDAExecutionProvider', {
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_use_max_workspace': '1',
                'enable_cuda_graph': '1',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
        onnx_session = onnxruntime.InferenceSession(self.onnx_model_path, None, providers)

        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name

        start_time = time.time()
        model_output = onnx_session.run([output_name], { input_name: input_numpy_array })
        end_time = time.time()

        inference_time = end_time - start_time
        log.info(f'[ONNX]: Inference prediction took {(inference_time * 1000):.2f} ms.')
        return (model_output[0], inference_time)
