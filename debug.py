import torch
import tensorrt
import onnxruntime as rt

import tensorrt
print(tensorrt.__version__)
assert tensorrt.Builder(tensorrt.Logger())


sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 8
providers = rt.get_available_providers()
ort_session = rt.InferenceSession(
    "gfpgan_1.4.onnx", providers=providers, session_options=sess_options
)
