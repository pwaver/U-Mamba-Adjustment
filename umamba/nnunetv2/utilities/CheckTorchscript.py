1+1
import torch
import onnx
import dill
from onnx2torch import convert

model = torch.jit.load("/home/ubuntu/U-Mamba-Adjustment/data/nets//UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript.pt")

onnxPath = "/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20.onnx"

torch_model_1 = convert(onnxPath)

torch.jit.save(torch_model_1, "/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript.pt")

scripted_model = torch.jit.script(torch_model_1)
scripted_model.save("/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript.pt")

random_tensor = torch.randn(1, 5, 512, 512)
random_tensor.shape

torch_model_1.compiled = torch.compile(torch_model_1)

traced_model = torch.jit.trace(torch_model_1, random_tensor)
traced_model.save("/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript-traced-compiled.pt")

traced_model_readback = torch.jit.load("/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript-traced.pt")

# test readback model by passign it random_tensor
result = traced_model_readback(random_tensor)

onnxModel = onnx.load(onnxPath)

print(onnxModel)

torch.backends.mps.is_available()
torch.cuda.is_available()
device = torch.device("cuda")

traced_model_readback=traced_model_readback.to(device)

random_tensor = random_tensor.to(device)

result = traced_model_readback(random_tensor)


# Test dill model import and infer
1+1
import torch
# import onnx
import dill

torch.cuda.is_available()
device = torch.device("cuda")

# dillModelPath="/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-dill.pth"
dillModelPath="/home/ubuntu/onnx-nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-dill.pth"


dillModel = torch.load(dillModelPath)

print(dillModel)

dillModel = dillModel.to(device)

random_tensor = torch.randn(1, 5, 512, 512)
random_tensor.shape
random_tensor = random_tensor.to(device)

result = dillModel(random_tensor)
result.shape

# Test torschscript traced model import and infer
1+1
import torch
import onnx

torch.cuda.is_available()
device = torch.device("cuda")

traced_model_readback = torch.jit.load("/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript-traced.pt")

device = torch.device("cuda")

traced_model_readback=traced_model_readback.to(device)

random_tensor = random_tensor.to(device)

result = traced_model_readback(random_tensor)

# try with onnx convert
from onnx2torch import convert

onnxPath = "/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20.onnx"

modelPerOnnx = convert(onnxPath)

modelPerOnnx = modelPerOnnx.to(device)

random_tensor = torch.randn(1, 5, 512, 512)
random_tensor.shape
random_tensor = random_tensor.to(device)

result = modelPerOnnx(random_tensor)
result.shape

# Scripted does not work
scriptedModelperOnnx = torch.jit.script(modelPerOnnx)
scripted_model.save("/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript.pt")

# Try traced # update: traced after re-import from onnx seems to work
tracedModelperOnnx = torch.jit.trace(modelPerOnnx, random_tensor)
# Error types: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
# TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results)
# TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.

tracedModelPath = "/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript-traced-onnx.pt"
tracedModelperOnnx.save(tracedModelPath)

# Read back in traced model path and see if works
import torch

torch.cuda.is_available()
device = torch.device("cuda")

tracedModelPath = "/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript-traced-onnx.pt"

tracedModel = torch.jit.load(tracedModelPath)

tracedModel = tracedModel.to(device)

random_tensor = torch.randn(1, 5, 512, 512)
random_tensor.shape
random_tensor = random_tensor.to(device)

result = tracedModel(random_tensor)
result.shape