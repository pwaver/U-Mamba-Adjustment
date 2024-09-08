1+1
import torch
import torch.nn as nn
import onnx
import dill
from onnx2torch import convert

import os
rootPath = '/mnt/SliskiDrive/AWI/AWIBuffer/' if os.name == 'posix' else '/Volumes/Crucial X8/AWI/Data/'

# onnxPath = rootPath + "UMambaBot-plans_unet_edge8_epochs250_2d-DC_and_CE_loss-w-1-20-20.onnx"
onnxPath = rootPath + "PlainConvUNet-nnUNetPlans_3d_fullres-unspecified-loss.onnx"

modelPerOnnx = convert(onnxPath)

gpuDevice = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

modelPerOnnx = modelPerOnnx.to(gpuDevice)

# For grins export back to onnx

random_tensor = torch.randn(1, 5, 512, 512, device=gpuDevice, dtype=torch.float32)
random_tensor.shape
# random_tensor = random_tensor.to(gpuDevice)

modelPerOnnx.eval()
with torch.inference_mode():
    result = modelPerOnnx(random_tensor)

result.shape

onnxOutputPath = onnxPath.replace(".onnx", "-torch-onnx.onnx")
with torch.inference_mode():
    torch.onnx.export(modelPerOnnx, random_tensor, onnxOutputPath, 
                  export_params=True, opset_version=18, 
                  do_constant_folding=True, verbose=True,
                  input_names=['input'], output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, 
                  training=torch.onnx.TrainingMode.EVAL)

# torch.onnx.export(modelPerOnnx, random_tensor, onnxOutputPath, 
#                           export_params=True, opset_version=18, 
#                           do_constant_folding=True, verbose=True,
#                           input_names=['input'], output_names=['output'], training=torch.onnx.TrainingMode.EVAL)


# torch.onnx.export(modelPerOnnx, random_tensor, onnxOutputPath, 
#                           export_params=True, opset_version=15, 
#                           do_constant_folding=True, verbose=True,
#                           input_names=['input'], output_names=['output'],
#                           dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, training=torch.onnx.TrainingMode.EVAL)



torchModelPath = onnxPath.replace(".onnx", "-torch-onnx.pt")

torch.save(modelPerOnnx, torchModelPath)

checkModel = torch.load(torchModelPath, weights_only=False)

checkModel.eval()
with torch.inference_mode():
    result = checkModel(random_tensor)

result.shape


# Define example inputs with different batch sizes
example_input_1 = torch.randn(2, 5, 512, 512).to(gpuDevice)
example_input_2 = torch.randn(4, 5, 512, 512).to(gpuDevice)

# Scripted does not work:
scriptedModelperOnnx = torch.jit.script(modelPerOnnx)
# scripted_model.save("/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript.pt")

# Try traced # update: traced after re-import from onnx seems to work
tracedModelperOnnx = torch.jit.trace(modelPerOnnx, example_input_1, check_trace=False)

tracedModelPath = onnxPath.replace(".onnx", "-torchscript-traced-onnx.pt")

tracedModelperOnnx.save(tracedModelPath)

# Read back in traced model path and see if works

# tracedModelPath = "/home/ubuntu/U-Mamba-Adjustment/data/nets/UMambaBot-plans_unet_edge8_2d-DC_and_CE_loss-w-1-20-20-torchscript-traced-onnx.pt"

tracedModel = torch.jit.load(tracedModelPath)

tracedModel = tracedModel.to(gpuDevice)

# random_tensor = torch.randn(2, 5, 512, 512)
# random_tensor.shape
# random_tensor = random_tensor.to(gpuDevice)

# tracedModel = nn.DataParallel(tracedModel)
tracedModel.eval()
with torch.inference_mode():
    result = tracedModel(example_input_2)

result.shape


# Trace the model with dynamic batch size
tracedModelperOnnx = torch.jit.trace(modelPerOnnx, example_input_1, check_trace=False)

# Test the traced model with a different batch size
tracedModelperOnnx(example_input_2)