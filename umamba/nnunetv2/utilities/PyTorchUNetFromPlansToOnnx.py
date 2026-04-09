# %% Imports and configuration
1+2
import torch
import os
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaTSEnc_LowDoseContrastSim import (
    nnUNetTrainerUMambaTSEnc_LowDoseContrastSim,
)

# Paths
pthModelPath = "/Volumes/X10Pro/AWIBuffer/NetModels/PyTorch/UMambaTSEnc_LowDoseContrastSim_final.pth"
outputDir = "/Volumes/X10Pro/AWIBuffer/NetModels/Onnx/"

# Derive output filename: strip ".pth" -> ".onnx"
baseName = os.path.splitext(os.path.basename(pthModelPath))[0] + ".onnx"
onnxOutputPath = os.path.join(outputDir, baseName)

print(f"Input:  {pthModelPath}")
print(f"Output: {onnxOutputPath}")

# %% Load checkpoint and reconstruct model from init_args
device = (
    torch.device("cuda:0") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

checkpoint = torch.load(pthModelPath, map_location=device, weights_only=False)
init_args = checkpoint['init_args']

# Reconstruct the model architecture from saved plans
plans_manager = PlansManager(init_args['plans'])
configuration_manager = plans_manager.get_configuration(init_args['configuration'])
dataset_json = init_args['dataset_json']
num_input_channels = len(dataset_json['channel_names'])

model = nnUNetTrainerUMambaTSEnc_LowDoseContrastSim.build_network_architecture(
    plans_manager, dataset_json, configuration_manager,
    num_input_channels, enable_deep_supervision=False,
)

# Load saved weights (strip "module." prefix from DDP if present)
state_dict = {}
for k, v in checkpoint['network_weights'].items():
    key = k[7:] if k.startswith('module.') else k
    state_dict[key] = v
model.load_state_dict(state_dict)

model = model.to(device).float()
model.eval()

# Optional: compile for potential speedup
model = torch.compile(model)
print(f"Model class: {model.__class__.__name__}")

# %% Test forward pass with dummy input
dummy_input = torch.randn(1, 5, 512, 512, device=device, dtype=torch.float32)

with torch.inference_mode():
    result = model(dummy_input)

# Handle deep supervision (returns list) vs single output
if isinstance(result, (list, tuple)):
    print(f"Output: list of {len(result)} tensors, first shape: {result[0].shape}")
else:
    print(f"Output shape: {result.shape}")

# %% Export to ONNX
print(f"Exporting ONNX to: {onnxOutputPath}")

# Move model and input to CPU for ONNX export compatibility
model_cpu = model.cpu()
dummy_input_cpu = torch.randn(1, 5, 512, 512, dtype=torch.float32)

# Static axes only — Mathematica's ONNX importer requires a fully static graph.
torch.onnx.export(
    model_cpu,
    dummy_input_cpu,
    onnxOutputPath,
    export_params=True,
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
)

print(f"ONNX model exported to: {onnxOutputPath}")

# %% Consolidate external weights into a single file
import onnx

onnx_model = onnx.load(onnxOutputPath, load_external_data=True)

# Internalize all external tensors so the .onnx is self-contained
for tensor in onnx_model.graph.initializer:
    if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
        tensor.ClearField("data_location")

# Promote 0-d scalar tensors to 1-d shape [1] for Mathematica compatibility
import numpy as np
from onnx import numpy_helper
for tensor in onnx_model.graph.initializer:
    arr = numpy_helper.to_array(tensor)
    if arr.ndim == 0:
        print(f"  Promoting scalar {tensor.name} ({arr.item()}) to shape [1]")
        new_tensor = numpy_helper.from_array(arr.reshape(1), name=tensor.name)
        tensor.CopyFrom(new_tensor)

# Remove weight/parameter entries from value_info — they are already
# described in graph.initializer and their non-batch shapes confuse
# Mathematica's batch dimension check.
init_names = {init.name for init in onnx_model.graph.initializer}
input_names = {inp.name for inp in onnx_model.graph.input}
to_remove = []
for vi in onnx_model.graph.value_info:
    if vi.name in init_names or vi.name in input_names:
        to_remove.append(vi)
    else:
        t = vi.type.tensor_type
        if t.HasField("shape") and len(t.shape.dim) == 0:
            # Promote scalar value_info to [1]
            t.shape.dim.add().dim_value = 1
for vi in to_remove:
    onnx_model.graph.value_info.remove(vi)
print(f"  Removed {len(to_remove)} weight/param entries from value_info")

# Downgrade IR version to 9 for Mathematica compatibility
if onnx_model.ir_version > 9:
    print(f"IR version: {onnx_model.ir_version} -> 9")
    onnx_model.ir_version = 9

onnx.save(onnx_model, onnxOutputPath)

# Clean up external data file if it exists
externalDataPath = onnxOutputPath + ".data"
if os.path.exists(externalDataPath):
    os.remove(externalDataPath)
    print(f"Removed external data file: {externalDataPath}")

# %% Validate the exported ONNX model
onnx_model = onnx.load(onnxOutputPath)
onnx.checker.check_model(onnx_model)
print("ONNX model validation passed.")
print(f"File size: {os.path.getsize(onnxOutputPath) / 1e6:.1f} MB")
