# %% Imports and configuration
import torch
import dill
import os

# Paths
pthModelPath = "/Volumes/X10Pro/AWIBuffer/NetModels/PyTorch/UXlstmBot-nnUNetPlans_2d-reduced3-DC_and_CE_loss-w-1-20-40-dill.pth"
outputDir = "/Volumes/X10Pro/AWIBuffer/NetModels/"

# Derive output filename: strip "-dill.pth" -> ".onnx"
baseName = os.path.basename(pthModelPath).replace("-dill.pth", ".onnx")
onnxOutputPath = os.path.join(outputDir, baseName)

print(f"Input:  {pthModelPath}")
print(f"Output: {onnxOutputPath}")

# %% Load the dill-pickled model
device = (
    torch.device("cuda:0") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

model = torch.load(pthModelPath, map_location=device, pickle_module=dill, weights_only=False)
model = model.to(device)
model.eval()
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

# Use dynamo exporter (legacy TorchScript fails on ViLBlock's dynamic group_norm).
# Dynamo exports weights to a separate .onnx.data file by default, so we
# consolidate them into a single self-contained .onnx file afterward.
# No dynamic_axes: Mathematica's ONNX importer requires a fully static graph.
# The mLSTM causal mask and other ViLBlock internals produce batch-varying
# intermediates that Mathematica incorrectly treats as constants when dynamic
# axes are present.
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

# %% Consolidate external weights and fix batch dimension for Mathematica
import onnx
import numpy as np
from onnx import numpy_helper

onnx_model = onnx.load(onnxOutputPath, load_external_data=True)

# Internalize all external tensors so the .onnx is self-contained
for tensor in onnx_model.graph.initializer:
    if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
        tensor.ClearField("data_location")

# Fix batch dimension for Mathematica compatibility.
# The MultiHeadLayerNorm in the ViLBlock merges batch (B=1) with sequence (S=16)
# via reshape [1,16,4,256] -> [16,4,256] for group_norm, then reshapes back.
# Mathematica requires dim 0 = batch throughout. We rewrite this section to
# keep batch=1 as a separate leading dimension:
#   [16,4,256] -> [1,16,4,256]  (InstanceNorm treats 16 as batch, 4 as channels)
#   [16,1024]  -> [1,16,1024]   (intermediate flatten)
print("Fixing batch dimension in MultiHeadLayerNorm nodes...")

def get_initializer(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None

def set_initializer_shape(model, name, new_shape):
    init = get_initializer(model, name)
    if init is not None:
        old_shape = np.frombuffer(init.raw_data, dtype=np.int64)
        init.raw_data = np.array(new_shape, dtype=np.int64).tobytes()
        print(f"  {name}: {old_shape} -> {new_shape}")
        return True
    return False

# Fix the Reshape constants that merge/split the batch dimension.
# Node 170: Reshape to [16, 4, 256] -> change to [1, 16, 4, 256]
# Node 172: Reshape to [16, 1024]   -> change to [1, 16, 1024]
for node in onnx_model.graph.node:
    if node.op_type == "Reshape":
        init = get_initializer(onnx_model, node.input[1])
        if init is not None:
            shape = np.frombuffer(init.raw_data, dtype=np.int64).copy()
            # Identify the problematic reshapes: first dim is B*S (not 1)
            if len(shape) >= 2 and shape[0] != 1 and shape[0] != -1:
                new_shape = np.concatenate([[1], shape])
                init.raw_data = new_shape.astype(np.int64).tobytes()
                # Update the dims metadata to match the new number of elements
                del init.dims[:]
                init.dims.append(len(new_shape))
                print(f"  Reshape {node.output[0]}: {shape} -> {list(new_shape)}")

# Now the InstanceNorm at node 171 receives [1,16,4,256] instead of [16,4,256].
# ONNX InstanceNorm normalizes over spatial dims (dims 2+), treating dim 0 as
# batch and dim 1 as channels. With input [1,16,4,256]:
#   batch=1, channels=16, spatial=[4,256] — this matches the original intent
#   (group_norm with 16 groups over 1024 features = InstanceNorm over 16 channels).
# The scale/bias have shape [16] which matches channels=16. No changes needed.

# %% Decompose operators unsupported by Mathematica's ONNX backend
# LayerNormalization (opset 17), Einsum (opset 12), and CumSum (opset 11)
# are replaced with primitive ops that Mathematica supports.
from onnx import helper, shape_inference

print("Decomposing unsupported operators for Mathematica...")

def make_unique(prefix, existing):
    """Generate a unique name."""
    i = 0
    while f"{prefix}_{i}" in existing:
        i += 1
    name = f"{prefix}_{i}"
    existing.add(name)
    return name

used_names = set()
for n in onnx_model.graph.node:
    for o in n.output:
        used_names.add(o)
for init in onnx_model.graph.initializer:
    used_names.add(init.name)

new_nodes = []
new_initializers = []

for node in onnx_model.graph.node:

    if node.op_type == "LayerNormalization":
        # Decompose: output = (x - mean) / sqrt(var + eps) * scale + bias
        x_name = node.input[0]
        scale_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 else None
        out_name = node.output[0]
        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = attr.f
            if attr.name == "axis":
                axis = attr.i

        mean = make_unique("ln_mean", used_names)
        centered = make_unique("ln_centered", used_names)
        sq = make_unique("ln_sq", used_names)
        var = make_unique("ln_var", used_names)
        eps_const = make_unique("ln_eps", used_names)
        var_eps = make_unique("ln_var_eps", used_names)
        std = make_unique("ln_std", used_names)
        normed = make_unique("ln_normed", used_names)
        scaled = make_unique("ln_scaled", used_names)

        # eps constant
        eps_tensor = numpy_helper.from_array(
            np.array([eps], dtype=np.float32), name=eps_const
        )
        new_initializers.append(eps_tensor)

        # Axes constant for ReduceMean (axis=-1 -> last dim)
        axes_name = make_unique("ln_axes", used_names)
        axes_tensor = numpy_helper.from_array(
            np.array([axis], dtype=np.int64), name=axes_name
        )
        new_initializers.append(axes_tensor)

        new_nodes.append(helper.make_node("ReduceMean", [x_name, axes_name], [mean], keepdims=1))
        new_nodes.append(helper.make_node("Sub", [x_name, mean], [centered]))
        new_nodes.append(helper.make_node("Mul", [centered, centered], [sq]))
        new_nodes.append(helper.make_node("ReduceMean", [sq, axes_name], [var], keepdims=1))
        new_nodes.append(helper.make_node("Add", [var, eps_const], [var_eps]))
        new_nodes.append(helper.make_node("Sqrt", [var_eps], [std]))
        new_nodes.append(helper.make_node("Div", [centered, std], [normed]))
        new_nodes.append(helper.make_node("Mul", [normed, scale_name], [scaled]))
        if bias_name:
            new_nodes.append(helper.make_node("Add", [scaled, bias_name], [out_name]))
        else:
            # rename scaled -> out_name
            new_nodes.append(helper.make_node("Identity", [scaled], [out_name]))

        print(f"  Decomposed LayerNormalization -> {out_name}")

    elif node.op_type == "InstanceNormalization":
        # Check if this is the MultiHeadLayerNorm InstanceNorm (whose input
        # shape was changed from [16,4,256] to [1,16,4,256] by our batch fix).
        # The scale/bias have shape [4] (NH=4 channels), but with our reshaped
        # input [1,16,4,256], ONNX InstanceNorm would interpret dim1=16 as
        # channels. We decompose this specific InstanceNorm into primitives
        # that normalize over the last dim (256) for each (batch, seq, head).
        #
        # For regular InstanceNorm nodes (encoder/decoder), we keep them as-is
        # since they already have batch=1 in dim 0.
        x_name = node.input[0]
        scale_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 else None
        out_name = node.output[0]
        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = attr.f

        # Check if scale has 4 elements (MultiHeadLayerNorm) vs larger (regular)
        scale_init = get_initializer(onnx_model, scale_name)
        is_multihead_instnorm = (scale_init is not None and
                                 len(numpy_helper.to_array(scale_init)) == 4)

        if is_multihead_instnorm:
            # Decompose: normalize over last dim (axis=3 for [1,16,4,256])
            # Then apply scale[4] and bias[4] which broadcast over dim 2.
            mean = make_unique("in_mean", used_names)
            centered = make_unique("in_centered", used_names)
            sq = make_unique("in_sq", used_names)
            var = make_unique("in_var", used_names)
            eps_const = make_unique("in_eps", used_names)
            var_eps = make_unique("in_var_eps", used_names)
            std = make_unique("in_std", used_names)
            normed = make_unique("in_normed", used_names)

            eps_tensor = numpy_helper.from_array(
                np.array([eps], dtype=np.float32), name=eps_const
            )
            new_initializers.append(eps_tensor)

            axes_name = make_unique("in_axes", used_names)
            axes_tensor = numpy_helper.from_array(
                np.array([-1], dtype=np.int64), name=axes_name
            )
            new_initializers.append(axes_tensor)

            # Unsqueeze scale [4] -> [1,1,4,1] for broadcasting with [1,16,4,256]
            scale_us_name = make_unique("in_scale_us", used_names)
            scale_us_axes = make_unique("in_scale_us_axes", used_names)
            scale_us_axes_tensor = numpy_helper.from_array(
                np.array([0, 1, 3], dtype=np.int64), name=scale_us_axes
            )
            new_initializers.append(scale_us_axes_tensor)
            new_nodes.append(helper.make_node("Unsqueeze", [scale_name, scale_us_axes], [scale_us_name]))

            new_nodes.append(helper.make_node("ReduceMean", [x_name, axes_name], [mean], keepdims=1))
            new_nodes.append(helper.make_node("Sub", [x_name, mean], [centered]))
            new_nodes.append(helper.make_node("Mul", [centered, centered], [sq]))
            new_nodes.append(helper.make_node("ReduceMean", [sq, axes_name], [var], keepdims=1))
            new_nodes.append(helper.make_node("Add", [var, eps_const], [var_eps]))
            new_nodes.append(helper.make_node("Sqrt", [var_eps], [std]))
            new_nodes.append(helper.make_node("Div", [centered, std], [normed]))

            scaled = make_unique("in_scaled", used_names)
            new_nodes.append(helper.make_node("Mul", [normed, scale_us_name], [scaled]))

            if bias_name:
                bias_us_name = make_unique("in_bias_us", used_names)
                bias_us_axes2 = make_unique("in_bias_us_axes", used_names)
                bias_us_axes_tensor2 = numpy_helper.from_array(
                    np.array([0, 1, 3], dtype=np.int64), name=bias_us_axes2
                )
                new_initializers.append(bias_us_axes_tensor2)
                new_nodes.append(helper.make_node("Unsqueeze", [bias_name, bias_us_axes2], [bias_us_name]))
                new_nodes.append(helper.make_node("Add", [scaled, bias_us_name], [out_name]))
            else:
                new_nodes.append(helper.make_node("Identity", [scaled], [out_name]))

            print(f"  Decomposed InstanceNorm (MultiHeadLayerNorm) -> {out_name}")
        else:
            # Regular InstanceNorm — keep as-is
            new_nodes.append(node)

    elif node.op_type == "Einsum":
        # Equation "...ab,acb->...ac" with shapes [1,16,256,4] x [256,4,4]
        # = for each head a: output[...,a,:] = input[...,a,:] @ weight[a,:,:]^T
        # Decompose: Transpose x dims 1↔2, MatMul with transposed weight, Transpose back.
        #   x [1,16,256,4] -> [1,256,16,4] @ w^T [256,4,4] -> [1,256,16,4] -> [1,16,256,4]
        eq = ""
        for attr in node.attribute:
            if attr.name == "equation":
                eq = attr.s.decode()
        assert eq == "...ab,acb->...ac", f"Unexpected Einsum equation: {eq}"

        x_name = node.input[0]    # [1, 16, 256, 4]
        w_name = node.input[1]    # [256, 4, 4]
        out_name = node.output[0]

        # Step 1: Transpose x: [1,16,256,4] -> [1,256,16,4] (swap dims 1,2)
        x_t = make_unique("einsum_xt", used_names)
        new_nodes.append(helper.make_node("Transpose", [x_name], [x_t], perm=[0, 2, 1, 3]))

        # Step 2: Transpose weight [NH,OUT_D,D] -> [NH,D,OUT_D] (swap dims 1,2)
        w_t = make_unique("einsum_wt", used_names)
        new_nodes.append(helper.make_node("Transpose", [w_name], [w_t], perm=[0, 2, 1]))

        # Step 3: MatMul: [1,256,16,4] @ [256,4,4] -> [1,256,16,4] (256 broadcasts)
        mm_out = make_unique("einsum_mm", used_names)
        new_nodes.append(helper.make_node("MatMul", [x_t, w_t], [mm_out]))

        # Step 4: Transpose back: [1,256,16,4] -> [1,16,256,4]
        new_nodes.append(helper.make_node("Transpose", [mm_out], [out_name], perm=[0, 2, 1, 3]))

        print(f"  Decomposed Einsum '{eq}' -> {out_name}")

    else:
        new_nodes.append(node)

# Replace all nodes
del onnx_model.graph.node[:]
onnx_model.graph.node.extend(new_nodes)

# Add new initializers
for init in new_initializers:
    onnx_model.graph.initializer.append(init)

# Clear stale value_info and re-infer shapes
del onnx_model.graph.value_info[:]
onnx_model = shape_inference.infer_shapes(onnx_model)

# Downgrade IR version from 10 to 9 for Mathematica compatibility.
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