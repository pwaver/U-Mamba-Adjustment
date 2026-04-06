# %% Imports and configuration
import torch
import dill
import numpy as np
import h5py
import os
from matplotlib import pyplot as plt

# --- Configurable parameters ---
datasetKey =  "Napari_48_rev" # "Angios_096_rev"  #/ "Napari_36_rev" / "Napari_29_rev"
datasetKey = "LPS_Patient_21_series8_9B3ABD3C"
h5_filename = "AngiogramsDistilledUInt8List.h5"  # or "WebknossosAngiogramsRevisedUInt8List.h5"
pytorch_model = "UMambaBot3-nnUNetPlans_2d-reduced3-lowdosesim-DC_and_CE_loss-w-1-10-20-dill.pth"

# --- Cross-platform path resolution ---
h5_candidates = [
    f"/Volumes/X10Pro/AWIBuffer/Angiostore/{h5_filename}",    # macOS
    f"/media/billb/WMDPP/Angiostore/{h5_filename}",           # Ubuntu / WDMPP
]
h5_path = next((p for p in h5_candidates if os.path.exists(p)), None)
if h5_path is None:
    raise FileNotFoundError(
        f"Could not find {h5_filename} at any candidate location:\n"
        + "\n".join(f"  - {p}" for p in h5_candidates)
    )

pytorch_model_folder_candidates = [
    "/Volumes/X10Pro/AWIBuffer/NetModels/PyTorch",            # macOS / X10Pro
    "/mnt/SliskiDrive/AWI/AWIBuffer/NetModels/PyTorch",       # Ubuntu
]
pytorch_model_folder = next((d for d in pytorch_model_folder_candidates if os.path.isdir(d)), None)
if pytorch_model_folder is None:
    raise FileNotFoundError(
        f"Could not find PyTorch model folder at any candidate location:\n"
        + "\n".join(f"  - {d}" for d in pytorch_model_folder_candidates)
    )
pytorch_model_path = os.path.join(pytorch_model_folder, pytorch_model)

print(f"H5 path:    {h5_path}")
print(f"Model path: {pytorch_model_path}")

# %% Load model
device = (
    torch.device("cuda:0") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

model = torch.load(pytorch_model_path, map_location=device, pickle_module=dill, weights_only=False)
model = model.to(device)
model.eval()
print(f"Model class: {model.__class__.__name__}")

# %% Load angiogram from HDF5
with h5py.File(h5_path, 'r') as f:
    available_keys = list(f.keys())
    print(f"Available keys ({len(available_keys)}): {available_keys[:10]}{'...' if len(available_keys) > 10 else ''}")
    angiograms_gt = np.array(f[datasetKey])

print(f"Angiogram '{datasetKey}': shape={angiograms_gt.shape}, dtype={angiograms_gt.dtype}")

# %% Z-score normalize and build 5-frame windows
def zNormalizeArray(arr):
    mean = np.mean(arr)
    std = np.std(arr) + 1e-4
    return (arr - mean) / std

z_data = zNormalizeArray(angiograms_gt.astype(np.float32))

n_frames = z_data.shape[0]
windows = np.stack([z_data[i - 2 : i + 3] for i in range(2, n_frames - 2)], axis=0)
print(f"Built {windows.shape[0]} five-frame windows, shape: {windows.shape}")

# %% Run batched inference
batch_size = 8
n_windows = windows.shape[0]
seg_masks = np.empty((n_windows, 512, 512), dtype=np.float32)

with torch.inference_mode():
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = torch.tensor(windows[start:end], dtype=torch.float32, device=device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        seg_masks[start:end] = probs[:, 2].cpu().numpy()  # vessel layer (index 2)
        if start % 64 == 0:
            print(f"  Processed {end}/{n_windows} windows")

print(f"Segmentation masks: shape={seg_masks.shape}, unique values={np.unique(seg_masks)}")

# %% Show frame 35 as a quick test
fig, (ax_orig, ax_seg) = plt.subplots(1, 2, figsize=(12, 6))
ax_orig.imshow(angiograms_gt[35], cmap='gray')
ax_orig.set_title('Frame 35 — original')
ax_orig.axis('off')
ax_seg.imshow(seg_masks[33], cmap='gray')  # window index 33 → center frame 35
ax_seg.set_title('Frame 35 — segmentation')
ax_seg.axis('off')
plt.tight_layout()
plt.show()

# %% Interactive slider inspector
import ipywidgets as widgets
from IPython.display import display, clear_output

out = widgets.Output()

def show_frame(idx):
    """idx is the window index (0-based); center frame is idx + 2."""
    frame_num = idx + 2
    with out:
        clear_output(wait=True)
        fig, (ax_orig, ax_seg) = plt.subplots(1, 2, figsize=(12, 6))
        ax_orig.imshow(angiograms_gt[frame_num], cmap='gray')
        ax_orig.set_title(f'Frame {frame_num}')
        ax_orig.axis('off')
        ax_seg.imshow(seg_masks[idx], cmap='gray')
        ax_seg.set_title(f'Segmentation (frame {frame_num})')
        ax_seg.axis('off')
        plt.tight_layout()
        plt.show()

frame_slider = widgets.IntSlider(
    value=0, min=0, max=n_windows - 1, step=1,
    description='Frame:', continuous_update=False,
    layout=widgets.Layout(width='80%'),
    style={'description_width': 'initial'}
)
frame_slider.observe(lambda change: show_frame(change['new']), names='value')

display(frame_slider, out)
show_frame(0)

# %% Export to ONNX
import onnx
from onnx import numpy_helper

onnx_model_folder_candidates = [
    "/Volumes/X10Pro/AWIBuffer/NetModels/Onnx",              # macOS / X10Pro
    "/mnt/SliskiDrive/AWI/AWIBuffer/NetModels/Onnx",         # Ubuntu
]
onnx_model_folder = next((d for d in onnx_model_folder_candidates if os.path.isdir(d)), None)
if onnx_model_folder is None:
    raise FileNotFoundError(
        f"Could not find ONNX model folder at any candidate location:\n"
        + "\n".join(f"  - {d}" for d in onnx_model_folder_candidates)
    )

onnx_basename = pytorch_model.replace("-dill.pth", ".onnx")
onnx_output_path = os.path.join(onnx_model_folder, onnx_basename)
print(f"Exporting ONNX to: {onnx_output_path}")

model_cpu = model.cpu()
dummy_input_cpu = torch.randn(1, 5, 512, 512, dtype=torch.float32)

torch.onnx.export(
    model_cpu,
    dummy_input_cpu,
    onnx_output_path,
    export_params=True,
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
)
print(f"ONNX exported: {onnx_output_path}")

# Move model back to original device for continued use
model = model.to(device)

# Consolidate external weights into a single self-contained file
onnx_model = onnx.load(onnx_output_path, load_external_data=True)

for tensor in onnx_model.graph.initializer:
    if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
        tensor.ClearField("data_location")

# Promote 0-d scalar tensors to 1-d shape [1] for Mathematica compatibility
for tensor in onnx_model.graph.initializer:
    arr = numpy_helper.to_array(tensor)
    if arr.ndim == 0:
        print(f"  Promoting scalar {tensor.name} ({arr.item()}) to shape [1]")
        new_tensor = numpy_helper.from_array(arr.reshape(1), name=tensor.name)
        tensor.CopyFrom(new_tensor)

# Remove weight/parameter entries from value_info
init_names = {init.name for init in onnx_model.graph.initializer}
input_names_set = {inp.name for inp in onnx_model.graph.input}
to_remove = []
for vi in onnx_model.graph.value_info:
    if vi.name in init_names or vi.name in input_names_set:
        to_remove.append(vi)
    else:
        t = vi.type.tensor_type
        if t.HasField("shape") and len(t.shape.dim) == 0:
            t.shape.dim.add().dim_value = 1
for vi in to_remove:
    onnx_model.graph.value_info.remove(vi)
print(f"  Removed {len(to_remove)} weight/param entries from value_info")

# Downgrade IR version to 9 for Mathematica compatibility
if onnx_model.ir_version > 9:
    print(f"IR version: {onnx_model.ir_version} -> 9")
    onnx_model.ir_version = 9

onnx.save(onnx_model, onnx_output_path)

# Clean up external data file if it exists
external_data_path = onnx_output_path + ".data"
if os.path.exists(external_data_path):
    os.remove(external_data_path)
    print(f"Removed external data file: {external_data_path}")

onnx.checker.check_model(onnx.load(onnx_output_path))
print(f"ONNX validation passed. File size: {os.path.getsize(onnx_output_path) / 1e6:.1f} MB")

# %% Reimport ONNX model as a check
import onnxruntime as ort

session = ort.InferenceSession(onnx_output_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"ONNX session loaded: input='{input_name}' {session.get_inputs()[0].shape}, "
      f"output='{output_name}' {session.get_outputs()[0].shape}")

# Run a test window through the ONNX model and compare to PyTorch
test_window = windows[0:1].astype(np.float32)
onnx_result = session.run([output_name], {input_name: test_window})[0]

model.eval()
with torch.inference_mode():
    torch_result = model(torch.tensor(test_window, device=device))
    torch_result = torch.softmax(torch_result, dim=1).cpu().numpy()

max_diff = np.abs(onnx_result - torch_result).max()
print(f"Max absolute difference (PyTorch vs ONNX): {max_diff:.6e}")
if max_diff < 1e-4:
    print("ONNX reimport check passed.")
else:
    print(f"Warning: difference is larger than expected ({max_diff:.4f})")

# %%
