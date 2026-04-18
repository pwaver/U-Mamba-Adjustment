"""
prepare_data_for_nnunetv2_3d.py

3D variant of prepare_data_for_nnunetv2.py. Produces a nnUNet-v2 dataset where
each case is a true 3-D NIfTI volume of shape (D=5, H=512, W=512). The 5-frame
temporal window is laid out along the z-axis (not packed into channels as in
the 2D prep). Labels for all 5 z-slices are written so the decoder receives
gradient at every temporal position; at inference the middle frame (index 2)
is sliced out.

Target layout under nnUNetRawFolder:
  Dataset430_Angiography3d/
    imagesTr/Angio_XXXX_0000.nii.gz    # (5, 512, 512) uint8
    labelsTr/Angio_XXXX.nii.gz         # (5, 512, 512) uint8, values {0,1,2}
    dataset.json
"""

# %%
import os
import json
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk

angiographyDataFile = str(Path.home() / "Angiostore/WebknossosAngiogramsRevisedUInt8List.h5")
annotationDataFile = str(Path.home() / "Angiostore/WebknossosAnnotationsRevisedUnitized-5.h5")
annotationIndicizedDataFile = str(Path.home() / "Angiostore/WebknossosAnnotationsRevisedIndicized-3.h5")
nnUNetRawFolder = str(Path.home() / "Angiostore/nnUnet_raw/Dataset430_Angiography3d")

SPACING = (1.0, 1.0, 1.0)
ORIGIN = (0.0, 0.0, 0.0)


# %%
def get_common_keys(angiographyDataFile, annotationDataFile):
    """Intersection of dataset keys between the two HDF5 files."""
    with h5py.File(angiographyDataFile, 'r') as f_angio, \
         h5py.File(annotationDataFile, 'r') as f_anno:
        angio_keys = set(f_angio.keys())
        anno_keys = set(f_anno.keys())
        common_keys = sorted(list(angio_keys.intersection(anno_keys)))

        print(f"Angiography keys: {len(angio_keys)}")
        print(f"Annotation keys: {len(anno_keys)}")
        print(f"Common keys: {len(common_keys)}")
        return common_keys


# %%
common_keys = get_common_keys(angiographyDataFile, annotationDataFile)
print("Common keys:", common_keys)


# %%
def _write_nifti(array_dhw: np.ndarray, path: str, spacing=SPACING, origin=ORIGIN,
                 cast_to_uint8: bool = True):
    """
    Write a (D, H, W) numpy array as a NIfTI volume via SimpleITK.

    sitk.GetImageFromArray takes NumPy (D, H, W) and produces an ITK image whose
    Size reports as (W, H, D) — this is what nnUNet's SimpleITKIO expects.
    """
    img = sitk.GetImageFromArray(array_dhw)
    if cast_to_uint8:
        img = sitk.Cast(img, sitk.sitkUInt8)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    sitk.WriteImage(img, path)


def export_angiography_to_nnunet_3d(angiographyDataFile, nnUNetRawFolder, common_keys):
    """
    Slice each angiography dataset into 5-frame z-stacks centered on each valid
    frame; write each stack as a single-channel 3-D NIfTI (Angio_XXXX_0000.nii.gz).

    Returns frameKeys: list of [dataset_name, center_idx] in blockCounter order
    so the downstream annotation export can pick the matching 5-frame label stack.
    """
    images_dir = os.path.join(nnUNetRawFolder, 'imagesTr')
    Path(images_dir).mkdir(parents=True, exist_ok=True)

    frameKeys = []

    with h5py.File(angiographyDataFile, 'r') as f:
        blockCounter = 0
        for dataset_name in common_keys:
            print(f"Processing angiogram dataset: {dataset_name}")
            angio_data = f[dataset_name][:]
            n_frames = angio_data.shape[0]

            for center_idx in range(2, n_frames - 2):
                volume = angio_data[center_idx - 2 : center_idx + 3]  # (5, 512, 512)

                if volume.dtype != np.uint8:
                    if volume.max() > 1:
                        volume = (volume / volume.max() * 255).astype(np.uint8)
                    else:
                        volume = (volume * 255).astype(np.uint8)

                filename = f'Angio_{blockCounter:04d}_0000.nii.gz'
                filepath = os.path.join(images_dir, filename)
                _write_nifti(volume, filepath)

                frameKeys.append([dataset_name, center_idx])
                blockCounter += 1

        print(f"Exported {blockCounter} 3-D angiography volumes (shape 5x512x512)")
        return frameKeys


# %%
frameKeys = export_angiography_to_nnunet_3d(angiographyDataFile, nnUNetRawFolder, common_keys)


# %%
def indicize_annotations(annotationDataFile, annotationIndicizedDataFile):
    """
    Transform 5-channel unitized annotations to single-channel indicized format.

    Rules:
    (0,0,0,*,*) -> 0  # background
    (0,1,0,*,*) -> 1  # catheter
    (0,0,1,0,*) -> 2  # vessel
    (0,0,*,1,*) -> 0  # stenosis (maps to background)
    """
    def dataset_transform(data):
        catheter = data[1]
        vessel = 2 * (data[2] - data[2] * data[3])
        result = catheter + vessel - data[1] * data[2]
        return result

    with h5py.File(annotationDataFile, 'r') as f_in, \
         h5py.File(annotationIndicizedDataFile, 'w') as f_out:
        for dataset_name in f_in.keys():
            print(f"Processing dataset: {dataset_name}")
            data = f_in[dataset_name][:]
            print(f"  Input shape: {data.shape}")
            transformed = dataset_transform(data)
            print(f"  Transformed shape: {transformed.shape}")

            dset_out = f_out.create_dataset(dataset_name, transformed.shape, dtype=np.uint8)
            dset_out[:] = transformed
            print(f"  Unique values: {np.unique(dset_out[:])}")
        print("Indicization complete.")


# %%
if os.path.exists(annotationIndicizedDataFile):
    print(f"Indicized file already exists at {annotationIndicizedDataFile} — skipping.")
else:
    indicize_annotations(annotationDataFile, annotationIndicizedDataFile)


# %%
def get_hdf5_keys(hdf5_file: str) -> list[str]:
    with h5py.File(hdf5_file, 'r') as f:
        keys = list(f.keys())
        print(f"Keys in {hdf5_file}:")
        for key in keys:
            print(f"  {key}")
        return keys


def get_dataset_data(hdf5_file: str, key: str) -> np.ndarray:
    with h5py.File(hdf5_file, 'r') as f:
        data = f[key][:]
        print(f"Shape of dataset {key}: {data.shape}")
        return data


keys = get_hdf5_keys(annotationIndicizedDataFile)
key = random.choice(keys)
example_data = get_dataset_data(annotationIndicizedDataFile, key)
print(key, example_data.shape)


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(example_data[30], cmap='gray')
plt.colorbar()
plt.title(f'{key} example_data[30]')
plt.axis('on')
plt.show()


# %%
def export_annotations_to_nnunet_3d(annotationIndicizedDataFile, nnUNetRawFolder, frameKeys):
    """
    For each (dataset_name, center_idx) in frameKeys, write the matching 5-frame
    label z-stack as Angio_XXXX.nii.gz (no _0000 suffix — nnUNet labels are
    single-file).
    """
    labels_dir = os.path.join(nnUNetRawFolder, 'labelsTr')
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

    with h5py.File(annotationIndicizedDataFile, 'r') as f:
        blockCounter = 0
        for dataset_name, center_idx in frameKeys:
            anno_data = f[dataset_name][center_idx - 2 : center_idx + 3]  # (5, 512, 512)

            filename = f'Angio_{blockCounter:04d}.nii.gz'
            filepath = os.path.join(labels_dir, filename)
            _write_nifti(anno_data.astype(np.uint8), filepath)

            blockCounter += 1

        print(f"Exported {blockCounter} label volumes (shape 5x512x512)")
        assert blockCounter == len(frameKeys), \
            f"Count mismatch: frameKeys={len(frameKeys)}, written={blockCounter}"


# %%
print("First 50 frame keys:")
for i, (dataset_name, center_idx) in enumerate(frameKeys[:50]):
    print(f"{i}: {dataset_name}, center_idx={center_idx}")


# %%
export_annotations_to_nnunet_3d(annotationIndicizedDataFile, nnUNetRawFolder, frameKeys)


# %%
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
datasetJson = {
    "_comment": (
        "3D variant — 5-frame temporal window laid out as the z-axis of a single-channel "
        "NIfTI volume. Labels: 0=background (includes stenosis), 1=catheter, 2=vessel. "
        f"Sourced from WebknossosAngiogramsRevisedUInt8List.h5 and "
        f"WebknossosAnnotationsRevisedIndicized-3.h5. Generated on {timestamp}."
    ),
    "channel_names": {"0": "0"},
    "labels": {
        "background": 0,
        "catheter": 1,
        "vessel": 2,
    },
    "numTraining": len(frameKeys),
    "file_ending": ".nii.gz",
}

dataset_json_path = os.path.join(nnUNetRawFolder, 'dataset.json')
with open(dataset_json_path, 'w') as f:
    json.dump(datasetJson, f, indent=4)

print(f"Created dataset.json at {dataset_json_path}")


# %%
# Sanity check: re-read a random pair and verify shape, dtype, and label values.
images_dir = os.path.join(nnUNetRawFolder, 'imagesTr')
labels_dir = os.path.join(nnUNetRawFolder, 'labelsTr')

image_files = sorted(f for f in os.listdir(images_dir) if f.endswith('.nii.gz'))
assert len(image_files) == len(frameKeys), \
    f"imagesTr count {len(image_files)} != frameKeys {len(frameKeys)}"

sample_idx = random.randint(0, len(image_files) - 1)
img_name = image_files[sample_idx]
lbl_name = img_name.replace('_0000.nii.gz', '.nii.gz')

img_sitk = sitk.ReadImage(os.path.join(images_dir, img_name))
lbl_sitk = sitk.ReadImage(os.path.join(labels_dir, lbl_name))
img_np = sitk.GetArrayFromImage(img_sitk)
lbl_np = sitk.GetArrayFromImage(lbl_sitk)

print(f"Sampled pair: {img_name} <-> {lbl_name}")
print(f"  image shape: {img_np.shape}, dtype: {img_np.dtype}, range: [{img_np.min()}, {img_np.max()}]")
print(f"  label shape: {lbl_np.shape}, dtype: {lbl_np.dtype}, unique: {np.unique(lbl_np)}")
print(f"  image spacing: {img_sitk.GetSpacing()}, label spacing: {lbl_sitk.GetSpacing()}")

assert img_np.shape == (5, 512, 512), f"image shape {img_np.shape} != (5,512,512)"
assert lbl_np.shape == (5, 512, 512), f"label shape {lbl_np.shape} != (5,512,512)"
assert img_np.dtype == np.uint8, f"image dtype {img_np.dtype} != uint8"
assert set(np.unique(lbl_np)).issubset({0, 1, 2}), \
    f"label has values outside {{0,1,2}}: {np.unique(lbl_np)}"
assert img_sitk.GetSpacing() == lbl_sitk.GetSpacing(), "image/label spacing mismatch"

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(img_np[2], cmap='gray')
axes[0].set_title(f'{img_name} — middle frame (z=2)')
axes[1].imshow(lbl_np[2], vmin=0, vmax=2, cmap='viridis')
axes[1].set_title(f'{lbl_name} — middle frame (z=2)')
for ax in axes:
    ax.axis('on')
plt.tight_layout()
plt.show()
