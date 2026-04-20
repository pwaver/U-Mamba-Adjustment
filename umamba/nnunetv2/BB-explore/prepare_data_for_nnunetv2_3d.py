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
1+1
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
annotationDataFile = str(Path.home() / "Angiostore/WebknossosAnnotationsRevisedUnitized-5-Bitfield.h5")
annotationIndicizedDataFile = str(Path.home() / "Angiostore/WebknossosAnnotationsRevisedIndicized-4.h5")
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
    Convert the bitfield annotation HDF5 file into a single-channel indicized HDF5 file
    suitable for nnUNet segmentation training.

    SOURCE FORMAT — WebknossosAnnotationsRevisedUnitized-5-Bitfield.h5
    -----------------------------------------------------------------------
    Each dataset has shape (frames, H, W), dtype uint8.
    Each voxel stores a BITFIELD where each bit encodes one annotation layer:

        Bit 0  mask=0x01  value=  1  channel 0: explicit background   (NEVER SET in practice)
        Bit 1  mask=0x02  value=  2  channel 1: catheter
        Bit 2  mask=0x04  value=  4  channel 2: vessel
        Bit 3  mask=0x08  value=  8  channel 3: stenosis
        Bit 4  mask=0x10  value= 16  channel 4: unknown/unannotated    (~90-96% of voxels)

    Multiple bits can be set simultaneously (e.g. value=6 = vessel+catheter overlap).
    Values outside the range 0–31 would indicate unexpected annotation layers.

    TARGET FORMAT — annotationIndicizedDataFile
    -----------------------------------------------------------------------
    Each dataset has shape (frames, H, W), dtype uint8.
    Each voxel holds a single class index:
        0 = background  (no catheter or vessel; stenosis also collapses to 0)
        1 = catheter
        2 = vessel

    PRIORITY RULES when bits overlap:
        - stenosis (bit 3) suppresses vessel: stenosis voxels → label 0
        - vessel > catheter when both present (original formula behavior)
        - catheter-only → label 1
        - background/unknown-only → label 0
    """

    if os.path.exists(annotationIndicizedDataFile):
        print(f"Indicized annotation file already exists, skipping: {annotationIndicizedDataFile}")
        return

    # ---- Upfront structural validation of the bitfield source file ----
    print("Validating bitfield annotation file structure...")
    with h5py.File(annotationDataFile, 'r') as f_in:
        sample_key = list(f_in.keys())[0]
        sample = f_in[sample_key][:]

        # Must be 3D (frames, H, W) — NOT 4D like the old 5-channel unitized format
        assert sample.ndim == 3, (
            f"Expected bitfield data to be 3D (frames, H, W), got ndim={sample.ndim}. "
            "Did you accidentally point to the old 5-channel unitized file?"
        )

        # Must be uint8 to hold packed bits
        assert sample.dtype == np.uint8, (
            f"Expected dtype uint8 for bitfield data, got {sample.dtype}."
        )

        # Values must fit within 5 bits (0–31). Bits 5–7 should never be set.
        max_val = int(sample.max())
        assert max_val <= 31, (
            f"Unexpected bitfield values > 31 found (max={max_val}). "
            "Bits 5–7 should be zero — this suggests extra annotation layers exist."
        )

        # Sanity: bit 4 (value 16, unknown/background) should dominate
        bit4_fraction = float((sample & 0x10).astype(bool).mean())
        assert bit4_fraction > 0.5, (
            f"Expected bit 4 (unknown/background) to dominate (>50%), "
            f"got {bit4_fraction:.1%}. Bitfield encoding may have changed."
        )
        print(f"  Validation passed: ndim=3, dtype=uint8, max_value={max_val}, "
              f"background_fraction={bit4_fraction:.1%}")

    def dataset_transform(bitfield: np.ndarray) -> np.ndarray:
        """
        Map a single (frames, H, W) uint8 bitfield array → (frames, H, W) uint8 label array.

        Bit extraction — each Boolean array is True where that annotation layer is active:
            catheter  = bit 1  (mask 0x02)
            vessel    = bit 2  (mask 0x04)
            stenosis  = bit 3  (mask 0x08)

        Combination formula (preserved from the original 5-channel indicize logic):
            vessel is suppressed where stenosis is also set
            catheter + vessel overlap resolves to vessel (formula gives label 2)

        Label arithmetic (all terms are 0 or 1 before scaling):
            result = catheter + 2*(vessel & ~stenosis) - catheter*vessel
        """
        catheter  = (bitfield & 0x02).astype(np.uint8) >> 1   # 0 or 1
        vessel    = (bitfield & 0x04).astype(np.uint8) >> 2   # 0 or 1
        stenosis  = (bitfield & 0x08).astype(np.uint8) >> 3   # 0 or 1

        # Vessel is suppressed wherever stenosis is active; scale vessel contribution to label 2
        vessel_masked = vessel * (1 - stenosis)                # 0 if stenosis, else vessel
        result = catheter + 2 * vessel_masked - catheter * vessel_masked
        return result.astype(np.uint8)

    print("Indicizing bitfield annotations → single-channel label file...")
    with h5py.File(annotationDataFile, 'r') as f_in, \
         h5py.File(annotationIndicizedDataFile, 'w') as f_out:
        for key in f_in.keys():
            bitfield_data = f_in[key][:]           # shape: (frames, H, W), dtype uint8
            label_data = dataset_transform(bitfield_data)
            f_out.create_dataset(key, data=label_data, dtype=np.uint8)
            print(f"  Indicized: {key}, shape={label_data.shape}, "
                  f"labels={np.unique(label_data).tolist()}")

    print(f"Indicized annotation file written: {annotationIndicizedDataFile}")


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

    Note: Annotation datasets may have fewer frames than angiography datasets.
    Frames near the edges where a 5-frame window cannot be formed are skipped.
    """
    labels_dir = os.path.join(nnUNetRawFolder, 'labelsTr')
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

    with h5py.File(annotationIndicizedDataFile, 'r') as f:
        blockCounter = 0
        skipped_count = 0
        for dataset_name, center_idx in frameKeys:
            # Verify the annotation dataset exists and has enough frames
            if dataset_name not in f:
                skipped_count += 1
                continue

            n_frames = f[dataset_name].shape[0]

            # Skip if center_idx is too close to the edges for a 5-frame window
            if center_idx - 2 < 0 or center_idx + 3 > n_frames:
                skipped_count += 1
                continue

            anno_data = f[dataset_name][center_idx - 2 : center_idx + 3]  # (5, 512, 512)

            # Sanity check: ensure we got 5 frames
            if anno_data.shape[0] != 5:
                print(f"  Warning: {dataset_name}[{center_idx-2}:{center_idx+3}] returned {anno_data.shape[0]} frames instead of 5")
                skipped_count += 1
                continue

            filename = f'Angio_{blockCounter:04d}.nii.gz'
            filepath = os.path.join(labels_dir, filename)
            _write_nifti(anno_data.astype(np.uint8), filepath)

            blockCounter += 1

        print(f"Exported {blockCounter} label volumes (shape 5x512x512)")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} frames due to boundary or missing annotation data")
        print(f"  Total frameKeys requested: {len(frameKeys)}, written: {blockCounter}")


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
