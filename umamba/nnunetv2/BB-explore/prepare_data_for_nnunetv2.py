# %%
import h5py
import numpy as np
import os
import cv2
from pathlib import Path

# Example paths (adjust as needed)
# angiographyDataFile = "/path/to/your/angiography.h5"
# nnUNetRawFolder = "/path/to/nnUNet/raw/folder"
angiographyDataFile = str(Path.home() / "Angiostore/WebknossosAngiogramsRevisedUInt8List.h5")
annotationDataFile = str(Path.home() / "Angiostore/WebknossosAnnotationsRevisedUnitized-5-Bitfield.h5")
annotationIndicizedDataFile = str(Path.home() / "Angiostore/WebknossosAnnotationsRevisedIndicized-4.h5")
nnUNetRawFolder = str(Path.home()/ "Angiostore/nnUnet_raw")


# %%
def get_common_keys(angiographyDataFile, annotationDataFile):
    """
    Get the intersection of keys between the two HDF5 files
    """
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
# Get common keys first
common_keys = get_common_keys(angiographyDataFile, annotationDataFile)

# If you want to see the keys before proceeding
print("Common keys:", common_keys)

# %%

def export_angiography_to_nnunet(angiographyDataFile, nnUNetRawFolder, common_keys):
    """
    Modified to only process common keys
    """
    images_dir = os.path.join(nnUNetRawFolder, 'imagesTr')
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    
    frameKeys=[]
    
    with h5py.File(angiographyDataFile, 'r') as f:
        blockCounter = 0
        
        # Only iterate through common keys
        for dataset_name in common_keys:
            print(f"Processing angiogram dataset: {dataset_name}")
            angio_data = f[dataset_name][:]
            
            n_frames = angio_data.shape[0]
            
            for center_idx in range(2, n_frames - 2):
                frameKeys.append([dataset_name, center_idx])
                frame_indices = range(center_idx - 2, center_idx + 3)
                frames = angio_data[frame_indices]
                
                for frame_num, frame in enumerate(frames):
                    if frame.dtype != np.uint8:
                        if frame.max() > 1:
                            frame = (frame / frame.max() * 255).astype(np.uint8)
                        else:
                            frame = (frame * 255).astype(np.uint8)
                    
                    filename = f'Angios_{blockCounter:04d}_{frame_num:04d}.png'
                    filepath = os.path.join(images_dir, filename)
                    cv2.imwrite(filepath, frame)
                
                blockCounter += 1
        
        print(f"Exported {blockCounter} sets of 5 frames each")
        print(f"Total number of PNG files created: {blockCounter * 5}")
        return frameKeys


# %%
frameKeys = export_angiography_to_nnunet(angiographyDataFile, nnUNetRawFolder, common_keys)


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
    """
    Get list of dataset keys in an HDF5 file.

    Args:
        hdf5_file: Path to the HDF5 file

    Returns:
        list[str]: List of dataset keys in the file
    """
    with h5py.File(hdf5_file, 'r') as f:
        keys = list(f.keys())
        print(f"Keys in {hdf5_file}:")
        for key in keys:
            print(f"  {key}")
        return keys

# Get keys from annotationIndicizedDataFile
keys = get_hdf5_keys(annotationIndicizedDataFile)


# %%
len(keys)

# %%
def get_dataset_data(hdf5_file: str, key: str) -> np.ndarray:
    """
    Get data from a specific dataset in an HDF5 file.

    Args:
        hdf5_file: Path to the HDF5 file
        key: Key of the dataset to retrieve

    Returns:
        np.ndarray: Data from the specified dataset
    """
    with h5py.File(hdf5_file, 'r') as f:
        data = f[key][:]
        print(f"Shape of dataset {key}: {data.shape}")
        return data

# Get data for "Napari_9_rev" dataset
key = random.choice(keys)
example_data = get_dataset_data(annotationIndicizedDataFile, key)
print(key, example_data.shape)


# %%
# Display as an image
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(example_data[30], cmap='gray')
plt.colorbar()
plt.title('example_data[30]')
plt.axis('on')
plt.show()


# %%

def export_annotations_to_nnunet(annotationDataFile, nnUNetRawFolder, frameKeys):
    """
    Modified to only process common keys and verify count matches angiography
    """
    labels_dir = os.path.join(nnUNetRawFolder, 'labelsTr')
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    
    with h5py.File(annotationDataFile, 'r') as f:
        blockCounter = 0
        
        # Only iterate through common keys
        for dataset_name, center_idx in frameKeys:
            print(f"Processing annotation dataset: {dataset_name}")
            anno_data = f[dataset_name][center_idx]
            
            filename = f'Angios_{blockCounter:04d}.png'
            filepath = os.path.join(labels_dir, filename)
            cv2.imwrite(filepath, anno_data)
                
            blockCounter += 1
        
        print(f"Exported {blockCounter} label files")
        
        # Verify we have the same number of cases as angiography
        assert blockCounter == len(frameKeys), \
            f"Mismatch in number of cases: Angiography had {len(frameKeys)}, Annotations had {blockCounter}"


# %%
# Print first 50 frame keys to inspect the data
print("First 50 frame keys:")
for i, (dataset_name, center_idx) in enumerate(frameKeys[:50]):
    print(f"{i}: {dataset_name}, center_idx={center_idx}")


# %%
export_annotations_to_nnunet(annotationIndicizedDataFile, nnUNetRawFolder, frameKeys)

# %%
len(frameKeys)

# %%
# ... existing code ...

from datetime import datetime

# Get current timestamp in a human-readable format
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

datasetJson = {
    "_comment": f"Dataset contains 5-channel angiography data at 7.5Hz (every other frame) with 5-frame neighborhoods. Labels include background (0), C (1), and V (2). Data sourced from WebknossosAngiogramsRevisedUInt8List.h5 and WebknossosAnnotationsRevisedIndicized-3.h5. Generated on {timestamp}.",
    "channel_names": {
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4"
    },
    "labels": {
        "background": 0,
        "catheter": 1,
        "vessel": 2
    },
    "numTraining": len(frameKeys),
    "file_ending": ".png"
}

# Write the dataset.json file
import json
dataset_json_path = os.path.join(nnUNetRawFolder, 'dataset.json')
with open(dataset_json_path, 'w') as f:
    json.dump(datasetJson, f, indent=4)

print(f"Created dataset.json at {dataset_json_path}")


