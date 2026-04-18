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
annotationDataFile = str(Path.home() / "Angiostore/WebknossosAnnotationsRevisedUnitized-5.h5")
annotationIndicizedDataFile = str(Path.home() / "Angiostore/WebknossosAnnotationsRevisedIndicized-3.h5")
nnUNetRawFolder = str(Path.home()/ "Angiostore/nnUnet_raw")

annotationBitfieldDataFile = str(Path.home()/ "Angiostore/WebknossosAnnotationsRevisedUnitized-5-Bitfield.h5")


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
    Transform 5-channel unitized annotations to single-channel indicized format.
    
    Rules:
    (0,0,0,*,*) -> 0  # background
    (0,1,0,*,*) -> 1  # catheter
    (0,0,1,0,*) -> 2  # vessel
    (0,0,*,1,*) -> 0  # stenosis (maps to background)
    """
    
    def dataset_transform(data):
        catheter = data[1]
        vessel =  2 * (data[2] - data[2]*data[3]) 
        result = catheter + vessel - data[1]*data[2] 
        
        return result
    
    # Open both files
    with h5py.File(annotationDataFile, 'r') as f_in, \
         h5py.File(annotationIndicizedDataFile, 'w') as f_out:
        
        # Process each dataset
        for dataset_name in f_in.keys():
            print(f"Processing dataset: {dataset_name}")
            
            # Read input data
            data = f_in[dataset_name][:]
            print(f"Dataset shape: {data.shape}")
            transformed = dataset_transform(data)
            print(f"Transformed shape: {transformed.shape}")
            
            # Create output dataset
            output_shape = transformed.shape
            dset_out = f_out.create_dataset(
                dataset_name,
                output_shape,
                dtype=np.uint8
            )
            
            # Write transformed data
            dset_out[:] = transformed
            
            # Verify unique values
            unique_values = np.unique(dset_out[:])
            print(f"  Dataset {dataset_name} unique values: {unique_values}")
            
        print("Transformation complete!")

# Example usage:
# transform_annotations(annotationDataFile, annotationIndicizedDataFile)

# For debugging, let's also print the shape of the first dataset
with h5py.File(annotationDataFile, 'r') as f:
    first_dataset_name = list(f.keys())[0]
    first_dataset = f[first_dataset_name][:]
    print(f"First dataset shape: {first_dataset.shape}")

# %%
# Example paths (adjust as needed)
# annotationDataFile = "/path/to/5channel/annotations.h5"
# annotationIndicizedDataFile = "/path/to/output/indicized_annotations.h5"

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


