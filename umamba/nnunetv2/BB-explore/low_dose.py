1+1
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch  # If needed for any random selection or device usage

from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform

def simulate_low_xray_dose(
    image: np.ndarray,
    show_intermediate: bool = False,
    noise_variance: tuple = (0.03, 0.08),
    blur_sigma: tuple = (0.4, 1.2),
    contrast_range: tuple = (0.7, 0.9),
    brightness_range: tuple = (0.6, 0.85),
    resolution_range: tuple = (0.7, 0.9)
) -> np.ndarray:
    """
    Simulates lower X-ray radiation dose by adding noise, blurring, lowering
    contrast and brightness, and reducing resolution.

    Args:
        image: Input angiographic image as a NumPy array, shape (H, W) or (1, H, W)
        show_intermediate: If True, shows a plot of the intermediate transforms
        noise_variance: Range of Gaussian noise variance to simulate quantum noise
        blur_sigma: Range of Gaussian blur sigma to simulate resolution loss
        contrast_range: Range of contrast reduction
        brightness_range: Multiplicative brightness range
        resolution_range: Range for reducing/zooming resolution

    Returns:
        np.ndarray: Simulated lower-dose angiogram image
    """
    # Ensure image is 4D: (B=1, C=1, H, W) for batchgeneratorsv2
    if image.ndim == 2:
        image = image[None, None, ...]
    elif image.ndim == 3:
        image = image[None, ...]
    
    # Dict expected by transforms
    data_dict = {'data': image.astype(np.float32)}

    transforms = [
        GaussianNoiseTransform(noise_variance=noise_variance),
        GaussianBlurTransform(blur_sigma=blur_sigma, p_per_channel=1),
        ContrastTransform(contrast_range=contrast_range, preserve_range=True, p_per_channel=1),
        MultiplicativeBrightnessTransform(multiplier_range=brightness_range, p_per_channel=1),
        SimulateLowResolutionTransform(scale=resolution_range, p_per_channel=1, ignore_axes=None)
    ]

    if show_intermediate:
        fig, axs = plt.subplots(1, len(transforms) + 1, figsize=(18, 4))
        axs = axs.ravel()
        axs[0].imshow(image[0, 0], cmap='gray')
        axs[0].set_title('Original')

        # Show effect of each transform in sequence
        temp_dict = {'data': data_dict['data'].copy()}
        for i, t in enumerate(transforms, start=1):
            t(temp_dict)
            axs[i].imshow(temp_dict['data'][0, 0], cmap='gray')
            axs[i].set_title(t.__class__.__name__)
            # Reset after displaying the step
            temp_dict = {'data': data_dict['data'].copy()}  

        plt.tight_layout()
        plt.show()

    # Apply transforms in a single pass
    pipeline = ComposeTransforms(transforms)
    pipeline(data_dict)
    
    # Return the 2D image
    return data_dict['data'][0, 0]

# if __name__ == "__main__":
# Load an actual coronary angiogram image from HDF5:
h5_path = "/home/billb/AWI/NetExploration/AngiogramsDistilledUInt8List.h5"
with h5py.File(h5_path, 'r') as f:
    keys = list(f.keys())
    key = np.random.choice(keys)
    print(f"Randomly selected key: {key}")
    frame_idx = 30
    # Load the 30th frame from the chosen key and normalize to [0,1]
    angio_frame = f[key][frame_idx].astype(float) / 255.0

# Simulate lower X-ray dose
low_dose_image = simulate_low_xray_dose(angio_frame, show_intermediate=True)

# Show the final result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(angio_frame, cmap='gray')
plt.title("Original Coronary Angiogram (30th Frame)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(low_dose_image, cmap='gray')
plt.title("Simulated Low X-ray Dose")
plt.axis("off")
plt.show() 