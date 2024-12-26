"""
Module for exploring X-ray dose reduction simulation using nnU-Net's data augmentation transforms.

This module provides tools to simulate low-dose X-ray images from normal-dose images using
various image transformations that approximate the physical effects of dose reduction in
X-ray imaging.
"""
1+1
from typing import Optional, Tuple, Union

from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import Compose
import numpy as np
import matplotlib.pyplot as plt
import h5py


def simulate_low_xray_dose(
    image: np.ndarray, 
    show_intermediate: bool = False,
    noise_variance: Tuple[float, float] = (0.05, 0.1),
    contrast_range: Tuple[float, float] = (0.6, 0.8),
    blur_sigma: Tuple[float, float] = (0.5, 2.0),
    brightness_range: Tuple[float, float] = (0.5, 0.7),
    resolution_range: Tuple[float, float] = (0.5, 0.7)
) -> np.ndarray:
    """
    Simulate low-dose X-ray image from normal-dose image.
    
    Args:
        image: Input image as numpy array of shape (H,W) or (1,H,W)
        show_intermediate: If True, shows effect of each transform
        noise_variance: Range of Gaussian noise variance to simulate quantum noise
        contrast_range: Range of contrast reduction
        blur_sigma: Range of Gaussian blur sigma to simulate resolution loss
        brightness_range: Range of brightness multiplication factors
        resolution_range: Range of resolution reduction factors
    
    Returns:
        np.ndarray: Simulated low-dose image with same shape as input
    """
    # Ensure image is in format (1,1,H,W) as expected by batchgenerators
    if image.ndim == 2:
        image = image[None, None, ...]
    elif image.ndim == 3:
        image = image[None, ...]
    
    # Create a batch dict as expected by transforms
    data_dict = {'data': image.astype(np.float32)}
    
    # Define transforms with p_per_sample=1 to ensure they're applied
    transforms = [
        GaussianNoiseTransform(
            p_per_sample=1, 
            noise_variance=noise_variance
        ),
        ContrastAugmentationTransform(
            p_per_sample=1, 
            contrast_range=contrast_range
        ),
        GaussianBlurTransform(
            blur_sigma=blur_sigma,
            different_sigma_per_channel=False,
            p_per_sample=1,
            p_per_channel=1
        ),
        BrightnessMultiplicativeTransform(
            multiplier_range=brightness_range,
            p_per_sample=1
        ),
        SimulateLowResolutionTransform(
            zoom_range=resolution_range,
            per_channel=False,
            p_per_channel=1,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=1,
            ignore_axes=None
        )
    ]

    if show_intermediate:
        plt.figure(figsize=(20, 4))
        plt.subplot(1, len(transforms)+1, 1)
        plt.imshow(image[0,0], cmap='gray')
        plt.title('Original')
        
        for i, transform in enumerate(transforms, 1):
            data_dict_tmp = {'data': data_dict['data'].copy()}
            transform(**data_dict_tmp)
            plt.subplot(1, len(transforms)+1, i+1)
            plt.imshow(data_dict_tmp['data'][0,0], cmap='gray')
            plt.title(transform.__class__.__name__)
        plt.show()

    # Apply all transforms
    transform = Compose(transforms)
    transform(**data_dict)
    
    return data_dict['data'][0,0]  # Return 2D image


def compare_xray_doses(
    image: np.ndarray,
    dose_levels: int = 3,
    base_noise: float = 0.05,
    base_contrast: float = 0.8,
    base_blur: float = 0.5,
    base_brightness: float = 0.7,
    base_resolution: float = 0.7
) -> None:
    """
    Compare different simulated dose levels of an X-ray image.
    
    Args:
        image: Input normal-dose image
        dose_levels: Number of different dose levels to simulate
        base_noise: Base noise level for lowest dose
        base_contrast: Base contrast level for lowest dose
        base_blur: Base blur level for lowest dose
        base_brightness: Base brightness level for lowest dose
        base_resolution: Base resolution level for lowest dose
    """
    plt.figure(figsize=(4*(dose_levels+1), 4))
    
    # Show original image
    plt.subplot(1, dose_levels+1, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original (100% dose)')
    plt.axis('off')
    
    # Show simulated dose levels
    for i in range(dose_levels):
        dose_factor = 1 - (i * (1/dose_levels))
        
        # Scale parameters based on dose level
        noise_var = (base_noise * (2-dose_factor), base_noise * (2-dose_factor) * 1.2)
        contrast = (base_contrast * dose_factor, base_contrast * dose_factor * 1.2)
        blur = (base_blur * (2-dose_factor), base_blur * (2-dose_factor) * 1.2)
        brightness = (base_brightness * dose_factor, base_brightness * dose_factor * 1.2)
        resolution = (base_resolution * dose_factor, base_resolution * dose_factor * 1.2)
        
        low_dose = simulate_low_xray_dose(
            image,
            noise_variance=noise_var,
            contrast_range=contrast,
            blur_sigma=blur,
            brightness_range=brightness,
            resolution_range=resolution
        )
        
        plt.subplot(1, dose_levels+1, i+2)
        plt.imshow(low_dose, cmap='gray')
        plt.title(f'{int(dose_factor*100)}% dose')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def simulate_low_contrast_medium(
    image: np.ndarray,
    show_intermediate: bool = False,
    contrast_range: Tuple[float, float] = (0.1, 0.3),
    blur_sigma: Tuple[float, float] = (0.3, 0.8)
) -> np.ndarray:
    """
    Simulate reduced iodinated contrast medium dose in angiographic images.
    This primarily affects the visibility of blood vessels while maintaining
    overall image quality.
    
    Args:
        image: Input image as numpy array of shape (H,W) or (1,H,W)
        show_intermediate: If True, shows effect of each transform
        contrast_range: Range of contrast reduction specifically for vessels
        blur_sigma: Range of slight blur to simulate contrast medium diffusion
    
    Returns:
        np.ndarray: Simulated low-contrast image with same shape as input
    """
    if image.ndim == 2:
        image = image[None, None, ...]
    elif image.ndim == 3:
        image = image[None, ...]
    
    data_dict = {'data': image.astype(np.float32)}
    
    transforms = [
        ContrastAugmentationTransform(
            p_per_sample=1,
            contrast_range=contrast_range
        ),
        GaussianBlurTransform(
            blur_sigma=blur_sigma,
            different_sigma_per_channel=False,
            p_per_sample=1,
            p_per_channel=1
        )
    ]

    if show_intermediate:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, len(transforms)+1, 1)
        plt.imshow(image[0,0], cmap='gray')
        plt.title('Original')
        
        for i, transform in enumerate(transforms, 1):
            data_dict_tmp = {'data': data_dict['data'].copy()}
            transform(**data_dict_tmp)
            plt.subplot(1, len(transforms)+1, i+1)
            plt.imshow(data_dict_tmp['data'][0,0], cmap='gray')
            plt.title(transform.__class__.__name__)
        plt.show()

    transform = Compose(transforms)
    transform(**data_dict)
    
    return data_dict['data'][0,0]


h5_path = "/home/billb/AWI/NetExploration/AngiogramsDistilledUInt8List.h5"
# Get list of keys and select a random one
with h5py.File(h5_path, 'r') as f:
    keys = list(f.keys())
    key = np.random.choice(keys)
    print(f"Randomly selected key: {key}")

# Get the number of frames for the selected sequence
with h5py.File(h5_path, 'r') as f:
    num_frames = f[key].shape[0]
    print(f"Number of frames in sequence {key}: {num_frames}")

# key = "/Napari_54_rev"
frame_idx = 30  # The frame we want to process

with h5py.File(h5_path, 'r') as f:
    # Load the specific frame
    angio_sequence = f[key][()]
    frame = angio_sequence[frame_idx].astype(float) / 255.0  # Normalize to [0,1] range

# Display the loaded frame
plt.figure(figsize=(8, 8))
plt.imshow(frame, cmap='gray')
plt.title(f'Frame {frame_idx}')
plt.axis('off')
plt.show()

simulate_low_xray_dose(frame, show_intermediate=True)

compare_xray_doses(frame, dose_levels=4)

lc=simulate_low_contrast_medium(frame, show_intermediate=False)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(frame, cmap='gray')
plt.title('Original Frame')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(lc, cmap='gray')
plt.title('Low Contrast')
plt.axis('off')

plt.tight_layout()
plt.show()

if __name__ == "__main__":
    # Example usage:
    # Load your image here
    # image = load_your_image()
    
    # Simulate single low dose version with intermediate steps
    # low_dose = simulate_low_xray_dose(image, show_intermediate=True)
    
    # Compare different dose levels
    # compare_xray_doses(image, dose_levels=4)
    
    print("Import this module and use simulate_low_xray_dose() or compare_xray_doses() with your images.") 