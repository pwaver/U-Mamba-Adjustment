from typing import Union

import torch
import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class PoissonNoiseTransform(AbstractTransform):
    """Custom transform to add Poisson noise to simulate low dose imaging artifacts.
    
    This transform simulates the quantum noise characteristic of low radiation dose
    imaging by applying Poisson noise to the image data. The noise level is controlled
    by scaling the data before applying the Poisson distribution.
    
    Args:
        p_per_sample: Probability of applying the transform per sample. Defaults to 1.0.
        data_key: Key for the data in the data dictionary. Defaults to 'data'.
    """
    
    def __init__(self, p_per_sample: float = 1.0, data_key: str = 'data') -> None:
        """Initialize the PoissonNoiseTransform.
        
        Args:
            p_per_sample: Probability of applying the transform per sample.
            data_key: Key for the data in the data dictionary.
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key

    def __call__(self, **data_dict) -> dict:
        """Apply Poisson noise to the image data.
        
        Args:
            **data_dict: Dictionary containing image data under the specified data_key.
            
        Returns:
            Dictionary with potentially modified image data.
        """
        if np.random.uniform() < self.p_per_sample:
            data = data_dict[self.data_key]
            # Convert to numpy if it's a tensor
            if torch.is_tensor(data):
                was_tensor = True
                original_device = data.device
                original_dtype = data.dtype
                data = data.cpu().numpy()
            else:
                was_tensor = False

            # Ensure data is positive and finite
            data = np.clip(data, 0, None)  # Clip negative values to 0
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs and infinities

            # Scale the data to a reasonable range for Poisson noise
            scale = 200.0  # Matches our tested value for maximum graininess
            scaled_data = data * scale

            # Generate Poisson noise
            noisy_data = np.random.poisson(scaled_data)

            # Scale back and ensure valid range
            noisy_data = np.clip(noisy_data / scale, 0, 1)

            # Convert back to tensor with the same device and precision as input
            if was_tensor:
                noisy_data = torch.from_numpy(noisy_data).to(original_device)
                # Ensure the same precision as the input
                noisy_data = noisy_data.to(original_dtype)

            data_dict[self.data_key] = noisy_data
        return data_dict


class RandomLowDoseTransform(AbstractTransform):
    """Transform that randomly applies low dose simulation with specified probability.
    
    This wrapper transform applies a low dose simulation transform chain with a specified
    probability, allowing for mixed training with both normal and low dose simulated images.
    
    Args:
        low_dose_transform: The transform to apply for low dose simulation.
        p: Probability of applying the low dose transform. Defaults to 0.3.
    """
    
    def __init__(self, low_dose_transform: AbstractTransform, p: float = 0.3) -> None:
        """Initialize the RandomLowDoseTransform.
        
        Args:
            low_dose_transform: The transform to apply for low dose simulation.
            p: Probability of applying the low dose transform.
        """
        self.low_dose_transform = low_dose_transform
        self.p = p

    def __call__(self, **data_dict) -> dict:
        """Apply low dose transform with specified probability.
        
        Args:
            **data_dict: Dictionary containing image and segmentation data.
            
        Returns:
            Dictionary with potentially modified image data.
        """
        if np.random.uniform() < self.p:
            return self.low_dose_transform(**data_dict)
        return data_dict 