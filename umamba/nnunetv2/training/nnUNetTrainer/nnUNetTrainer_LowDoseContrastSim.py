import torch
import numpy as np
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from nnunetv2.training.data_augmentation.custom_transforms.LowDoseTransform import (
    PoissonNoiseTransform, 
    RandomLowDoseTransform
)

# At deployment:
# nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD -tr nnUNetTrainer_LowDoseContrastSim


class nnUNetTrainer_LowDoseContrastSim(nnUNetTrainer):
    """nnUNet trainer with low dose contrast simulation for fluoroscopic cardiac angiography.
    
    This trainer adds data augmentation that simulates physically low radiation dose images
    during training by applying contrast reduction, Poisson noise, and Gaussian noise.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')) -> None:
        """Initialize the low dose contrast simulation trainer.
        
        Args:
            plans: Training plans dictionary.
            configuration: Configuration name.
            fold: Cross-validation fold number.
            dataset_json: Dataset configuration.
            unpack_dataset: Whether to unpack the dataset.
            device: Torch device for training.
        """
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.print_to_log_file("Using Custom Trainer nnUNetTrainer_LowDoseContrastSim with 30% low dose simulation")

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        """Get training transforms including low dose simulation.
        
        Args:
            patch_size: Size of training patches.
            rotation_for_DA: Rotation angles for data augmentation.
            deep_supervision_scales: Scales for deep supervision.
            mirror_axes: Axes for mirroring augmentation.
            do_dummy_2d_data_aug: Whether to use dummy 2D augmentation.
            order_resampling_data: Interpolation order for data resampling.
            order_resampling_seg: Interpolation order for segmentation resampling.
            border_val_seg: Border value for segmentation.
            use_mask_for_norm: Whether to use mask for normalization.
            is_cascaded: Whether this is a cascaded model.
            foreground_labels: Labels to consider as foreground.
            regions: Regions for region-based training.
            ignore_label: Label to ignore during training.
            
        Returns:
            Composed transform including low dose simulation.
        """
        # Get the default augmentations from the parent class
        base_transforms = super().get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, order_resampling_data, order_resampling_seg,
            border_val_seg, use_mask_for_norm, is_cascaded, foreground_labels,
            regions, ignore_label
        )

        # Define low dose simulation transforms using our tested parameters
        low_dose_sim = Compose([
            ContrastAugmentationTransform(
                contrast_range=(0.2, 0.4), 
                p_per_sample=1.0, 
                preserve_range=True
            ),
            PoissonNoiseTransform(p_per_sample=1.0),
            GaussianNoiseTransform(
                noise_variance=(0.05, 0.1), 
                p_per_sample=1.0
            )
        ])

        # Create random low dose transform with 30% probability
        random_low_dose = RandomLowDoseTransform(low_dose_transform=low_dose_sim, p=0.3)

        # Since U-Mamba uses regular Compose instead of ComposeTransforms,
        # we need to create a combined transform that applies both base transforms
        # and our low dose simulation
        class CombinedTransform(AbstractTransform):
            """Combined transform that applies base transforms and low dose simulation."""
            
            def __init__(self, base_transform: AbstractTransform, low_dose_transform: AbstractTransform) -> None:
                self.base_transform = base_transform
                self.low_dose_transform = low_dose_transform
            
            def __call__(self, **data_dict) -> dict:
                """Apply base transforms followed by low dose simulation.
                
                Args:
                    **data_dict: Dictionary containing image and segmentation data.
                    
                Returns:
                    Dictionary with transformed data.
                """
                # Apply base transforms first
                data_dict = self.base_transform(**data_dict)
                # Then apply low dose simulation
                data_dict = self.low_dose_transform(**data_dict)
                return data_dict

        # Combine base transforms with low dose simulation
        all_transforms = CombinedTransform(base_transforms, random_low_dose)

        return all_transforms 