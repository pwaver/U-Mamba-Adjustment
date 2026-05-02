from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LowDoseContrastSim import nnUNetTrainer_LowDoseContrastSim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBot3Mamba3_2d import get_umamba_bot3_mamba3_2d_from_plans


class nnUNetTrainerUMambaBot3Mamba3_LowDoseContrastSim(nnUNetTrainer_LowDoseContrastSim):
    """U-Mamba bottleneck variant using the unofficial pure-PyTorch Mamba-3.

    Identical to nnUNetTrainerUMambaBot3_LowDoseContrastSim except the
    bottleneck SSM is Mamba-3 (rishikksh20/mamba3-pytorch, vendored) instead
    of Mamba-2 from state-spaces/mamba. Pure PyTorch — no CUDA/Triton kernels —
    so it runs on CPU, Turing, Apple Silicon, etc., at the cost of an O(L)
    serial scan (fine for U-Net bottleneck sequence lengths ~100-300).
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot3_mamba3_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                        num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            raise NotImplementedError("UMambaBot3Mamba3 3D not yet implemented")
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("UMambaBot3Mamba3_LowDoseContrastSim: {}".format(model))

        return model
