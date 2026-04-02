from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LowDoseContrastSim import nnUNetTrainer_LowDoseContrastSim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaTSEnc_3d import get_umamba_ts_enc_3d_from_plans
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans


class nnUNetTrainerUMambaTSEnc_LowDoseContrastSim(nnUNetTrainer_LowDoseContrastSim):
    """Temporal-first (temporospatial) variant of nnUNetTrainerUMambaTSEnc
    with low dose contrast simulation data augmentation.

    For 3D data, uses UMambaTSEnc which permutes the SSM scan order so that
    the temporal (depth) axis varies fastest — forcing the SSM to model
    pixel-wise time series before spatial context.

    For 2D data, falls back to the standard UMambaEnc (no temporal axis).
    """
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_ts_enc_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("UMambaTSEnc_LowDoseContrastSim: {}".format(model))

        return model
