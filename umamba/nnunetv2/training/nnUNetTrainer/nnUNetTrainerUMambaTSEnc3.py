from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaTSEnc3_3d import get_umamba_ts_enc3_3d_from_plans
from nnunetv2.nets.UMambaEnc3_2d import get_umamba_enc3_2d_from_plans

class nnUNetTrainerUMambaTSEnc3(nnUNetTrainer):
    """Temporal-first variant using Mamba-3 (complex-valued SSM).

    For 3D data, uses UMambaTSEnc3 which combines temporal-first scan order
    with Mamba-3's complex-valued state updates — enabling the SSM to
    represent oscillatory cardiac dynamics along the temporal axis.
    Mamba layer at every encoder stage.

    For 2D data, falls back to UMambaEnc3 (Mamba-2, no temporal axis).
    """
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_enc3_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_ts_enc3_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")


        print("UMambaTSEnc3: {}".format(model))

        return model
