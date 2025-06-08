from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LowDoseContrastSim import nnUNetTrainer_LowDoseContrastSim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UxLSTMBot_3d import get_uxlstm_bot_3d_from_plans
from nnunetv2.nets.UxLSTMBot_2d import get_uxlstm_bot_2d_from_plans


class nnUNetTrainerUxLSTMBot_LowDose(nnUNetTrainer_LowDoseContrastSim):
    """nnUNet trainer combining UxLSTM network architecture with low dose contrast simulation.
    
    This trainer uses the UxLSTM network architecture while applying low dose simulation
    data augmentation for fluoroscopic cardiac angiography training.
    """
    
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """Build UxLSTM network architecture.
        
        Args:
            plans_manager: Plans manager containing training configuration.
            dataset_json: Dataset configuration dictionary.
            configuration_manager: Configuration manager for the specific setup.
            num_input_channels: Number of input channels.
            enable_deep_supervision: Whether to enable deep supervision.
            
        Returns:
            Configured UxLSTM network model.
        """
        if len(configuration_manager.patch_size) == 2:
            model = get_uxlstm_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_uxlstm_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        
        print("UxLSTMBot with Low Dose Simulation: {}".format(model))

        return model 