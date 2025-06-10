from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LowDoseContrastSim import nnUNetTrainer_LowDoseContrastSim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UxLSTMBot_3d import get_uxlstm_bot_3d_from_plans
from nnunetv2.nets.UxLSTMBot_2d import get_uxlstm_bot_2d_from_plans
from typing import Union, List, Tuple


class nnUNetTrainerUxLSTMBot_LowDose(nnUNetTrainer_LowDoseContrastSim):
    """nnUNet trainer combining UxLSTM network architecture with low dose contrast simulation.
    
    This trainer uses the UxLSTM network architecture while applying low dose simulation
    data augmentation for fluoroscopic cardiac angiography training.
    """
    
    @staticmethod
    def build_network_architecture(*args, **kwargs) -> nn.Module:
        """Build UxLSTM network architecture.
        
        This method supports both calling conventions:
        1. New format: build_network_architecture(plans_manager, dataset_json, configuration_manager, 
                                                 num_input_channels, enable_deep_supervision=True)
        2. Old format: build_network_architecture(architecture_class_name, arch_init_kwargs, 
                                                 arch_init_kwargs_req_import, num_input_channels, 
                                                 num_output_channels, enable_deep_supervision=True)
        
        Returns:
            Configured UxLSTM network model.
        """
        # Detect calling convention by checking the number of positional arguments
        if len(args) >= 3 and isinstance(args[0], PlansManager):
            # New format: (plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
            plans_manager = args[0]
            dataset_json = args[1]
            configuration_manager = args[2]
            num_input_channels = args[3]
            enable_deep_supervision = args[4] if len(args) > 4 else kwargs.get('enable_deep_supervision', True)
            
            return nnUNetTrainerUxLSTMBot_LowDose._build_network_from_plans(
                plans_manager, dataset_json, configuration_manager, 
                num_input_channels, enable_deep_supervision
            )
        
        elif len(args) >= 5 and isinstance(args[0], str):
            # Old format: (architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, 
            #              num_input_channels, num_output_channels, enable_deep_supervision)
            architecture_class_name = args[0]
            arch_init_kwargs = args[1]
            arch_init_kwargs_req_import = args[2]
            num_input_channels = args[3]
            num_output_channels = args[4]
            enable_deep_supervision = args[5] if len(args) > 5 else kwargs.get('enable_deep_supervision', True)
            
            # For UxLSTM networks, we can't build them without the plans structure
            # So we need to raise an informative error
            raise RuntimeError(
                f"UxLSTMBot_LowDose trainer requires the new calling convention with PlansManager, "
                f"but received old format with architecture_class_name='{architecture_class_name}'. "
                f"The UxLSTM networks require additional planning information that is not available "
                f"in the old architecture format. Please use the newer nnUNet version or update "
                f"your plans files."
            )
        
        else:
            raise ValueError(f"Invalid arguments to build_network_architecture: args={args}, kwargs={kwargs}")
    
    @staticmethod
    def _build_network_from_plans(plans_manager: PlansManager,
                                  dataset_json,
                                  configuration_manager: ConfigurationManager,
                                  num_input_channels,
                                  enable_deep_supervision: bool = True) -> nn.Module:
        """Build UxLSTM network architecture from plans.
        
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