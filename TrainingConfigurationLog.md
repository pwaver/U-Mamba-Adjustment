024-12-25 21:21:47.054810: Patch size changed from [512, 512] to [512, 512]
2024-12-25 21:21:48.272626: Using optimizer AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-05
    foreach: None
    fused: None
    initial_lr: 0.0001
    lr: 0.0001
    maximize: False
    weight_decay: 0.01
)
2024-12-25 21:21:48.272794: Using scheduler <nnunetv2.training.lr_scheduler.polylr.PolyLRScheduler object at 0x7e0c601c86a0>
RobustCrossEntropyLoss with class weights (weight)  tensor([ 1., 20., 20.], device='cuda:0')

This is the configuration used by this training:
Configuration name: 2d
 {'data_identifier': 'nnUNetPlans_2d', 'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 6, 'patch_size': [512, 512], 'median_image_size_in_voxels': [512.0, 512.0], 'spacing': [1.0, 1.0], 'normalization_schemes': ['ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization', 'ZScoreNormalization'], 'use_mask_for_norm': [False, False, False, False, False], 'UNet_class_name': 'PlainConvUNet', 'UNet_base_num_features': 32, 'n_conv_per_stage_encoder': [2, 2, 2, 2, 2, 2, 2, 2], 'n_conv_per_stage_decoder': [2, 2, 2, 2, 2, 2, 2], 'num_pool_per_axis': [7, 7], 'pool_op_kernel_sizes': [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]], 'conv_kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]], 'unet_max_num_features': 512, 'resampling_fn_data': 'resample_data_or_seg_to_shape', 'resampling_fn_seg': 'resample_data_or_seg_to_shape', 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape', 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'batch_dice': True} 

These are the global plan.json settings:
 {'dataset_name': 'Dataset332_Angiography', 'plans_name': 'nnUNetPlans', 'original_median_spacing_after_transp': [999.0, 1.0, 1.0], 'original_median_shape_after_transp': [1, 512, 512], 'image_reader_writer': 'NaturalImage2DIO', 'transpose_forward': [0, 1, 2], 'transpose_backward': [0, 1, 2], 'experiment_planner_used': 'ExperimentPlanner', 'label_manager': 'LabelManager', 'foreground_intensity_properties_per_channel': {'0': {'max': 255.0, 'mean': 128.83082120117274, 'median': 128.0, 'min': 0.0, 'percentile_00_5': 47.0, 'percentile_99_5': 215.0, 'std': 30.65544790876828}, '1': {'max': 255.0, 'mean': 125.46021537534533, 'median': 125.0, 'min': 0.0, 'percentile_00_5': 44.0, 'percentile_99_5': 211.0, 'std': 30.87612196864302}, '2': {'max': 255.0, 'mean': 117.41717650203141, 'median': 116.0, 'min': 0.0, 'percentile_00_5': 41.0, 'percentile_99_5': 202.0, 'std': 29.663179432865363}, '3': {'max': 255.0, 'mean': 125.08628344076286, 'median': 124.0, 'min': 0.0, 'percentile_00_5': 44.0, 'percentile_99_5': 211.0, 'std': 30.914632633602274}, '4': {'max': 255.0, 'mean': 128.32018867431148, 'median': 128.0, 'min': 0.0, 'percentile_00_5': 46.0, 'percentile_99_5': 214.0, 'std': 30.735461117834287}}} 
