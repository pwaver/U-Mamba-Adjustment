We assume we are in the conda environment (in this case called nnunet or umamba, depending on the github repository). We use here the example of a dataset named Dataset332_Angiography. It is in the nnUNet standard location of ./data/nnUNet_raw. Our data are furnished as images frames of size 512x512 pixels. The default 2d UNet planner compresses it in the encoder to 512x4x4, then decodes it with the help of skip connections back to 512x512, one for each class. On the hypothesis that 512x4x4 leads to over-regularization, we wish to reduce the compression. In the example here, we wish the encoder to go to 512x16x16. This is the meaning of the term "edge16" in the nnUNet plan.

The below assumes you are in the correct directory, have directory environment variables set up as necessary, and have activate the correctly installed conda environment. The enviroment variables seem to be needed for nnunetv2 but not umamba. The environment variables that give this file system structure can be given in the shell by calls as
export nnUNet_raw="/billb/github/nnUNet-Adjustment/data/nnUNet_raw"
export nnUNet_preprocessed="/billb/github/nnUNet-Adjustment/data/nnUNet_preprocessed"
export nnUNet_results="/billb/github/nnUNet-Adjustment/data/nnUNet_results"

I think these can be snuck into the conda environment. Execute conda env list to find the conda environments and their paths. Go to the environment that has nnUNet installed and look for etc/conda/activate.d. There you will create or find a file called env_vars.sh. Open it and place the export statements into it.

The first step is to check the data integrity with a call such as 

nnUNetv2_plan_and_preprocess -d 332  --verify_dataset_integrity

First step is to create a new experiment planner this can inherit from the class ExperimentPlanner that is in the file default_experiment_planner.py. In this example, the inheritance class is called CompressionAdjustmentExperimentPlanner. The constructor has a call to super.__init__(….) as below and has within it a new reference value and a new value for min edge length.

We can limit the number of epochs by adding a new parameter to the constructor nnUNetTrainer, self.max_num_epochs=250. This is done in the file nnUNetTrainer.py. Watchout for the confusion since the adjustments to the training are specified in two files, nnUNetTrainer.py and ExperimentPlanner.py.

In default_experiment_planner.py we have

class CompressionAdjustmentExperimentPlanner(ExperimentPlanner):
	def __init__(….):
		super.__init__(….)
		….
		self.UNet_featuremap_min_edge_length = 16
		….

Then one specifies this experiment planner class in a call to the command line program that plans the experiment (this is nnUNet terminology for designing the nnUNet structure), nnUNetv2_plan_experiment. The name of the experiment planner class is passed via a parameter -pl. The generated plan name is specified via parameter -overwrite_plans_name. In the plans name, specify the modifications to the trainer (such as max_num_epochs and the loss function) and the experiment planner (such as the compression edge length).

nnUNetv2_plan_experiment -d 332 -pl CompressionAdjustmentExperimentPlanner -overwrite_plans_name plans_unet_edge8_epochs250

This creates a new experiment plan file in json format. The command line program nnUNetv2_plan_experiment places the plan file in its standard place within the file structure under the data directory. In our naming  convention, the 16 refers to the edge length of maximal compression of the encoding. If say the maximal edge in compression is 32 then one would use a name as plans_unet_edge32. If so, then in the instructions below rewrite 16 to 32 in the various plans-related names.

A habit to consider is to append information about the planned loss function to the plans file name. In the example above, the loss function is DC_and_CE_loss-w-1-20-20. This information otherwise may be specified in the lossFunctionSpecifier key of the call to prediction as per the file predict_from_raw_data_with_model_exports.py. The risk of error is that the naming of the loss function in the plans file does not match the loss function that is actually used. This is because at this time the plans file is created the loss information is not programmatically propagated to the trainer. So if the loss function is specified in the plans file name, but not in the lossFunctionSpecifier, then the planner will not find the loss function that is actually used.

To do this one observes the creation by the experiment planner a file under nnUNet_preprocessed/Dataset332_Angiography called plans_unet_edge16.json as specified in the call to nnUNetv2_plan_experiment above. The subsequent call to trainer then expects to find this file and furthermore find the training data in the expected location. Note that the directory where it expects to find the 2d data has "_2d" appended to the name, as plans_unet_edge16_2d, illustrated below.

You may find a directory named nnUNetPlans_2d that has the training data under it. A quick an dirty step is to rename the directory nnUNetPlans_2d to plans name specified above. In the example above, the directory would be renamed to plans_unet_edge8_epochs250_2d. This can be done from within the cursor GUI.

This is to recap how to find the plans file and training data. File structure:

data
	nnUNet_preprocessed
		Dataset332_Angiography
			plans_unet_edge8_epochs250_2d
				Angio_0001_seq.npy (expect to copy the data here or rename its wrapper directory)
				….
			plans_unet_edge8_epochs250.json (created by the call to nnUNetv2_plan_experiment)

Once this is set up one calls the training program and specifies the training configuration in a parameter -p.


To train and later to predict, the plans file needs to be included under the parameter -p. This causes two effects. The first is to look for the plans file in the path nnUNet_preprocessed/Dataset332_Angiography/plans_unet_edge16.json. The other is to look for the data in the path nnUNet_preprocessed/Dataset332_Angiography/plans_unet_edge16_2d.

With the above set up, the training program command line call is something like:

nnUNetv2_train 332 2d all -tr nnUNetTrainerUMambaBot -p plans_unet_edge8_epochs250

The "-tr" parameter specifies the trainer class. In this example case, it is nnUNetTrainerUMambaBot. If get torch.cuda.OutOfMemoryError, then the batch size needs to be reduced. This may be done in the plans file plans_unet_edge16.json. The batch size is specified in the plans file under the key "batch_size". Reducing the batch size will reduce the memory required for the model. The "all" positional parameter specifies that all folds are to be trained.

The prediction call is something like:

nnUNetv2_predict -i path-to-input-folder -o path-to-output-folder -d 332 -c 2d -tr nnUNetTrainer —disable_tta -f 1 -p plans_unet_edge16 -chk checkpoint_best.pth

check:

nnUNetv2_predict_with_model_exports -i /home/billb/github/U-Mamba-Adjustment/data/nnUNet_input -o /home/billb/github/U-Mamba-Adjustment/data/nnUNet_output  -d 332  -c 2d -tr nnUNetTrainerUMambaBot  --disable_tta -f all -lossFunctionSpecifier DC_and_CE_loss-w-1-20-20 -p plans_unet_edge8_epochs250

The U-Mamba distribution contains a path at nnunetv2/nets that contains models for inference that are brought in with restoration by dill. For example, we have

$ ls /home/ubuntu/U-Mamba-Adjustment/umamba/nnunetv2/nets
EMUNet.py  UMambaBot_2d.py  UMambaBot_3d.py  UMambaEnc_2d.py  UMambaEnc_3d.py

At inference, we need to add the path to the nnunetv2/nets directory to the PYTHONPATH environment variable . This is done by adding the following to the .bashrc file:

export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/U-Mamba-Adjustment/umamba/nnunetv2/nets"

or by amending the python path within the interpreter, as 

import sys
sys.path.append("/home/ubuntu/U-Mamba-Adjustment/umamba/nnunetv2/nets")

This may require cuda per se and not run on mps since mamba-ssm is cuda-dependent.

BTW, an alternate strategy to conserve GPU RAM is to boot not into the GUI when training. So as follows:

     sudo systemctl set-default multi-user.target

	 sudo reboot
or
	gnome-session-quit

then afterward go back to GUI mode

     sudo systemctl set-default graphical.target
     sudo reboot
or
	sudo systemctl start gdm3


By the way, in the onnx export call we wish to specify the input shape and further that the first dimension is the batch size. This is done as follows:

onnx.export(netModel, dummy_input, "model.onnx", verbose=True, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})


## **Export to PyTorch**

We wish to export the model to PyTorch for consideration of deployment in the AWI cloud. There are several steps involved and not all of the program environments are compatible with one another.

### **Export to ONNX**

We start with the export to ONNX from within the nnunetv2 or umamba workspaces. As of this writing 3 Sep 2024, umamba seems to train best with the umamba conda environment not having the most recent PyTorch versions. For export, however, we need a more recent PyTorch. So we need to create a new conda environment for export, umamba-upgrade. 

To export for Wolfram, we seem to need to use the umamba conda environment with the line
torch.onnx.export(self.network, dummy_input, onnx_model_path, export_params=True, opset_version=18, do_constant_folding=True, verbose=True)
in the method _internal_maybe_mirror_and_predict of the file predict_from_raw_data_with_model_exports.py.
The specification of dynamic axes seems to confuse Wolfram import, so we need to remove that.

### **From ONNX to PyTorch**

We wish to export to PyTorch as a stage for deploying to the AWI cloud, which is PyTorch based. This seems to require the umamba-upgrade conda environment, having the most recent PyTorch, 2.4.0 at this writing. The downstream onnx import prefers opset 15 to 18. It is able to make use of dynamic axes. We seem to need to export with the line        torch.onnx.export(self.network, dummy_input, onnx_model_path, export_params=True, opset_version=15, do_constant_folding=True, verbose=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, training=torch.onnx.TrainingMode.EVAL)
in the method _internal_maybe_mirror_and_predict of the file predict_from_raw_data_with_model_exports.py.

As an aside, as noted above, from within the umamba-upgrade conda environment, the command line program is as this example:
nnUNetv2_predict_with_model_exports -i /home/billb/github/U-Mamba-Adjustment/data/nnUNet_input -o /home/billb/github/U-Mamba-Adjustment/data/nnUNet_output  -d 332  -c 2d -tr nnUNetTrainerUMambaBot  --disable_tta -f all -lossFunctionSpecifier DC_and_CE_loss-w-1-20-20 -p plans_unet_edge8_epochs250

This generates a file named nnUNetTrainerUMambaBot_2d_332_plans_unet_edge8_epochs250_DC_and_CE_loss-w-1-20-20.onnx in the output directory, here /home/billb/github/U-Mamba-Adjustment/data/nnUNet_output. By our internal convention, the onnx file is copied to the net model archive directory as per the python code rootPath = '/mnt/SliskiDrive/AWI/AWIBuffer/' if os.name == 'posix' else '/Volumes/Crucial X8/AWI/Data/'. This is the directory where later python and Wolfram code looks for the model.

We wish a PyTorch version of the model that is self-contained. Options for this seem to be with the onnx framework, which seems to work, and the PyTorch jit and torchscript frameworks, which do not seem to work. They do not work in the sense that they are not aware that the 1st dimension of the input is the batch size. The onnx framework seems to know it, so our roundabout strategy is to export to onnx then import and convert it to a PyTorch version. At this time we do this with the program OnnxToTorchscript.py in the nnunetv2/utilities directory. We run this in the umamba-upgrade conda environment.

Then we apply the onnx-derived PyTorch model to the data, again in the umamba-upgrade conda environment. It seems to need the latest PyTorch for both creation and use.

The following causes a segmentation fault:

We get a PyTorch copy of the onnx model, adjust the file name to specify provenance, and save it with the lines

rootPath = '/mnt/SliskiDrive/AWI/AWIBuffer/' if os.name == 'posix' else '/Volumes/Crucial X8/AWI/Data/'

onnxPath = rootPath + "UMambaBot-plans_unet_edge8_epochs250_2d-DC_and_CE_loss-w-1-20-20.onnx"

modelPerOnnx = convert(onnxPath)

torchModelPath = onnxPath.replace(".onnx", "-torch-onnx.pt")

torch.save(modelPerOnnx, torchModelPath)

These onnx and torch models seem to understand the batch size as the first dimension.
