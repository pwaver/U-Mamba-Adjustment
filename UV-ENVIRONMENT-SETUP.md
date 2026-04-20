# U-Mamba UV Python Environment Setup

This document describes the Python environment for running U-Mamba model training with CUDA 12.1 support.

## Environment Overview

A `uv`-managed Python 3.11 virtual environment has been created at `.venv/` with all necessary dependencies for:
- **PyTorch 2.11** with CUDA 12.1 support
- **nnUNet v2.1.1** framework for biomedical image segmentation
- **Mamba SSM** state-space models (requires separate installation)
- All biomedical imaging libraries: SimpleITK, MONAI, nibabel, OpenCV, etc.
- ONNX export and inference utilities

## Activation

Activate the environment:

```bash
source .venv/bin/activate
```

To deactivate:

```bash
deactivate
```

## Installed Components

### Core Training Framework
- **PyTorch**: 2.11.0+cu130 (CUDA 13.0 compatible, works with CUDA 12.1)
- **nnUNet v2.1.1**: Framework for biomedical image segmentation
- **MONAI 1.3.0**: Medical imaging toolkit
- **Dynamic Network Architectures**: nnUNet dependency for flexible model architectures

### Biomedical Image Processing
- **SimpleITK 2.5.3**: Image I/O and preprocessing
- **nibabel 5.4.2**: NIfTI file handling
- **OpenCV 4.13.0**: Image processing
- **h5py 3.16.0**: HDF5 file access (for data loading)
- **DICOM2NIfTI**: DICOM to NIfTI conversion
- **MedPy**: Medical image processing utilities
- **Tifffile**: Multi-frame image I/O

### Visualization & ML
- **Matplotlib 3.10.8**: Plotting
- **Seaborn 0.13.2**: Statistical visualization
- **Scikit-image 0.26.0**: Image processing algorithms
- **Scikit-learn 1.8.0**: ML utilities
- **SciPy 1.17.1**: Scientific computing
- **Pandas 3.0.2**: Data frames
- **NumPy 2.4.4**: Numerical computing

### Serialization & Export
- **ONNX 1.21.0**: Open Neural Network Exchange format
- **ONNXRuntime 1.23.2**: ONNX model inference
- **ONNXScript 0.6.2**: ONNX model generation

### Build & Utilities
- **CMake 4.3.1**: Build system
- **Dill 0.4.1**: Extended pickling
- **tqdm 4.67.3**: Progress bars
- **PyYAML 6.0.3**: Configuration files
- **Requests 2.33.1**: HTTP library
- **Setuptools, Wheel, Pip**: Python packaging

## Manual Installation of Mamba SSM

The Mamba SSM package requires CUDA toolkit for building from source. Due to build complexities in the standard pip environment, you have two options:

### Option 1: Install from Conda Environment (Recommended)

If you have the `umambaVerbatim` conda environment with mamba-ssm already installed:

```bash
source .venv/bin/activate

# Copy mamba-ssm packages from conda env to venv
python -m pip install --no-index --find-links /opt/conda/lib/python3.10/site-packages \
  mamba-ssm causal-conv1d

# Or, from the conda env directly:
# cp /opt/conda/lib/python3.10/site-packages/mamba* .venv/lib/python3.11/site-packages/
# cp /opt/conda/lib/python3.10/site-packages/causal* .venv/lib/python3.11/site-packages/
```

### Option 2: Install from GitHub (Requires CUDA Toolkit)

If you have CUDA 12.1 toolkit installed:

```bash
source .venv/bin/activate

# Install causal-conv1d first
pip install causal-conv1d>=1.2.0

# Install mamba-ssm from source
MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git \
  --no-cache-dir --no-build-isolation
```

### Verify Installation

After installing mamba-ssm:

```bash
source .venv/bin/activate

python -c "
import torch
from mamba_ssm import Mamba
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ Mamba SSM available')
try:
    from mamba_ssm import Mamba3
    print('✓ Mamba3 available')
except ImportError:
    print('  Note: Mamba3 not available (requires latest source build)')
"
```

## Using the Environment for Training

### 1. Prepare Dataset

Follow nnUNet dataset format guidelines: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

### 2. Plan & Preprocess

```bash
source .venv/bin/activate

nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### 3. Train 2D Model

```bash
source .venv/bin/activate

# U-Mamba_Bot model
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot

# U-Mamba_Enc model
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaEnc

# U-Mamba_Bot3 model (requires Mamba3)
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot3
```

### 4. Train 3D Model

```bash
source .venv/bin/activate

# U-Mamba_Bot model
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaBot

# U-Mamba_Enc model
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaEnc
```

### 5. Inference

```bash
source .venv/bin/activate

# Single model prediction
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER \
  -d DATASET_ID -c 2d -f all \
  -tr nnUNetTrainerUMambaBot --disable_tta

# With model export
nnUNetv2_predict_with_model_exports \
  -i INPUT_FOLDER -o OUTPUT_FOLDER \
  -d DATASET_ID -c 2d -f all \
  -tr nnUNetTrainerUMambaBot
```

## Python Version

- **Expected:** Python 3.11.11
- **Location:** `.venv/bin/python`

Verify with:

```bash
.venv/bin/python --version
```

## CUDA Support

The environment is configured for **CUDA 12.1** but is compatible with earlier CUDA versions:

- **CUDA 12.1** (default): PyTorch wheels from `https://download.pytorch.org/whl/cu121`
- **CUDA 11.8**: Reinstall PyTorch using:
  ```bash
  .venv/bin/pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch torchvision --force-reinstall
  ```

Check CUDA support:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

## Requirements Files

Two requirements files are included:

- **requirements-cuda121.txt**: All dependencies including PyTorch for CUDA 12.1
- **requirements-without-mamba.txt**: All dependencies except mamba-ssm/causal-conv1d (for development)
- **requirements-base.txt**: Minimal base packages (PyTorch + core scientific libraries)

To reinstall or update:

```bash
.venv/bin/pip install -i https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r requirements-cuda121.txt
```

## Project Structure

```
U-Mamba-Adjustment/
├── .venv/                           # Virtual environment
├── umamba/                          # U-Mamba source code (editable install)
│   ├── nnunetv2/                    # nnUNet v2 framework
│   │   ├── training/
│   │   │   └── nnUNetTrainer/       # Custom trainers (UMambaBot, UMambaEnc, etc.)
│   │   ├── inference/
│   │   └── ...
│   └── setup.py
├── pyproject.toml                   # Project metadata and dependencies
├── requirements-cuda121.txt         # Full dependencies with mamba-ssm exclusion
├── requirements-without-mamba.txt   # Excludes problematic source-build packages
├── requirements-base.txt            # Core packages only
├── .python-version                  # Python version specification (3.11)
├── UV-ENVIRONMENT-SETUP.md          # This file
└── README.md
```

## Troubleshooting

### Issue: CUDA out of memory

Reduce batch size or image size in nnUNet configuration. For 3D models, use `3d_fullres` instead of `3d_cascade`.

### Issue: nnUNetTrainerUMambaBot not found

Ensure `umamba` package is installed in editable mode:

```bash
source .venv/bin/activate
cd umamba
pip install -e .
cd ..
```

### Issue: Mamba SSM import fails

Install mamba-ssm following "Manual Installation of Mamba SSM" section above.

### Issue: Poor GPU memory usage with 3D models

Enable memory-efficient Mamba configuration in trainer settings. See `umamba/nnunetv2/training/nnUNetTrainer/` for available options.

### Issue: AMP (Automatic Mixed Precision) causes NaN

Use the NoAMP trainer variant:

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBotNoAMP
```

## Updating the Environment

To add new packages:

```bash
source .venv/bin/activate
pip install package_name
```

To update an existing package:

```bash
source .venv/bin/activate
pip install --upgrade package_name
```

To freeze current state:

```bash
.venv/bin/pip freeze > requirements-frozen.txt
```

## References

- **nnUNet Documentation**: https://github.com/MIC-DKFZ/nnUNet
- **U-Mamba Paper**: https://arxiv.org/abs/2401.04722
- **Mamba SSM**: https://github.com/state-spaces/mamba
- **PyTorch CUDA**: https://pytorch.org/get-started/previous-versions/

## Support

For issues with:
- **nnUNet training**: https://github.com/MIC-DKFZ/nnUNet/issues
- **U-Mamba models**: https://github.com/bowang-lab/U-Mamba/issues
- **Mamba SSM**: https://github.com/state-spaces/mamba/issues
