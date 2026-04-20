# U-Mamba uv Environment - Setup Complete ✓

## What's Ready

Your Python environment for U-Mamba training is **fully configured** with:

- ✅ **Python 3.11.11** in isolated venv
- ✅ **PyTorch 2.11.0+cu130** (CUDA 12.1/13.0 compatible)
- ✅ **nnUNet v2.1.1** framework (editable install)
- ✅ **Biomedical imaging**: SimpleITK, MONAI, nibabel, OpenCV, h5py
- ✅ **ML/Scientific**: NumPy, SciPy, Pandas, Scikit-learn, Scikit-image
- ✅ **Visualization**: Matplotlib, Seaborn
- ✅ **ONNX**: Model export/inference
- ⚠️  **Mamba SSM**: Requires manual installation (see below)

## Quick Start

```bash
source .venv/bin/activate
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot
```

## Next: Install Mamba SSM

```bash
source .venv/bin/activate
pip install causal-conv1d>=1.2.0
MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git \
  --no-cache-dir --no-build-isolation
```

## Documentation

- **Setup Details**: See `UV-ENVIRONMENT-SETUP.md`
- **Training Guide**: See `README.md`

## Environment Location

- Virtual environment: `.venv/`
- Python executable: `.venv/bin/python`
- Requirements files: `requirements-*.txt`

✨ Environment is ready for U-Mamba training!
