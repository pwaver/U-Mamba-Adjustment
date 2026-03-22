# Mamba-3 Installation on AWS (CUDA)

Activate your conda environment, then:

```bash
# Check current mamba-ssm version
pip show mamba-ssm

# Upgrade to latest (includes Mamba3)
pip install mamba-ssm --no-cache-dir --no-build-isolation

# Ensure causal-conv1d is current
pip install causal-conv1d>=1.2.0

# Verify Mamba3 is available
python -c "from mamba_ssm import Mamba3; print('Mamba3 available')"
```

If the PyPI release doesn't yet include Mamba3, install from source:

```bash
MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git --no-cache-dir --no-build-isolation
```

## Training with UMambaBot3

```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot3
```
