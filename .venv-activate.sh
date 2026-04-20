#!/bin/bash
# Quick activation script for the U-Mamba uv environment

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Run: /home/ubuntu/.local/bin/uv sync"
    exit 1
fi

echo "🐍 Activating U-Mamba environment..."
source "${VENV_PATH}/bin/activate"

echo "✓ Environment activated"
echo ""
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "📖 For setup details, see UV-ENVIRONMENT-SETUP.md"
echo ""
echo "To train a model:"
echo "  nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot"
echo ""
echo "To install mamba-ssm (if needed):"
echo "  pip install 'causal-conv1d>=1.2.0'"
echo "  MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git --no-cache-dir --no-build-isolation"
echo ""
