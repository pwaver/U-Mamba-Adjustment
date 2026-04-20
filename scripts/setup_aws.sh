#!/usr/bin/env bash
# setup_aws.sh — bootstrap a fresh clone into a trainable environment.
#
# Reproduces the local Ubuntu-x86 + Quadro RTX 4000 setup on an AWS GPU
# instance (Ubuntu 22.04/24.04). Installs:
#   1. uv         — Python package manager (if missing)
#   2. micromamba — standalone conda-channel tool for CUDA toolchain (if missing)
#   3. CUDA 12.8 toolchain (nvcc + cudart-dev + cccl) into ~/micromamba/envs/cuda128
#   4. Python deps via `uv sync --extra gpu` — builds causal-conv1d and
#      mamba-ssm from source against the toolchain in step 3
#   5. .envrc helper that activates the combined environment in one shot
#
# Usage:
#   bash scripts/setup_aws.sh                    # auto-detect GPU arch
#   TORCH_CUDA_ARCH_LIST="8.0" bash scripts/setup_aws.sh   # override
#
# Override hints by instance:
#   g5  (A10G) → 8.6
#   g6  (L4)   → 8.9
#   p4d (A100) → 8.0
#   p5  (H100) → 9.0
#
# No sudo required.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CUDA_ENV_PREFIX="$HOME/micromamba/envs/cuda128"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"

log() { printf '\n\e[1;34m[setup_aws]\e[0m %s\n' "$*"; }

# ─────────────────────────────────────────────────────────────────────────────
# 1. uv
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installer puts it in ~/.local/bin; make it reachable in this shell
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. micromamba  (only needed for CUDA toolchain)
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -x "$MICROMAMBA_BIN" ]; then
    log "Installing micromamba → $MICROMAMBA_BIN"
    mkdir -p "$(dirname "$MICROMAMBA_BIN")"
    curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C "$(dirname "$(dirname "$MICROMAMBA_BIN")")" bin/micromamba
fi
log "micromamba version: $($MICROMAMBA_BIN --version)"

# ─────────────────────────────────────────────────────────────────────────────
# 3. CUDA 12.8 toolchain
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -x "$CUDA_ENV_PREFIX/bin/nvcc" ]; then
    log "Creating CUDA 12.8 toolchain env at $CUDA_ENV_PREFIX"
    "$MICROMAMBA_BIN" create -y -p "$CUDA_ENV_PREFIX" \
        --root-prefix "$HOME/micromamba" \
        -c conda-forge -c nvidia \
        'cuda-nvcc=12.8' 'cuda-cudart-dev=12.8' 'cuda-cccl=12.8'
fi
log "nvcc: $($CUDA_ENV_PREFIX/bin/nvcc --version | tail -1)"

export CUDA_HOME="$CUDA_ENV_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pick the right sm_ target for this GPU
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Query compute capability from the first GPU; map "8.6" form directly
        CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
        if [ -n "$CAP" ]; then
            export TORCH_CUDA_ARCH_LIST="$CAP"
            log "Detected GPU compute capability: $CAP"
        fi
    fi
fi
# Fat binary fallback if we couldn't detect — covers Turing → Hopper
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"
log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Build + install all Python deps (this is the slow step, ~15–30 min)
# ─────────────────────────────────────────────────────────────────────────────
export MAX_JOBS="${MAX_JOBS:-4}"
export MAMBA_FORCE_BUILD=TRUE

log "Syncing Python deps (gpu extras — will compile causal-conv1d + mamba-ssm)"
uv sync --extra gpu --no-build-isolation

# ─────────────────────────────────────────────────────────────────────────────
# 6. Smoke test
# ─────────────────────────────────────────────────────────────────────────────
log "Running smoke test"
.venv/bin/python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
from mamba_ssm import Mamba3
from nnunetv2.nets.UMambaTSBot3_3d import MambaLayer3
m = MambaLayer3(dim=320).cuda()
x = torch.randn(1, 320, 5, 8, 8).cuda()
y = m(x)
assert y.shape == x.shape
print("Mamba3 forward OK — output shape:", tuple(y.shape))
PY

# ─────────────────────────────────────────────────────────────────────────────
# 7. Drop a .envrc so future shells just do: `source .envrc`
# ─────────────────────────────────────────────────────────────────────────────
cat > .envrc <<EOF
# Auto-generated by scripts/setup_aws.sh — source this (or use direnv).
export CUDA_HOME="\$HOME/micromamba/envs/cuda128"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}"
source "$(pwd)/.venv/bin/activate"
EOF

log "Done. Activate future shells with:  source .envrc"
