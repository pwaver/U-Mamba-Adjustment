# syntax=docker/dockerfile:1.6
#
# U-Mamba / nnUNetv2 training container.
#
# Mirrors scripts/setup_aws.sh inside a container, on top of NVIDIA's CUDA
# 12.8 devel image (nvcc shipped -> no micromamba toolchain needed).
#
# Build:
#   docker build \
#     --build-arg TORCH_CUDA_ARCH_LIST="8.6" \
#     -t umamba-ts:local .
#
# Arch hints (set via --build-arg TORCH_CUDA_ARCH_LIST):
#   RTX 4000 local -> 7.5      p4d (A100) -> 8.0
#   g5  (A10G)     -> 8.6      p5  (H100) -> 9.0
#   g6  (L4)       -> 8.9
#   fat binary (default)       -> "7.5;8.0;8.6;8.9;9.0"
#
# Run (on a host with NVIDIA Container Toolkit):
#   docker run --gpus all --rm \
#     -v ~/Angiostore/nnUNet_raw:/data/raw \
#     -v ~/Angiostore/nnUNet_preprocessed:/data/preprocessed \
#     -v ~/Angiostore/nnUNet_results:/data/results \
#     umamba-ts:local \
#     nnUNetv2_train 430 3d_fullres 0 -tr nnUNetTrainerUMambaTSBot3

ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ARG MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    MAMBA_FORCE_BUILD=TRUE \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System deps:
#   git             pulling mamba-ssm from a pinned commit
#   curl + unzip    uv installer + AWS CLI v2 installer
#   build-essential C/C++ extension compiles
#   ca-certificates TLS for git/pip/curl
#   libgl1 libglib  runtime for opencv-python / nibabel / SimpleITK
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl unzip ca-certificates build-essential \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2 (needed if the entrypoint runs `aws s3 sync` to fetch data)
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscli.zip /tmp/aws

# uv — installs to /root/.local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# CUDA already at /usr/local/cuda in the devel image — point the compile env at it
ENV CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:/root/.local/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /app

# Slow layer: resolve + build torch, causal-conv1d, mamba-ssm (~20 min).
# Cached unless pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./

# Bootstrap: create a Python 3.11 venv pre-seeded with setuptools + wheel +
# packaging + ninja. causal-conv1d and mamba-ssm don't declare setuptools /
# torch as build deps, so under --no-build-isolation they must already be in
# the venv. ninja is the build tool for the CUDA kernel compiles.
RUN uv venv --python 3.11 --seed /app/.venv \
 && uv pip install --python /app/.venv/bin/python \
        setuptools wheel packaging ninja

# Two-pass install so mamba-ssm's setup.py finds torch already installed:
#  Pass 1: base deps (brings in torch 2.10 + its cuda userspace wheels).
#  Pass 2: gpu extras (builds causal-conv1d + mamba-ssm against installed torch).
RUN uv sync --frozen --no-build-isolation
RUN uv sync --frozen --no-build-isolation --extra gpu

# Put the project venv on PATH so later RUN / ENTRYPOINT find nnUNetv2_train
ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV=/app/.venv

# Install the local nnunetv2 package (not referenced from pyproject.toml).
# Pulls in its biomedical-stack deps declared in umamba/setup.py
# (batchgenerators, dynamic-network-architectures, monai, SimpleITK, etc.).
# Done as its own layer so edits to trainer code don't bust the mamba-ssm build.
COPY umamba/ ./umamba/
RUN uv pip install ./umamba

# Remaining repo bits the trainer / entrypoint might need
COPY scripts/ ./scripts/
COPY README.md ./

# Standard nnUNet dir layout — mount host dirs over these at `docker run` time
RUN mkdir -p /data/raw /data/preprocessed /data/results
ENV nnUNet_raw=/data/raw \
    nnUNet_preprocessed=/data/preprocessed \
    nnUNet_results=/data/results

# Build-time import check (no GPU needed). Catches packaging regressions early.
RUN python -c "import torch; print('torch', torch.__version__); \
from mamba_ssm import Mamba3; \
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaTSBot3 import nnUNetTrainerUMambaTSBot3; \
print('import OK: Mamba3 + nnUNetTrainerUMambaTSBot3')"

RUN chmod +x /app/scripts/docker-entrypoint.sh
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["nnUNetv2_train", "--help"]
