#!/usr/bin/env bash
# docker-build.sh — build the training image.
#
# Usage:
#   bash scripts/docker-build.sh                     # fat-binary image
#   TORCH_CUDA_ARCH_LIST="8.6" bash scripts/docker-build.sh   # g5/A10G-only
#   IMAGE_TAG=umamba-ts:g5  bash scripts/docker-build.sh
#
# The build does NOT require a GPU on the build host — the CUDA kernels are
# compiled against the arch(es) listed in TORCH_CUDA_ARCH_LIST.
#
# Arch hints:
#   RTX 4000 (home)  -> 7.5        p4d (A100) -> 8.0
#   g5  (A10G)       -> 8.6        p5  (H100) -> 9.0
#   g6  (L4)         -> 8.9
#   fat binary       -> "7.5;8.0;8.6;8.9;9.0"   (default)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARCH="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"
TAG="${IMAGE_TAG:-umamba-ts:local}"
JOBS="${MAX_JOBS:-4}"

echo "[docker-build] image tag:             $TAG"
echo "[docker-build] TORCH_CUDA_ARCH_LIST:  $ARCH"
echo "[docker-build] MAX_JOBS:              $JOBS"

docker build \
    --build-arg TORCH_CUDA_ARCH_LIST="$ARCH" \
    --build-arg MAX_JOBS="$JOBS" \
    -t "$TAG" \
    .

echo
echo "[docker-build] done. Image: $TAG"
docker image ls "$TAG"
