"""Vendored copy of rishikksh20/mamba3-pytorch.

Source:  https://github.com/rishikksh20/mamba3-pytorch
Pinned:  master @ 1aa707ddddaec4dcbc2b02042bd4d9630cecfaaf (2026-03-18)
Paper:   Lahoti et al., "Mamba-3: Improved Sequence Modeling using State Space
         Principles", arXiv:2603.15569 (2026-03-16)

Pure PyTorch + einops — no CUDA / Triton / TileLang / CuteDSL kernels.
Runs on CPU, Turing, Ampere, Hopper, Apple Silicon, etc., at the cost of an
O(L) serial scan (acceptable for U-Net-bottleneck-sized sequences).
"""
from .mamba3 import Mamba3

__all__ = ["Mamba3"]
