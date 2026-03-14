from __future__ import annotations

import os

import torch


def enable_cuda_perf_flags() -> None:
    """Enable safe CUDA performance flags for Ampere+ GPUs."""
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def maybe_mark_cudagraph_step_begin() -> None:
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(marker):
        marker()
