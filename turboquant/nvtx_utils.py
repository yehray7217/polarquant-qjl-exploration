from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(name: str):
    """
    Lightweight NVTX helper.

    If CUDA is unavailable, behaves as a no-op context manager.
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield
