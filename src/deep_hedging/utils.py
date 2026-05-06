"""Reproducibility and device helpers shared across the project."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG that affects this project's outputs.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU and, when
    available, CUDA / MPS). Call this at the top of every script
    *before* any tensor allocation or ``torch.nn`` module
    instantiation, since weight initialization consumes torch's
    global RNG.

    Parameters
    ----------
    seed : int
        Non-negative integer seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> str:
    """Return the best available torch device string.

    Re-exported from :mod:`deep_hedging.simulator` for convenience.
    """
    from deep_hedging.simulator import get_device as _get_device
    return _get_device()


__all__ = ["set_seed", "get_device"]
