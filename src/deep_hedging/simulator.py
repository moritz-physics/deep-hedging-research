"""Geometric Brownian Motion path simulator."""

from __future__ import annotations

import numpy as np
import torch


def get_device() -> str:
    """Return the best available torch device string.

    Returns
    -------
    str
        ``"cuda"`` if a CUDA device is available, ``"mps"`` on Apple
        Silicon when MPS is available, otherwise ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Simulate Geometric Brownian Motion paths.

    Paths follow ``S_{t+dt} = S_t * exp((mu - 0.5 * sigma^2) * dt +
    sigma * sqrt(dt) * Z)`` with ``Z ~ N(0, 1)``. Implementation builds
    log-increments and takes a cumulative sum along the time axis,
    which avoids accumulated floating-point drift from iterative
    multiplication.

    Parameters
    ----------
    S0 : float
        Initial spot price. Must be positive.
    mu : float
        Annualized drift.
    sigma : float
        Annualized volatility. Must be non-negative.
    T : float
        Time horizon in years. Must be positive.
    n_steps : int
        Number of time steps. Must be positive.
    n_paths : int
        Number of independent paths. Must be positive.
    seed : int or None, optional
        If given, seeds both ``numpy`` and ``torch`` for
        reproducibility. Default ``None``.
    device : str, optional
        Torch device string for the returned tensor. Default
        ``"cpu"``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_paths, n_steps + 1)`` and dtype
        ``float32``. Column 0 equals ``S0`` exactly.
    """
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    dt = T / n_steps
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * (dt ** 0.5)

    Z = torch.randn(n_paths, n_steps, device=device)
    log_increments = drift + diffusion * Z

    zero_col = torch.zeros(n_paths, 1, device=device)
    log_path = torch.cat([zero_col, log_increments.cumsum(dim=1)], dim=1)
    return S0 * torch.exp(log_path)
