"""Payoff and Black-Scholes utilities for European options.

Black-Scholes computations use ``scipy.stats.norm`` for the cumulative
normal. They are not part of any computational graph in this project
(the neural hedger does not differentiate through the analytic BS
price), so the non-differentiable scipy backend is fine and matches
the textbook reference implementation.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from scipy.stats import norm

ScalarOrTensor = Union[float, torch.Tensor]


def european_call_payoff(S_T: torch.Tensor, K: float) -> torch.Tensor:
    """Terminal payoff of a European call option.

    Parameters
    ----------
    S_T : torch.Tensor
        Terminal underlying prices.
    K : float
        Strike price.

    Returns
    -------
    torch.Tensor
        ``max(S_T - K, 0)``, same shape as ``S_T``.
    """
    return torch.clamp(S_T - K, min=0.0)


def _to_numpy(x: ScalarOrTensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _wrap_like(value: np.ndarray, ref: ScalarOrTensor) -> ScalarOrTensor:
    if isinstance(ref, torch.Tensor):
        return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    if value.ndim == 0:
        return float(value)
    return value


def bs_call_price(
    S: ScalarOrTensor,
    K: float,
    T: ScalarOrTensor,
    r: float,
    sigma: float,
) -> ScalarOrTensor:
    """Black-Scholes price of a European call.

    Parameters
    ----------
    S : float or torch.Tensor
        Spot price(s).
    K : float
        Strike.
    T : float or torch.Tensor
        Time to maturity in years. Values ``<= 0`` are treated as
        already expired and return the payoff ``max(S - K, 0)``.
    r : float
        Continuously compounded risk-free rate (annualized).
    sigma : float
        Volatility (annualized). Must be non-negative.

    Returns
    -------
    float or torch.Tensor
        Call price, matching the type of ``S`` (float in, float out;
        tensor in, tensor out on the same device/dtype).
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    S_np = _to_numpy(S)
    T_np = _to_numpy(T)

    expired = T_np <= 0
    safe_T = np.where(expired, 1.0, T_np)  # placeholder to avoid /0
    sqrtT = np.sqrt(safe_T)

    if sigma == 0.0:
        # Degenerate: discounted intrinsic against forward strike.
        forward_payoff = np.maximum(S_np - K * np.exp(-r * safe_T), 0.0)
        live = forward_payoff
    else:
        d1 = (np.log(S_np / K) + (r + 0.5 * sigma * sigma) * safe_T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        live = S_np * norm.cdf(d1) - K * np.exp(-r * safe_T) * norm.cdf(d2)

    intrinsic = np.maximum(S_np - K, 0.0)
    price = np.where(expired, intrinsic, live)
    return _wrap_like(price, S)


def bs_call_delta(
    S: ScalarOrTensor,
    K: float,
    T: ScalarOrTensor,
    r: float,
    sigma: float,
) -> ScalarOrTensor:
    """Black-Scholes delta of a European call (``N(d1)``).

    Parameters
    ----------
    S, K, T, r, sigma
        See :func:`bs_call_price`.

    Returns
    -------
    float or torch.Tensor
        Call delta. For ``T <= 0`` returns ``1`` if ``S > K`` else
        ``0``.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    S_np = _to_numpy(S)
    T_np = _to_numpy(T)

    expired = T_np <= 0
    safe_T = np.where(expired, 1.0, T_np)
    sqrtT = np.sqrt(safe_T)

    if sigma == 0.0:
        live = (S_np > K * np.exp(-r * safe_T)).astype(np.float64)
    else:
        d1 = (np.log(S_np / K) + (r + 0.5 * sigma * sigma) * safe_T) / (sigma * sqrtT)
        live = norm.cdf(d1)

    at_expiry = (S_np > K).astype(np.float64)
    delta = np.where(expired, at_expiry, live)
    return _wrap_like(delta, S)


__all__ = [
    "european_call_payoff",
    "bs_call_price",
    "bs_call_delta",
]
