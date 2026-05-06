"""Risk measures and the end-to-end training loop for the deep hedger.

Two risk measures are provided as candidate training objectives:

* :func:`cvar_loss` — Conditional Value-at-Risk (expected shortfall).
  Equals minus the mean of the worst ``alpha`` fraction of P&Ls.
  Implemented with :func:`torch.topk`, which is differentiable
  through the selected elements.

* :func:`entropic_risk` — entropic risk measure
  ``log E[exp(-lambda * pnl)] / lambda``. Smooth in P&L, no top-k
  selection, often easier to optimise than CVaR.

Both are coherent risk measures (Bühler et al. 2019, §2.2) and lower
is better. CVaR-50% is the default.

The training loop simulates a fresh batch of paths every epoch
(simulation-based training; no path is reused), runs the policy
forward over that batch, computes terminal P&L, and steps Adam on
the chosen risk measure. Validation uses a fixed seed every epoch so
its trajectory is comparable across epochs.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from deep_hedging.hedgers import compute_pnl
from deep_hedging.policies import HedgePolicy, NeuralHedger


def cvar_loss(pnl: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Expected shortfall at level ``alpha``.

    Returns the mean of the worst ``alpha`` fraction of P&Ls,
    negated so that "lower is better" as a loss. Concretely::

        cvar_loss = mean of the smallest ceil(alpha * n) values of -pnl
                  = - mean of the smallest ceil(alpha * n) values of pnl

    Parameters
    ----------
    pnl : torch.Tensor
        1-D tensor of terminal P&Ls (positive = good).
    alpha : float, optional
        Fraction of paths considered worst case. Default ``0.5``
        (median-and-below).

    Returns
    -------
    torch.Tensor
        Scalar loss; minimise to maximise expected shortfall.
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    n = pnl.shape[0]
    k = max(1, int(math.ceil(alpha * n)))
    worst_losses, _ = torch.topk(-pnl, k)
    return worst_losses.mean()


def entropic_risk(pnl: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Entropic risk measure ``log E[exp(-lambda * pnl)] / lambda``.

    Smooth in ``pnl`` and grows exponentially with the size of
    losses, but does not introduce a top-k selection. As
    ``lambda -> 0`` it converges to ``-mean(pnl)``.

    Parameters
    ----------
    pnl : torch.Tensor
        1-D tensor of terminal P&Ls.
    lambda_ : float, optional
        Risk aversion parameter; must be positive. Default ``1.0``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda must be positive, got {lambda_}")
    n = pnl.shape[0]
    log_n = torch.log(torch.tensor(float(n), dtype=pnl.dtype, device=pnl.device))
    return (torch.logsumexp(-lambda_ * pnl, dim=0) - log_n) / lambda_


def train_hedger(
    policy: HedgePolicy,
    simulate_paths: Callable[..., torch.Tensor],
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    premium: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    cost_bps: float,
    n_epochs: int = 200,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    risk_measure: Callable[[torch.Tensor], torch.Tensor] = cvar_loss,
    val_seed: int = 1_000_000,
    val_batch_size: int = 4096,
    log_every: int = 10,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train ``policy`` end-to-end on simulated GBM paths.

    Parameters
    ----------
    policy : HedgePolicy
        Network to train (mutated in place).
    simulate_paths : callable
        Returns simulated paths given keyword arguments
        ``n_paths`` and (optionally) ``seed``. Typically a
        ``functools.partial`` over GBM parameters.
    payoff_fn : callable
        Maps terminal spot tensor to payoff tensor.
    premium : float
        Initial premium received by the seller. Used in the P&L
        accounting only — the policy does not see it.
    K, T, r, sigma : float
        Option and market parameters passed through to the hedger.
    cost_bps : float
        Proportional transaction cost (basis points).
    n_epochs : int, optional
        Number of training steps (each step = one fresh batch).
        Default ``200``.
    batch_size : int, optional
        Number of paths per training step. Default ``1024``.
    learning_rate : float, optional
        Adam learning rate. Default ``1e-3``.
    risk_measure : callable, optional
        Loss function on P&L. Default :func:`cvar_loss`.
    val_seed : int, optional
        Seed for the validation batch (held fixed across epochs so
        trajectories are comparable). Default ``1_000_000``.
    val_batch_size : int, optional
        Validation batch size. Default ``4096``.
    log_every : int, optional
        Compute val loss / log progress every ``log_every`` epochs.
        Default ``10``.
    device : str, optional
        Torch device. Default ``"cpu"``.
    verbose : bool, optional
        Print per-log step. Default ``True``.

    Returns
    -------
    dict
        ``{"history": {"epoch": [...], "loss": [...], "val_loss": [...]}}``.
    """
    policy.to(device)
    hedger = NeuralHedger(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {"epoch": [], "loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        paths = simulate_paths(n_paths=batch_size).to(device)
        positions = hedger.positions(paths, K=K, T=T, r=r, sigma=sigma)
        payoff = payoff_fn(paths[:, -1])
        pnl = compute_pnl(
            paths=paths, positions=positions, payoff=payoff,
            cost_bps=cost_bps, initial_premium=premium,
        )
        loss = risk_measure(pnl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                val_paths = simulate_paths(
                    n_paths=val_batch_size, seed=val_seed,
                ).to(device)
                val_positions = hedger.positions(
                    val_paths, K=K, T=T, r=r, sigma=sigma,
                )
                val_payoff = payoff_fn(val_paths[:, -1])
                val_pnl = compute_pnl(
                    paths=val_paths, positions=val_positions, payoff=val_payoff,
                    cost_bps=cost_bps, initial_premium=premium,
                )
                val_loss_value = risk_measure(val_pnl).item()
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["val_loss"].append(val_loss_value)
            if verbose:
                print(
                    f"  epoch {epoch:4d}: train={loss.item():+8.4f}  "
                    f"val={val_loss_value:+8.4f}"
                )

    return {"history": history}


__all__ = ["cvar_loss", "entropic_risk", "train_hedger"]
