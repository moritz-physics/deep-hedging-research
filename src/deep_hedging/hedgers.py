"""Hedger base class, Black-Scholes delta hedger, and P&L accounting."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from deep_hedging.instruments import bs_call_delta


class Hedger(ABC):
    """Base class for any hedging policy.

    A hedger maps simulated underlying paths to a sequence of share
    positions. The position ``h_n`` (column ``n`` of the returned
    tensor) is the number of shares held during the interval
    ``[t_n, t_{n+1})``, i.e. between rebalance ``n`` and rebalance
    ``n+1``. The position vector therefore has ``n_steps`` columns,
    one less than the number of price observations.
    """

    @abstractmethod
    def positions(
        self,
        paths: torch.Tensor,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> torch.Tensor:
        """Compute share positions held during each rebalance interval.

        Parameters
        ----------
        paths : torch.Tensor
            Shape ``(n_paths, n_steps + 1)``. Column 0 is the spot at
            ``t = 0``; column ``n_steps`` is the spot at maturity.
        K : float
            Strike of the option being hedged.
        T : float
            Time to maturity (in years) at ``t = 0``.
        r : float
            Risk-free rate (annualized, continuous compounding).
        sigma : float
            Volatility used by the policy. For the BS-delta hedger
            this is the model parameter; for learned hedgers it may
            be ignored.

        Returns
        -------
        torch.Tensor
            Shape ``(n_paths, n_steps)``.
        """


class BlackScholesDeltaHedger(Hedger):
    """Holds the Black-Scholes call delta computed at each rebalance.

    The position held during ``[t_n, t_{n+1})`` is
    ``Delta_BS(S_{t_n}, K, T - t_n, r, sigma)``. At the final
    observation ``t_N = T`` no new position is taken — the option
    has expired.
    """

    def positions(
        self,
        paths: torch.Tensor,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> torch.Tensor:
        n_paths, n_obs = paths.shape
        n_steps = n_obs - 1
        dt = T / n_steps

        # Build (n_steps,) tensor of remaining times: T, T - dt, ..., dt.
        time_index = torch.arange(n_steps, device=paths.device, dtype=paths.dtype)
        remaining = T - time_index * dt

        S_at_rebalance = paths[:, :n_steps]  # (n_paths, n_steps)
        # Broadcast remaining (n_steps,) over n_paths.
        remaining_b = remaining.expand(n_paths, n_steps)

        deltas = bs_call_delta(
            S=S_at_rebalance, K=K, T=remaining_b, r=r, sigma=sigma,
        )
        # bs_call_delta returns a tensor with the dtype/device of S.
        assert isinstance(deltas, torch.Tensor)
        return deltas


def compute_pnl(
    paths: torch.Tensor,
    positions: torch.Tensor,
    payoff: torch.Tensor,
    cost_bps: float,
    initial_premium: float,
) -> torch.Tensor:
    """Terminal P&L of an option seller running the given hedge.

    The seller receives ``initial_premium`` at ``t = 0``, runs the
    hedge described by ``positions``, pays the option ``payoff`` at
    maturity, and pays linear transaction costs proportional to
    notional traded. Premium is a required argument so that callers
    must explicitly choose the framing (e.g. fair BS price, mid-market
    premium, zero for pure replication-error analysis); the helper
    does not implicitly assume the option is sold at the BS price.

    Cost model (basis points of notional):
    ``cost_n = (cost_bps / 10000) * |h_n - h_{n-1}| * S_{t_n}`` with
    ``h_{-1} = 0`` (entering from flat) and ``h_N = 0`` (unwinding at
    expiry). The unwind cost at ``t_N`` uses ``S_{t_N}``.

    Parameters
    ----------
    paths : torch.Tensor
        Shape ``(n_paths, n_steps + 1)``.
    positions : torch.Tensor
        Shape ``(n_paths, n_steps)``. Position ``h_n`` is held during
        ``[t_n, t_{n+1})``.
    payoff : torch.Tensor
        Shape ``(n_paths,)``. Option payoff at maturity, paid by the
        seller.
    cost_bps : float
        Per-trade proportional cost in basis points. Pass ``0.0`` for
        the frictionless case.
    initial_premium : float
        Premium received at ``t = 0``. Pass ``0.0`` for pure
        replication-error framing.

    Returns
    -------
    torch.Tensor
        Shape ``(n_paths,)``. Terminal P&L per path.
    """
    if positions.shape[0] != paths.shape[0]:
        raise ValueError(
            "paths and positions must agree on n_paths "
            f"(got {paths.shape[0]} vs {positions.shape[0]})"
        )
    if positions.shape[1] != paths.shape[1] - 1:
        raise ValueError(
            "positions must have n_steps columns; expected "
            f"{paths.shape[1] - 1}, got {positions.shape[1]}"
        )
    if payoff.shape != (paths.shape[0],):
        raise ValueError(
            f"payoff must have shape ({paths.shape[0]},), got {payoff.shape}"
        )

    # Hedge gains: sum_n h_n * (S_{t_{n+1}} - S_{t_n}).
    price_increments = paths[:, 1:] - paths[:, :-1]   # (n_paths, n_steps)
    hedge_pnl = (positions * price_increments).sum(dim=1)

    # Trades: from h_{-1}=0 to h_0, ..., h_{N-1} to h_N=0.
    n_paths, n_steps = positions.shape
    zero_col = torch.zeros(n_paths, 1, device=paths.device, dtype=paths.dtype)
    h_with_bookends = torch.cat([zero_col, positions, zero_col], dim=1)
    trades = (h_with_bookends[:, 1:] - h_with_bookends[:, :-1]).abs()
    # The trade at column k happens at time t_k against price S_{t_k}.
    # Columns of trades correspond to t_0, t_1, ..., t_N.
    cost_rate = cost_bps / 10_000.0
    costs = cost_rate * (trades * paths).sum(dim=1)

    return hedge_pnl - costs - payoff + initial_premium


__all__ = [
    "Hedger",
    "BlackScholesDeltaHedger",
    "compute_pnl",
]
