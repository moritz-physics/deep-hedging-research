"""Neural-network hedging policy and the hedger that drives it.

Architecture follows the modern simplification of Bühler et al.
(2019): a single feedforward network with weights shared across
timesteps, called once per rebalance with time-to-maturity passed in
as an input feature. This is more parameter-efficient and trains
faster than the original "one network per timestep" formulation
while remaining expressive enough for the optimal hedge.

Inputs at time ``t_n`` (one row per path):

    [ log(S_n / K),  (T - t_n) / T,  h_{n-1},  BS_delta(S_n, T - t_n) ]

The Black-Scholes delta is included as a feature so the network can
trivially copy it in the frictionless limit and learn corrections to
it under frictions, rather than having to discover the entire shape
of the optimal hedge from scratch.
"""

from __future__ import annotations

import torch
from torch import nn

from deep_hedging.hedgers import Hedger
from deep_hedging.instruments import bs_call_delta


N_FEATURES = 4


class HedgePolicy(nn.Module):
    """Time-conditioned MLP from state to share position.

    Parameters
    ----------
    hidden_layers : list of int, optional
        Hidden layer widths. Default ``[32, 32, 32]`` — three
        layers of 32 units each (~3.4k parameters), more than
        sufficient since the optimal hedge is a smooth function of
        a low-dimensional state.
    """

    def __init__(self, hidden_layers: list[int] | None = None) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [32, 32, 32]
        widths = [N_FEATURES, *hidden_layers]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(widths[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map state features to a position scalar.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(batch, 4)``. Columns are
            ``(log_moneyness, ttm_frac, prev_position, bs_delta)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)``. Position in shares.
        """
        return self.net(features)


class NeuralHedger(Hedger):
    """Hedger that rolls a :class:`HedgePolicy` forward through time.

    The position recursion ``h_n = policy([state_n, h_{n-1}])``
    requires a Python loop over timesteps because each step's input
    depends on the previous step's output. Within each step, all
    paths are computed in parallel via the network's vectorized
    forward pass. This is fast because ``n_steps`` is small (50)
    and the per-step batch is large (typically 1024+ paths).

    Notes
    -----
    Black-Scholes deltas at every ``(path, step)`` are pre-computed
    once before the temporal recursion and indexed per step. The BS
    delta is an input feature only — its computation goes through
    SciPy and is not part of the autograd graph, so this caching is
    purely an efficiency optimization.
    """

    def __init__(self, policy: HedgePolicy) -> None:
        self.policy = policy

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

        # Spot prices and remaining time at each rebalance instant.
        S_grid = paths[:, :n_steps]                                  # (P, N)
        time_index = torch.arange(n_steps, device=paths.device, dtype=paths.dtype)
        ttm_grid_1d = T - time_index * dt                            # (N,)
        ttm_grid = ttm_grid_1d.expand(n_paths, n_steps)              # (P, N)

        # BS-delta feature, computed once for the entire grid.
        bs_delta_grid = bs_call_delta(
            S=S_grid, K=K, T=ttm_grid, r=r, sigma=sigma,
        )
        assert isinstance(bs_delta_grid, torch.Tensor)
        bs_delta_grid = bs_delta_grid.detach()  # not part of autograd graph

        log_moneyness_grid = torch.log(S_grid / K)
        ttm_frac_grid = ttm_grid / T

        h_prev = torch.zeros(n_paths, device=paths.device, dtype=paths.dtype)
        positions_per_step: list[torch.Tensor] = []
        for n in range(n_steps):
            features = torch.stack(
                [
                    log_moneyness_grid[:, n],
                    ttm_frac_grid[:, n],
                    h_prev,
                    bs_delta_grid[:, n],
                ],
                dim=1,
            )
            h_n = self.policy(features).squeeze(-1)
            positions_per_step.append(h_n)
            h_prev = h_n

        return torch.stack(positions_per_step, dim=1)


__all__ = ["HedgePolicy", "NeuralHedger", "N_FEATURES"]
