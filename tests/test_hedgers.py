"""Tests for the Hedger interface, BlackScholesDeltaHedger, and compute_pnl."""

from __future__ import annotations

import torch

from deep_hedging.hedgers import BlackScholesDeltaHedger, compute_pnl
from deep_hedging.instruments import bs_call_price, european_call_payoff
from deep_hedging.simulator import simulate_gbm


def test_bs_delta_hedger_shape() -> None:
    paths = simulate_gbm(
        S0=100.0, mu=0.05, sigma=0.2, T=1.0,
        n_steps=20, n_paths=500, seed=0,
    )
    hedger = BlackScholesDeltaHedger()
    positions = hedger.positions(paths, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert positions.shape == (500, 20)


def test_bs_delta_hedger_initial_position_matches_bs_at_t0() -> None:
    """At t = 0 every path is at S0, so h_0 must equal Delta(S0, K, T, r, sigma)."""
    from deep_hedging.instruments import bs_call_delta

    paths = simulate_gbm(
        S0=100.0, mu=0.05, sigma=0.2, T=1.0,
        n_steps=20, n_paths=200, seed=1,
    )
    hedger = BlackScholesDeltaHedger()
    positions = hedger.positions(paths, K=100.0, T=1.0, r=0.05, sigma=0.2)
    expected = bs_call_delta(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert torch.allclose(positions[:, 0], torch.full((200,), float(expected)), atol=1e-6)


def test_bs_delta_hedger_terminal_call_in_money() -> None:
    """At the last rebalance (T - dt remaining), if S >> K then delta -> 1."""
    paths = torch.full((10, 21), 200.0)  # 20 steps, deeply ITM throughout
    hedger = BlackScholesDeltaHedger()
    positions = hedger.positions(paths, K=100.0, T=1.0, r=0.05, sigma=0.2)
    last = positions[:, -1]
    assert torch.all(last > 0.99)


def test_bs_delta_hedger_terminal_call_out_of_money() -> None:
    paths = torch.full((10, 21), 50.0)  # deeply OTM
    hedger = BlackScholesDeltaHedger()
    positions = hedger.positions(paths, K=100.0, T=1.0, r=0.05, sigma=0.2)
    last = positions[:, -1]
    assert torch.all(last < 0.01)


def test_compute_pnl_zero_position_zero_cost() -> None:
    """Holding zero shares with zero costs and no premium yields P&L = -payoff."""
    paths = simulate_gbm(
        S0=100.0, mu=0.05, sigma=0.2, T=1.0,
        n_steps=10, n_paths=200, seed=0,
    )
    payoff = european_call_payoff(paths[:, -1], K=100.0)
    positions = torch.zeros(200, 10)
    pnl = compute_pnl(paths, positions, payoff, cost_bps=0.0, initial_premium=0.0)
    assert torch.allclose(pnl, -payoff)


def test_compute_pnl_no_position_with_premium() -> None:
    """No hedge but premium received -> mean P&L near zero under r=0.

    Under ``r = 0`` the BS price equals the expected payoff, so an
    unhedged seller's mean P&L should be zero. The unhedged variance
    is large, so we use enough paths that the sample-mean standard
    error is well below the test tolerance.
    """
    n_paths = 50_000
    paths = simulate_gbm(
        S0=100.0, mu=0.0, sigma=0.2, T=1.0,
        n_steps=10, n_paths=n_paths, seed=0,
    )
    payoff = european_call_payoff(paths[:, -1], K=100.0)
    positions = torch.zeros(n_paths, 10)
    premium = float(bs_call_price(S=100.0, K=100.0, T=1.0, r=0.0, sigma=0.2))
    pnl = compute_pnl(paths, positions, payoff, cost_bps=0.0, initial_premium=premium)
    assert abs(pnl.mean().item()) < 0.2


def test_compute_pnl_costs_only_with_static_unit_position() -> None:
    """Hold exactly 1 share throughout. Initial entry trade is 1 share at S0.
    Final unwind is 1 share at S_T. No intermediate trades. Total cost should
    equal cost_rate * (S_0 + S_T)."""
    n_paths, n_steps = 5, 4
    paths = torch.tensor(
        [[100.0, 101.0, 102.0, 101.0, 105.0]] * n_paths
    )
    positions = torch.ones(n_paths, n_steps)
    payoff = torch.zeros(n_paths)
    cost_bps = 10.0  # 10 bps = 0.001
    pnl = compute_pnl(paths, positions, payoff, cost_bps=cost_bps, initial_premium=0.0)
    # Hedge P&L = 1 * (105 - 100) = 5. Costs = 0.001 * (100 + 105) = 0.205.
    expected = 5.0 - 0.001 * (100.0 + 105.0)
    assert torch.allclose(pnl, torch.full((n_paths,), expected), atol=1e-5)


def test_compute_pnl_shape_validation() -> None:
    paths = torch.zeros(10, 6)
    positions_bad = torch.zeros(10, 4)
    payoff = torch.zeros(10)
    try:
        compute_pnl(paths, positions_bad, payoff, cost_bps=0.0, initial_premium=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched n_steps")
