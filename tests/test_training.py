"""Tests for risk measures and the training loop."""

from __future__ import annotations

from functools import partial

import torch

from deep_hedging.instruments import bs_call_price, european_call_payoff
from deep_hedging.policies import HedgePolicy
from deep_hedging.simulator import simulate_gbm
from deep_hedging.training import cvar_loss, entropic_risk, train_hedger
from deep_hedging.utils import set_seed


def test_cvar_loss_known() -> None:
    """Hand-computable: P&L = [-3, -1, 0, 2], alpha=0.5 -> worst 2 are
    -3 and -1, mean of (3, 1) = 2.0."""
    pnl = torch.tensor([-3.0, -1.0, 0.0, 2.0])
    assert cvar_loss(pnl, alpha=0.5).item() == 2.0


def test_cvar_loss_alpha_one_is_neg_mean() -> None:
    pnl = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert cvar_loss(pnl, alpha=1.0).item() == -2.5


def test_cvar_loss_differentiable() -> None:
    x = torch.randn(100, requires_grad=True)
    pnl = x * 2.0 + 1.0
    loss = cvar_loss(pnl, alpha=0.5)
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_entropic_risk_small_lambda_approximates_neg_mean() -> None:
    """As lambda -> 0, entropic risk -> -mean(pnl). Use lambda=1e-3."""
    pnl = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    er = entropic_risk(pnl, lambda_=1e-3).item()
    expected = -pnl.mean().item()
    assert abs(er - expected) < 1e-2


def test_entropic_risk_differentiable() -> None:
    x = torch.randn(50, requires_grad=True)
    pnl = x.exp()
    loss = entropic_risk(pnl, lambda_=0.5)
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def _gbm_factory(S0: float, mu: float, sigma: float, T: float, n_steps: int):
    """Return a `simulate_paths(n_paths, seed=...)` closure."""

    def _simulate(n_paths: int, seed: int | None = None) -> torch.Tensor:
        return simulate_gbm(
            S0=S0, mu=mu, sigma=sigma, T=T,
            n_steps=n_steps, n_paths=n_paths, seed=seed,
        )
    return _simulate


def test_train_reduces_loss() -> None:
    """End-to-end smoke test: training should reduce val loss."""
    set_seed(0)
    policy = HedgePolicy(hidden_layers=[16, 16])
    simulate_paths = _gbm_factory(S0=100.0, mu=0.0, sigma=0.2, T=1.0, n_steps=10)
    premium = float(bs_call_price(S=100.0, K=100.0, T=1.0, r=0.0, sigma=0.2))
    out = train_hedger(
        policy=policy,
        simulate_paths=simulate_paths,
        payoff_fn=partial(european_call_payoff, K=100.0),
        premium=premium,
        K=100.0, T=1.0, r=0.0, sigma=0.2,
        cost_bps=0.0,
        n_epochs=30,
        batch_size=256,
        learning_rate=5e-3,
        log_every=5,
        verbose=False,
    )
    val_history = out["history"]["val_loss"]
    assert val_history[-1] < val_history[0], (
        f"val loss did not decrease: {val_history}"
    )


def test_train_seeded_reproducibility() -> None:
    """Same seed and hyperparams -> identical loss history."""

    def run_once() -> list[float]:
        set_seed(123)
        policy = HedgePolicy(hidden_layers=[16, 16])
        simulate_paths = _gbm_factory(
            S0=100.0, mu=0.0, sigma=0.2, T=1.0, n_steps=10,
        )
        premium = float(bs_call_price(S=100.0, K=100.0, T=1.0, r=0.0, sigma=0.2))
        out = train_hedger(
            policy=policy,
            simulate_paths=simulate_paths,
            payoff_fn=partial(european_call_payoff, K=100.0),
            premium=premium,
            K=100.0, T=1.0, r=0.0, sigma=0.2,
            cost_bps=5.0,
            n_epochs=20,
            batch_size=256,
            learning_rate=1e-3,
            log_every=5,
            verbose=False,
        )
        return out["history"]["val_loss"]

    a = run_once()
    b = run_once()
    assert a == b, f"non-deterministic: {a} vs {b}"
