"""Tests for the GBM path simulator."""

from __future__ import annotations

import math

import torch

from deep_hedging.simulator import simulate_gbm


def test_gbm_shape() -> None:
    paths = simulate_gbm(
        S0=100.0, mu=0.05, sigma=0.2, T=1.0,
        n_steps=50, n_paths=1_000, seed=0,
    )
    assert paths.shape == (1_000, 51)


def test_gbm_starts_at_S0() -> None:
    S0 = 100.0
    paths = simulate_gbm(
        S0=S0, mu=0.05, sigma=0.2, T=1.0,
        n_steps=50, n_paths=1_000, seed=0,
    )
    assert torch.all(paths[:, 0] == S0)


def test_gbm_seeded_reproducibility() -> None:
    kwargs = dict(
        S0=100.0, mu=0.05, sigma=0.2, T=1.0,
        n_steps=50, n_paths=1_000, seed=42,
    )
    a = simulate_gbm(**kwargs)
    b = simulate_gbm(**kwargs)
    assert torch.equal(a, b)


def test_gbm_log_return_stats() -> None:
    mu, sigma, T, n_steps = 0.05, 0.2, 1.0, 252
    n_paths = 100_000
    paths = simulate_gbm(
        S0=100.0, mu=mu, sigma=sigma, T=T,
        n_steps=n_steps, n_paths=n_paths, seed=123,
    )
    log_returns = torch.log(paths[:, 1:] / paths[:, :-1])

    dt = T / n_steps
    expected_mean = (mu - 0.5 * sigma * sigma) * dt
    expected_std = sigma * math.sqrt(dt)

    empirical_mean = log_returns.mean().item()
    empirical_std = log_returns.std().item()

    # Sample-mean std for a single increment: expected_std / sqrt(N).
    n_samples = n_paths * n_steps
    mean_tol = 4.0 * expected_std / math.sqrt(n_samples)
    assert abs(empirical_mean - expected_mean) < mean_tol
    # Std estimate is much tighter than the mean estimate; 0.5% is plenty.
    assert abs(empirical_std - expected_std) / expected_std < 5e-3


def test_gbm_no_negative_prices() -> None:
    paths = simulate_gbm(
        S0=100.0, mu=0.05, sigma=0.5, T=1.0,
        n_steps=252, n_paths=10_000, seed=7,
    )
    assert torch.all(paths > 0)


def test_gbm_martingale_under_zero_drift() -> None:
    """With ``mu = 0`` the process ``S_t`` is a martingale, so
    ``E[S_T / S_0] = 1``. ``simulate_gbm`` treats ``mu`` as the SDE
    drift and applies the ``-0.5 * sigma**2`` Itô correction
    internally, so the martingale condition is ``mu = 0``. A sign
    error on that correction would push the empirical mean to
    ``exp(sigma**2 * T) ~= 1.04`` — well outside tolerance.
    """
    sigma = 0.2
    paths = simulate_gbm(
        S0=100.0, mu=0.0, sigma=sigma, T=1.0,
        n_steps=252, n_paths=100_000, seed=2024,
    )
    ratio_mean = (paths[:, -1] / paths[:, 0]).mean().item()
    assert abs(ratio_mean - 1.0) < 5e-3
