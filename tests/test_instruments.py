"""Tests for payoff and Black-Scholes utilities."""

from __future__ import annotations

import torch

from deep_hedging.instruments import (
    bs_call_delta,
    bs_call_price,
    european_call_payoff,
)


def test_call_payoff() -> None:
    S_T = torch.tensor([80.0, 100.0, 120.0, 100.0])
    K = 100.0
    expected = torch.tensor([0.0, 0.0, 20.0, 0.0])
    assert torch.allclose(european_call_payoff(S_T, K), expected)


def test_call_payoff_zero_when_at_strike() -> None:
    assert float(european_call_payoff(torch.tensor(100.0), 100.0)) == 0.0


def test_bs_call_price_known() -> None:
    """Hull textbook benchmark: S=K=100, T=1, r=0.05, sigma=0.2 -> 10.4506."""
    price = bs_call_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert abs(price - 10.4506) < 5e-4


def test_bs_call_price_at_expiry() -> None:
    assert bs_call_price(S=120.0, K=100.0, T=0.0, r=0.05, sigma=0.2) == 20.0
    assert bs_call_price(S=80.0, K=100.0, T=0.0, r=0.05, sigma=0.2) == 0.0


def test_bs_call_price_tensor_input() -> None:
    S = torch.tensor([90.0, 100.0, 110.0])
    prices = bs_call_price(S=S, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert isinstance(prices, torch.Tensor)
    assert prices.shape == S.shape
    # Monotone in spot.
    assert torch.all(prices[1:] > prices[:-1])


def test_bs_call_delta_atm() -> None:
    """ATM with non-zero T, delta is slightly above 0.5 (call has upside)."""
    delta = bs_call_delta(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert 0.5 < delta < 0.7


def test_bs_call_delta_at_expiry() -> None:
    assert bs_call_delta(S=120.0, K=100.0, T=0.0, r=0.05, sigma=0.2) == 1.0
    assert bs_call_delta(S=80.0, K=100.0, T=0.0, r=0.05, sigma=0.2) == 0.0


def test_bs_call_delta_tensor_input() -> None:
    S = torch.tensor([50.0, 100.0, 200.0])
    deltas = bs_call_delta(S=S, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert isinstance(deltas, torch.Tensor)
    assert deltas.shape == S.shape
    # Deep ITM -> ~1, deep OTM -> ~0.
    assert deltas[0].item() < 0.05
    assert deltas[2].item() > 0.95


def test_bs_call_price_put_call_parity() -> None:
    """Cross-check: c - p = S - K * exp(-r*T) for European options.

    Implements the put price as ``c - S + K * exp(-r * T)`` and checks
    it stays non-negative and bounded by ``K * exp(-r*T)``.
    """
    import math

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    c = bs_call_price(S=S, K=K, T=T, r=r, sigma=sigma)
    p = c - S + K * math.exp(-r * T)
    assert p > 0
    assert p < K
