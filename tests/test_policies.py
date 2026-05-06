"""Tests for HedgePolicy and NeuralHedger."""

from __future__ import annotations

import torch

from deep_hedging.policies import HedgePolicy, NeuralHedger
from deep_hedging.simulator import simulate_gbm
from deep_hedging.utils import set_seed


def test_policy_forward_shape() -> None:
    set_seed(0)
    policy = HedgePolicy()
    out = policy(torch.zeros(100, 4))
    assert out.shape == (100, 1)


def test_neural_hedger_positions_shape() -> None:
    set_seed(0)
    policy = HedgePolicy()
    hedger = NeuralHedger(policy)
    paths = simulate_gbm(
        S0=100.0, mu=0.0, sigma=0.2, T=1.0,
        n_steps=20, n_paths=200, seed=1,
    )
    positions = hedger.positions(paths, K=100.0, T=1.0, r=0.0, sigma=0.2)
    assert positions.shape == (200, 20)


def test_neural_hedger_differentiable() -> None:
    """Backprop a dummy loss through the recursion and verify all
    network parameters receive gradient signal."""
    set_seed(0)
    policy = HedgePolicy()
    hedger = NeuralHedger(policy)
    paths = simulate_gbm(
        S0=100.0, mu=0.0, sigma=0.2, T=1.0,
        n_steps=10, n_paths=50, seed=2,
    )
    positions = hedger.positions(paths, K=100.0, T=1.0, r=0.0, sigma=0.2)
    positions.sum().backward()

    found_any = False
    for p in policy.parameters():
        assert p.grad is not None, "missing gradient on a parameter"
        if p.grad.abs().sum().item() > 0:
            found_any = True
    assert found_any, "no parameter received nonzero gradient"


def test_no_lookahead_in_features() -> None:
    """Perturbing path values *after* timestep n must not change any
    position decision at or before n.

    Mechanical proof of causality: the network's recurrence is
    ``h_n = f(state_n, h_{n-1})``, so positions through index n
    depend only on path values through index n. We perturb later
    columns of ``paths`` and assert the early columns of
    ``positions`` are byte-identical."""
    set_seed(0)
    policy = HedgePolicy()
    hedger = NeuralHedger(policy)
    paths = simulate_gbm(
        S0=100.0, mu=0.0, sigma=0.2, T=1.0,
        n_steps=20, n_paths=64, seed=3,
    )
    cutoff = 10  # perturb columns from cutoff+1 onward (i.e., S_{cutoff+1} ...)

    with torch.no_grad():
        positions_a = hedger.positions(paths, K=100.0, T=1.0, r=0.0, sigma=0.2)

        paths_perturbed = paths.clone()
        paths_perturbed[:, cutoff + 1 :] *= 1.5  # arbitrary future-only perturbation

        positions_b = hedger.positions(paths_perturbed, K=100.0, T=1.0, r=0.0, sigma=0.2)

    # Positions h_0..h_cutoff use S_0..S_cutoff exclusively.
    assert torch.equal(positions_a[:, : cutoff + 1], positions_b[:, : cutoff + 1])
    # Sanity: positions after the cutoff *do* differ.
    assert not torch.equal(positions_a[:, cutoff + 1 :], positions_b[:, cutoff + 1 :])


def test_neural_hedger_zero_weights_yields_constant_position() -> None:
    """If we zero every linear layer in the network, the output
    everywhere is the bias of the final linear layer (no activation
    after the final layer). This confirms the BS-delta input is a
    *feature* and not the network *output* — i.e., the network does
    not pass BS delta straight through.
    """
    set_seed(0)
    policy = HedgePolicy()
    with torch.no_grad():
        for module in policy.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        # Set the final layer's bias to a known value.
        last_linear = [m for m in policy.modules() if isinstance(m, torch.nn.Linear)][-1]
        last_linear.bias.fill_(0.7)

    hedger = NeuralHedger(policy)
    paths = simulate_gbm(
        S0=100.0, mu=0.0, sigma=0.2, T=1.0,
        n_steps=10, n_paths=32, seed=4,
    )
    positions = hedger.positions(paths, K=100.0, T=1.0, r=0.0, sigma=0.2)
    assert torch.allclose(positions, torch.full_like(positions, 0.7))
