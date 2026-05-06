"""Neural hedger vs Black-Scholes delta — comparison across cost levels.

Trains a single time-conditioned MLP per cost level and evaluates it
against the Black-Scholes delta benchmark on a *shared held-out
batch* of 100k paths. Cost levels: 0, 5, 25, 50 basis points.

Premium framing
---------------
The seller receives a fixed premium equal to the *frictionless*
Black-Scholes price at all cost levels. This intentionally exposes
the cost burden directly in the histograms (mean P&L drops as
cost_bps grows) rather than absorbing it into a cost-adjusted
price. Both the BS and neural hedgers see the same premium, so the
comparison is policy-against-policy with cost as the only varying
external parameter.

Held-out evaluation
-------------------
A single eval seed (``EVAL_SEED``) is shared across cost levels, so
the *only* axis varying across rows of the comparison table is the
cost level — the simulated paths and BS-delta positions are
identical from row to row.

Anti-look-ahead audit
---------------------
1. Network input at timestep ``n`` is
   ``(log(S_n / K), (T - t_n) / T, h_{n-1}, BS_delta(S_n, T - t_n))``
   — every component is observable at or before ``t_n``.
2. ``h_{n-1}`` is the network's own output from the previous step,
   which by induction depends only on ``S_0..S_{n-1}``.
3. The training loss is terminal P&L, computed from positions
   determined by the causal recursion above. Path data after the
   position decision never influences the position decision.
4. The held-out evaluation uses a different seed than training.
5. ``tests/test_policies.py::test_no_lookahead_in_features`` proves
   the recursion is causal mechanically: perturbing future path
   columns leaves earlier position decisions byte-identical.

Wall-clock estimate
-------------------
~3-5 minutes total on CPU (Apple Silicon). Each cost level trains
for 200 epochs of 1024-path batches; eval is on 100k paths × 50
timesteps.
"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_hedging import (
    BlackScholesDeltaHedger,
    HedgePolicy,
    NeuralHedger,
    bs_call_price,
    compute_pnl,
    cvar_loss,
    european_call_payoff,
    set_seed,
    simulate_gbm,
    train_hedger,
)


# -- Configuration --------------------------------------------------------

S0 = 100.0
K = 100.0
T = 1.0
R = 0.0
SIGMA = 0.20
N_STEPS = 50

TRAIN_SEED = 42
EVAL_SEED = 999_999
EVAL_PATHS = 100_000

BATCH_SIZE = 1024
LR = 1e-3
HIDDEN_LAYERS = [32, 32, 32]

COST_LEVELS_BPS = [0.0, 5.0, 25.0, 50.0]

# Per-cost training budget. 0 bps is at optimum and 5 bps is
# noise-dominated by ~200 epochs; 25 / 50 bps benefit from longer
# training because the optimal policy diverges most from BS there.
EPOCHS_PER_COST = {
    0.0: 200,
    5.0: 200,
    25.0: 500,
    50.0: 500,
}

DEVICE = "cpu"  # See plan Q1: stay on CPU for session 02.

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# -- Helpers --------------------------------------------------------------

def cvar50(pnl: torch.Tensor) -> float:
    sorted_pnl, _ = torch.sort(pnl)
    cutoff = sorted_pnl.numel() // 2
    return sorted_pnl[:cutoff].mean().item()


def summary_stats(pnl: torch.Tensor) -> dict[str, float]:
    return {
        "mean": pnl.mean().item(),
        "std": pnl.std().item(),
        "q05": torch.quantile(pnl, 0.05).item(),
        "q95": torch.quantile(pnl, 0.95).item(),
        "cvar50": cvar50(pnl),
    }


def gbm_factory() -> Callable[..., torch.Tensor]:
    """Return a `simulate_paths(n_paths, seed=...)` closure with
    fixed (S0, mu=R, sigma, T, n_steps).

    Note: training uses ``seed=None`` (uses current global RNG state)
    for fresh paths each epoch, while validation passes a fixed
    seed so the trajectory is comparable across epochs."""
    def _simulate(n_paths: int, seed: int | None = None) -> torch.Tensor:
        return simulate_gbm(
            S0=S0, mu=R, sigma=SIGMA, T=T,
            n_steps=N_STEPS, n_paths=n_paths, seed=seed,
        )
    return _simulate


def convergence_diagnostic(
    history: dict, cost: float, n_epochs: int,
) -> tuple[float, str | None]:
    """Inspect the last 5 logged validation losses for residual descent.

    Computes the relative drop ``(prev - curr) / |prev|`` across the 4
    consecutive windows formed by the last 5 log points and returns
    their mean. Fires a warning when the mean drop per window exceeds
    0.3%, which indicates the run was undertrained at the chosen
    epoch budget. Replaces the earlier 2-point / 1% detector, which
    was tighter than the actual decay rate at 50 bps and therefore
    failed to fire on a clearly-still-descending trajectory.

    Returns
    -------
    mean_drop_per_window : float
        Average relative drop per 10-epoch window over the last 4
        windows. Positive = still descending.
    warning : str or None
        A formatted warning string when the threshold trips, else
        ``None``.
    """
    val = history["val_loss"]
    if len(val) < 5:
        return 0.0, None
    last5 = val[-5:]
    drops = []
    for prev, curr in zip(last5[:-1], last5[1:]):
        if prev == 0.0:
            continue
        drops.append((prev - curr) / abs(prev))
    if not drops:
        return 0.0, None
    mean_drop = sum(drops) / len(drops)
    threshold = 0.003  # 0.3% per 10-epoch window
    if mean_drop > threshold:
        warn = (
            f"  WARNING: cost_bps={cost} appears undertrained at "
            f"{n_epochs} epochs.\n"
            f"           Mean per-window val-loss drop over last 5 "
            f"log points: {mean_drop * 100:+.3f}% (>{threshold * 100:.1f}%).\n"
            f"           Last 5 val losses: {[round(v, 4) for v in last5]}"
        )
        return mean_drop, warn
    return mean_drop, None


# -- Main pipeline --------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Session 02 — Neural hedger vs Black-Scholes delta")
    print("=" * 72)
    print(f"  S0={S0}, K={K}, T={T}, r={R}, sigma={SIGMA}")
    print(f"  n_steps={N_STEPS}, train batch={BATCH_SIZE}, lr={LR}")
    print(f"  cost levels (bps): {COST_LEVELS_BPS}")
    print(f"  epochs per cost: {EPOCHS_PER_COST}")
    print(f"  device: {DEVICE}")

    simulate_paths = gbm_factory()
    payoff_fn = partial(european_call_payoff, K=K)
    premium = float(bs_call_price(S=S0, K=K, T=T, r=R, sigma=SIGMA))
    print(f"  premium (BS price at t=0): {premium:.4f}")

    # Held-out evaluation set — shared across all cost levels and both
    # policies, so the only thing varying across rows of the report is the
    # cost level.
    print(f"\nGenerating shared held-out set: "
          f"{EVAL_PATHS} paths, seed={EVAL_SEED}.")
    eval_paths = simulate_paths(n_paths=EVAL_PATHS, seed=EVAL_SEED)
    eval_payoff = payoff_fn(eval_paths[:, -1])

    bs_hedger = BlackScholesDeltaHedger()
    bs_positions = bs_hedger.positions(eval_paths, K=K, T=T, r=R, sigma=SIGMA)

    # -- Train one policy per cost level ---------------------------------
    histories: dict[float, dict] = {}
    eval_results: list[dict] = []
    neural_positions_per_cost: dict[float, torch.Tensor] = {}

    for cost in COST_LEVELS_BPS:
        n_epochs = EPOCHS_PER_COST[cost]
        print(f"\n--- Training neural hedger at cost_bps = {cost} "
              f"({n_epochs} epochs) ---")
        # Seed before policy instantiation so weight init is reproducible.
        set_seed(TRAIN_SEED + int(cost))
        policy = HedgePolicy(hidden_layers=HIDDEN_LAYERS)
        out = train_hedger(
            policy=policy,
            simulate_paths=simulate_paths,
            payoff_fn=payoff_fn,
            premium=premium,
            K=K, T=T, r=R, sigma=SIGMA,
            cost_bps=cost,
            n_epochs=n_epochs,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            risk_measure=cvar_loss,
            val_seed=EVAL_SEED + 1,  # not the same as the held-out seed
            val_batch_size=4096,
            log_every=10,
            device=DEVICE,
            verbose=True,
        )
        histories[cost] = out["history"]
        mean_drop, warn = convergence_diagnostic(
            out["history"], cost, n_epochs,
        )
        print(
            f"  convergence: mean per-window val drop over last 5 log "
            f"points = {mean_drop * 100:+.3f}%"
        )
        if warn:
            print(warn)

        # Evaluate on shared held-out set.
        with torch.no_grad():
            neural_hedger = NeuralHedger(policy)
            neural_positions = neural_hedger.positions(
                eval_paths, K=K, T=T, r=R, sigma=SIGMA,
            )
        neural_positions_per_cost[cost] = neural_positions

        bs_pnl = compute_pnl(
            paths=eval_paths, positions=bs_positions, payoff=eval_payoff,
            cost_bps=cost, initial_premium=premium,
        )
        nn_pnl = compute_pnl(
            paths=eval_paths, positions=neural_positions, payoff=eval_payoff,
            cost_bps=cost, initial_premium=premium,
        )
        nn_beats_bs = (nn_pnl > bs_pnl).float().mean().item()
        eval_results.append({
            "cost_bps": cost,
            "bs": summary_stats(bs_pnl),
            "nn": summary_stats(nn_pnl),
            "nn_beats_bs_frac": nn_beats_bs,
            "bs_pnl": bs_pnl,
            "nn_pnl": nn_pnl,
        })

    # -- Zero-cost gate --------------------------------------------------
    # cvar50() returns the mean of the worst 50% of P&L (signed); higher
    # (less negative) is better. The user's spec wrote the gate in
    # loss-form (lower better); the equivalent here is
    #   nn_cvar > bs_cvar + threshold     <=>     nn beats BS by > threshold.
    # BS is provably optimal in the frictionless limit, so any material
    # outperformance at 0 bps indicates a look-ahead leak or training bug.
    zero = next(r for r in eval_results if r["cost_bps"] == 0.0)
    bs_cvar = zero["bs"]["cvar50"]
    nn_cvar = zero["nn"]["cvar50"]
    threshold = max(0.05, 0.05 * abs(bs_cvar))
    if nn_cvar > bs_cvar + threshold:
        raise RuntimeError(
            "Zero-cost gate triggered: neural hedger beats BS by "
            f"{nn_cvar - bs_cvar:+.4f} CVaR-50% at 0 bps "
            f"(threshold = {threshold:.4f}). BS is provably optimal "
            "in the frictionless limit; this indicates a look-ahead "
            "leak or a bug in the training/eval pipeline. "
            "Investigate before reporting hedge results."
        )
    print(f"\nZero-cost gate: BS CVaR-50% = {bs_cvar:+.4f}, "
          f"NN CVaR-50% = {nn_cvar:+.4f}, "
          f"diff = {nn_cvar - bs_cvar:+.4f} (threshold {threshold:.4f}). OK.")

    # -- Comparison table ------------------------------------------------
    print("\n" + "=" * 92)
    print("Comparison: BS-delta vs Neural hedger on held-out 100k paths")
    print("=" * 92)
    header = (
        f"  {'cost(bps)':>10} | "
        f"{'BS mean':>9}  {'BS std':>8}  {'BS CVaR50':>10} | "
        f"{'NN mean':>9}  {'NN std':>8}  {'NN CVaR50':>10} | "
        f"{'NN-BS CVaR':>11}  {'NN>BS frac':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in eval_results:
        bs, nn = r["bs"], r["nn"]
        diff = nn["cvar50"] - bs["cvar50"]
        print(
            f"  {r['cost_bps']:>10.1f} | "
            f"{bs['mean']:+9.4f}  {bs['std']:8.4f}  {bs['cvar50']:+10.4f} | "
            f"{nn['mean']:+9.4f}  {nn['std']:8.4f}  {nn['cvar50']:+10.4f} | "
            f"{diff:+11.4f}  {r['nn_beats_bs_frac']:>10.3f}"
        )

    # -- CSV -------------------------------------------------------------
    csv_path = RESULTS_DIR / "02_neural_hedger_comparison.csv"
    fieldnames = [
        "cost_bps",
        "bs_mean", "bs_std", "bs_q05", "bs_q95", "bs_cvar50",
        "nn_mean", "nn_std", "nn_q05", "nn_q95", "nn_cvar50",
        "nn_minus_bs_cvar50",
        "nn_beats_bs_frac",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in eval_results:
            bs, nn = r["bs"], r["nn"]
            writer.writerow({
                "cost_bps": r["cost_bps"],
                **{f"bs_{k}": v for k, v in bs.items()},
                **{f"nn_{k}": v for k, v in nn.items()},
                "nn_minus_bs_cvar50": nn["cvar50"] - bs["cvar50"],
                "nn_beats_bs_frac": r["nn_beats_bs_frac"],
            })
    print(f"\nWrote {csv_path}")

    # -- Figure 1: training loss curves ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for cost in COST_LEVELS_BPS:
        h = histories[cost]
        ax.plot(h["epoch"], h["val_loss"], label=f"{cost:.0f} bps")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation CVaR-50% (lower = better)")
    ax.set_title("Training curves across cost levels")
    ax.legend(title="cost_bps", frameon=False)
    fig.tight_layout()
    p1 = RESULTS_DIR / "02_loss_curves.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print(f"Wrote {p1}")

    # -- Figure 2: P&L histograms 2x2 ------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    for ax, r in zip(axes.flat, eval_results):
        bs_pnl = r["bs_pnl"].numpy()
        nn_pnl = r["nn_pnl"].numpy()
        lo = float(min(bs_pnl.min(), nn_pnl.min()))
        hi = float(max(bs_pnl.max(), nn_pnl.max()))
        bins = np.linspace(lo, hi, 100)
        ax.hist(bs_pnl, bins=bins, density=True, color="steelblue",
                alpha=0.55, label=f"BS (CVaR50 {r['bs']['cvar50']:+.3f})")
        ax.hist(nn_pnl, bins=bins, density=True, color="firebrick",
                alpha=0.55, label=f"NN (CVaR50 {r['nn']['cvar50']:+.3f})")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"cost = {r['cost_bps']:.0f} bps")
        ax.set_xlabel("Terminal P&L")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("BS-delta vs Neural hedger — terminal P&L distributions",
                 fontsize=13)
    fig.tight_layout()
    p2 = RESULTS_DIR / "02_pnl_comparison.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    print(f"Wrote {p2}")

    # -- Figure 3: position paths for one sample path --------------------
    sample_idx = int(torch.argmax(eval_paths[:, -1] - eval_paths[:, 0]).item())
    times = np.linspace(0.0, T, N_STEPS + 1)
    fig, ax_price = plt.subplots(figsize=(10, 5))
    ax_price.plot(times, eval_paths[sample_idx].numpy(),
                  color="black", linewidth=1.4, label="S(t)")
    ax_price.set_xlabel("t (years)")
    ax_price.set_ylabel("Stock price S(t)")
    ax_price.legend(loc="upper left", frameon=False)
    ax_pos = ax_price.twinx()
    ax_pos.plot(
        times[:-1], bs_positions[sample_idx].numpy(),
        color="steelblue", linewidth=1.2, label="BS delta",
    )
    # Plot neural positions for the highest-cost level — shows largest delta.
    high_cost = max(COST_LEVELS_BPS)
    ax_pos.plot(
        times[:-1], neural_positions_per_cost[high_cost][sample_idx].numpy(),
        color="firebrick", linewidth=1.2,
        label=f"NN ({int(high_cost)} bps)",
    )
    ax_pos.set_ylabel("Position (shares)")
    ax_pos.legend(loc="upper right", frameon=False)
    ax_price.set_title(
        f"Sample path #{sample_idx}: BS-delta vs neural positions "
        f"({int(high_cost)} bps)"
    )
    fig.tight_layout()
    p3 = RESULTS_DIR / "02_position_paths.png"
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    print(f"Wrote {p3}")

    # -- Figure 4: policy difference vs spot, per cost level -------------
    # For each cost level, bin (S_n, h_nn - h_bs) over all (path, step)
    # entries and plot the conditional mean difference.
    n_bins = 30
    s_low, s_high = 60.0, 160.0
    bin_edges = np.linspace(s_low, s_high, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(9, 5))
    bs_pos_flat = bs_positions.numpy().flatten()
    s_flat = eval_paths[:, :-1].numpy().flatten()
    bin_idx = np.digitize(s_flat, bin_edges) - 1
    mask = (bin_idx >= 0) & (bin_idx < n_bins)

    for cost in COST_LEVELS_BPS:
        nn_pos_flat = neural_positions_per_cost[cost].numpy().flatten()
        diff = nn_pos_flat - bs_pos_flat
        means = np.full(n_bins, np.nan)
        for b in range(n_bins):
            sel = mask & (bin_idx == b)
            if sel.sum() > 100:  # skip thinly-populated bins
                means[b] = diff[sel].mean()
        ax.plot(bin_centers, means, marker=".", linewidth=1.4,
                label=f"{cost:.0f} bps")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvline(K, color="grey", linestyle=":", linewidth=0.8,
               label=f"K = {K:.0f}")
    ax.set_xlabel("Spot S_t")
    ax.set_ylabel("Mean(h_NN - h_BS)")
    ax.set_title("Policy difference vs spot, by cost level")
    ax.legend(title="cost_bps", frameon=False)
    fig.tight_layout()
    p4 = RESULTS_DIR / "02_policy_difference.png"
    fig.savefig(p4, dpi=120)
    plt.close(fig)
    print(f"Wrote {p4}")


if __name__ == "__main__":
    main()
