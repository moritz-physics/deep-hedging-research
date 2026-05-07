# deep-hedging-research

A clean implementation of the Deep Hedging framework
(Bühler, Gonon, Teichmann, Wood, 2019) on a single European call
option under transaction costs and discrete rebalancing. A neural
network is trained end-to-end via CVaR-50% on terminal P&L through
a fully differentiable simulation pipeline, and benchmarked against
Black-Scholes delta hedging on the same paths and same costs at
0, 5, 25, and 50 basis points. The objective is reproducing the
paper's central finding — a learned policy beats the classical
delta hedge once frictions are non-trivial — and recovering an
*interpretable* economic structure from the trained network.

![NN minus BS position, by spot and cost level](results/02_policy_difference.png)

*Figure: difference between the neural hedger's position and the
Black-Scholes delta, averaged by spot price across the validation
paths, for each cost level. The non-zero shape at high cost shows
the network has learned a "trade ahead, save on later
rebalancing" policy that BS does not implement.*

## Findings

**F1 — Pipeline validation.** With 50 rebalances and zero costs,
the BS-delta hedger's terminal P&L over 100,000 paths is centred
at $0.002 with standard deviation $0.970. The dispersion is driven
purely by discrete-rebalance error; the mean sits at zero up to
Monte Carlo noise. This reproduces the textbook result and
validates the simulator, pricer, and cost-aware P&L accounting end
to end. (`results/01_bs_delta_frictionless.png`.)

**F2 — Cost impact on BS.** Switching to 5 bps costs on the same
paths shifts the BS-delta P&L mean to -$0.163 with standard
deviation $0.977. The cost bleed is consistent with a back-of-the-
envelope ATM-gamma calculation: turnover from gamma rebalancing
times the per-trade cost. The classical hedger pays for its
sensitivity to spot. (`results/01_bs_delta_with_costs.png`.)

**F3 — Zero-cost gate.** At 0 bps the neural hedger matches BS
within 0.009 in CVaR-50% on a 100,000-path held-out set. Black-
Scholes is provably optimal in the frictionless limit, so any
material outperformance at 0 bps would indicate a look-ahead leak.
The gate doubles as the project's primary correctness check on the
training pipeline. (`results/02_neural_hedger_comparison.csv`.)

**F4 — Neural beats BS under costs, monotonically.** The NN-BS
delta in CVaR-50% is -0.011 at 5 bps (noise-dominated), +0.099 at
25 bps, and +0.374 at 50 bps. The fraction of paths on which the
neural hedger outperforms BS rises from 0.46 at 0 bps to 0.67 at
50 bps. The improvement scales monotonically with the friction
the policy is being asked to manage. (`results/02_pnl_comparison.png`,
`results/02_neural_hedger_comparison.csv`.)

**F5 — The learned policy is interpretable.** Binning the
(spot, position-difference) plane shows the network *front-loads*
shares in the slightly-OTM region ($85-$100 spot), peaking at
roughly +0.047 shares above BS at 50 bps, and *under-hedges* in
the deep-ITM tail. The economic reading is "trade ahead, save on
later rebalancing": gamma-driven rebalancing through the strike is
the single most expensive turnover under proportional costs, and
the network learns to pre-position rather than chase. The
structure was not designed-in — the network was given BS-delta as
a feature and nothing else. (`results/02_policy_difference.png`,
`results/02_position_paths.png`.)

**F6 — High-cost regime is architecture-limited, identifying
clear next steps.** Pushing training at 50 bps from 200 to 500
epochs gained only +0.040 in CVaR-50%, and the convergence
diagnostic confirms the val-loss plateau. The 32×32×32 MLP with
four features has hit a local optimum within its own expressive
capacity — meaning the gap between the learned policy and the
true cost-aware optimum is bounded by the architecture, not by
training time. Larger networks, richer features (e.g.
log-moneyness × time-to-maturity interactions), or a
stochastic-vol underlying are the natural next axes.
(`results/02_loss_curves.png`.)

## Methodology

- **Underlying.** Geometric Brownian motion vectorised on the log
  scale via cumulative log-increments; one tensor of shape
  `(n_paths, n_steps + 1)` per simulation call.
- **Instrument.** European call payoff. Closed-form Black-Scholes
  price and delta utilities for benchmarking and as a feature.
- **Hedger interface.** Abstract `Hedger` base class with a
  `positions(...)` method; concrete `BlackScholesDeltaHedger` and
  `NeuralHedger` implementations share the same evaluation path.
- **Network.** A single time-conditioned MLP shared across
  timesteps. Input features at step *n*: log-moneyness
  `log(S_n / K)`, time-to-maturity fraction `(T - t_n) / T`,
  previous position `h_{n-1}`, and BS-delta evaluated at
  `(S_n, T - t_n)`. Hidden layers `[32, 32, 32]` with ReLU; linear
  scalar output gives the new position in shares.
- **Loss.** Coherent risk measure on terminal P&L; default
  CVaR-50% (mean of the worst half). Entropic risk is implemented
  as an alternative.
- **Training.** Fresh 1024-path batches each epoch, Adam at
  `1e-3`, 200 epochs at 0/5 bps and 500 at 25/50 bps. Validation
  pass on a held-out batch at a separate seed every 10 epochs.
- **Anti-look-ahead.** Each input feature at step *n* is a
  function of `S_0..S_n` only. `h_{n-1}` is the network's own
  prior output, causal by construction. A held-out evaluation set
  uses a different seed than training. A mechanical test in
  `tests/test_policies.py` perturbs future path columns and
  asserts earlier position decisions are byte-identical.
- **Cost model.** Linear in absolute share change times spot:
  `cost_n = (bps / 10_000) * |h_n - h_{n-1}| * S_n`. The seller
  receives a fixed premium equal to the frictionless BS price
  across all cost levels, so cost burden appears directly in
  P&L histograms instead of being absorbed into the price.

## Why neural hedging

Real markets have neither: rebalancing is discrete, and every
trade pays a spread. The classical hedge
over-rebalances when costs matter, especially through the strike
where gamma is largest, and the policy that minimises a coherent
risk measure of *terminal* P&L under those frictions is no longer
analytic.

Deep Hedging reformulates the problem as end-to-end optimisation
of the risk measure through a fully differentiable pipeline: the
GBM simulator, the cost model, the position recursion, and the
loss are all torch operations, and the neural-network policy is
one learnable component in that chain. This is the same
"differentiable everything" pattern that powers neural ODEs and
physics-informed networks, applied to a derivatives-pricing
problem in which the risk-neutral story is well-understood and the
cost-adjusted story is not. The project doubles as a clean
illustration of the paradigm.

## Repository layout

```
src/deep_hedging/      simulator, instruments, hedgers, policies, training, utils
scripts/               runnable entry points (01_*.py validation, 02_*.py main result)
tests/                 mirrors src/ module structure; pytest, <30s total
results/               committed CSV + PNG artefacts produced by the scripts
docs/                  research-note writeup (writeup.md)
notebooks/             walkthrough notebook over the committed artefacts
```

## Quick start

Clone and sync:

```bash
git clone <repo-url> deep-hedging-research
cd deep-hedging-research
uv sync
```

Run the cheap sanity check (~9 seconds on CPU, Apple Silicon
M-series):

```bash
uv run python scripts/01_gbm_bs_validation.py
```

Run the main result (~41 seconds on CPU, Apple Silicon M-series;
trains a separate policy per cost level and evaluates all four on
a shared 100k-path held-out set):

```bash
uv run python scripts/02_neural_hedger.py
```

## Limitations and honest caveats

- All paths are simulated; no historical data validation. The
  result is "the trick learns the structure on the model's own
  paths," not "the trick generalises to real markets."
- GBM assumes constant volatility. Real options markets have
  stochastic volatility (Heston, rough volatility). Extending the
  simulator to Heston is the most natural next step.
- The linear cost model ignores bid-ask spread structure, market
  impact, and instrument-specific effects. It is the standard toy
  friction.
- The 32×32×32 MLP is small. Per F6, the 50-bps regime is
  architecture-limited, not data-limited.
- Single-asset, single-option setting. Portfolios of options and
  multi-asset hedging are not addressed.

## What I learned

- Backpropagation through a *simulator* is the central trick:
  once the path generator, the cost model, and the loss are all
  torch operations, training the policy is just gradient descent
  on a long composite function.
- CVaR-trained policies have visibly broader P&L distributions
  than BS — they trade some centrality for better tails. Reading
  histograms with this in mind is more informative than a single
  scalar comparison.
- A learned policy is interpretable if you ask the right
  question. Plotting `h_NN - h_BS` against spot turns a black
  box into a one-line economic story; plotting `h_NN` alone does
  not.
- Convergence diagnostics catch under-training, not local
  minima. The 50-bps curve looked converged by the 0.3%-per-
  window threshold; longer training and a wider architecture
  still found gain.
- The zero-cost gate is the load-bearing correctness check. It is
  the cheapest reproducible way to detect a look-ahead leak, and
  worth more than any number of unit tests on its own.

## Related repository

[Backtest-Engine](../Backtest-Engine) is listed as an editable
dependency to keep cost-model semantics aligned across projects;
the current Deep Hedging code implements the linear cost model
inline rather than importing it, since the simulation/training
problem here is structurally different from historical-data
backtesting.

## References

- Bühler, Gonon, Teichmann, Wood (2019), "Deep Hedging."
- Bühler, Gonon, Teichmann, Wood, Mohan (2019), "Deep Hedging:
  Hedging Derivatives Under Generic Market Frictions Using
  Reinforcement Learning."
- Hull, *Options, Futures, and Other Derivatives.*
- Glasserman, *Monte Carlo Methods in Financial Engineering.*

## License

MIT. See `LICENSE`.

## Author

Moritz Heidtmann.
