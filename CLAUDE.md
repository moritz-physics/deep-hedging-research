# CLAUDE.md — deep-hedging-research

## Project

Research project implementing the Deep Hedging framework
(Bühler, Gonon, Teichmann, Wood, 2019) — training a neural
network to learn optimal hedging policies for derivatives
under realistic frictions (transaction costs, discrete
rebalancing), and comparing performance against Black-Scholes
delta hedging.

Owner: physics/AI master's student. The project is intended
to demonstrate competence at the intersection of stochastic
modeling, neural networks, and quantitative finance.
Prioritize correctness, clarity, and pedagogical value over
performance optimization.

## Relationship to Backtest-Engine

Backtest-Engine is installed as an editable dependency. We may
import the LinearCost model and the metrics module if useful,
but Deep Hedging is structurally different from
historical-data backtesting: we simulate paths and learn a
policy, rather than feeding signals into a backtest engine.
Most of the project is independent of the framework.

## Stack

- Python 3.12+
- numpy, pandas, scipy (numerical)
- torch (PyTorch, neural networks and autograd)
- matplotlib (plotting)
- pytest, ruff (dev)
- Backtest-Engine (editable, optional use)

## Domain Conventions

- Time is measured in trading years (1.0 = 1 trading year =
  252 days). Discount factor uses risk-free rate per year.
- Stock prices are in dollars. Default S_0 = 100.
- Volatility is annualized (e.g., σ = 0.2 means 20%/yr).
- Risk-free rate r is annualized continuous compounding.
- Path index runs 0..N where N is the number of rebalance
  events (usually 30 or 50). t_n = n * dt where dt = T/N.
- Hedge position h_n is in number of shares (not dollar-weighted).
- All randomness is seeded; every script must produce identical
  output across runs given the same seed.

## Non-Negotiables

1. The neural network's input at time n must use only
   information observable at or before time n. Specifically:
   no peeking at future paths during training the policy
   evaluation. (During training we have full paths in memory,
   but the policy network only sees state at t_n.)
2. Training loss is a coherent risk measure of terminal P&L
   (default: CVaR-50% or entropic risk). Document the choice.
3. Every result is benchmarked against Black-Scholes delta
   hedging on the same paths and same costs. No "neural net
   beat the no-hedge baseline" reporting — that's trivial.
4. Reproducibility: all torch.manual_seed and numpy seed calls
   set explicitly. Document expected wall-clock training time
   per script.
5. Costs are always modeled. The whole point is to study
   hedging under frictions; reporting frictionless results
   is for sanity checks only.

## Code Style

- src/deep_hedging/ holds reusable logic: simulators,
  payoff functions, policy networks, training loops, metrics.
- scripts/NN_description.py are runnable entry points.
- Tests in tests/ mirror module structure. Mock or seed
  everything; tests must run in <30s total.
- Prefer torch operations over numpy when inside the
  computational graph; convert to numpy only for plotting
  or final reporting.
- Type hints on public functions. NumPy-style docstrings.

## Performance Notes

- PyTorch device: dynamically choose MPS (Apple Silicon) or
  CUDA if available, else CPU. Wrap in a get_device() helper.
- Training a small policy (2-3 layers, 32-64 hidden units) on
  100k paths × 30 timesteps takes seconds on CPU. No GPU
  required for the baseline experiments.
- Vectorize across paths. The model processes one timestep at
  a time across all paths in parallel — this is the key
  efficiency pattern.

## References

- Bühler, Gonon, Teichmann, Wood (2019), "Deep Hedging"
- Bühler, Gonon, Teichmann, Wood, Mohan (2019), "Deep Hedging: 
  Hedging Derivatives Under Generic Market Frictions Using 
  Reinforcement Learning"
- Hull, "Options, Futures, and Other Derivatives" (Black-Scholes 
  reference)
- Glasserman, "Monte Carlo Methods in Financial Engineering"
  (path simulation reference)