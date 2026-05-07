# Deep Hedging on a European call: a clean reproduction

*A research note accompanying the* `deep-hedging-research` *repository.*

## 1. Motivation

Black-Scholes delta hedging is one of the cleanest results in
mathematical finance: under continuous trading, constant
volatility, and zero transaction costs, the option seller can
eliminate market risk by holding `Δ(S_t, T - t)` shares of the
underlying at all times. The hedge is self-financing and the P&L
at expiry is identically zero. None of those assumptions survive
contact with a real trading desk. Rebalancing is discrete, costs
are non-zero, volatility is stochastic, and the *terminal P&L*
distribution under any fixed policy is a real object with a
non-trivial shape.

Once frictions enter, the analytic optimum disappears. There is
no closed form for the hedge that minimises a coherent risk
measure of terminal P&L under proportional costs and discrete
rebalancing — even for a vanilla European call. Bühler, Gonon,
Teichmann and Wood (2019) propose what is, with hindsight, the
obvious thing to do: write the entire pipeline (path simulation,
cost-aware position recursion, payoff, terminal P&L, risk
measure) as a differentiable program; insert a neural network in
place of the unknown optimal policy; and train end-to-end. The
present note reproduces their central finding on a single
European call, on simulated GBM paths, with linear proportional
costs at four cost levels (0, 5, 25, 50 bps), and benchmarks the
learned policy against discrete Black-Scholes delta on the same
paths.

## 2. Setup

Let `(S_t)` be a geometric Brownian motion with drift `μ`,
volatility `σ`, and initial value `S_0`. The seller writes a
European call with strike `K` and maturity `T`, receives a fixed
premium `p` at `t = 0`, and may rebalance a hedge in the
underlying at the discrete grid `t_n = nT/N`, `n = 0, …, N - 1`.
The hedge position over the interval `[t_n, t_{n+1})` is
`h_n ∈ ℝ` shares; at each rebalance the hedger pays a
proportional cost

`c_n = (κ / 10⁴) · |h_n - h_{n-1}| · S_{t_n}`,

where `κ` is the bps cost rate and `h_{-1} := 0`. The terminal
P&L is

`PnL = p + Σ_n h_n (S_{t_{n+1}} - S_{t_n}) - Σ_n c_n - (S_T - K)⁺`.

We use `S_0 = K = 100`, `T = 1` year, `μ = r = 0`, `σ = 0.20`,
and `N = 50` rebalances. The premium `p` is fixed at the
*frictionless* Black-Scholes price `BS(S_0, K, T, r, σ) ≈ 7.97`
across all cost levels, so cost burden appears directly in the
P&L mean rather than being absorbed into `p`. The training loss
is the CVaR-50% of `PnL`, i.e. the mean of the worst half. CVaR
is coherent (in the Artzner et al. sense), aligned with the
seller's downside concern, and stable to optimise.

## 3. The differentiable pipeline

The structural choice that makes the whole project tractable is
that *every* step in the chain — simulation, cost computation,
P&L accumulation, risk measure — is implemented in torch. A call
to `simulate_gbm(...)` returns a tensor of paths; the policy
evaluates as a torch module on a feature vector at each step;
costs and P&L are tensor reductions; the loss is a sort and a
mean. Gradients flow from the loss back through the entire
recursion to the policy weights.

The recursion is genuinely sequential: `h_n` depends on the
output of `h_{n-1}` through the previous-position feature.
Vectorisation is across *paths*, not across time. Concretely,
the training loop holds a `(B, N + 1)` tensor of paths in
memory, runs a Python `for` loop over the `N` rebalance times,
and at each step evaluates the policy on a `(B, 4)` feature
tensor producing the `(B,)` new positions. Within each step
everything is parallel; across steps it is serial. On 50 steps
and 1024-path batches this is fast enough that no GPU is needed.

A subtle but important choice is to feed the network the
Black-Scholes delta as one of its inputs. The network is then
not learning the hedge from scratch; it is learning the
*correction* to Black-Scholes that minimises the cost-adjusted
risk measure. This is a warm start in the most literal sense:
without that input the network must rediscover an `S`-shape from
its own gradient signal, and converges substantially more slowly
to a worse solution.

## 4. Architecture and training

The policy is a single MLP shared across timesteps (one
"time-conditioned" network rather than one network per step).
Inputs at step `n`, all observable at `t_n`:

- `log(S_n / K)` — log-moneyness;
- `(T - t_n) / T` — time-to-maturity fraction;
- `h_{n-1}` — previous position (the network's own prior output);
- `Δ_BS(S_n, T - t_n)` — closed-form Black-Scholes delta.

Three hidden layers of 32 ReLU units; linear scalar output. The
shared-weights design is a stronger inductive bias than one-net-
per-step: it forces the policy to be a *function of state*, not
of timestep label. Time enters only via the explicit feature.

Training: Adam at `1e-3`, fresh batches of 1024 paths drawn at
each epoch, validation on a held-out batch (different seed) every
10 epochs. 200 epochs at 0 and 5 bps; 500 epochs at 25 and 50
bps, where the optimal policy diverges most from BS and longer
training helps. A convergence diagnostic checks the mean per-
window relative drop in validation loss across the last five log
points and flags the run as undertrained if the drop exceeds
0.3% per 10-epoch window.

## 5. Frictionless baseline (session 01)

Before turning to the neural hedger we validate the simulator,
the BS pricer, and the P&L accounting end to end. Run BS-delta
hedging at 0 bps over 100,000 paths:

![BS delta, frictionless, 100k paths](../results/01_bs_delta_frictionless.png)

The terminal P&L is centred at $0.002 with standard deviation
$0.970. The mean sits at zero up to Monte Carlo noise; the
dispersion is pure discrete-rebalance error. This reproduces the
textbook discrete-BS result and confirms that the price-payoff-
position chain is wired up correctly.

Switching the same paths to 5 bps proportional cost shifts the
distribution leftward:

![BS delta, frictionless vs 5 bps overlay](../results/01_bs_delta_with_costs.png)

Mean P&L drops to -$0.163 at standard deviation $0.977. The
shape is preserved; the entire distribution is translated. This
is the friction-induced bleed an optimal cost-aware hedger
should be able to reduce.

## 6. Neural hedger comparison (session 02)

A separate neural policy is trained at each cost level. All four
trained policies are evaluated on a *shared* 100,000-path held-
out set (a single eval seed shared across cost levels), so the
only thing varying across rows below is the cost the policy was
trained against:

| cost (bps) | BS mean | BS CVaR-50% | NN mean | NN CVaR-50% | NN−BS CVaR-50% | NN > BS frac. |
|----------:|--------:|------------:|--------:|------------:|---------------:|--------------:|
| 0  | +0.002 | -0.727 | +0.002 | -0.736 | -0.009 | 0.461 |
| 5  | -0.163 | -0.896 | -0.160 | -0.907 | -0.011 | 0.478 |
| 25 | -0.823 | -1.597 | -0.690 | -1.499 | +0.099 | 0.593 |
| 50 | -1.649 | -2.521 | -1.247 | -2.146 | +0.374 | 0.666 |

Three observations.

**Zero-cost row is a sanity check, not a result.** BS is provably
optimal at 0 bps; the NN should not beat it. The CVaR-50% gap of
-0.009 is well inside Monte Carlo noise. The training pipeline
includes an explicit zero-cost gate that raises if the NN beats
BS by more than `max(0.05, 5%·|bs_cvar|)` at 0 bps, which would
indicate a look-ahead leak. The gate did not trigger.

**The 5 bps row is noise-dominated.** The CVaR gap of -0.011 is
the same order of magnitude as the 0-bps gap. At this cost
level the optimal policy and BS are too close in P&L space for
1024-path batches and 200 epochs to resolve a meaningful
improvement.

**The 25 and 50 bps rows are where the framework earns its keep.**
The CVaR-50% gap rises monotonically (+0.099, +0.374). The
fraction of held-out paths on which the neural policy strictly
beats BS rises from a coin flip at 0 bps to two-thirds at 50 bps.

The training curves and held-out P&L distributions:

![Training curves per cost level](../results/02_loss_curves.png)
![BS vs NN terminal P&L distributions](../results/02_pnl_comparison.png)

The histograms make the qualitative trade-off legible: at high
cost the NN distribution is *wider* than BS but *less left-
skewed*.

The CVaR-50% loss is doing visible work here: it rewards exactly
that trade-off. A mean-variance objective would have penalised
the wider NN distribution for its variance and pulled the policy
back toward the BS shape; CVaR-50% only sees the worst half of
P&Ls, so it accepts extra dispersion on the *upside* in exchange
for a less-bad left tail. The choice of risk measure is not a
detail — it is what determines the qualitative shape of the
optimal policy under frictions.

## 7. Interpreting the learned policy

The most distinctive finding of the project is that the network
learns a *legible* deviation from BS, not just a numerically
better one. Binning every `(S_n, h_{NN,n} - h_{BS,n})` pair from
the held-out set by spot, and plotting the conditional mean
position difference per cost level:

![NN minus BS position vs spot, by cost level](../results/02_policy_difference.png)

The 0- and 5-bps lines hover around zero. The 25- and 50-bps
lines have clear *S*-shape: the NN holds *more* than BS in the
slightly-OTM region (`S ∈ [85, 100]`, peaking at roughly +0.047
shares above BS at 50 bps), and *less* than BS in the deep-ITM
tail.

The economic reading is straightforward once you look at where
gamma lives. Around the strike, BS-delta moves fastest with
spot; under proportional costs, that is exactly where
rebalancing turnover is most expensive. The NN learns to
*pre-position* — to hold a bit more delta when the spot is in
the slightly-OTM region from which a path is most likely to
sweep through the strike, and to *not chase* every late
in-the-money move that BS would. "Trade ahead, save on later
rebalancing." None of this was designed in: the network's only
economic prior is the BS-delta input feature.

A single path makes the same point concretely:

![Sample path: stock, BS delta, NN at 50 bps](../results/02_position_paths.png)

The black trace is `S_t`; the blue trace is `h_BS`; the red
trace is the 50-bps neural policy. The blue line is the
familiar `S`-shape of delta; the red line traces the same shape
but with a visible smoothing — fewer, smaller rebalances in the
high-gamma window.

## 8. Anti-look-ahead checks

The whole pipeline holds full paths in memory during training,
so the discipline of "the policy at time `t_n` sees only data up
to `t_n`" has to be enforced by construction and verified by
test. Three checks in the project:

1. **Feature causality by inspection.** Each of the four input
   features at step `n` is a function of `S_0, …, S_n` only.
   `h_{n-1}` is the network's own previous output, which by
   induction depends on `S_0, …, S_{n-1}`.
2. **Mechanical no-look-ahead test.** A unit test in
   `tests/test_policies.py` calls the policy on a batch of paths,
   then perturbs the *future* path columns (`S_{n+1}, …, S_N`)
   and re-runs the policy. The earlier position decisions are
   asserted byte-identical. Any future-information leak would
   trip this immediately.
3. **Evaluation-seed separation.** Training uses a per-cost
   training seed; held-out evaluation uses a different seed
   from any of the training-time validation seeds.

All three pass. The zero-cost gate (Section 6) is the fourth
line of defence: if anything subtle were leaking, BS would lose
to the NN by a noticeable margin at 0 bps, and the gate would
raise.

## 9. Limitations

The result of this project is "the Deep Hedging trick reproduces
cleanly on a textbook problem and recovers an interpretable
policy." Several things it is *not*:

- *No historical data.* All paths are simulated. The demonstration
  is that the policy learns the right structure on the model's
  own paths, not that it generalises to real-market dynamics.
  Real markets are not GBM.
- *Constant volatility.* Real options markets have stochastic
  volatility (Heston, SABR, rough volatility). Extending the
  simulator and the BS-feature to a stochastic-vol underlying is
  the most natural next step and changes the qualitative shape
  of the cost-optimal policy.
- *Toy cost model.* The linear-bps cost is a useful default but
  ignores bid-ask spread structure, market impact, and
  instrument-specific effects. Real costs are convex in
  size and dependent on regime.
- *Architecture-limited at high cost.* Per finding F6, going from
  200 to 500 epochs at 50 bps gained only +0.040 in CVaR-50%, and
  the convergence diagnostic was below threshold. The 32×32×32
  MLP with four features has hit a local optimum. A wider
  network or richer features (e.g. log-moneyness × time-to-
  maturity interactions, second-order BS Greeks) might extract
  more.
- *Single instrument, single asset.* Portfolios of options and
  multi-asset hedging are not addressed. The framework extends
  cleanly; the current code does not.

## 10. Future directions

Three concrete extensions follow directly from the limitations
above:

- *Heston paths.* Replace `simulate_gbm` with a Heston path
  generator and re-train. The cost-optimal policy under
  stochastic vol is qualitatively different; it should reveal a
  vol-dependent slice of the front-loading structure visible in
  Section 7.
- *Richer features and a wider network.* Add interaction terms
  and second-order Greeks to the input; widen to `[64, 64, 64]`
  or larger. The finding F6 prediction is that the 50-bps gap
  closes further; the experiment is cheap.
- *Alternative risk measures.* The training loss is one knob.
  Entropic risk is implemented as an alternative; quadratic loss
  recovers a mean-variance hedge; spectral risk measures
  interpolate between CVaR levels. The shape of the policy
  difference plot in Section 7 is presumably sensitive to this
  choice in interesting ways.

Beyond these, exotic payoffs (barrier, Asian) and multi-asset
hedging are the obvious larger steps.

## 11. References

- Bühler, H.; Gonon, L.; Teichmann, J.; Wood, B. (2019).
  "Deep Hedging."
- Bühler, H.; Gonon, L.; Teichmann, J.; Wood, B.; Mohan, B.
  (2019). "Deep Hedging: Hedging Derivatives Under Generic
  Market Frictions Using Reinforcement Learning."
- Hull, J. *Options, Futures, and Other Derivatives.*
- Glasserman, P. *Monte Carlo Methods in Financial Engineering.*
- Artzner, P.; Delbaen, F.; Eber, J.-M.; Heath, D. (1999).
  "Coherent Measures of Risk."
