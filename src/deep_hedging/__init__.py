"""Deep Hedging research package.

Public API for scripts and notebooks. The session-by-session build
adds policies and training utilities to this surface; for now it
exposes the GBM simulator, Black-Scholes utilities, and the BS-delta
benchmark hedger.
"""

from deep_hedging.hedgers import (
    BlackScholesDeltaHedger,
    Hedger,
    compute_pnl,
)
from deep_hedging.instruments import (
    bs_call_delta,
    bs_call_price,
    european_call_payoff,
)
from deep_hedging.simulator import get_device, simulate_gbm

__all__ = [
    "BlackScholesDeltaHedger",
    "Hedger",
    "bs_call_delta",
    "bs_call_price",
    "compute_pnl",
    "european_call_payoff",
    "get_device",
    "simulate_gbm",
]
