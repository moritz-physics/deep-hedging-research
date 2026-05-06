"""Deep Hedging research package.

Public API for scripts and notebooks.
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
from deep_hedging.policies import HedgePolicy, NeuralHedger
from deep_hedging.simulator import simulate_gbm
from deep_hedging.training import cvar_loss, entropic_risk, train_hedger
from deep_hedging.utils import get_device, set_seed

__all__ = [
    "BlackScholesDeltaHedger",
    "Hedger",
    "HedgePolicy",
    "NeuralHedger",
    "bs_call_delta",
    "bs_call_price",
    "compute_pnl",
    "cvar_loss",
    "entropic_risk",
    "european_call_payoff",
    "get_device",
    "set_seed",
    "simulate_gbm",
    "train_hedger",
]
