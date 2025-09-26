# Re-export context assemblers
from core.context import build_streams, pricing_baseline, loan_contexts, compute_pnl_aggregates

__all__ = [
    "build_streams",
    "pricing_baseline",
    "loan_contexts",
    "compute_pnl_aggregates",
]
