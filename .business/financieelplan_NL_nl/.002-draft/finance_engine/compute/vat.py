from __future__ import annotations

from typing import List


def vat_examples(vat_rate_pct: float) -> List[dict]:
    """Compute set-aside for €1k/€10k/€100k inflow examples."""
    rate = vat_rate_pct / 100.0
    examples = [1000, 10000, 100000]
    return [{
        "inflow_eur": x,
        "vat_set_aside_eur": round(x * rate, 2),
    } for x in examples]
