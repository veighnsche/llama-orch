from __future__ import annotations
from typing import List
import math

"""
TAM & saturation utilities.

- apply_tam_cap: hard cap actives at TAM
- smooth_saturation: smooth approach to TAM using a logistic-like transform
"""


def apply_tam_cap(actives: List[float], tam: float) -> List[float]:
    tam = max(0.0, float(tam))
    return [min(max(0.0, float(a)), tam) for a in actives]


def smooth_saturation(actives: List[float], tam: float, softness: float = 0.2) -> List[float]:
    """Compress actives toward TAM smoothly.

    softness in (0,1]; smaller = sharper knee near TAM.
    We use: sat = TAM * (1 - exp(- (a/TAM) / softness))
    """
    tam = max(1e-9, float(tam))
    s = max(1e-6, min(float(softness), 1.0))
    out: List[float] = []
    for a in actives:
        x = max(0.0, float(a)) / tam
        out.append(tam * (1.0 - math.exp(-x / s)))
    return out
