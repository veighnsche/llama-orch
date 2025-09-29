from __future__ import annotations
from typing import Dict, Any, List


def compute_kpis(ebitda_m: List[float], interest_m: List[float], principal_m: List[float], ending_cash_m: List[float]) -> Dict[str, Any]:
    dscr_m = []
    for i in range(len(ebitda_m)):
        debt_service = interest_m[i] + principal_m[i]
        dscr = float('inf') if debt_service <= 0 else (ebitda_m[i] / max(debt_service, 1e-9))
        dscr_m.append(dscr)
    icr_m = [float('inf') if interest_m[i] <= 0 else (ebitda_m[i] / max(interest_m[i], 1e-9)) for i in range(len(ebitda_m))]
    return {
        "dscr_min": (min(dscr_m) if dscr_m else None),
        "icr_min": (min(icr_m) if icr_m else None),
        "cash_min": (min(ending_cash_m) if ending_cash_m else None),
        "runway_months": next((i for i, c in enumerate(ending_cash_m) if c < 0), len(ending_cash_m)),
    }
