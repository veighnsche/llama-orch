from __future__ import annotations
from typing import Dict, Any, List, Optional
import math


def sum_by_month(rows: List[Dict[str, str]], month_col: str, val_col: str, scenario: Optional[str] = None) -> List[float]:
    months: Dict[int, float] = {}
    for r in rows:
        if scenario is not None:
            if (r.get("scenario") or "").strip().lower() != scenario:
                continue
        try:
            m = int(float(r.get(month_col, 0) or 0))
            v = float(r.get(val_col, 0) or 0)
        except Exception:
            continue
        months[m] = months.get(m, 0.0) + v
    if not months:
        return []
    n = max(months.keys()) + 1
    return [months.get(i, 0.0) for i in range(n)]


def ebitda_series(revenue_m: List[float], cogs_m: List[float], opex_fixed_m: List[float]) -> List[float]:
    n = len(revenue_m)
    opex_fixed_m = opex_fixed_m + [0.0] * (n - len(opex_fixed_m))
    return [revenue_m[i] - cogs_m[i] - opex_fixed_m[i] for i in range(n)]


def ebt_tax_net(ebitda_m: List[float], dep_m: List[float], interest_m: List[float], corporate_pct: float) -> tuple[List[float], List[float], List[float]]:
    corp_frac = corporate_pct / 100.0
    ebt_m = [ebitda_m[i] - dep_m[i] - interest_m[i] for i in range(len(ebitda_m))]
    tax_m = [max(0.0, ebt_m[i]) * corp_frac for i in range(len(ebitda_m))]
    net_income_m = [ebt_m[i] - tax_m[i] for i in range(len(ebitda_m))]
    return ebt_m, tax_m, net_income_m


def pad_series(xs: List[float], n: int) -> List[float]:
    return xs + [0.0] * (n - len(xs)) if len(xs) < n else xs[:n]


def fixed_opex_series(finance: Dict[str, Any], n: int) -> List[float]:
    opex_fixed = 0.0
    fixed = (finance or {}).get("fixed_costs_monthly_eur", {})
    if isinstance(fixed, dict):
        try:
            opex_fixed = sum(float(v) for v in fixed.values())
        except Exception:
            opex_fixed = 0.0
    return [opex_fixed] * n


def depreciation_series(dep_sched: Any, dep_const: float, n: int) -> List[float]:
    dep_m = [dep_const] * n
    if isinstance(dep_sched, list) and dep_sched:
        dep_m = [0.0] * n
        for a in dep_sched:
            try:
                amt = float(a.get("amount_eur", 0) or 0)
                months = int(a.get("months", 0) or 0)
                start = int(a.get("start_month_index", 0) or 0)
                if amt > 0 and months > 0 and start >= 0:
                    per = amt / months
                    for i in range(start, min(n, start + months)):
                        dep_m[i] += per
            except Exception:
                continue
    return dep_m


def revenue_and_cogs(pub_rev_m: List[float], prv_rev_m: List[float], pub_cost_m: List[float], prv_cost_m: List[float]) -> tuple[List[float], List[float]]:
    n = max(len(pub_rev_m), len(prv_rev_m), len(pub_cost_m), len(prv_cost_m))
    def g(xs: List[float]) -> List[float]:
        return xs + [0.0] * (n - len(xs))
    pub_rev_m, prv_rev_m, pub_cost_m, prv_cost_m = map(g, [pub_rev_m, prv_rev_m, pub_cost_m, prv_cost_m])
    revenue_m = [pub_rev_m[i] + prv_rev_m[i] for i in range(n)]
    cogs_m = [pub_cost_m[i] + prv_cost_m[i] for i in range(n)]
    return revenue_m, cogs_m


def vat_cash_series(tax: Dict[str, Any], wc: Dict[str, Any], revenue_m: List[float], cogs_m: List[float]) -> List[float]:
    vat_pct = float((tax or {}).get("vat_pct", 0.0) or 0.0) / 100.0
    vat_lag = int((wc or {}).get("vat_payment_lag_months", 0) or 0)
    vat_in_m = [revenue * vat_pct for revenue in revenue_m]
    vat_on_purchases_m = [cogs * vat_pct for cogs in cogs_m]
    vat_payable_m = [vat_in_m[i] - vat_on_purchases_m[i] for i in range(len(revenue_m))]
    vat_cash_m = [0.0] * len(revenue_m)
    for i in range(len(revenue_m)):
        j = i + vat_lag
        if 0 <= j < len(vat_cash_m):
            vat_cash_m[j] += max(0.0, vat_payable_m[i])
    return vat_cash_m


def working_capital_deltas(wc: Dict[str, Any], revenue_m: List[float], cogs_m: List[float]) -> List[float]:
    ar_days = float((wc or {}).get("ar_days", 0.0) or 0.0)
    ap_days = float((wc or {}).get("ap_days", 0.0) or 0.0)
    def level(revenue: float, cogs: float) -> float:
        ar = revenue * (ar_days / 30.0)
        ap = cogs * (ap_days / 30.0)
        return ar - ap
    levels = [level(revenue_m[i], cogs_m[i]) for i in range(len(revenue_m))]
    return [0.0] + [levels[i] - levels[i-1] for i in range(1, len(levels))]


# Percentile consolidation across runs
def _percentile(xs: List[float], p: float) -> float:
    """Compute the pth percentile (0-100) using nearest-rank on sorted data.

    If xs is empty, returns 0.0. For singletons, returns the value.
    """
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    ys = sorted(xs)
    # Clamp p to [0, 100]
    p = max(0.0, min(100.0, float(p)))
    # Nearest-rank index (1-based), then convert to 0-based
    import math
    k = int(math.ceil((p / 100.0) * len(ys)))
    k = max(1, min(k, len(ys)))
    return ys[k - 1]


def percentiles_sum_by_month(
    rows: List[Dict[str, str]],
    month_col: str,
    val_col: str,
    percentiles: List[float],
    scenario: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Aggregate values per month by summing across models within the same run, then compute percentiles across runs.

    Run-identity is inferred from common index columns if present (any of):
      ["grid_index", "replicate_index", "mc_index", "random_run_index", "run_id", "seed"]
    If none are present, all rows are treated as a single run.
    The returned dict maps str(perc) -> series list (length = max month + 1).
    """
    # Detect run id columns present in the data
    idx_candidates = ["grid_index", "replicate_index", "mc_index", "random_run_index", "run_id", "seed"]
    present_cols: List[str] = []
    for r in rows:
        present_cols = [c for c in idx_candidates if c in r]
        if present_cols:
            break
    # Build month -> run_key -> sum(value)
    month_run_sums: Dict[int, Dict[tuple, float]] = {}
    for r in rows:
        if scenario is not None:
            if (r.get("scenario") or "").strip().lower() != scenario:
                continue
        try:
            m = int(float(r.get(month_col, 0) or 0))
            v = float(r.get(val_col, 0) or 0)
        except Exception:
            continue
        if present_cols:
            key = tuple((r.get(c) or "").strip() for c in present_cols)
        else:
            key = ("_single_run",)
        inner = month_run_sums.get(m)
        if inner is None:
            inner = {}
            month_run_sums[m] = inner
        inner[key] = inner.get(key, 0.0) + v
    if not month_run_sums:
        return {str(int(p)): [] for p in percentiles}
    n = max(month_run_sums.keys()) + 1
    # For each month, compute percentile on the distribution of per-run sums
    result: Dict[str, List[float]] = {str(int(p)): [0.0] * n for p in percentiles}
    for m in range(n):
        dist = list(month_run_sums.get(m, {}).values())
        if not dist:
            continue
        ys = sorted(dist)
        L = len(ys)
        for p in percentiles:
            # Nearest-rank percentile on pre-sorted data
            pc = max(0.0, min(100.0, float(p)))
            k = int(math.ceil((pc / 100.0) * L))
            k = 1 if k < 1 else (L if k > L else k)
            result[str(int(p))][m] = ys[k - 1]
    return result
