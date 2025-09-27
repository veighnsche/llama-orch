from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class FileReport:
    name: str
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    count: int | None = None  # rows for CSVs, keys for YAML
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightResult:
    ok: bool
    files: List[FileReport]
    warnings: List[str] = field(default_factory=list)


# ---- Generic helpers (kept minimal; to be replaced by schema libs over time) ----

def require_keys(obj: Dict[str, Any], path: List[str]) -> Tuple[bool, Any]:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return False, None
        cur = cur[p]
    return True, cur


def is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def non_empty_string(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def build_preflight_markdown(res: PreflightResult) -> str:
    lines: List[str] = []
    lines.append("## Preflight")
    status = "OK (validation)" if res.ok else "FAILED (validation)"
    lines.append(f"STATUS: {status}")
    lines.append("")
    lines.append("Files:")
    for fr in res.files:
        if fr.count is not None:
            lines.append(f"- {fr.name}: {'\u2713' if fr.ok else '✗'}{'' if fr.count is None else f' ({fr.count} rows/keys)'}")
        else:
            lines.append(f"- {fr.name}: {'\u2713' if fr.ok else '✗'}")
    # Warnings
    if res.warnings or any(fr.warnings for fr in res.files):
        lines.append("")
        lines.append("Warnings:")
        for fr in res.files:
            for w in fr.warnings:
                lines.append(f"- {fr.name}: {w}")
        for w in res.warnings:
            lines.append(f"- {w}")
    # Errors (only if failed)
    if not res.ok:
        lines.append("")
        lines.append("Errors:")
        for fr in res.files:
            for e in fr.errors:
                lines.append(f"- {fr.name}: {e}")
        lines.append("")
        lines.append("How to fix:")
        # Provide simple hints by file
        hints = {
            "gpu_rentals.csv": "Add missing column(s) gpu, vram_gb, hourly_usd_min, hourly_usd_max and ensure non-negative numbers.",
            "price_sheet.csv": "Ensure columns sku, category, unit exist; for public_tap, unit must be 1k_tokens or 1M_tokens.",
            "tps_model_gpu.csv": "Ensure columns model_name, gpu, throughput_tokens_per_sec exist and throughput is non-negative.",
            "oss_models.csv": "Ensure name column is present and non-empty; include at least one spec column.",
            "config.yaml": "Ensure currency is non-empty; percentages are 0-100; booleans are booleans.",
            "lending_plan.yaml": "Provide loan_request.amount_eur > 0; repayment_plan.term_months > 0; interest_rate_pct ≥ 0.",
            "costs.yaml": "Provide one or more numeric amounts ≥ 0 so a total can be computed.",
        }
        for fname, hint in hints.items():
            lines.append(f"- {fname}: {hint}")
    return "\n".join(lines) + "\n"
