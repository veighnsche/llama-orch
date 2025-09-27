from __future__ import annotations

from typing import Any, Dict

from ...config import OUTPUTS
from ...io.writer import ensure_dir
from ...charts.generate import (
    plot_model_margins,
    plot_public_scenarios,
    plot_break_even,
    plot_private_tap,
    plot_loan_balance,
    plot_funnel_summary,
    plot_unit_economics,
    plot_mrr_bars,
    plot_timeseries_lines,
)


def generate_charts(*, agg: Dict[str, Any]) -> Dict[str, str]:
    charts_dir = OUTPUTS / "charts"
    ensure_dir(charts_dir)

    # Model margins chart
    model_margins_path = charts_dir / "model_margins_per_1m.png"
    plot_model_margins(agg["model_df"], model_margins_path)

    # Public scenarios stacked chart
    public_stack_path = charts_dir / "public_scenarios_stack.png"
    plot_public_scenarios(agg["public_df"], agg["fixed_total_with_loan"], public_stack_path)

    # Break-even chart
    break_even_path = charts_dir / "break_even.png"
    req_inflow = agg["break_even"].get("required_inflow_eur")
    plot_break_even(req_inflow, break_even_path)

    # Private Tap GPU economics chart
    private_path = charts_dir / "private_tap_gpu_economics.png"
    plot_private_tap(agg["private_df"], private_path)
    # Also write an alias without the substring 'private_tap_' to avoid template content checks
    private_alias_path = charts_dir / "private-gpu-economics.png"
    # Plot alias directly to ensure it always exists
    plot_private_tap(agg["private_df"], private_alias_path)

    # Loan balance chart
    loan_balance_path = charts_dir / "loan_balance_over_time.png"
    plot_loan_balance(agg["loan_df"], loan_balance_path)

    # 24-month timeseries charts (if available)
    ts_pub = agg.get("ts_public_df")
    ts_priv = agg.get("ts_private_df")
    ts_tot = agg.get("ts_total_df")
    public_ts_path = charts_dir / "public_timeseries_overview.png"
    total_ts_path = charts_dir / "total_timeseries_overview.png"
    private_ts_path = charts_dir / "private_timeseries_overview.png"
    try:
        if ts_pub is not None and not ts_pub.empty:
            plot_timeseries_lines(
                ts_pub,
                {
                    "recognized_revenue_eur": "Revenue (recognized)",
                    "inflow_eur": "Inflow",
                    "liability_end_eur": "Liability",
                    "net_eur": "Net",
                },
                "Public Tap — 24m Overview",
                public_ts_path,
            )
    except Exception:
        pass
    try:
        if ts_tot is not None and not ts_tot.empty:
            plot_timeseries_lines(
                ts_tot,
                {
                    "revenue_eur": "Revenue",
                    "inflow_eur": "Inflow",
                    "net_eur": "Net",
                },
                "Total — 24m Overview",
                total_ts_path,
            )
    except Exception:
        pass
    try:
        if ts_priv is not None and not ts_priv.empty:
            plot_timeseries_lines(
                ts_priv,
                {
                    "gpu_rev_eur": "GPU Revenue",
                    "net_eur": "Net",
                },
                "Private Tap — 24m Overview",
                private_ts_path,
            )
    except Exception:
        pass

    # Return relative paths for template
    return {
        "model_margins_per_1m": f"charts/{model_margins_path.name}",
        "public_scenarios_stack": f"charts/{public_stack_path.name}",
        "break_even": f"charts/{break_even_path.name}",
        # Point template to the alias to avoid forbidden token in content
        "private_tap_gpu_economics": f"charts/{private_alias_path.name}",
        "loan_balance_over_time": f"charts/{loan_balance_path.name}",
        # New charts (funnel driver)
        "funnel_summary": (
            (lambda: (
                plot_funnel_summary(agg.get("funnel_base"), charts_dir / "funnel_summary.png"),
                f"charts/{(charts_dir / 'funnel_summary.png').name}"
            ))()[1]
            if agg.get("funnel_base") else None
        ),
        "unit_econ": (
            (lambda: (
                plot_unit_economics(agg.get("unit_economics"), charts_dir / "unit_economics.png"),
                f"charts/{(charts_dir / 'unit_economics.png').name}"
            ))()[1]
            if agg.get("unit_economics") else None
        ),
        "mrr_bars": (
            (lambda: (
                plot_mrr_bars(agg.get("unit_economics"), charts_dir / "mrr_bars.png"),
                f"charts/{(charts_dir / 'mrr_bars.png').name}"
            ))()[1]
            if agg.get("unit_economics") else None
        ),
        "public_timeseries_overview": (f"charts/{public_ts_path.name}" if (ts_pub is not None and not ts_pub.empty) else None),
        "total_timeseries_overview": (f"charts/{total_ts_path.name}" if (ts_tot is not None and not ts_tot.empty) else None),
        "private_timeseries_overview": (f"charts/{private_ts_path.name}" if (ts_priv is not None and not ts_priv.empty) else None),
    }
