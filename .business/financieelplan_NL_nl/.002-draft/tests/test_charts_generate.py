from __future__ import annotations

from pathlib import Path
import pandas as pd

from finance_engine.charts.generate import (
    plot_model_margins,
    plot_public_scenarios,
    plot_break_even,
    plot_private_tap,
    plot_loan_balance,
)


def test_plot_model_margins(tmp_path: Path):
    df = pd.DataFrame({
        "model": ["A", "B"],
        "margin_per_1m_med": [100.0, 50.0],
    })
    out = tmp_path / "model.png"
    plot_model_margins(df, out)
    assert out.exists()


def test_plot_public_scenarios(tmp_path: Path):
    df = pd.DataFrame({
        "case": ["worst", "base", "best"],
        "revenue_eur": [1000.0, 2000.0, 3000.0],
        "cogs_eur": [400.0, 800.0, 1200.0],
        "marketing_reserved_eur": [150.0, 300.0, 450.0],
        "net_eur": [100.0, 200.0, 300.0],
    })
    out = tmp_path / "scen.png"
    plot_public_scenarios(df, 3500.0, out)
    assert out.exists()


def test_plot_break_even(tmp_path: Path):
    out = tmp_path / "be.png"
    plot_break_even(12000.0, out)
    assert out.exists()


def test_plot_private_tap(tmp_path: Path):
    df = pd.DataFrame({
        "gpu": ["L4", "L40S"],
        "provider_eur_hr_med": [0.6, 0.9],
        "sell_eur_hr": [0.9, 1.3],
        "margin_eur_hr": [0.3, 0.4],
        "markup_pct": [50.0, 45.0],
    })
    out = tmp_path / "priv.png"
    plot_private_tap(df, out)
    assert out.exists()


def test_plot_loan_balance(tmp_path: Path):
    df = pd.DataFrame({
        "month": [1, 2, 3],
        "balance_end_eur": [29500.0, 29000.0, 28500.0],
    })
    out = tmp_path / "loan.png"
    plot_loan_balance(df, out)
    assert out.exists()
