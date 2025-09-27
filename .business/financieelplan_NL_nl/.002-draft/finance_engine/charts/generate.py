from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # safe for headless rendering
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def plot_model_margins(model_df: pd.DataFrame, out_path: Path) -> None:
    if model_df.empty:
        return
    df = model_df.copy()
    df = df.sort_values(by="margin_per_1m_med", ascending=False)
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(x), df["margin_per_1m_med"], color="#4e79a7")
    ax.set_title("Gross Margin per 1M Tokens (Median)")
    ax.set_ylabel("€/1M tokens")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_public_scenarios(public_df: pd.DataFrame, fixed_total_with_loan: float, out_path: Path) -> None:
    if public_df.empty:
        return
    df = public_df.copy()
    cases = ["worst", "base", "best"]
    df = df.set_index("case").loc[cases]
    # Grouped bars: revenue, cogs, marketing, fixed+loan, net
    labels = ["Revenue", "COGS", "Marketing", "Fixed+Loan", "Net"]
    revenue = df["revenue_eur"].values
    cogs = df["cogs_eur"].values
    marketing = df["marketing_reserved_eur"].values
    fixed = [fixed_total_with_loan] * len(cases)
    net = df["net_eur"].values

    width = 0.15
    x = range(len(cases))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 2*width for i in x], revenue, width, label="Revenue", color="#4e79a7")
    ax.bar([i - width for i in x], cogs, width, label="COGS", color="#e15759")
    ax.bar(x, marketing, width, label="Marketing", color="#f28e2b")
    ax.bar([i + width for i in x], fixed, width, label="Fixed+Loan", color="#76b7b2")
    ax.bar([i + 2*width for i in x], net, width, label="Net", color="#59a14f")
    ax.set_xticks(list(x), cases)
    ax.set_title("Public Scenarios — Monthly Components")
    ax.legend(loc="best")
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_break_even(required_inflow: float | None, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Break-even Prepaid Inflow")
    ax.set_xlabel("Inflow €")
    ax.set_yticks([])
    if required_inflow is None:
        ax.text(0.5, 0.5, "No finite break-even (margin ≤ marketing)", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.axvline(required_inflow, color="#e15759", linestyle="--", label=f"Break-even €{required_inflow:,.0f}")
        ax.legend(loc="best")
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


essential_private_cols = ["gpu", "provider_eur_hr_med", "sell_eur_hr", "margin_eur_hr", "markup_pct"]


def plot_private_tap(private_df: pd.DataFrame, out_path: Path) -> None:
    if private_df.empty:
        return
    df = private_df[essential_private_cols].copy()
    df = df.sort_values(by="margin_eur_hr", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = range(len(df))
    ax.bar([i - 0.15 for i in idx], df["provider_eur_hr_med"], width=0.3, label="Provider €/hr", color="#e15759")
    ax.bar([i + 0.15 for i in idx], df["sell_eur_hr"], width=0.3, label="Sell €/hr", color="#4e79a7")
    ax.set_xticks(list(idx), df["gpu"], rotation=30, ha="right")
    ax.set_title("Private Tap — GPU Economics")
    ax.legend(loc="best")
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_loan_balance(loan_df: pd.DataFrame, out_path: Path) -> None:
    if loan_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loan_df["month"], loan_df["balance_end_eur"], color="#4e79a7")
    ax.set_title("Loan Balance Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("€")
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
