# Qredits Lender Pack – Analyzed Report

This folder contains a printable HTML report built from the simulation outputs under `../outputs/` (18‑month horizon). It converts raw data into lender‑friendly charts, tables, and narrative focused on ability to repay.

## Files

- `index.html` – Self‑contained report. Open in a browser and Print → Save as PDF.
- `styles.css` – Clean layout and print‑optimized styling.

## How to use

1. Open `index.html` in your browser.
2. Review the Executive Summary and Repayment sections.
3. Use the “Print to PDF” button (top‑right) for a polished PDF.

Notes:
- Data is inlined from `../outputs/{pnl_by_month.csv,cashflow_by_month.csv,loan_schedule.csv,consolidated_summary.json,kpi_summary.json}` at the time this file was generated.
- If you rerun the simulation and want fresh numbers, ask Cascade to “refresh the analyzed report,” or we can add a small generator script that rebuilds `index.html` directly from the latest `outputs/`.

## What lenders see

- Executive KPIs: total revenue/cost/margin (18m), utilization/SLA, runway.
- Charts: Revenue vs Costs, EBITDA vs Net Income, Cash vs Debt Service.
- Repayment: latest payment, outstanding principal, break‑even months, trough cash, and an approximate DSCR from EBITDA / payment.
- Detailed tables: full P&L, cashflow, and amortization schedule.

## Assumptions & realism

- Autoscaling utilization (avg/p95) and 0 SLA violations evidence capacity headroom.
- Growth path is measured (based on simulated demand and pricing), not a single jump.
- Unit economics improve with scale; EBITDA crosses positive before the end of the horizon.
- Sensitivity statistics are summarized in the report.

## Customization

- Language: If you prefer a Dutch version, tell Cascade and we’ll localize headings and narrative.
- DSCR method: By default we show an approximate DSCR using EBITDA/payment. We can switch to a more conservative cash‑flow‑based DSCR if desired.
- Additional sections: We can add covenant checks, scenario overlays (P50/P10), or a one‑page summary.

## Regeneration (optional)

We can add a tiny Python script (or a Rust `xtask`) to parse `../outputs/` and regenerate `index.html` with current data. This avoids stale inlined data and keeps one command to refresh the report.
