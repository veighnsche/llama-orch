# Finance Engine v1 — README (.002-draft)

Korte projectbeschrijving voor de Python-engine die het financiële plan rekent en de lender-sjabloon vult.

## Doel
- Lees alle invoer uit `inputs/` (YAML/CSV).
- Valideer verplichte velden en schrijf `outputs/validation_report.json`.
- Leg alle aannames vast in `outputs/assumptions.yaml` (FX, TPS, median GPU per model).
- Bereken alle tabellen (kosten €/1M tokens, scenario’s, private tap, break-even, lening, BTW).
- Render `template.md` → `outputs/template_filled.md` (geen `{{...}}` meer).
- Schrijf `outputs/run_summary.md` (timestamp, versie, file-hashes, FX, TPS, KPI’s).

Deterministisch: geen netwerk; alle constants komen uit inputs of worden vastgelegd in `assumptions.yaml`.

## Belangrijkste invoer
- `inputs/config.yaml`
  - Vereist: `currency`, `pricing_inputs.fx_buffer_pct`, `tax_billing.vat_standard_rate_pct`, `finance.marketing_allocation_pct_of_inflow`.
  - Vaste kosten: `finance.fixed_costs_monthly_eur.{personal,business?}` (business kan `null` → waarschuwing).
- `inputs/costs.yaml` (referentiebanden; informatief voor v1).
- `inputs/lending_plan.yaml` (loan: `amount_eur`, `term_months`, `interest_rate_pct`; optioneel `monthly_payment_eur`, `total_repayment_eur`).
- `inputs/price_sheet.csv` (modelprijzen voor Public Tap; rijen voor private: `private_tap_management_fee`, `private_tap_gpu_hour_markup`).
- `inputs/oss_models.csv` (modelcatalogus)
- `inputs/gpu_rentals.csv` (USD/hr min–max per GPU)
- `inputs/tps_model_gpu.csv` (optioneel benchmarks; voorkeur `aggregate`, `vLLM`/`TensorRT-LLM`)
- `inputs/extra.yaml` (overrides: FX, scenario’s, per-model mix, TPS, median GPU, prijs-fallbacks, VAT)

## Outputs (must exist)
- `outputs/model_price_per_1m_tokens.csv`
- `outputs/public_tap_scenarios.csv`
- `outputs/private_tap_economics.csv`
- `outputs/break_even_targets.csv`
- `outputs/loan_schedule.csv`
- `outputs/vat_set_aside.csv`
- `outputs/assumptions.yaml`
- `outputs/validation_report.json`
- `outputs/run_summary.md`
- `outputs/template_filled.md`

## Kernlogica (samenvatting)
- FX & providerprijzen: USD/hr → EUR/hr met `eur_usd_rate` en `fx_buffer_pct`; median = (min+max)/2.
- Model↔GPU pairing v1: 6–9B→RTX4090/L4; 30–34B→L40S/A100 80GB; 70–72B→A100 80GB/H100; MoE 8x7B→A100 80GB/L40S. Opslaan in `assumptions.yaml` (overrides via `extra.yaml`).
- Kosten per 1M tokens per model:
  - `tokens_per_hour = tps * 3600`
  - `cost_per_1M = eur_hr / (tokens_per_hour/1_000_000)` (min/median/max)
  - `sell_per_1M = unit_price_per_1k * 1000`
- Public Tap scenario’s (maand): worst/base/best M tokens; gewogen gemiddeld (gelijke weging of `extra.per_model_mix`); `net = revenue − cogs − fixed − marketing`.
- Private Tap: `sell_hr = eur_hr_median * (1 + markup%)`; `margin_hr = sell_hr − eur_hr_median` + management fee.
- Break-even: `required_inflow = (fixed_total + marketing_reserve) / margin_rate` (baseline margin-rate uit Public Tap median).
- Lening: flat interest, 60 mnd, tabel 1..60.
- BTW: voorbeelden voor €1k/€10k/€100k.

## Template-rendering
- Vervang alle scalars in `template.md`.
- Tabellen injecteren uit CSV’s:
  - `{{tables.model_price_per_1m_tokens}}`
  - `{{tables.private_tap_gpu_economics}}`
  - `{{tables.loan_schedule}}`

## Validatie & Acceptatie
- `validation_report.json` bevat `errors[]/warnings[]/info[]`; bij errors → non-zero exitcode.
- `template_filled.md` mag geen `{{...}}` meer bevatten.

## Implementatieplan (verwachtingen voor Python dev)
1) Loader/validator (pandas + yaml; geen netwerk). Schrijf `validation_report.json` en stop bij errors.
2) `assumptions.yaml` samenstellen (FX, TPS-selectie uit benchmark of defaults, median GPU map, VAT, provider selector).
3) Berekeningen 3.1–3.8 uit `prompt.md` implementeren; CSV’s schrijven.
4) Renderer: placeholders vervangen en tabellen injecteren; `template_filled.md` schrijven.
5) `run_summary.md` met timestamp, versie (`v1.0.0`), input-hashes, FX, TPS, KPI’s.
6) Optioneel `run.py` en `Makefile` target `make run`.
7) Eenvoudige rooktest (pytest): run en assert op non-empty outputs.

## Gebruik (indicatief)
- `python3 engine.py` of `python3 run.py`
- Alle artefacten verschijnen onder `outputs/`.

### TPS selection rules (if inputs/tps_model_gpu.csv is present)
- For each model, try to select a benchmark row in this order:
  1) gpu == assumptions.public_tap.median_gpu_for_model,
  2) measurement_type == "aggregate",
  3) engine in ["vLLM","TensorRT-LLM"].
- If multiple match, pick the row with the highest throughput_tokens_per_sec.
- If none match, relax (1) and use any GPU for that model; still prefer "aggregate".
- Record the chosen row under outputs/assumptions.yaml → public_tap.selected_tps[model].

Precedence: inputs/extra.yaml overrides config.yaml, costs.yaml, and price_sheet.csv **only** for the documented keys; all overrides must be echoed in outputs/run_summary.md.

If a public-tap model is missing in price_sheet.csv and not provided in extra.yaml.price_overrides,
emit a WARNING and exclude that model from public revenue/margin calculations.
