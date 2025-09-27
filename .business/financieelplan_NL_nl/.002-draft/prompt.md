# IDE AI Assignment — Finance Engine v1

# Repo: /home/vince/Projects/llama-orch

# Workspace: .business/financieelplan_NL_nl/.002-draft

# Goal: Read inputs/, compute financial outputs, fill template.md → template_filled.md

# and write a set of CSVs that justify all numbers

## 0) Constraints & Style

- Language: Python 3.11+.
- Libraries: stdlib + pandas (prefer), numpy (if needed). No network calls.
- Deterministic: all constants come from inputs files or are written to outputs/assumptions.yaml.
- Do not change input files. Write all artifacts under `.002-draft/outputs/`.
- Keep code in a single script `engine.py` plus a tiny `run.py` entrypoint (optional).
- Log a concise run summary to `outputs/run_summary.md`.

## 1) Inputs (all files already exist)

- .002-draft/inputs/config.yaml
- .002-draft/inputs/costs.yaml
- .002-draft/inputs/lending_plan.yaml
- .002-draft/inputs/price_sheet.csv      # per-model public tap pricing + private fees/markup
- .002-draft/inputs/oss_models.csv
- .002-draft/inputs/gpu_rentals.csv
- .002-draft/template.md                  # lender-facing markdown with {{placeholders}}
- .002-draft/calculation_hints.md         # dev-facing notes

### Validation rules

- YAMLs parse; required fields present:
  - config.yaml: currency, prepaid_policy, pricing_inputs.fx_buffer_pct, tax_billing.vat_standard_rate_pct, finance.marketing_allocation_pct_of_inflow
  - costs.yaml: fixed monthly personal, business (may be null → warn)
  - lending_plan.yaml: amount_eur, term_months, interest_rate_pct (flat)
- price_sheet.csv must contain rows for each public model with columns:
  `sku,category,unit,unit_price_eur_per_1k_tokens,margin_target_pct,notes`
- gpu_rentals.csv columns: `gpu,vram_gb,hourly_usd_min,hourly_usd_max,sources`
- oss_models.csv columns: `name,variant_size_b,context_tokens,license`

Emit `outputs/validation_report.json` with `errors[]/warnings[]/info[]`. If errors exist, still write the report and exit non-zero.

## 2) Assumptions (v1 placeholders to make math possible)

Because throughput (tokens/sec) isn’t measured yet, use an explicit assumptions file:

- Create `outputs/assumptions.yaml` with:
  - fx:
      eur_usd_rate: <read from config if present, else 1.08>   # constant used this run
      buffer_pct: <from config.pricing_inputs.fx_buffer_pct>
  - public_tap:

    # default assumed TPS per model *at median GPU used for that model*

    # If a per-model TPS map is not provided in config, set a conservative default (e.g., 35 tps for 7–9B, 15 tps for 30–34B, 10 tps for MoE 8x7B active)

    # and record these values here for transparency

      assumed_tps:
        "Llama-3.1-8B": 35
        "Llama-3.1-70B": 10
        "Mixtral-8x7B": 12
        "Mixtral-8x22B": 8
        "Qwen2.5-7B": 35
        "Qwen2.5-32B": 15
        "Qwen2.5-72B": 10
        "Yi-1.5-6B": 35
        "Yi-1.5-9B": 30
        "Yi-1.5-34B": 15
        "DeepSeek-Coder-6.7B": 30
        "DeepSeek-Coder-33B": 14
        "DeepSeek-Coder-V2-16B": 18
        "DeepSeek-Coder-V2-236B": 9
  - provider_price_selector: median   # use min/median/max when producing ranges
  - billing:
      vat_rate_pct: <from config.tax_billing.vat_standard_rate_pct>

These values MUST be echoed in `outputs/run_summary.md`.

## Optional: Real TPS Benchmarks
- .002-draft/inputs/tps_benchmarks.csv (optional)
  Schema (CSV headers, required):
    model_id,model_name,engine,precision,gpu,gpu_count,batch,input_tokens,output_tokens,
    throughput_tokens_per_sec,measurement_type,scenario_notes,source_tag
  Selection rules:
    - For each model, pick TPS from this file if available; else fall back to assumptions.yaml.
    - Prefer rows where:
        a) gpu == median_gpu_for_model (see assumptions.yaml),
        b) measurement_type == "aggregate" (throughput under batching/concurrency),
        c) engine in ["vLLM","TensorRT-LLM"].
      If multiple match, choose the row with the highest throughput_tokens_per_sec.
      If none match, relax (a) → same model on any GPU; still prefer "aggregate".
    - Record the chosen row (model_id, gpu, engine, source_tag, TPS) under
      outputs/assumptions.yaml → public_tap.selected_tps[model].

## 3) Core calculations

### 3.1 Currency & provider price normalization

- Convert GPU USD/hr to EUR/hr using `eur_usd_rate` and apply `fx.buffer_pct`.
- For each GPU: compute `eur_hr_min`, `eur_hr_median`, `eur_hr_max` from the input min/max (median = (min+max)/2 if not present).

### 3.2 Model ↔ GPU pairing (simple v1)

- Choose a “median GPU” per model with this rule (overrideable later by config):
  - 6–9B dense → RTX 4090 or L4
  - 30–34B dense → L40S or A100 80GB
  - 70–72B dense → A100 80GB or H100 80GB
  - MoE 8x7B → A100 80GB or L40S (assume fits via quant/activation)
- Record chosen median GPU per model into `outputs/assumptions.yaml` under `public_tap.median_gpu_for_model`.

### 3.3 Cost per 1M tokens (per model)

For each model:

- tokens_per_hour = assumed_tps * 3600
- cost_per_1k_tokens_(min/median/max) = eur_hr_(min/median/max) / (tokens_per_hour / 1000)
- cost_per_1M_tokens_(min/median/max) = cost_per_1k_tokens_(min/median/max) * 1000
Write `outputs/model_price_per_1m_tokens.csv` with columns:
`model, median_gpu, eur_hr_min, eur_hr_median, eur_hr_max, assumed_tps, cost_per_1m_min, cost_per_1m_median, cost_per_1m_max, sell_per_1m, margin_per_1m_median, margin_pct_median`
- `sell_per_1m` = `price_sheet.unit_price_eur_per_1k_tokens * 1000` for that model.

### 3.4 Public Tap scenarios (monthly)

Provide three scenario rows using tokens sold per month (M):

- Read from config if present; else set defaults: worst=1M, base=5M, best=15M.
- weighted average `sell_per_1M` and `cost_per_1M_median` across models included in price_sheet (equal weighting unless a per-model mix is provided; if not, log a warning).
- revenue = m_tokens * weighted_sell
- cogs = m_tokens * weighted_cost
- gross = revenue − cogs
- marketing = revenue * config.finance.marketing_allocation_pct_of_inflow/100
- net = gross − (fixed.personal + fixed.business + loan.monthly_payment)
Write `outputs/public_tap_scenarios.csv` with:
`case,m_tokens,revenue_eur,cogs_eur,gross_eur,gross_margin_pct,fixed_plus_loan_eur,marketing_eur,net_eur`

### 3.5 Private Tap economics

- From price sheet: `private_tap_gpu_hour_markup` (percent_over_cost) and `private_tap_management_fee`.
- For each GPU: sell_price_eur_hr = eur_hr_median * (1 + markup%)
- margin_eur_hr = sell − eur_hr_median
Write `outputs/private_tap_economics.csv` with:
`gpu, eur_hr_median, markup_pct, sell_eur_hr, margin_eur_hr, mgmt_fee_eur_month`

### 3.6 Break-even targets

- fixed_total = fixed.personal + fixed.business + loan.monthly_payment
- required_margin = fixed_total + marketing_reserve, where marketing_reserve = target_inflow * marketing_pct (solve self-consistently → required_inflow = required_margin / margin_rate; if margin_rate unknown, use weighted margin from §3.4 baseline; otherwise write `NA` and warn).
Write `outputs/break_even_targets.csv` with:
`fixed_total_eur, marketing_pct, margin_rate_pct, required_margin_eur, required_prepaid_inflow_eur`

### 3.7 Loan schedule (flat interest, 60 months)

- monthly_payment = provided in lending_plan or compute: (loan + loan*rate*term/12)/term
- Build table month 1..60 with opening, interest, principal, payment, closing.
Write `outputs/loan_schedule.csv`.

### 3.8 VAT set-aside examples

- Use sample gross revenues: 1000, 10000, 100000 EUR.
- vat_set_aside = gross * vat_rate_pct/100; net = gross − vat_set_aside.
Write `outputs/vat_set_aside.csv`.

## 4) Template rendering

- Read `.002-draft/template.md`. Replace all `{{...}}` placeholders with computed values.
  - Scalars come from YAMLs/CSVs or derived numbers above.
  - Tables: inject pre-rendered Markdown for:
    - `{{tables.model_price_per_1m_tokens}}` from `model_price_per_1m_tokens.csv`
    - `{{tables.private_tap_gpu_economics}}` from `private_tap_economics.csv`
    - `{{tables.loan_schedule}}` from `loan_schedule.csv`
- Write to `.002-draft/outputs/template_filled.md`.

## 5) Run summary

Write `.002-draft/outputs/run_summary.md` including:

- Timestamp, engine version (start with `v1.0.0`), file hashes of inputs, FX rate used, TPS map used.
- Any warnings (e.g., missing business fixed costs, assumed per-model mix).
- Quick KPIs: fixed_total, monthly loan payment, baseline scenario net.

## 6) Acceptance criteria (must pass)

- All outputs exist:
  - outputs/model_price_per_1m_tokens.csv
  - outputs/public_tap_scenarios.csv
  - outputs/private_tap_economics.csv
  - outputs/break_even_targets.csv
  - outputs/loan_schedule.csv
  - outputs/vat_set_aside.csv
  - outputs/assumptions.yaml
  - outputs/validation_report.json
  - outputs/run_summary.md
  - outputs/template_filled.md
- No validation errors in `validation_report.json`.
- `template_filled.md` contains no raw `{{...}}` placeholders.

## 7) Nice-to-have (if trivial)

- Add `make run` target that executes the engine and writes outputs.
- Add a light `pytest` with a smoke test: run engine on provided inputs and assert files exist and are non-empty.

# End of assignment
