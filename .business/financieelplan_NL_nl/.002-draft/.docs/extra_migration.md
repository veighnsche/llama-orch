# Extra.yaml Removal Mapping (Step B)

This document inventories all current usages of `extra[...]` across the finance engine and proposes their first-class destinations.

## Inventory (source → usage)

- compute/pricing.py
  - `extra.fx.eur_usd_rate` → used in `_get_fx_and_buffer()`
  - `extra.include_models` → used in `_select_models()`
  - `extra.price_overrides[sku].unit_price_eur_per_1k_tokens` → used in `_sell_per_1m()`
  - `extra.assumed_tps[model]` / `.default` / `[gpu]` → used in `_model_tps()`
  - `extra.median_gpu_for_model[model]` → used in `_model_gpu()`
- engine_pkg/steps/compute.py
  - `extra.scenarios.monthly.(worst/base/best)_m_tokens` → `_finance_knobs()`
  - `extra.per_model_mix` → `_finance_knobs()` (weights across models)
  - `extra.pricing_inputs.private_tap_markup_by_gpu[GPU]` → `_compute_private_block()`
  - `extra.loan.{amount_eur,term_months,interest_rate_pct}` → fallback existed; removed in Step B planning
- engine_pkg/steps/context.py
  - `extra.prepaid.*` → included in rendered context
  - `extra.private.management_fee_eur_per_month` → included in context
  - `extra.price_overrides` → surfaced in context `pricing.model_prices`
  - `extra.runway_target_months` → included in context

## Destination mapping (target files and keys)

- FX and buffers
  - `extra.fx.eur_usd_rate` → `config.yaml.fx.eur_usd_rate`
- Scenario inflows and model weights
  - `extra.scenarios.monthly.*` → `inputs/scenarios.yaml.monthly.*` (validator added)
  - `extra.per_model_mix` → `inputs/scenarios.yaml.per_model_mix` (new key)
  - `extra.include_models` → `inputs/scenarios.yaml.include_skus` (new key)
- Model capacity and GPU pairing overrides
  - `extra.assumed_tps[model][default|<GPU>]` → Prefer tabular in `tps_model_gpu.csv` (add columns: `assumed_tps_override`, `assumed_tps_gpu_override`), or new `inputs/capacity_overrides.yaml` if not easily tabular.
  - `extra.median_gpu_for_model[model]` → `inputs/capacity_overrides.yaml.median_gpu_for_model[model]` (or additional column in `tps_model_gpu.csv`)
- Pricing overrides and markups
  - `extra.price_overrides[sku].unit_price_eur_per_1k_tokens` → Prefer `inputs/price_sheet.csv` as explicit rows/filled values; otherwise `inputs/overrides.yaml.price_overrides` as a temporary migration target enforced by validator.
  - `extra.pricing_inputs.private_tap_markup_by_gpu[GPU]` → `inputs/gpu_pricing.yaml.private_tap_markup_by_gpu[GPU]`
- Private tap, prepaid, runway
  - `extra.private.management_fee_eur_per_month` → `config.yaml.prepaid_policy.private_tap.management_fee_eur_per_month`
  - `extra.prepaid.*` → `config.yaml.prepaid_policy.credits.*`
  - `extra.runway_target_months` → `config.yaml.finance.runway_target_months`
- Loans (remove fallback)
  - `extra.loan.*` → already exists in `lending_plan.yaml`; remove any fallback use in compute.

## Migration plan

1. Introduce new files incrementally:
   - `inputs/scenarios.yaml` (added)
   - `inputs/gpu_pricing.yaml` (markups per GPU)
   - `inputs/overrides.yaml` (temporary for price overrides if `price_sheet.csv` is not filled yet)
   - `inputs/capacity_overrides.yaml` (if not embedding into `tps_model_gpu.csv`)
2. Update validators:
   - Add required validators for new files; optional initially, strict-required with `ENGINE_STRICT_VALIDATION=1`.
3. Refactor reads:
   - Replace `extra[...]` reads with new files (above).
   - Delete `extra.yaml` handlers.
4. Harden:
   - Flip coverage and missing-price to errors (strict)
   - Enforce single unit mode across `public_tap` (strict)
5. Transition note:
   - For first hardened run, include an “Extra removal mapping” section in `outputs/run_summary.md`.

## Strict mode toggle

Set `ENGINE_STRICT_VALIDATION=1` to flip warnings → errors and enforce mixed-units failure. In CI we will turn this on when migrations are complete.
