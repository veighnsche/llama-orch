## Preflight
STATUS: OK (validation)

Files:
- config.yaml: ✓ (11 rows/keys)
- costs.yaml: ✓ (13 rows/keys)
- lending_plan.yaml: ✓ (3 rows/keys)
- gpu_rentals.csv: ✓ (22 rows/keys)
- oss_models.csv: ✓ (17 rows/keys)
- price_sheet.csv: ✓ (19 rows/keys)
- tps_model_gpu.csv: ✓ (22 rows/keys)
- scenarios.yaml: ✓ (1 rows/keys)
- gpu_pricing.yaml: ✓ (9 rows/keys)
- capacity_overrides.yaml: ✓ (2 rows/keys)
- overrides.yaml: ✓ (0 rows/keys)
- acquisition.yaml: ✓ (2 rows/keys)
- funnel_overrides.yaml: ✓ (3 rows/keys)
- private_sales.yaml: ✓ (7 rows/keys)
- seasonality.yaml: ✓ (12 rows/keys)
- timeseries.yaml: ✓ (4 rows/keys)
- billing.yaml: ✓ (1 rows/keys)
- competitor_benchmarks.yaml: ✓ (3 rows/keys)

Warnings:
- coverage: GPUs referenced in tps_model_gpu.csv not found in gpu_rentals.csv: A100, A100 40G, A100 80G, A40, H100, H200 (Step A: warn)
- coverage: 13 pseudo SKUs from oss_models.csv not found in price_sheet.csv (Step A: warn)
- units: public_tap unit(s) observed: 1k_tokens

Overrides Applied:
- capacity:
  - Llama-3-3-70B: tps=16.6 gpu=H100 expires_on=None
  - DeepSeek-R1-Distill-Llama-8B: tps=58.5 gpu=A100 expires_on=None
# Run Summary
- Engine version: v1.0.0
- Run at (UTC): 2025-09-28T10:32:35Z
