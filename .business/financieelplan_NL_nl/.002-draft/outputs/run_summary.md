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
- scenarios.yaml: ✓ (3 rows/keys)
- gpu_pricing.yaml: ✓
- capacity_overrides.yaml: ✓
- overrides.yaml: ✓

Warnings:
- gpu_pricing.yaml: gpu_pricing.yaml not present (optional)
- coverage: GPUs referenced in tps_model_gpu.csv not found in gpu_rentals.csv: A100, A100 40G, A100 80G, A40, H100, H200 (Step A: warn)
- coverage: 13 pseudo SKUs from oss_models.csv not found in price_sheet.csv (Step A: warn)
- units: public_tap unit(s) observed: 1k_tokens
# Run Summary
- Engine version: v1.0.0
- Run at (UTC): 2025-09-27T17:30:54Z
