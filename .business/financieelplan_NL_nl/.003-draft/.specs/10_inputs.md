# 10 — Inputs Spec (Formats, Merge, Validation)

Status: Draft
Version: 0.1.0

## 1. Purpose & Scope

- **Doel.** Definieert alle inputregels voor de D3-engine.
- **Scope.** Bestandsstructuur, toegestane formats (YAML/CSV), split tussen `public_data` en `operator_{public,private}`, merge/precedentie, determinisme, en validatorregels.

## 2. Input Bundles

- **Public Data (exogeen).** Buiten operator‑controle.
  - YAML: `public_data.yaml` (alleen: `meta.version`, `meta.currency`, `fx.eur_usd_rate`).
  - CSV: `throughput_tps.csv`, `gpu_rentals.csv`, `insurances.csv`, `acquisition_benchmarks.csv`.
- **Operator Config (onder jouw controle).** Gesplitst in drie:
  - YAML: `operator_general.yaml` (globale financiën en simulatie-instellingen):
    - `finance.fixed_costs_monthly_eur.{personal,business}`
    - `finance.marketing_allocation_pct_of_inflow`
    - `insurances.selected` (koppelt naar public `insurances.csv`)
    - `loan.*` (bedrag, looptijd, rente)
    - `simulation.run_horizon_months` (bijv. 24)
  - YAML: `operator_public.yaml` (seed, fx buffer, catalog, public pricing/packs, acquisitie voor Public).
  - YAML: `operator_private.yaml` (seed, private pricing/fees, vendor weights, acquisitie voor Private).
  - CSV (optioneel): `catalog_models.csv`, `catalog_gpus.csv` (gebruikt door Public pijplijn).

## 3. Directory Layout

```
.003-draft/
  inputs/
    public_data.yaml
    operator_general.yaml
    operator_public.yaml
    operator_private.yaml
    throughput_tps.csv
    gpu_rentals.csv
    insurances.csv
    acquisition_benchmarks.csv
    catalog_models.csv
    catalog_gpus.csv
```

## 4. Supported Formats (ONLY YAML & CSV)

- **YAML** voor configuratie/kleine datasets.
- **CSV** voor grote tabellen en lijsten.
- **Niet toegestaan:** JSON, Excel, Parquet, SQL, API‑calls. Alleen lokale YAML/CSV.

## 5. Merge & Precedence

- **Strikte scheiding.**
  - `public_data.yaml`: `meta.version`, `meta.currency`, `fx.eur_usd_rate`, (optioneel) YAML‑vorm `throughput_tps` en `gpu_rentals`.
  - `operator_general.yaml` (globaal): `finance.*`, `insurances.selected`, `loan.*`, `simulation.run_horizon_months`.
  - `operator_public.yaml` (alleen Public pijplijn): `meta.seed`, `fx.fx_buffer_pct`, `catalog.*`, `pricing_policy.public_tap.*`, `prepaid_policy.credits.*`, `scenarios.*`, `acquisition.*`.
  - `operator_private.yaml` (alleen Private pijplijn): `meta.seed`, `fx.fx_buffer_pct` (MAY; MUST match public indien in beide gezet), `pricing_policy.private_tap.*`, `prepaid_policy.private_tap.*` (+ `base_fee*`, `vendor_weights`), `scenarios.*`, `acquisition.*` (private‑specifiek).
- **Verboden overlap.** Keys in het verkeerde bestand → **ERROR**.
- **Effectieve config.** Er zijn drie operator‑contexten: `O_general`, `O_public`, `O_private`. De engine laadt `public_data` + `O_general` + de relevante operatorfile per pijplijn. Voor consolidatie worden outputs gecombineerd; gedeelde operatorparameters (bijv. `fx_buffer_pct`) MUST consistent zijn (indien in beide aanwezig).
- **CSV wint.** Als YAML én CSV voor dezelfde dataset aanwezig zijn, MUST CSV prevaleren; YAML wordt genegeerd met **WARNING** (shadowed input).

### 5.1 Deprecation

- `operator_config.yaml` wordt niet langer ondersteund in D3 (geen backwards compat pre‑1.0). Gebruik `operator_public.yaml` en `operator_private.yaml`.

## 6. Determinisme & Seeding

- Elke pijplijn gebruikt zijn eigen seed: `operator_public.yaml: meta.seed` en `operator_private.yaml: meta.seed`.
- Seeds mogen gelijk zijn; indien verschillend, zijn runs nog steeds deterministisch per pijplijn.
- `run_summary.{json,md}` MUST de seed, input‑hashes en merge/precedence beslissingen loggen (incl. shadowing‑warnings).

## 7. CSV Schemas (MUST)

- `throughput_tps.csv`:
  - Kolommen: `model,gpu,tps` (float, `tps>0`).
  - Uniek per `(model,gpu)`.
- `gpu_rentals.csv`:
  - Kolommen: `gpu,vram_gb,provider,usd_hr` (`vram_gb>0`, `usd_hr>0`).
  - Geen min/max/percentkolommen.
- `catalog_models.csv`:
  - Kolom: `model` (exacte naam als in catalog).
- `catalog_gpus.csv`:
  - Kolom: `gpu` (exacte naam als in catalog).

- `insurances.csv` (Public):
  - Kolommen: `insurance_id,provider,product,category,monthly_premium_eur,deductible_eur,coverage_limit_eur,notes`.
  - Domeinen: `monthly_premium_eur>=0`, `deductible_eur>=0`, `coverage_limit_eur>=0` of leeg.

- `acquisition_benchmarks.csv` (Public):
  - Kolommen: `channel,cvr_mean,cvr_sd,cac_mean,cac_sd` (alle ≥0; `cvr_*` in 0..1 met clipping in simulatie).

## 8. Validator Rules (MUST)

- **Schema‑checks:** types, domeinen, kolomnamen exact volgens §7; anders **ERROR**.
- **Throughput coverage:** voor elke `catalog.models` ≥ 1 `(model,gpu)` entry; anders **ERROR**.
- **Curated lijsten:** alleen `catalog.models`/`catalog.gpus` worden meegenomen; onbekenden → **WARNING/ERROR** afhankelijk van impact.
- **Providerprijzen:** `usd_hr>0` en plausibel (SHOULD `<50`).
- **FX regels:** `fx.eur_usd_rate` (public) en `fx_buffer_pct` (operator_public/private) aanwezig.
- **Precedentie‑log:** CSV > YAML shadowing MUST als **WARNING** gelogd worden (met bestandsnamen).
- **Insurances:** `insurances.csv` schema/domeinen valide. Indien `operator_general.yaml → insurances.selected` verwijst naar onbekende `insurance_id` → **ERROR**.
- **Horizon:** `simulation.run_horizon_months` MUST ≥ 1 (aanbevolen 24). Engine MUST loan‑uitvoer leveren voor `loan.term_months` (typisch 60).
- **Acquisitie benchmarks:** schema/domeinen valide; geen onbekende kanalen als operator allocaties opgegeven zijn.

## 9. Prohibited / Safety

- Geen netwerk tijdens run; geen externe bronnen.
- Geen andere file formats; geen dynamische code in YAML/CSV.

## 10. Examples

```yaml
# public_data.yaml (kern)
meta: { version: 3, currency: EUR }
fx: { eur_usd_rate: 1.08 }
```

```csv
# throughput_tps.csv
model,gpu,tps
Llama-3-1-8B,A100-40GB-PCIe,3200
Llama-3-1-8B,L4,1800
```

```csv
# insurances.csv
insurance_id,provider,product,category,monthly_premium_eur,deductible_eur,coverage_limit_eur,notes
PRO-PL-001,ACME,Professional Liability,liability,29.5,250,1000000,EU coverage
OFF-CONT-001,ACME,Contents,property,12.9,0,25000,Office equipment
```

```csv
# acquisition_benchmarks.csv
channel,cvr_mean,cvr_sd,cac_mean,cac_sd
google_ads,0.04,0.01,4.0,1.5
linkedin_ads,0.015,0.005,8.0,2.0
events,0.08,0.03,120.0,40.0
```

## 12. Zie ook

- `99_calc_or_input.md` — matrix met Calc/Input classificatie en acquisitie‑simulatieontwerp.

## 11. Change Policy

- Pre‑1.0: geen backwards compat‑garantie. Schema’s kunnen wijzigen (release notes verplicht).
