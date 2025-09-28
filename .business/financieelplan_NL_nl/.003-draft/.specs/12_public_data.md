# 12 — Public Data (Shapes & Examples)

Status: Draft
Version: 0.1.0

## 1. Scope

- Exogene input die buiten operator‑controle valt.
- Bronnen: markt/benchmarks, geen netwerk tijdens run; data staat lokaal.
- Bestanden: `public_data.yaml`, optioneel `throughput_tps.csv`, `gpu_rentals.csv`, `insurances.csv`, `acquisition_benchmarks.csv`.

## 2. YAML — `public_data.yaml`

```yaml
meta:
  version: 3
  currency: EUR
fx:
  eur_usd_rate: 1.08   # spot/marktkoers (exogeen)
# Optioneel (kleine datasets): throughput_tps en gpu_rentals kunnen ook in YAML,
# maar CSV is aanbevolen bij grotere tabellen.
```

## 3. CSV

- `throughput_tps.csv`
  - Kolommen: `model,gpu,tps` (float, `tps>0`).
  - Uniek per `(model,gpu)`.
  - Voor elke `catalog.models` in operator‑config bestaat ≥1 rij (validator eist dit).

```csv
model,gpu,tps
Llama-3-1-8B,A100-40GB-PCIe,3200
Llama-3-1-8B,L4,1800
Qwen2-5-7B,A100-40GB-PCIe,3000
Qwen2-5-7B,L4,1700
```

- `gpu_rentals.csv`
  - Kolommen (granulair per provider — pre‑1.0 normatief): `gpu,vram_gb,provider,usd_hr`.
  - Domeinen: `vram_gb>0`, `usd_hr>0`.
  - Geen min/max of percent‑velden; mediaan/statistiek wordt door de engine berekend.

```csv
gpu,vram_gb,provider,usd_hr
A100-40GB-PCIe,40,acmecloud,1.40
A100-40GB-PCIe,40,othercloud,1.26
A100-80GB,80,acmecloud,2.30
H100-80GB,80,acmecloud,3.90
H200-141GB,141,acmecloud,9.20
L4,24,othercloud,0.82
L40S,48,othercloud,0.95
RTX-3090,24,labcloud,0.80
RTX-4090,24,labcloud,0.95
```

- `insurances.csv`
  - Kolommen: `insurance_id,provider,product,category,monthly_premium_eur,deductible_eur,coverage_limit_eur,notes`.
  - Domeinen: `monthly_premium_eur>=0`, `deductible_eur>=0`, `coverage_limit_eur>=0` of leeg.

```csv
insurance_id,provider,product,category,monthly_premium_eur,deductible_eur,coverage_limit_eur,notes
PRO-PL-001,ACME,Professional Liability,liability,29.5,250,1000000,EU coverage
OFF-CONT-001,ACME,Contents,property,12.9,0,25000,Office equipment
```

- `acquisition_benchmarks.csv`
  - Kolommen: `channel,cvr_mean,cvr_sd,cac_mean,cac_sd` (≥0; `cvr_*` in 0..1 met clipping in simulatie).

```csv
channel,cvr_mean,cvr_sd,cac_mean,cac_sd
google_ads,0.04,0.01,4.0,1.5
linkedin_ads,0.015,0.005,8.0,2.0
events,0.08,0.03,120.0,40.0
```

## 4. Precedentie

- Als dezelfde dataset zowel in YAML als CSV aanwezig is, dan **MUST CSV winnen**.
- YAML wordt genegeerd met een duidelijke **WARNING** (shadowed input) in `run_summary` en logs.

## 5. Validator‑regels (public)

- `throughput_tps.csv`: kolommen exact, unieke `(model,gpu)`, `tps>0`.
- `gpu_rentals.csv`: kolommen exact `[gpu,vram_gb,provider,usd_hr]`, `usd_hr>0`, `vram_gb>0`.
- `fx.eur_usd_rate` aanwezig in `public_data.yaml`.
- Geen onbekende of ontbrekende tabellen; fout → **ERROR**.

## 6. Gebruik in simulatie

- Public Tap (`PublicTapSim`): `eur_hr(g)` op basis van minimale providerprijs per GPU; per model gekozen GPU + provider gelogd.
- Private Tap (`PrivateTapSim`): mediaan prijs per GPU‑klasse voor verkoopquote; provider-aanbeveling op minimale `eur_hr` (of score als geconfigureerd).
- Insurances: maandelijkse premies uit `insurances.csv` worden opgeteld voor policies geselecteerd in `operator_config.yaml → insurances.selected`.
- Acquisitie: benchmark‑parameters voeden de seeded simulatie (zie `99_calc_or_input.md`).

## 7. Zie ook

- `11_operator_inputs.md` — operator selectie van verzekeringen en acquisitie‑assumpties.
- `99_calc_or_input.md` — calc/input matrix en acquisitie‑simulatieontwerp.

## 7. Afbakening

- Geen operator‑policies of catalogus in public bundle.
- Geen extra formaten of externe calls. Alleen lokale YAML/CSV, deterministisch verwerkbaar.
