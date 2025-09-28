# 10 — Inputs Spec (Formats, Merge, Validation)

Status: Draft
Version: 0.1.0

## 1. Purpose & Scope

- **Doel.** Definieert alle inputregels voor de D3-engine.
- **Scope.** Bestandsstructuur, toegestane formats (YAML/CSV), strikte scheiding tussen **constanten** (operator‑YAML), **variabelen** (CSV) en **facts** (exogene data), overlay/precedentie, determinisme, en validatorregels.

## 2. Input Bundles

- **Constants (operator).** Onder jouw controle; veranderen niet binnen een simulatie:
  - YAML: `inputs/operator/general.yaml` (globale financiën, loan, tax, kalender, afschrijving, verzekeringen selectie),
           `inputs/operator/public_tap.yaml`, `inputs/operator/private_tap.yaml` (pijplijnspecifieke policies die constant blijven, zoals floors/caps, billing‑units, non‑refundables, seeds).
  - CSV curated lijsten: `inputs/operator/curated_public_tap_models.csv`, `inputs/operator/curated_gpu.csv` (whitelists en providerprijzen/offers).
- **Variables (operator‑overlays).** CSV “knoppen” die tijdens runs variëren:
  - `inputs/variables/general.csv`, `inputs/variables/public_tap.csv`, `inputs/variables/private_tap.csv`.
  - Zie `12_oprator_variables.md` voor schema, treatments en determinisme.
- **Facts (exogeen).** Buiten operator‑controle; referentiedata:
  - CSV/YAML in `inputs/facts/`: `ads_channels.csv`, `agency_fees.csv`, `insurances.csv`, `market_env.yaml` (incl. `finance.eur_usd_fx_rate`).
- **Simulation plan.**
  - YAML: `inputs/simulation.yaml` (runparameters, stochasticiteit, stress, consolidatie, UI, logging). Zie `15_simulation_constants.md`.

## 3. Directory Layout

```
.003-draft/
  inputs/
    simulation.yaml
    operator/
      general.yaml
      public_tap.yaml
      private_tap.yaml
      curated_public_tap_models.csv
      curated_gpu.csv
    variables/
      general.csv
      public_tap.csv
      private_tap.csv
    facts/
      ads_channels.csv
      agency_fees.csv
      insurances.csv
      market_env.yaml
```

## 4. Supported Formats (ONLY YAML & CSV)

- **YAML** voor configuratie/kleine datasets.
- **CSV** voor grote tabellen en lijsten.
- **Niet toegestaan:** JSON, Excel, Parquet, SQL, API‑calls. Alleen lokale YAML/CSV.

## 5. Merge & Precedence

- **Strikte scheiding.**
  - Constants: alleen in `inputs/operator/*.yaml` en curated CSV’s onder `inputs/operator/`.
  - Variables: alleen in `inputs/variables/*.csv` (overlays op operator‑paths; geen facts‑overschrijvingen).
  - Facts: alleen in `inputs/facts/*` (read‑only; geen operator/policy in facts).
- **Overlayvolgorde.**
  1) Laad constants (operator YAML + curated CSV’s)
  2) Pas variables per scope toe (zie `12_oprator_variables.md`)
  3) Lees facts (exogeen) ter referentie in berekeningen
- **Verboden overlap.** Variabelen buiten toegestane roots of operator‑keys in facts → **ERROR**.
- **CSV wint (datasets).** Indien eenzelfde dataset zowel in YAML als CSV bestaat (zeldzaam), MUST CSV prevaleren; YAML wordt genegeerd met **WARNING** (shadowed input) in `run_summary`.

### 5.1 Deprecation

- `operator_config.yaml` wordt niet langer ondersteund in D3 (geen backwards compat pre‑1.0). Gebruik `operator_public.yaml` en `operator_private.yaml`.

## 6. Determinisme & Seeding

- Seed‑resolutie volgorde:
  1) `inputs/simulation.yaml → stochastic.random_seed` (indien gezet)
  2) `inputs/simulation.yaml → run.random_seed` (indien gezet)
  3) `inputs/operator/public_tap.yaml → meta.seed` of `private_tap.yaml → meta.seed` (per pijplijn)
  4) Geen seed → **ERROR**
- Identieke inputs + seed → byte‑gelijke outputs over machines.
- `run_summary.{json,md}` MUST de seed(s), input‑hashes, overlay‑beslissingen en shadowing‑warnings loggen.

## 7. CSV/YAML Schemas (MUST)

- `inputs/operator/curated_public_tap_models.csv`:
  - Minimale kolommen: `Model,Variant,Developer,Quantization/Runtime,Typical_VRAM_for_4bit_or_MXFP4_GB,License,Download,Benchmarks,Notes`.
  - Extra kolommen (bijv. `weights_vram_4bit_gb_est`) zijn toegestaan.
- `inputs/operator/curated_gpu.csv` (provider‑prijzen/offers):
  - Minimale kolommen: `provider,gpu_model,gpu_vram_gb,num_gpus,price_usd_hr` of `price_per_gpu_hr`.
  - Engine MUST normaliseren naar de interne rentals‑vorm `[gpu, vram_gb, provider, usd_hr]`, waarbij `usd_hr = price_per_gpu_hr` (indien aanwezig) anders `price_usd_hr/num_gpus`.
  - Geen min/max/percent‑velden voor prijzen in de bron.
- `inputs/variables/*.csv`: zie strikt schema in `12_oprator_variables.md`.
- `inputs/facts/ads_channels.csv`:
  - Kolommen: `channel_id,platform,region,objective,cost_model,unit,currency,typical_value,low_value,high_value,notes,source_url,source_date`.
- `inputs/facts/agency_fees.csv`:
  - Kolommen: `agency_id,agency_name,city,region,service_type,fee_model,unit,currency,value,low_value,high_value,setup_fee_currency,setup_fee_value,notes,source_url,source_date`.
- `inputs/facts/insurances.csv`:
  - Kolommen: `insurer_id,insurer_name,product,focus_segment,coverage_highlights,example_limit_eur,example_deductible_eur,example_premium_eur_per_month,source_url,notes`.
  - `inputs/operator/general.yaml → insurances.selected` MUST refereren naar `insurer_id` waarden.
- `inputs/facts/market_env.yaml`:
  - Bevat secties (`demographics`,`economy`,`energy`,`finance`). FX koers op `finance.eur_usd_fx_rate.value` (MUST aanwezig).

## 8. Validator Rules (MUST)

- **Schema‑checks:** types, domeinen, kolomnamen exact volgens §7; anders **ERROR**.
- **Curated lijsten:** alleen modellen/GPUs uit de curated CSV’s worden meegenomen; onbekenden → **WARNING/ERROR** afhankelijk van impact.
- **Providerprijzen:** genormaliseerde `usd_hr>0` en plausibel (SHOULD `<50`).
- **FX regels:** `inputs/facts/market_env.yaml → finance.eur_usd_fx_rate.value` MUST aanwezig; `meta.fx_buffer_pct` aanwezig in operator‑YAML per pijplijn.
- **Precedentie‑log:** dataset‑CSV > YAML shadowing MUST als **WARNING** gelogd worden (met bestandsnamen). Variabele‑overrides worden gelogd als `variable_override`.
- **Insurances:** facts‑schema valide. `operator/general.yaml → insurances.selected` MUST verwijzen naar bestaande `insurer_id`; anders **ERROR**.
- **Horizon & loan:** `loan.*` valide; consolidatie MUST de volledige looptijd afdekken (typisch 60 mnd).
- **Acquisitiekanalen:** `channel_id`/kanalen uit variables allocaties MUST bestaan in `facts/ads_channels.csv` (anders **ERROR**).

## 9. Prohibited / Safety

- Geen netwerk tijdens run; geen externe bronnen.
- Geen andere file formats; geen dynamische code in YAML/CSV.

## 10. Examples

```yaml
# market_env.yaml (kern)
finance:
  eur_usd_fx_rate:
    value: 1.10
```

```csv
# curated_public_tap_models.csv (subset)
Model,Variant,Developer,Quantization/Runtime,Typical_VRAM_for_4bit_or_MXFP4_GB,License,Download,Benchmarks,Notes
Mistral 7B,7.3B,Mistral AI,GGUF(Q4_K_M); llama.cpp; vLLM; Transformers,6-8,Apache-2.0,mistralai/Mistral-7B-Instruct-v0.3,see model card,v0.3 adds function calling
```

```csv
# insurances.csv (facts)
insurer_id,insurer_name,product,focus_segment,coverage_highlights,example_limit_eur,example_deductible_eur,example_premium_eur_per_month,source_url,notes
centraal_beheer_avb,Centraal Beheer,Bedrijfsaansprakelijkheidsverzekering (AVB),ZZP/MKB,"AVB voor letsel- en zaakschade",,250,9.40,https://www.centraalbeheer.nl/zzp,
```

```csv
# ads_channels.csv
channel_id,platform,region,objective,cost_model,unit,currency,typical_value,low_value,high_value,notes,source_url,source_date
google_search_cpc_nl_est,Google Ads,NL,Search Clicks,cpc,per_click,USD,4.66,,,,https://www.wordstream.com/blog/2024-google-ads-benchmarks,2025-05-20
```

## 12. Zie ook

- `12_oprator_variables.md` — variabelen‑CSV schema, treatments en determinisme.
- `11_operator_constants.md` — operator‑YAML/curated CSV shapes.
- `19_facts.md` — facts‑datasets en gebruik.

## 11. Change Policy

- Pre‑1.0: geen backwards compat‑garantie. Schema’s kunnen wijzigen (release notes verplicht).
