# 12 — Public Data / Facts (Shapes & Examples)

Status: Draft
Version: 0.1.0

## 1. Scope

- Exogene input die buiten operator‑controle valt.
- Bronnen: markt/benchmarks; geen netwerk tijdens run. Data staat lokaal onder `inputs/facts/`.
- Bestanden (huidige set):
  - `ads_channels.csv`
  - `agency_fees.csv`
  - `insurances.csv`
  - `market_env.yaml`

## 2. CSV — `ads_channels.csv`

- Kolommen (MUST):

```csv
channel_id,platform,region,objective,cost_model,unit,currency,typical_value,low_value,high_value,notes,source_url,source_date
```

- Domeinen/richtlijnen:
  - `cost_model` enum bijv. `cpc|cpm` en `unit` bijv. `per_click|per_1000_impressions`.
  - Valuta vrij (EUR/USB/...), engine converteert niet automatisch; gebruik als planning‑referentie.
  - Missing `low_value/high_value` toegestaan.

Voorbeeld (subset):

```csv
google_search_cpc_nl_est,Google Ads,NL,Search Clicks,cpc,per_click,USD,4.66,,,Global 2024 avg CPC; use NL multiplier if you adopt a country factor.,https://www.wordstream.com/blog/2024-google-ads-benchmarks,2025-05-20
```

## 3. CSV — `agency_fees.csv`

- Kolommen (MUST):

```csv
agency_id,agency_name,city,region,service_type,fee_model,unit,currency,value,low_value,high_value,setup_fee_currency,setup_fee_value,notes,source_url,source_date
```

- Domeinen/richtlijnen:
  - `fee_model` bijv. `flat_monthly|flat_hour` en bijpassende `unit`.
  - Monetary velden ≥ 0; `low/high` optioneel.

## 4. CSV — `insurances.csv`

- Kolommen (MUST):

```csv
insurer_id,insurer_name,product,focus_segment,coverage_highlights,example_limit_eur,example_deductible_eur,example_premium_eur_per_month,source_url,notes
```

- Domeinen/richtlijnen:
  - `example_*` velden zijn indicatief; engine mag buffers toepassen via variabelen (`insurances.premiums_buffer_pct` uit `variables/general.csv`).
  - `inputs/operator/general.yaml → insurances.selected` MUST verwijzen naar bestaande `insurer_id` waarden.

## 5. YAML — `market_env.yaml`

- Structuur (indicatief):

```yaml
demographics:
  population_nl: { value: 17600000, unit: people }
  internet_penetration_nl: { value: 0.97, unit: fraction }
economy:
  sme_count_nl: { value: 2200000, unit: companies }
energy:
  avg_electricity_price_nl: { value: 0.32, unit: EUR_per_kWh }
finance:
  eur_usd_fx_rate: { value: 1.10, unit: ratio }
```

- Normatief:
  - `finance.eur_usd_fx_rate.value` MUST aanwezig zijn; engine gebruikt dit i.c.m. operator `fx_buffer_pct` (indien van toepassing) voor conversies.

## 6. Validator‑regels (MUST)

- CSV headers exact overeenkomend met de schema’s in §2–§4; anders **ERROR**.
- `insurances.selected` (operator) MUST mappen op `insurer_id` in facts/insurances; anders **ERROR**.
- Geen providerprijzen of curated catalogus in facts (die horen in `inputs/operator/`).
- Geen netwerk; alleen lokale files.

## 7. Gebruik in simulatie (SHOULD)

- `ads_channels.csv` en `agency_fees.csv` worden als referentie gebruikt voor acquisitie‑ en agency‑kostenbanden in het rapport.
- `insurances.csv` voedt premies/coverage‑context; daadwerkelijke premie gebruikt buffers/percentages volgens operator/variables.
- `market_env.yaml` levert context (demografie/energie) en de FX koers.

## 8. Zie ook

- `11_operator_constants.md` — koppeling `insurances.selected`.
- `12_oprator_variables.md` — buffers/allocaties als variabelen.
