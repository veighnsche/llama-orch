# 80 · Unit Economics (v1 — deterministisch)

Brondata: `../dataset.v1.json`
Disclaimers: Indicatief; prijspeil 2025; excl. BTW; geen fiscale advisering.

## 1) Public Tap (prepaid credits)

- Definities:
  - `Revenue_per_1M_tokens = €1,20`
  - `GM_public = 55%`  ⇒ `GP_per_1M = 1,20 × 0,55 = €0,66`  |  `COGS_per_1M = 1,20 − 0,66 = €0,54`
- Effektieve tarieven per pack (afgeleid uit prijs / tokens_in_million):
  - Starter €50 / 41M ⇒ €1,2195 per 1M
  - Builder €200 / 166M ⇒ €1,2048 per 1M
  - Pro €500 / 416M ⇒ €1,2019 per 1M
- Kengetallen (deterministisch, formulevorm):
  - `ARPU_public_monthly ≈ (Σ pack_omzet × usage_load_factor) / #accounts`
  - `Gross_Profit_public = Revenue_public × 0,55`
  - `PSP_cost ≈ 0,018 × omzet_transactie + €0,30 × #transacties`

## 2) Private Tap (dedicated GPU API — voorbeeldtarieven)

- Doel GM: 40% (cloud‑scenario)
- A100 80GB:
  - `Revenue_per_GPU_hour = €1,80`
  - `GP_per_GPU_hour = 1,80 × 0,40 = €0,72` | `COGS = €1,08`
  - `Base_fee_monthly = €250` (kan overhead dekken; GM hierop beleidsmatig vast te stellen)
- H100 80GB:
  - `Revenue_per_GPU_hour = €3,80`
  - `GP_per_GPU_hour = 3,80 × 0,40 = €1,52` | `COGS = €2,28`
- On‑prem energiekost (indicatief):
  - `Power_monthly = 0,35 €/kWh × 0,6 kW × 24 × 30 ≈ €151,20`

## 3) Vaste lasten (ter referentie uit 70_breakeven.md)

- Payroll totaal (incl. 25% wg‑lasten, 8% vakantiegeld) ≈ €6.982,50/m
- Overige vaste kosten ≈ €950,00/m
- OPEX vast ≈ €7.932,50/m

## 4) CAPEX‑afschrijving (lineair)

- Workstation €2.500 / 36 mnd ⇒ `€69,44 / mnd`
- Upgrade €1.500 / 36 mnd (vanaf 2026‑07) ⇒ `€41,67 / mnd`

## 5) Blended GM en implicaties

- `GM_blended = w_public × 0,55 + w_private × 0,40` met `w_public + w_private = 1`
- Bij `w_public = w_private = 0,5` ⇒ `GM_blended = 0,475`
- Break‑even omzet per maand (herhaling): `Omzet = OPEX_vast / GM_blended`

## 6) KPI‑referenties (consistent met product‑metrics)

- Public Tap: `tokens_out`, `gross_margin`, `ARPU`, `active_accounts`
- Private Tap: `gpu_utilization`, `tokens_out`, `gross_margin`, `SLA_uptime`

## 7) Notities

- GM‑doelen zijn beleidsmatig: verhogen vergt betere inkoop, batching of hardware‑mix.
- Base fees (Private) verbeteren stabiliteit MRR en kunnen operationele overhead dekken.
