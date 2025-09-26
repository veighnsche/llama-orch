# 90 · Werkkapitaal (v1 — deterministisch)

Brondata: `../dataset.v1.json`
Disclaimers: Indicatief; prijspeil 2025; excl. BTW; geen fiscale advisering.

## 1) Parameters uit dataset

- DSO (debiteuren):
  - PSP‑kanalen (Public Tap): **2 dagen** (`working_capital.dso_days.psp_channels`)
  - Facturen (Private Tap): **14 dagen** (`working_capital.dso_days.invoices_private_tap`)
- DPO (crediteuren):
  - Leveranciers infra: **7 dagen** (`working_capital.dpo_days.suppliers_infra`)
  - SaaS: **30 dagen** (`working_capital.dpo_days.saas`)
- Deferred revenue (vooruitbetaald):
  - Public Tap jaar‑vooruit aandeel: **20%** (`working_capital.deferred_revenue.annual_prepaid_share_public_tap`)
- BTW: 21% (kwartaal, netto‑positie varieert)

## 2) Formules (deterministisch)

- Weighted DSO (zonder jaarlijkse prepay):
  - `DSO_weighted = w_psp × 2d + w_invoice × 14d`, met `w_psp + w_invoice = 1`
- Effect van annual prepaid (Public Tap):
  - Jaarvooruit omzetdeel verlaagt AR; voor dat deel `DSO ≈ 0d` en ontstaat verplichting (deferred revenue).
- Weighted DPO:
  - `DPO_weighted = w_infra × 7d + w_saas × 30d`, met gewichten naar kostenmix.
- Cash Conversion Cycle (CCC, geen voorraad):
  - `CCC = DSO_weighted − DPO_weighted` (DIO = 0)

## 3) Bandbreedtes (zonder aannames over omzetmix)

- Ondergrens (alles prepaid/PSP): `DSO ≈ 2d`, `CCC ≈ 2 − 7 = **−5 dagen**` (gunstig)
- Bovenkant (alles op factuur): `DSO ≈ 14d`, `CCC ≈ 14 − 7 = **+7 dagen**`
- Met 20% jaarvooruit Public Tap daalt DSO verder voor dat segment naar ≈ 0d en verschuift liquiditeit naar vooruitontvangen posten.

## 4) BTW‑cashflow (samenvatting)

- BTW 21%, kwartaalritme. Netto‑positie afhankelijk van mix: prepaid/vouchers versus afdracht bij levering.
- Praktisch advies (operationeel): houd **≥ 1 maand** verwachte netto‑BTW als buffer.

## 5) Beleid en credit control (aanbevolen, niet bindend)

- Private Tap facturen 14d met automatische herinneringen (7/14/21d).
- PSP/credits standaard voor Public Tap om DSO te minimaliseren.
- Deferred revenue voor jaarpacks correct boeken; monthly performance pro rata vrijvallen.
