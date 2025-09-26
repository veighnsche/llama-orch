# FINAPLAN v2 — Deterministic Finance Engine Specification

Status: Draft (active)
Owner: .business/financieelplan_NL_nl

## 1. Purpose & Scope

This specification defines the deterministic Python-based finance engine ("FINAPLAN v2") that transforms a single, human-editable input file into Markdown reports and CSV detail tables consumed by business stakeholders and documentation pipelines.

The engine is designed to be:

- Deterministic (identical outputs for identical inputs)
- Auditable (no hidden state, simple math, clear rounding)
- Scriptable (single CLI, single canonical input)
- Extensible (explicit templates, placeholder whitelist)

Out-of-scope: Any Excel-based workflow (legacy). No network calls, no randomness, no timestamps.

## 2. Normative Requirements (FP-1xxx)

- FP-1001 Single Input: The engine MUST accept exactly one canonical input file (`config.yaml` by default) and MUST NOT read other business inputs at runtime.
- FP-1002 Format: Input MUST be YAML or JSON. If YAML is used, PyYAML MUST be available; otherwise the CLI MUST error with a clear message.
- FP-1003 Determinism: All numeric calculations MUST use decimal arithmetic with `ROUND_HALF_UP`. Month ordering and CSV column order MUST be stable.
- FP-1004 Rounding: Monetary outputs MUST be shown with two decimals and EUR formatting in Markdown. CSVs MUST serialize raw decimals in two-decimal fixed-point strings.
- FP-1005 Templates: Unknown or missing placeholders in `.md.tpl` templates MUST fail-fast with a clear error. A whitelist is parsed from `templates/placeholders/placeholders.md`.
- FP-1006 Stress: The engine MUST provide the following stress recomputations deterministically: (a) omzet −30%, (b) DSO +30 days, (c) OPEX +20%, and (d) a combined scenario. Placeholders MUST be populated accordingly.
- FP-1007 VAT: The engine MUST support VAT (BTW) accrual models and scheduled payments by period (monthly/quarterly). KOR/vrijstelling MUST set VAT payable to zero.
- FP-1008 Loans: The engine MUST support annuity loans with grace months (interest-only) and export a full amortization schedule.
- FP-1009 Depreciation: The engine MUST compute straight-line depreciation per investment item, prorated from the start month.
- FP-1010 DSCR/Runway: The engine MUST compute monthly DSCR and the minimum across the horizon, and MUST compute runway (max consecutive months with cash ≥ 0).
- FP-1011 Outputs: The engine MUST emit all CSVs and render all Markdown templates under `out/` in a single run.
- FP-1012 Credits-first: When modeling a prepaid credits stream, DSO MUST be configurable and SHOULD be set to 0 for that stream or globally, reflecting prepaid cash.

## 3. Input Schema (YAML/JSON)

Top-level keys with defaults shown:

```yaml
schema_version: 1
bedrijf:
  naam: "Onbekend"          # display only
  rechtsvorm: "IB"          # "IB" (eenmanszaak) or "BV"
  start_maand: "YYYY-MM"
  valuta: "EUR"
horizon_maanden: 12           # ≥ 1
scenario: "base"             # base | best | worst
vat_period: "monthly"        # monthly | quarterly
btw:
  btw_pct: 21
  model: "omzet_enkel"       # or "omzet_minus_simple_kosten"
  kor: false
  btw_vrij: false
  mkb_vrijstelling_pct: 0     # %

# Revenue streams. Credits-first example (Public Tap): price per 1M tokens, volume in M tokens.
omzetstromen:
  - naam: "Public Tap (credits)"
    prijs: 1.20                 # EUR per 1M tokens
    volume_pm: [50, 75, 100]    # short arrays are carried forward to the horizon
    var_kosten_per_eenheid: 0.85 # infra cost per 1M tokens
    btw_pct: 21
    dso_dagen: 0                 # prepaid credits → DSO ≈ 0

# Fixed OPEX per month
opex_vast_pm:
  personeel: []                 # [{ rol: string, bruto_pm: number }]
  marketing: 0
  software: 0
  huisvesting: 0
  overig: 0

# Investments (CAPEX)
investeringen:
  - omschrijving: "Item"
    bedrag: 0
    levensduur_mnd: 36
    start_maand: "YYYY-MM"

# Financing
financiering:
  eigen_inbreng: 0
  leningen:                        # zero or more
    - verstrekker: "Qredits"
      hoofdsom: 20000
      rente_nominaal_jr: 7.0       # or rente_nominaal_jr_pct
      looptijd_mnd: 60
      grace_mnd: 3
      alleen_rente_in_grace: true

# Working capital
werkkapitaal:
  dso_dagen: 0
  dpo_dagen: 14
  dio_dagen: 0
  vooruitbetaald_pm: 0
  deferred_revenue_pm: 0

# Assumptions
essumpties:
  discount_rate_pct: 10.0
  kas_buffer_norm: 2500
```

Defaults/normalization rules:

- Missing arrays are filled to `horizon_maanden` by carrying forward the last provided value.
- Missing fields are defaulted as in the example above.
- Unknown keys are ignored by the engine, but templates still enforce a placeholder whitelist.

## 4. Computation Semantics

- Calendar: Build a month axis from `bedrijf.start_maand` over `horizon_maanden` months.
- Depreciation: Straight-line per item; prorated starting at the item’s `start_maand`.
- Loans: Annuity with monthly rate = (nominal yearly / 12); grace months optionally interest-only. Full schedule exported.
- Revenue & COGS: For each stream, revenue = `prijs × volume`; COGS = `var_kosten_per_eenheid × volume`.
- OPEX: Sum of scalar categories plus monthly sum of personeel bruto_pm.
- EBITDA: (revenue − COGS − OPEX) per month.
- VAT: Accrual computed from model; payment scheduled by `vat_period` (last month of month/quarter). KOR/vrijstelling → VAT set to zero.
- Cashflow: Includes DSO/DPO timing, CAPEX outflows, VAT payments, loan interest/principal, and initial inflows (eigen inbreng, loan proceeds).
- DSCR: Monthly DSCR = EBITDA / (interest + principal); min DSCR across the horizon is reported.
- Runway: Max consecutive months with end-cash ≥ 0.
- Stress recompute: Omzet −30%, DSO +30d, OPEX +20%, and a combined scenario. Liquidity, DSCR, EBITDA, and end-cash metrics are recomputed for stress placeholders.
- Indicative Income Tax: A simple, transparent estimate:
  - IB: apply `mkb_vrijstelling_pct`, then an indicative flat rate (current implementation ~28%).
  - VPB: indicative flat ~25.8%.
  - Populates `belastingdruk_pct` and `indicatieve_heffing_jaar` (CSV + MD). This is not fiscal advice.

## 5. Templates & Placeholders

- Location: `.business/financieelplan_NL_nl/templates/*.md.tpl`
- Whitelist: Parsed from `templates/placeholders/placeholders.md`. Renderer fails on unknown/missing placeholders.
- Example templates: `00_overview.md.tpl`, `10_investering_financiering.md.tpl`, `20_liquiditeit.md.tpl`, `30_exploitatie.md.tpl`, `40_qredits_maandlasten.md.tpl`, `50_belastingen.md.tpl`, `60_pricing.md.tpl`, `70_breakeven.md.tpl`, `80_unit_economics.md.tpl`, `90_working_capital.md.tpl`.

## 6. Outputs

CSV files (stable headers):

- `10_investering.csv`: `omschrijving, levensduur_mnd, start_maand, afschrijving_pm, bedrag`
- `10_financiering.csv`: `verstrekker, hoofdsom, rente_nominaal_jr_pct, looptijd_mnd, grace_mnd, termijn_bedrag`
- `20_liquiditeit_monthly.csv`: `maand, begin_kas, in_omzet, in_overig, uit_cogs, uit_opex, uit_btw, uit_rente, uit_aflossing, eind_kas`
- `30_exploitatie.csv`: `maand, omzet, cogs, marge, opex_<categorie>..., opex_totaal, afschrijving, rente, ebitda, resultaat_vb`
- `40_amortisatie.csv`: `maand, verstrekker, rente_pm, aflossing_pm, restschuld`
- `50_tax.csv`: `regime, btw_pct, btw_model, kor, btw_vrij, mkb_vrijstelling_pct, belastingdruk_pct, indicatieve_heffing_jaar`

Markdown files: one output per `*.md.tpl` rendered to `out/<name>.md`.

## 7. CLI Contract

Program: `.business/financieelplan_NL_nl/fp`

Commands:

- `./fp validate <config.yaml|json> [--months N]` — Validate and print applied defaults.
- `./fp run <config.yaml|json> --out ./out --months N --scenario base|best|worst --vat-period monthly|quarterly` — Run engine and write reports.
- `./fp check-templates` — Verify templates against whitelist and generate `templates/TEMPLATE_KEYS.md`.
- `./fp list-keys` — Print all whitelisted placeholders.

Makefile convenience (single input machine):

- `make templates validate run` — uses `INPUT=config.yaml` by default.
- `make migrate-v1 INPUT=<old.yml> OUT=config.yaml` — converts v1 schema to v2.

## 8. Determinism & Reproducibility

- Decimal context: `ROUND_HALF_UP` for monetary rounding.
- Stable order: Months in chronological order; CSV headers fixed; template order fixed.
- No randomness, no timestamps.
- Unknown/missing placeholders cause errors; YAML parsing is strict.

## 9. Migration (v1 → v2)

- Tool: `tools/migrate_v1_to_v2.py` converts legacy `omzetmodel/opex_pm` into v2’s `omzetstromen/opex_vast_pm`.
- Output path: Defaults to `config.yaml` for single config workflows.

## 10. Non-Goals

- No Excel/Jinja pipeline; all reports are Markdown + CSV.
- No external API calls, no network, no caching.

## 11. Versioning & Change Control

- Any change to input schema, outputs, or placeholder set MUST update this spec and the placeholder whitelist.
- Template changes MUST be validated by `./fp check-templates`.
- Breaking changes are allowed pre-1.0.0 but MUST be documented.

## 12. Verification Plan (FV-2xxx)

- FV-2001 Determinism: Running `make run` twice with the same `config.yaml` MUST yield identical bytes under `out/`.
- FV-2002 Templates: `./fp check-templates` MUST pass with 0 unknown placeholders.
- FV-2003 CSV Headers: Each CSV MUST contain the exact header sets listed in §6.
- FV-2004 Stress: Stress placeholders in Markdown MUST be populated without errors.
- FV-2005 VAT: KOR/vrijstelling set to true/false MUST toggle VAT payable accordingly.
- FV-2006 Loans: A `leningen` entry MUST produce a non-empty `40_amortisatie.csv` and DSCR metrics.
- FV-2007 Depreciation: An `investeringen` entry MUST produce `10_investering.csv` with correct `afschrijving_pm`.
- FV-2008 Credits-first: With a stream having `dso_dagen: 0`, cash receipts timing MUST reflect prepaid behavior.

## 13. Example Minimal Config

```yaml
schema_version: 1
bedrijf:
  naam: "Demo BV"
  rechtsvorm: "IB"
  start_maand: "2026-01"
horizon_maanden: 12
scenario: "base"
vat_period: "monthly"

btw: { btw_pct: 21, model: "omzet_enkel", kor: false, btw_vrij: false, mkb_vrijstelling_pct: 0 }

omzetstromen:
  - naam: "Public Tap (credits)"
    prijs: 1.20
    volume_pm: [50, 75, 100]
    var_kosten_per_eenheid: 0.85
    btw_pct: 21
    dso_dagen: 0

opex_vast_pm: { personeel: [], marketing: 400, software: 120, huisvesting: 800, overig: 350 }

investeringen:
  - { omschrijving: "GPU", bedrag: 12000, levensduur_mnd: 36, start_maand: "2026-02" }

financiering:
  eigen_inbreng: 8000
  leningen:
    - { verstrekker: "Qredits", hoofdsom: 20000, rente_nominaal_jr: 7.0, looptijd_mnd: 60, grace_mnd: 3 }

werkkapitaal: { dso_dagen: 0, dpo_dagen: 14, dio_dagen: 0, vooruitbetaald_pm: 0, deferred_revenue_pm: 0 }
assumpties: { discount_rate_pct: 10.0, kas_buffer_norm: 2500 }
```
