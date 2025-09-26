# ALIGNMENT — Template Sections, Headings, and Placeholders (DRAFT)

Scope: This document locks the template structure and the placeholder contract derived from one-time reading of `.002-draft`. It is DEV-only guidance. The runtime generator will not access `.002-draft` or any scan outputs.

Determinism: Outputs must be byte-identical across runs for the same inputs. All prose is fixed in templates. Only numbers and short deterministic verdicts change when inputs change.

Unknown keys: If a template contains an unknown placeholder, the generator MUST fail fast with a clear error: `Unknown template key: <KEY>`.

## Global Template Rules

- Engine: minimal, stdlib-only.
- Placeholders: `{{KEY}}`.
- Repeats: Mustache-style blocks `{{#list}}...{{/list}}` (keys inside use the item’s fields). Ordering is the original input order unless explicitly sorted.
- No timestamps or non-deterministic content.
- Decimal: `ROUND_HALF_UP`, fixed cents.
- Currency in Markdown prose uses `€ 12.345,67`. CSV uses dot-decimal strings (quantized to cents), stable column order.

## Files and Sections

Templates live under `templates/` with the following filenames and headings. CSVs are appended by the program and not templated.

### 1) 00_overview.md.tpl — “Overzicht”

Heading: `# Overzicht`

Required placeholders (minimum):
- `{{start_maand}}`
- `{{scenario}}`
- `{{vat_period}}`
- `{{total_investering}}`
- `{{total_financiering}}`
- `{{eigen_inbreng}}`
- `{{eigen_inbreng_pct}}` (0–100 with `%` sign)
- `{{debt_equity_pct}}` (formatted as `DEBT% / EQUITY%`)
- `{{eerste_kas}}` (eigen inbreng + hoofdsommen in maand 1)
- `{{laagste_kas}}`
- `{{maandlast_dekking}}` (e.g., `1.25x`)

Example sentences:
- `- Start kas (eigen inbreng + leningen in maand 1): {{eerste_kas}}`
- `- Laagste kasstand: {{laagste_kas}}`
- `- Maandlast-dekking (gem. EBITDA / gem. schuldendienst): {{maandlast_dekking}}`
- `- Belastingregime: {{regime}}, BTW-model: {{btw_model}}, periode: {{vat_period}}.`
- `_Indicatief; geen fiscale advisering._`

### 2) 10_investering_financiering.md.tpl — “Investeringen & Financiering”

Heading: `# Investeringen & Financiering`

Required placeholders:
- `{{inv_totaal}}`
- `{{inv_count}}`
- `{{fin_eigen}}`
- `{{fin_schuld}}`
- `{{lening_count}}`

Optional repeat block for listing loans:
```
{{#leningen}}
- {{verstrekker}} — hoofdsom {{hoofdsom}}, rente {{rente_nominaal_jr}}%, looptijd {{looptijd_mnd}} mnd, grace {{grace_mnd}} mnd
{{/leningen}}
```

CSV emitted alongside:
- `10_investering.csv`
- `10_financiering.csv`

### 3) 20_liquiditeit.md.tpl — “Liquiditeit”

Heading: `# Liquiditeit`

Required placeholders:
- `{{kas_begin}}` (first month begin)
- `{{kas_eind}}` (last month end)
- `{{laagste_kas}}`
- `{{btw_model}}`

Example sentence:
- `Kasstroomtabel met DSO/DPO verschuivingen en BTW-afdracht per {{vat_period}}.`

CSV emitted alongside:
- `20_liquiditeit_monthly.csv`

### 4) 30_exploitatie.md.tpl — “Exploitatie (maandelijks)”

Heading: `# Exploitatie (maandelijks)`

Required placeholders (totals over horizon):
- `{{omzet_totaal}}`
- `{{brutomarge_totaal}}`
- `{{opex_totaal}}`
- `{{afschrijving_totaal}}`
- `{{rente_totaal}}`
- `{{resultaat_vb_totaal}}`

Example sentence:
- `Omzet → COGS → marge → OPEX → afschrijving → rente → resultaat (v/bel).`

CSV emitted alongside:
- `30_exploitatie.csv` with columns `maand, omzet, cogs, marge, opex_<categorie>..., opex_totaal, afschrijving, rente, resultaat_vb`.

### 5) 40_qredits_maandlasten.md.tpl — “Qredits / Maandlasten”

Heading: `# Qredits / Maandlasten`

Minimum placeholders (per-lening and/or aggregated):
- `{{termijn_bedrag}}` (annuity per month, if single loan) OR expose repeat block below.
- `{{rente_nominaal_jr}}`
- `{{looptijd_mnd}}`
- `{{grace_mnd}}`

Recommended repeat block for multi-loan narrative:
```
{{#leningen}}
- {{verstrekker}} — termijn {{termijn_bedrag}} p/m, rente {{rente_nominaal_jr}}%, looptijd {{looptijd_mnd}} mnd, grace {{grace_mnd}} mnd
{{/leningen}}
```

CSV emitted alongside:
- `40_amortisatie.csv` (full schedule)

### 6) 50_belastingen.md.tpl — “Belastingen (indicatief)”

Heading: `# Belastingen (indicatief)`

Required placeholders:
- `{{regime}}` (IB|VPB)
- `{{btw_pct}}`
- `{{btw_vrij}}`
- `{{kor}}`
- `{{mkb_vrijstelling_pct}}`
- `{{btw_model}}`

Example sentences:
- `Regime: {{regime}}. BTW-model: {{btw_model}}. KOR: {{kor}}. BTW-vrij: {{btw_vrij}}.`
- `_Indicatief; geen fiscale advisering._`

CSV emitted alongside:
- `50_tax.csv`

### 7) zz_schema.md.tpl — “Schema (mapping input → output)”

Heading: `# Schema (mapping input → output)`

Required placeholder:
- `{{mapping_block}}` — a fixed, pre-rendered block listing the mapping between input fields and output tables/columns (NL labels), based on `DEV_NOTES.md`.

## Placeholder Semantics and Formats

- Monetary placeholders are pre-formatted with `money()` (e.g., `€ 12.345,67`).
- Percentage placeholders include `%` where applicable.
- Month placeholders use `YYYY-MM`.
- `{{debt_equity_pct}}` is formatted: `DEBT% / EQUITY%`.
- Lists (`{{#leningen}}...{{/leningen}}`) iterate in input order; inner keys available: `verstrekker, hoofdsom, rente_nominaal_jr, looptijd_mnd, grace_mnd, termijn_bedrag`.

## Enforcement

- Unknown placeholder → fail.
- Missing required placeholders in a template → fail with `Template missing required placeholder: <KEY>`.
- No runtime reading of `.002-draft`, `.specs`, `frontend/`, or `consumers/`.
- Deterministic outputs: sorting, formatting, and numeric precision are fixed.

## Example Snippet (Overview)

```
# Overzicht

- Start kas (eigen inbreng + leningen in maand 1): {{eerste_kas}}
- Totaal investering: {{total_investering}}; totaal financiering: {{total_financiering}}
- Eigen inbreng: {{eigen_inbreng}} ({{eigen_inbreng_pct}}); Debt/Equity: {{debt_equity_pct}}
- Laagste kasstand: {{laagste_kas}}
- Maandlast-dekking (gem. EBITDA / gem. schuldendienst): {{maandlast_dekking}}
- Regime: {{regime}}, BTW-model: {{btw_model}}, periode: {{vat_period}}

_Indicatief; geen fiscale advisering._
```
