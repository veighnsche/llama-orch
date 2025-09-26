# Migration Notes: v1 → v2 Finance Engine

This guide helps you migrate inputs and expectations from the older v1 configuration to the v2 deterministic engine under `core/`.

## Summary of Key Changes

- **Input schema:** v2 uses simple, explicit keys. Replace `omzetmodel` with `omzetstromen` and `opex_vast_pm`.
- **Determinism:** all math via `decimal.Decimal` with ROUND_HALF_UP; outputs are byte-identical for the same inputs.
- **Templates:** Markdown is rendered from `.md.tpl` files using a fail-fast Mustache-like renderer with a whitelist parsed from `templates/placeholders/placeholders.md`.
- **CSV outputs:** fixed/stable column order for all CSVs. See `templates/placeholders/placeholders.md` → CSV section.
- **CLI:** `./fp` provides `run`, `validate`, `list-keys`, and `check-templates`.

## v1 → v2 Field Mapping

- `omzetmodel.omzet_pm`, `omzetmodel.cogs_pct`, `omzetmodel.seizoen` → v2 splits into streams under `omzetstromen`:
  - `omzetstromen[*].prijs`
  - `omzetstromen[*].var_kosten_per_eenheid`
  - `omzetstromen[*].volume_pm` (array; short arrays are carried forward)
- `omzetmodel.opex_pm` → v2 `opex_vast_pm` with:
  - `personeel: [{ rol, bruto_pm }, ...]`
  - scalars: `marketing`, `software`, `huisvesting`, `overig`
- `belastingen.btw_*` → v2 `btw` block (`btw_pct`, `model`, `kor`, `btw_vrij`).
- `belastingen.vat_period` → top-level `vat_period`.
- Investments/loans are similar but normalized:
  - `investeringen[*]: { omschrijving, bedrag, levensduur_mnd, start_maand }`
  - `financiering: { eigen_inbreng, leningen: [{ verstrekker, hoofdsom, rente_nominaal_jr, looptijd_mnd, grace_mnd }] }`
- Working capital: `werkkapitaal: { dso_dagen, dpo_dagen, dio_dagen, vooruitbetaald_pm, deferred_revenue_pm }`.

## CLI Usage

- Validate input only:
  ```bash
  ./fp validate examples/v2_example.json --months 24
  ```
- Check templates against whitelist and generate `templates/TEMPLATE_KEYS.md`:
  ```bash
  ./fp check-templates
  ```
- Run the engine and write outputs to `./out/`:
  ```bash
  ./fp run examples/v2_example.json --out ./out --months 24 --scenario base --vat-period monthly
  ```

## What changed in outputs

- Markdown outputs now come from templates in `templates/*.md.tpl`.
- CSV outputs remain stable and are produced by `core/io_.py`:
  - `10_investering.csv`
  - `10_financiering.csv`
  - `20_liquiditeit_monthly.csv`
  - `30_exploitatie.csv`
  - `40_amortisatie.csv`
  - `50_tax.csv`

## Determinism & Guardrails

- No timestamps or randomness in generated files.
- Template render fails fast on unknown or missing keys.
- Filesize guards: core modules ≤ 300 LOC; CLI ≤ 200 LOC. Use `make check`.

## FAQ

- Q: Can I still use YAML?  
  A: Yes, but `pyyaml` must be installed. Otherwise use JSON.

- Q: My v1 `omzetmodel` input produces zeros.  
  A: Migrate to `omzetstromen` and `opex_vast_pm` as shown above, or use the v2 example dataset in `examples/`.

- Q: How do I add new placeholders?  
  A: Update `templates/placeholders/placeholders.md` and re-run `./fp check-templates` to regenerate `TEMPLATE_KEYS.md`. Update `core/mapping.py` to provide values.
