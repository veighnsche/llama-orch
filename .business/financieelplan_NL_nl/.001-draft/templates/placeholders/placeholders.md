# Finance Engine — Deterministic Spec (for IDE AI)

**Goal**
Build a modular, deterministic Python engine that takes a simple YAML/JSON input and produces all Markdown + CSV outputs required by our templates. Calculations must be realistic, auditable, and byte-identical for the same inputs.

**Non-negotiables**

* **Deterministic math:** Python `decimal.Decimal` with `ROUND_HALF_UP`, fixed cents. No randomness, no timestamps.
* **Reproducible ordering:** stable month order `YYYY-MM`, stable column orders.
* **No runtime repo scanning:** All prose is template-fixed. Only numbers/short verdicts change.
* **Small files:** modules ≤ **300 LOC** each.
* **Fail-fast templates:** unknown placeholder → hard error; missing required placeholder → hard error.

**Recommended module layout**

```
core/
  money.py          # Decimal ctx, currency/percent formatting
  calendar.py       # month series, DSO/DPO shifts, VAT timing
  depreciation.py   # straight-line + prorata start
  loan.py           # annuity, grace, amortization tables
  pnl.py            # omzet → cogs → marge → opex → ebitda → dep → interest → resultaat
  cashflow.py       # operating/investing/financing cash, VAT payments, runway
  tax.py            # indicative IB/VPB + VAT summary (no advice)
  scenario.py       # base/best/worst/stress deltas
  metrics.py        # DSCR, ICR, breakeven, IRR, NPV, payback, KPIs
  io_.py            # read yaml/json, write csv/md, render templates (whitelist)
  schema.py         # validate inputs & defaults, versioning
```

---

## Calculation principles (closest to reality)

* **Timeline:** monthly buckets across horizon (e.g., 12–36 months).
* **Accrual P\&L, cash timing in CF:** revenue recognized by month; cash receipts shifted by **DSO**; payables by **DPO**; VAT paid monthly/quarterly.
* **Depreciation:** straight-line per item, prorated from start month.
* **Loans:** annuity by default; support grace (interest-only) months; compute full amortization table; effective APR informative.
* **VAT:** model variants (`omzet_enkel` vs “output − input simplified”), payment in last month of period; KOR/BTW-vrij zeroes out payments.
* **Metrics:**

  * **EBITDA** = gross margin − OPEX.
  * **DSCR (monthly)** = `EBITDA_m / (interest_m + principal_m)`; min/avg across horizon.
  * **ICR** = `EBIT / interest`.
  * **Runway** = max consecutive months with end-cash ≥ 0.
  * **Breakeven units** = `fixed_costs_pm / margin_per_unit`; **breakeven omzet** = units × price.
  * **IRR/NPV/Payback:** project cash flows (capex out, loan inflows, operating cash, debt service). NPV at given discount; IRR via fixed-iteration Newton; **payback months** = first month cum. project cash ≥ 0.
* **CCC (optional DIO):** `DSO + DIO − DPO`.
* **Stress tests:** apply deterministic deltas (e.g., omzet −30%, OPEX +10%, DSO +30d) and recompute summary metrics.

---

## Input schema (beginner-friendly YAML)

Users can fill only the basics; defaults cover the rest. (JSON equivalent is 1:1.)

```yaml
schema_version: 1
bedrijf:
  naam: "Jouw BV"
  rechtsvorm: "BV"         # of "IB"
  start_maand: "2026-01"
  valuta: "EUR"

horizon_maanden: 24
scenario: "base"           # base | best | worst
vat_period: "monthly"      # monthly | quarterly
btw:
  btw_pct: 21
  model: "omzet_enkel"     # of "omzet_minus_simple_kosten"
  kor: false
  btw_vrij: false

omzetstromen:              # voeg 1..n stromen toe (simpel houden!)
  - naam: "API"
    prijs: 49.0            # per eenheid
    volume_pm: [100, 120, 140, 160]   # eerste maanden; daarna laatst herhalen
    var_kosten_per_eenheid: 10.0      # hosting/API/support per unit
    btw_pct: 21
    dso_dagen: 30

opex_vast_pm:              # vaste kosten per maand (bedragen incl. of excl. BTW – kies 1 beleid)
  personeel:
    - rol: "Founder"
      bruto_pm: 3000
    - rol: "Contractor 0.5 FTE"
      bruto_pm: 1750
  marketing: 800
  software: 400
  huisvesting: 600
  overig: 300

investeringen:             # CAPEX items
  - omschrijving: "GPU server"
    bedrag: 12000
    levensduur_mnd: 36
    start_maand: "2026-02"

financiering:
  eigen_inbreng: 8000
  leningen:
    - verstrekker: "Qredits"
      hoofdsom: 20000
      rente_nominaal_jr: 7.0  # %
      looptijd_mnd: 60
      grace_mnd: 3            # alleen rente tijdens grace

werkkapitaal:
  dso_dagen: 30              # klanten betalen na 30 dagen
  dpo_dagen: 14              # leveranciers betaald na 14 dagen
  dio_dagen: 0               # 0 als geen voorraad
  vooruitbetaald_pm: 0       # prepaid contracten/kosten
  deferred_revenue_pm: 0

assumpties:
  discount_rate_pct: 10.0    # voor NPV/IRR
  kas_buffer_norm: 2500      # gewenste minimale kasbuffer

stress:                      # optioneel; engine gebruikt defaults indien leeg
  omzet_pct: -30
  opex_pct: 10
  dso_plus_dagen: 30
```

**Design notes for IDE AI**

* Validate inputs and **apply defaults** if arrays shorter than horizon (carry last value forward).
* Accept *numbers only* from users; percentages as plain numbers (e.g., 7 = 7%).
* If the user is unsure, **print the defaults used** in DEV\_NOTES.

---

## Placeholders — short descriptions

Below are **all placeholders** referenced by templates (consolidated) with a short meaning. Use this as the **whitelist** and as mapping targets for the engine’s outputs. *(List derived from our template set.)*&#x20;

**General / Overview**

* `horizon_maanden` — aantal maanden in de planning.
* `eerste_maand` / `laatste_maand` — eerste/laatste maand (YYYY-MM).
* `scenario` — gekozen scenario (base/best/worst).
* `regime` — fiscaal regime: IB of VPB.
* `btw_model` — BTW-berekeningsmodel.
* `vat_period` — BTW-afdrachtfrequentie (maand/kwartaal).
* `total_investering` / `total_financiering` — som CAPEX / som financiering.
* `eigen_inbreng` / `eigen_inbreng_pct` — eigen geld en % t.o.v. investering.
* `debt_equity_pct` — verhouding schuld/eigen vermogen.
* `eerste_kas` — kas in maand 1 na inbreng + leningontvangst.
* `laagste_kas` / `kas_diepste_maand` — minimum kas en in welke maand.
* `kas_waarschuwing` — korte tekstsignaal (OK/negatief).
* `kas_eind` — eindkas na horizon.
* `maandlast_dekking` — Gem. DSCR (EBITDA/schuldendienst).
* `dekking_verdict` — tekstsignaal (voldoende/onvoldoende).
* `irr_project` — IRR van projectcashflows.
* `discount_rate_pct` — disconteringsvoet voor NPV.
* `payback_maanden` — maanden tot cumulatieve cash ≥ 0.
* `runway_maanden_base` / `runway_maanden_worst` — maanden kas ≥ 0 (base/worst).

**Investeringen & Financiering**

* `inv_count` / `inv_totaal` — # en som investeringen.
* `inv_avg_levensduur` — gemiddelde levensduur (mnd).
* `inv_max_omschrijving` / `inv_max_bedrag` — grootste investering.
* `fin_eigen` / `fin_schuld` — eigen inbreng vs vreemd vermogen.
* `lening_count` / `lening_avg_looptijd` / `lening_maandlasten_gem` — leningen samenvatting.
* `inv_pct_eigen` / `inv_pct_schuld` — % dekking investering door eigen/schuld.
* **Per lening (repeat `leningen`):**

  * `verstrekker` — kredietverstrekker.
  * `hoofdsom` — geleend bedrag.
  * `looptijd_mnd` — looptijd in maanden.
  * `rente_nominaal_jr` / `rente_nominaal_maand` — rente p/jr & p/maand.
  * `grace_mnd` — maanden alleen rente.
  * `termijn_bedrag` — annuïtaire maandlast.
  * `rente_effectief_jaar` — effectieve jaarrente (APR, indicatief).
  * `rente_totaal` — totale rente over looptijd.
  * `restschuld_einde` — restschuld aan het einde (meestal 0).

**Liquiditeit**

* `kas_begin` / `kas_eind` — begin/eind kas.
* `kas_negatief_count` / `kas_negatief_verdict` — # maanden < 0 + signaal.
* `btw_totaal` — som BTW-betalingen over horizon.
* `btw_max_bedrag` / `btw_max_maand` — piek-BTW betaling.
* `dso_dagen` / `dpo_dagen` / `dio_dagen` — betaal/ontvangst/voorraaddagen.
* `ccc_dagen` — Cash Conversion Cycle (DSO + DIO − DPO).
* `stress_laagste_kas` / `stress_runway_maanden` — stress-resultaat kas/runway.
* `kas_na_stress_omzet` / `kas_na_stress_dso` / `kas_na_stress_opex` / `kas_na_stress_combo` — eindkas onder stress.
* `runway_stress_omzet` / `runway_stress_dso` / `runway_stress_opex` / `runway_stress_combo` — runway onder stress.
* `kas_buffer_norm` — gewenste minimale buffer.

**Exploitatie (P\&L)**

* `omzet_totaal` / `brutomarge_totaal` — som omzet / bruto marge.
* `brutomarge_pct` — brutomarge% van omzet.
* `opex_totaal` / `afschrijving_totaal` / `rente_totaal` — sommen kosten.
* `resultaat_vb_totaal` — resultaat vóór belasting totaal.
* `ebitda_pm_avg` — gemiddelde EBITDA per maand.
* `nettowinst_pct` — (indicatief) nettomarge%.
* `resultaat_best` / `maand_best` — beste maand.
* `resultaat_worst` / `maand_worst` — slechtste maand.
* `verlies_maanden` / `verlies_verdict` — # verliesmaanden + signaal.
* `stress_result_avg` / `stress_verlies_maanden` / `stress_verdict` — P\&L stress signaal.
* `breakeven_omzet_pm` / `veiligheidsmarge_pct` — zie ook BE-sectie.

**Qredits / Maandlasten**

* `dscr_min` / `dscr_maand_min` — laagste DSCR en maand.
* `dscr_below_1_count` / `dscr_verdict` — # maanden DSCR<1 + signaal.
* `stress_dscr_min` / `stress_dscr_below_1_count` / `stress_dscr_verdict` — DSCR onder stress.

**Belastingen (indicatief)**

* `btw_pct` / `kor` / `btw_vrij` / `mkb_vrijstelling_pct` — fiscale parameters.
* `belastingdruk_pct` — heffing als % van resultaat v/bel (indicatief).
* `indicatieve_heffing_jaar` — totaal indicatieve heffing.
* `stress_heffing_range` / `stress_kasimpact` — indicatieve variatie.

**Pricing**

* `prijs_per_eenheid` / `eenheden_pm` / `omzet_basis` — pricing kern.
* `marge_per_eenheid` / `marge_pct` — marge per unit en in %.
* `cac_eur` / `payback_maanden` — acquisitiekosten & payback.
* **Price sensitivity:** `prijs_min10/prijs_basis/prijs_plus10`, bijbehorende `omzet_*`, `marge_*`, `runway_*`.
* **Elasticity combos:** `volume_plus15/volume_min10/volume_basis`, `omzet_pricevol1/2`, `marge_pricevol1/2`.
* **Streams (repeat `omzetstromen`):** `naam`, `prijs`, `volume_pm`, `omzet_pm`, `marge_pm`.

**Break-even**

* `vaste_kosten_pm` / `variabele_kosten_per_eenheid` / `prijs_per_eenheid`.
* `marge_per_eenheid` / `marge_pct`.
* `breakeven_eenheden_pm` / `breakeven_omzet_pm`.
* `breakeven_omzet_plus10_opex` / `breakeven_omzet_min10_marge` — stress op BE.
* `veiligheidsmarge_pct` — afstand van basisomzet tot BE.

**Unit economics**

* `ltv_eur` / `ltv_cac_ratio` — LTV en verhouding t.o.v. CAC.
* `contribution_margin` — marge na variabele saleskosten/CAC.
* `contrib_basis/min10/plus10` — contribution in scenarios.
* `marge_pricemin10` / `marge_varplus10` — stress op unit marge.
* **Stream table (optional repeat):** `naam`, `prijs`, `var_kosten`, `marge_eh`, `marge_pct`.

**Working capital**

* `debiteuren_openstaand` / `crediteuren_openstaand` — openstaande posten.
* `voorraad` / `borg_depositos` — kapitaal vastgezet.
* `vooruitbetaald` / `deferred_revenue` — prepaids en vooruitontvangen omzet.
* `werkkapitaal_totaal` — netto WC behoefte.
* `runway_stress_*` / `kas_na_stress_*` — stress op kas & runway.

*(If a listed placeholder isn’t used by your current templates, keep it optional in the whitelist.)*

---

## CSV outputs (stable columns)

* **`10_investering.csv`**: `omschrijving, levensduur_mnd, start_maand, afschrijving_pm, bedrag`
* **`10_financiering.csv`**: `verstrekker, hoofdsom, rente_nominaal_jr_pct, looptijd_mnd, grace_mnd, termijn_bedrag`
* **`20_liquiditeit_monthly.csv`**: `maand, begin_kas, in_omzet, in_overig, uit_cogs, uit_opex, uit_btw, uit_rente, uit_aflossing, eind_kas`
* **`30_exploitatie.csv`**: `maand, omzet, cogs, marge, opex_<categorie>..., opex_totaal, afschrijving, rente, ebitda, resultaat_vb`
* **`40_amortisatie.csv`**: `maand, verstrekker, rente_pm, aflossing_pm, restschuld`
* **`50_tax.csv`**: `regime, btw_pct, btw_model, kor, btw_vrij, mkb_vrijstelling_pct, belastingdruk_pct, indicatieve_heffing_jaar`

---

## Edge cases & guardrails

* **Missing/short arrays:** extend with last value.
* **Zero/negative prices or volumes:** treat as **0** with warning in DEV\_NOTES.
* **Loan with 0% interest:** allowed; APR = 0; schedule still computed.
* **VAT quarterly:** group three months; pay in last month of quarter.
* **KOR/BTW-vrij:** VAT payable forced to zero with clear template note.
* **IRR fallback:** if no sign change in cashflow → IRR “n.v.t.” (string), do **not** crash.

---

## What the IDE AI must deliver

1. **Parser & schema validation** (defaults applied, clear errors).
2. **Computations** (modules above) producing in-memory model.
3. **Render engine** with placeholder whitelist + fail-fast.
4. **Writers** for MD (templates) and CSV (stable columns).
5. **Determinism tests** (same input → identical bytes).
6. **Docs**: a generated `TEMPLATE_KEYS.md` from whitelist; update `DEV_NOTES.md` with computed defaults actually used.
