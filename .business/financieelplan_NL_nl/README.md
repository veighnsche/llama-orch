# 3. Financieel Plan — Build README

> NOTE (2025-09): Deterministic Engine v2 is active. This repo uses a single canonical input file and renders Markdown + CSV deterministically. The legacy Excel/Jinja workflow below is kept for historical context.

## Deterministic Engine v2 (single input)

- Canonical input file (YAML): `config.yaml` in this folder.
- Run checks, validate, and render:

```
make templates validate run
```

- Migrate an older v1 file (with `omzetmodel`/`opex_pm`) to v2:

```
make migrate-v1 INPUT=<path/to/your_v1.yml> OUT=config.yaml
```

- Outputs go to `out/` as `.md` (per template) and `.csv` (stable columns).

Only edit `config.yaml`. No other inputs are used by the engine.

Dit is de historische **build-index** voor de Excel/Jinja pipeline. De huidige v2-engine gebruikt géén Excel meer en rendert direct naar Markdown + CSV op basis van `config.yaml`.

---

## 0. Structuur (overzicht)

```

daar finance/
finance\_input.yaml           # AI/IDE invulbare brondata (business getallen + aannames)
excel\_cellmap.yaml           # mapping van YAML-sleutels naar Excel-cellen/tabelranges
financieel-plan-1.xlsx       # Qredits-sjabloon met formules (bron)
gen\_finplan.py               # main generator: YAML → Excel (filled) + MD-samenvatting
templates/
finance\_summary.md.j2      # Jinja2: samenvattende rapportage (wordt MD → PDF)
out/
financieel-plan-1.filled.xlsx
finance\_summary.md
finance\_summary.pdf
charts/                    # optioneel: gegenereerde PNG’s (omzet/kosten/liquiditeit)

```

---

## 1. Invoer (door IDE-AI in te vullen)

- `finance_input.yaml`  
  - **Investeringsbegroting**: items met categorie / omschrijving / bedrag  
  - **Financiering**: eigen inbreng, lening (bedrag, rente%, looptijd)  
  - **Exploitatie**: omzet en kosten per maand  
  - **Liquiditeit**: (optioneel) specifieke posten per maand  
  - **Privébegroting** (optioneel): vaste lasten, netto privé  
  - **Belasting** (optioneel): IB/VPB parameters  
  - **Meta**: bedrijfsnaam, datum, valuta, btw-tarief, aangiftefrequentie

> Richtlijn: houd velden “plat & simpel” zodat AI consistent kan invullen.

---

## 2. Mapping naar Excel

- `excel_cellmap.yaml`  
  - `workbook`: verwijst naar `financieel-plan-1.xlsx`  
  - `sheets`: per tabblad de **fields** (cel → scalar) en **tables** (startcel + kolommen → rijen)  
  - Voorbeeld-tabs (pas cellen aan jouw sheet aan):
    - **"VRAGENLIJST"**: bedrijf_naam, datum, valuta, btw_tarief_pct, btw_aangifte  
    - **"Investering & Financiering"**: investeringen_totaal, tabel `investeringen`  
    - **"Exploitatie"**: tabellen `omzet` (maand, waarde), `kosten` (maand, waarde)  
    - **"Liquiditeit"**: (optioneel) maandelijkse posten  
    - **"Qredits maandlasten"**: privé vaste lasten (velden/tabel)  
    - **"IB VPB"**: belastingparameters

> Belangrijk: Excel-formules blijven **onaangetast**; we vullen alleen invoercellen en tabellen.

---

## 3. Generator

- `gen_finplan.py` voert uit:
  1. Lees `finance_input.yaml`  
  2. Schrijf waarden in `financieel-plan-1.xlsx` volgens `excel_cellmap.yaml`  
  3. Bewaar als `out/financieel-plan-1.filled.xlsx`  
  4. Render `templates/finance_summary.md.j2` → `out/finance_summary.md`  
  5. (Optioneel) Genereer grafieken → `out/charts/`

**Uitvoering:**
```bash
cd finance
python3 gen_finplan.py
````

---

## 4. Templates & Rapportage

* `templates/finance_summary.md.j2` produceert een **leesbare samenvatting**:

  * Investeringsbegroting (tabel + totaal)
  * Financieringsbegroting (eigen inbreng, lening)
  * Exploitatie (maand: omzet, kosten, resultaat)
  * Liquiditeit (optioneel)
  * Privébegroting / belasting (indien aanwezig)
  * (Optioneel) Grafieken insluiten als afbeeldingen

**PDF maken (samenvatting):**

```bash
pandoc out/finance_summary.md -o out/finance_summary.pdf
```

**PDF maken (Excel als bron):**

* Open `out/financieel-plan-1.filled.xlsx` in Excel/LibreOffice → **Opslaan als PDF**.
  Dit behoudt **alle formules en Qredits-indeling**.

---

## 5. Kwaliteitschecks (minimaal)

* **Mapping check:** alle vereiste tabs/cellen bestaan en krijgen waarden
* **Som- en range-checks:** totalen kloppen (Excel herberekent bij openen)
* **Compleetheid:** 3.1–3.4 **altijd** gevuld; 3.5/3.6 indien vereist door Qredits
* **Samenvatting:** bedragen in MD ≈ bedragen in Excel (spot-check)

---

## 6. Navigatie (inhoudsopgave financieel plan)

> Dit is **documentatie-navigatie** (geen 1:1 MD-files), gericht op de pipeline.

* **3.1 Investeringsbegroting** → ingevuld via `finance_input.yaml` → mapping naar *Investering & Financiering*
* **3.2 Financieringsbegroting** → idem (eigen inbreng + leningvelden)
* **3.3 Exploitatiebegroting** → tabellen `omzet`, `kosten` op *Exploitatie*
* **3.4 Liquiditeitsbegroting** → (optioneel) tab/range *Liquiditeit*
* **3.5 Privébegroting / maandlasten** → tab *Qredits maandlasten* (optioneel/verplicht per case)
* **3.6 IB / VPB** → tab *IB VPB* (optioneel/verplicht per case)
* **3.7 Bijlagen & toelichting aannames** → vrije tekst in `finance_summary.md.j2` + referentie naar `out/*.filled.xlsx`

---

## 7. Makefile (optioneel)

```make
gen:
\tpython3 finance/gen_finplan.py

pdf:
\tpandoc finance/out/finance_summary.md -o finance/out/finance_summary.pdf

all: gen pdf
```

---

## 8. Workflow (samengevat)

1. **IDE-AI** vult `finance_input.yaml`
2. **Dev** controleert/actualiseert `excel_cellmap.yaml`
3. Run `python3 gen_finplan.py` → schrijf Excel + MD
4. Maak PDF (samenvatting en/of Excel)
5. Review & indien nodig waarden aanpassen → rerun

---

*NB: als Qredits de Excel verplicht als bron wil, lever dan **altijd** `out/financieel-plan-1.filled.xlsx` mee. De PDF-rapportage is aanvullend voor leesbaarheid.*
