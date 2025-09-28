# Financial Plan — Draft 3 (D3) Specificatie

Status: Draft
Versie: 0.1.0
Laatst bijgewerkt: 2025-09-28

## 1. Doel & Scope

- **Doel.** D3 levert een deterministisch financieel simulatiemodel met een lokaal te bedienen UI (voorkeur: Vue) dat invoerparameters wijzigt, simulaties draait (met seed) en direct een lender-rapport (Markdown + grafieken + CSV/JSON) toont.
- **Scope.** Deze specificatie beschrijft inputs, validatie, simulatiekern, per-model prijsoptimalisatie, curated catalogi (modellen/GPUs), outputartefacten, UI-vereisten en acceptatie/validatiecriteria.
- **Non‑goals.** Geen cloud-SaaS, geen externe API-calls, geen historische telemetrie-inname in D3 (kan later). Geen backwards compatibility met D2-bestandsindeling.

## 2. Terminologie

- **MUST/SHOULD/MAY** volgens RFC‑2119.
- "Seed" = RNG-initialisatie voor gesimuleerde variatie; reproduceerbaar.
- "Curated" = whitelist die expliciet in de invoer is vastgelegd.
- "Public Tap" = per‑token modelverkoop via publiek aanbod.
- "Private Tap" = prepaid GPU-uren met management fee.

## 3. Afbakening en verschillen t.o.v. D2

- **Curated i.p.v. autodetect.** D3 MUST uitsluitend werken met door de gebruiker **gecurateerde** model- en GPU‑lijsten. D2‑logica om zelf de “beste” modellen/GPU’s te kiezen wordt verwijderd.
- **Per‑model prijs vanuit de basis.** D3 MUST de verkoopprijs per model afleiden uit grondkosten (provider USD/hr → EUR/hr, TPS) en beleidsdoelen, niet uit handmatige startprijzen. Geen guestimates.
- **Eén geconsolideerde YAML‑invoer.** D3 MUST starten met één hoofd‑YAML die alle parameters bevat. Later MAY een gelaagd schema worden toegevoegd.
- **Determinisme met seed.** D3 MUST alle stochastiek seed‑gedreven maken en de seed MUST in outputs worden vastgelegd.
- **UI.** D3 MUST een lokale website leveren (voorkeur: Vue) die invoer wijzigt en simulaties start en de rapportage toont.
- **Herbruikbare D2‑artefacten.** D2‑templates en rapportstructuur zijn bruikbaar en SHOULD worden geüpdatet; D2’s multi‑YAML validators en autodiscovery logica worden niet overgenomen.

## 4. Architectuuroverzicht

- **Engine Core (Python).** MUST implementeren: loader, validator, simulatie, per‑model prijsoptimalisatie, outputs, charts.
- **UI (Vue + Vite).** MUST parameters bewerken, seed instellen, “Run simulation” triggeren, voortgang en resultaten tonen, en outputs (MD/CSV/PNG/JSON) previewen en opslaan.
- **Runner.** MUST een lokale runner aanbieden (CLI of kleine HTTP‑bridge) die de Engine start zonder netwerkverkeer. Tijdens dev MAY Vite proxy naar de runner gebruiken.
- **Templates.** MUST een nieuwe D3‑template leveren en bestaande D2‑template uitbreiden (sectie 10).

## 5. Invoer (één geconsolideerde YAML)

De hoofdinvoer is één YAML‑bestand. De engine MUST deze exacte vorm accepteren (namen/typen indicatief; uitbreidbaar zonder BC‑garantie pre‑1.0):

```yaml
meta:
  version: 3
  seed: 12345          # RNG seed (int)
  currency: EUR
  fx:
    eur_usd_rate: 1.08
    fx_buffer_pct: 5.0

catalog:
  models:               # curated allow-list (exact namen)
    - Llama-3-1-8B
    - Qwen2-5-7B
    - Mixtral-8x7B
    - DeepSeek-R1-Distill-Llama-8B
    - Llama-3-3-70B
  gpus:                 # curated GPU‑set
    - A10
    - A100-40GB-PCIe
    - A100-80GB
    - H100-80GB
    - H200-141GB
    - L4
    - L40S
    - RTX-3090
    - RTX-4090
  throughput_tps:       # REQUIRED: tps per (model,gpu)
    Llama-3-1-8B:
      A100-40GB-PCIe: 3200.0
      L4: 1800.0
    Qwen2-5-7B:
      A100-40GB-PCIe: 3000.0
      L4: 1700.0
    # ... voor alle curated modellen en minstens 1 curated GPU

gpu_rentals:           # MUST enforce schema: [gpu, vram_gb, provider, usd_hr]
  - gpu: A100-40GB-PCIe
    vram_gb: 40
    provider: acmecloud
    usd_hr: 1.40
  - gpu: A100-40GB-PCIe
    vram_gb: 40
    provider: othercloud
    usd_hr: 1.26
  # ... meerdere providers per GPU toegestaan

pricing_policy:
  public_tap:
    target_margin_pct: 55.0      # beleidsdoel
    round_increment_eur_per_1k: 0.01
    min_floor_eur_per_1k: 0.05   # MAY ontbreken
    max_cap_eur_per_1k: 3.00     # MAY ontbreken
  private_tap:
    default_markup_over_provider_cost_pct: 50.0

prepaid_policy:
  credits:
    min_topup_eur: 5
    max_topup_eur: 1000
    expiry_months: 12
    non_refundable: true
    auto_refill_default_enabled: false
    auto_refill_cap_eur: null
  private_tap:
    billing_unit_minutes: 15
    management_fee_eur_per_month: 99.0

finance:
  fixed_costs_monthly_eur:
    personal: 3000
    business: 0
  marketing_allocation_pct_of_inflow: 20.0

loan:
  amount_eur: 30000
  term_months: 60
  interest_rate_pct_flat: 9.95

scenarios:              # seed‑gedreven multiplicatoren
  worst:
    budget_multiplier: 0.7
    cvr_multiplier: 0.6
    cac_multiplier: 1.5
  base:
    budget_multiplier: 1.0
    cvr_multiplier: 1.0
    cac_multiplier: 1.0
  best:
    budget_multiplier: 1.3
    cvr_multiplier: 1.2
    cac_multiplier: 0.8
```

### 5.1 Normatieve input‑vereisten

- **GPU rentals schema.** De tabel MUST exact de kolommen `gpu, vram_gb, provider, usd_hr` bevatten. Geen min/median/max velden in de invoer (de engine berekent statistiek zelf).
- **Throughput coverage.** Voor iedere curated `model` MUST er ten minste één `(model,gpu)` entry in `throughput_tps` zijn; ontbrekende entries zijn **ERROR**.
- **Curated lijsten.** Alleen items uit `catalog.models` en `catalog.gpus` worden meegenomen; overige invoer wordt genegeerd of is **WARNING**.
- **Currency & FX.** `eur_usd_rate` en `fx_buffer_pct` MUST aanwezig zijn; engine MUST EUR als rekeneenheid gebruiken voor marges/outputs.

## 6. Globale Validator

De validator MUST:

- **Schema‑checks** uitvoeren op alle topniveaus en types (inclusief numerieke grenzen waar relevant). Bij fouten → **ERROR** en non‑zero exit.
- **Referentiële volledigheid** afdwingen: elke curated `model` heeft ten minste één TPS‑entry en wordt door ten minste één curated GPU ondersteund.
- **Providerprijzen** controleren op `usd_hr > 0` en plausibiliteit (SHOULD `usd_hr < 50`).
- **Financiële parameters** controleren: vaste kosten ≥ 0; marketing % tussen 0 en 100; loan velden aanwezig en valide.
- **Policy‑consistentie**: `pricing_policy.public_tap.target_margin_pct` tussen 0 en 95; `round_increment_eur_per_1k` > 0.
- **Warnings** produceren voor ontbrekende floors/caps en voor modellen die enkel op 1 GPU draaien (risicoconcentratie).
- **Determinisme‑echo**: seed MUST gelogd worden in `run_summary.json` en `run_summary.md`.

## 7. Kosten en prijsbepaling per model (Public Tap)

### 7.1 Grondkosten

Voor elk model m en GPU g:

- `eur_hr(g) = min_provider_over(g, usd_hr) * (1 + fx_buffer_pct/100) / eur_usd_rate` (MUST).
- `tokens_per_hour(m,g) = throughput_tps[m][g] * 3600` (MUST).
- `cost_per_1M_tokens(m,g) = eur_hr(g) / (tokens_per_hour(m,g)/1_000_000)` (MUST).
- De engine MUST de GPU g* kiezen die `cost_per_1M_tokens(m,g)` minimaliseert; ties: SHOULD kiezen op laagste `eur_hr`, daarna alfabetisch `gpu`.

### 7.2 Verkoopprijs per model (recursieve optimalisatie)

- Doel: prijs `sell_per_1k_tokens(m)` vinden die:
  - MUST target marge ≥ `pricing_policy.public_tap.target_margin_pct` tegen de gekozen g*.
  - SHOULD niet lager zijn dan `min_floor_eur_per_1k` (indien gezet); SHOULD niet hoger dan `max_cap_eur_per_1k` (indien gezet).
  - MUST worden afgerond op `round_increment_eur_per_1k`.
- "Recursief" betekent: prijs hangt af van g* die zelf afhangt van `eur_hr`→`usd_hr`→`fx`. Elke wijziging in deze keten triggert herberekening stroomopwaarts (MUST).
- Robustness: de engine SHOULD een **p90‑stress** check doen met een +X% providerprijs drift (configurabel, default 10%) en **WARNING** loggen bij negatieve marge op p90.

Pseudocode:

```python
for m in catalog.models:
  costs = [cost_per_1M_tokens(m,g) for g in catalog.gpus if tps_available(m,g)]
  g_star, cost1M = argmin(costs)
  sell1M = solve_min_price_for_margin(cost1M, target_margin_pct)
  sell1k = round_to_increment(sell1M / 1000, round_increment)
  sell1k = apply_floor_cap(sell1k, floor, cap)
  output[m] = {gpu: g_star, cost_per_1M: cost1M, sell_per_1k: sell1k}
```

### 7.3 Toelatingsregels Public vs Private

- Indien voor model m geen positieve marge haalbaar bij redelijke floors/caps, MUST m naar **Private Tap only** verplaatst worden en op Public niet getoond.
- Modellen met stabiele positieve marge SHOULD op Public Tap blijven.

## 8. Private Tap (GPU‑uren)

- `provider_eur_hr_med(g)` = mediaan over providers van `eur_hr(g)` (MUST).
- `sell_eur_hr(g) = provider_eur_hr_med(g) * (1 + default_markup_over_provider_cost_pct/100)` (MUST).
- Management fee per klant per maand MUST worden opgeteld in voorbeelden/rapportage.
- Output MUST tabel‑ en grafiekvormen voor GPU‑uren (per uur kosten/verkoop/marge) leveren.

## 9. Simulatie & determinisme

- Alle stochastiek (funnel, CAC, budgetvariatie) MUST via een **seeded RNG** lopen. Bij dezelfde invoer + seed MUST alle outputs bit‑voor‑bit gelijk zijn.
- Scenario’s `worst/base/best` MUST door multiplicatoren (en seed) worden aangestuurd; ranges MAY later worden toegevoegd.
- Engine MUST alle gebruikte seeds, parameters en input‑hashes opnemen in `outputs/run_summary.{json,md}`.

## 10. Templates & rapportage

- D3 MUST een nieuwe template `financial_plan_v3.md` aanleveren die minimaal de volgende secties bevat:
  - Executive Summary (prepaid model, targets)
  - Inputs & Catalog (curated modellen/GPUs)
  - Public Tap per‑model economieën (inclusief gekozen GPU per model en afleiding van prijs)
  - Private Tap economics (GPU‑uren + management fee)
  - Scenarios (maand/jaar/60m), Break‑even, Loan schedule
  - VAT & compliance, Safeguards
  - Data Quality legend
- Bestaande D2‑template onderdelen SHOULD worden hergebruikt en uitgebreid met een nieuwe subsectie “Prijsafleiding per model”.
- Tabellen MUST als CSV worden weggeschreven én als Markdown in het rapport worden geïnjecteerd.

## 11. UI (Vue) vereisten

- **Lokale bediening.** De UI MUST lokaal draaien (Vite). “Run simulation” MUST de engine starten via een lokale runner (CLI of HTTP‑bridge) zonder netwerkdependenties.
- **Functionaliteit.**
  - Parameters bewerken (form + YAML‑editor).
  - Seed kiezen/locken.
  - Run starten → voortgang en logs tonen.
  - Resultaten tonen: Markdown‑rapport (preview), grafieken, tabellen, en downloadknoppen.
  - Profielbeheer MAY toegevoegd worden (meerdere inputprofielen opslaan).
- **Robuustheid.** UI MUST validatiefouten prominent tonen en het uitvoeren blokkeren bij **ERROR**.

## 12. Outputs & bestandsstructuur

- MUST de volgende artefacten produceren onder `.003-draft/outputs/` (namen indicatief):
  - `model_price_per_1m_tokens.csv`
  - `public_tap_prices_per_model.csv` (model, gpu*, cost €/1M, sell €/1k, margin %)
  - `private_tap_economics.csv`
  - `public_tap_scenarios.csv`
  - `break_even_targets.csv`
  - `loan_schedule.csv`
  - `vat_set_aside.csv`
  - `financial_plan_v3.md` (gerenderd)
  - `charts/*.png`
  - `run_summary.json` + `run_summary.md` (incl. seed en input‑hashes)

## 13. Acceptatiecriteria (MUST)

- Identieke invoer + seed → identieke outputs (hash‑gelijk) over meerdere runs en machines.
- Validator vangt ontbrekende TPS, ongeldige GPU‑schemas, ongeldige loan‑velden met **ERROR**.
- Voor elk curated model wordt een GPU gekozen en een prijs afgeleid die de target marge haalt bij basisaanname; negatieve marge op p90 levert **WARNING**, niet **ERROR**.
- Public‑ vs Private‑toewijzing is consistent met margeregels.
- UI toont bewerkingsformulieren, seed‑controle, run‑knop, validatiefeedback en alle outputs.

## 14. Verificatieplan (SHOULD)

- **Unit tests** voor validator (schema, referenties, grenzen).
- **Golden tests** voor prijsafleiding per model (kleine fixture met 2 modellen × 2 GPU’s).
- **Determinism test**: twee runs met dezelfde seed geven byte‑gelijke CSV/MD.
- **UI e2e smoke**: invullen → run → preview toont niet‑lege secties en grafieken.

## 15. Operationele en data‑eisen

- Geen netwerk tijdens engine‑run (MUST). Alle data komt uit de enkele YAML en lokale tabellen.
- Alle geldbedragen in EUR (MUST). USD/hr wordt geconverteerd o.b.v. `eur_usd_rate` + `fx_buffer_pct` (MUST).
- GPU‑rental invoer MUST de kolommen `[gpu, vram_gb, provider, usd_hr]` hebben; dit wordt strikt gehandhaafd (geen min/max/percent‑velden in de bron).

## 16. Migratie vanaf D2

- D2 multi‑YAML wordt **niet** ondersteund in D3 (geen BC). Een migratiescript MAY worden geleverd om relevante velden samen te voegen naar de D3 YAML.
- D2 charts/secties die correct zijn SHOULD worden hergebruikt; inconsistenties worden herrekend o.b.v. D3‑regels.

## 17. Open punten / vervolg

- Competitor prijsfloors/price‑parity regels (optioneel) — MAY in een latere iteratie.
- Telemetrie‑input (meet‑TPS, prijsdrift) — MAY later; nu manueel via YAML.
- Gelaagde invoer (overrides per profiel) — MAY in v0.2.

---

### Bijlage A — Indicatieve CSV‑kolommen

- `public_tap_prices_per_model.csv`: `model,gpu,cost_eur_per_1M,sell_eur_per_1k,margin_pct`
- `private_tap_economics.csv`: `gpu,provider_eur_hr_med,markup_pct,sell_eur_hr,margin_eur_hr`

### Bijlage B — Safeguards (overgenomen en aangescherpt)

- Prepaid only; non‑refundable credits (MUST). Geen service zonder prepaid balans (MUST).
- GPU’s alleen huren na betaling (MUST) → geen idle exposure.
- Marketing als % van inflow reserveren (SHOULD; default 20%).
- VAT correct scheiden en rapporteren (MUST; NL/EU regels).
