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

## 3. Afbakening en verschillen t.o.v. D2

- **Curated i.p.v. autodetect.** D3 MUST uitsluitend werken met door de gebruiker **gecurateerde** model- en GPU‑lijsten. D2‑logica om zelf de “beste” modellen/GPU’s te kiezen wordt verwijderd.
- **Per‑model prijs vanuit de basis.** D3 MUST de verkoopprijs per model afleiden uit grondkosten (provider USD/hr → EUR/hr, TPS) en beleidsdoelen, niet uit handmatige startprijzen. Geen guestimates.
- **Gesplitste invoer (Public + Operator).** D3 MUST werken met `public_data.yaml` (exogeen) en twee operatorbestanden: `operator_public.yaml` (Public pijplijn) en `operator_private.yaml` (Private pijplijn). Geen single‑file modus.
- **Determinisme met seed.** D3 MUST alle stochastiek seed‑gedreven maken en de seed MUST in outputs worden vastgelegd.
- **UI.** D3 MUST een lokale website LEVEREN (voorkeur: Vue) die invoer wijzigt en simulaties start en de rapportage toont.
- **Herbruikbare D2‑artefacten.** D2‑templates en rapportstructuur zijn bruikbaar en SHOULD worden geüpdatet; D2’s multi‑YAML validators en autodiscovery logica worden niet overgenomen.

## 4. Architectuuroverzicht

- **Engine Core (Python).** MUST implementeren: loader, validator, simulatie, per‑model prijsoptimalisatie, outputs, charts.
- **UI (Vue + Vite).** MUST parameters bewerken, seed instellen, “Run simulation” triggeren, voortgang en resultaten tonen, en outputs (MD/CSV/PNG/JSON) previewen en opslaan.
- **Runner.** MUST een lokale runner aanbieden (CLI of kleine HTTP‑bridge) die de Engine start zonder netwerkverkeer. Tijdens dev MAY Vite proxy naar de runner gebruiken.
- **Templates.** MUST een nieuwe D3‑template leveren en bestaande D2‑template uitbreiden (sectie 10).

## 5. Invoer (overzicht)

De invoerregels, formats en schema’s zijn verplaatst naar dedictated docs:

- `10_inputs.md` — formats (YAML/CSV), directory layout, merge/precedence, determinisme, validatorregels.
- `11_operator_inputs.md` — volledige shapes/voorbeelden voor operator‑config (YAML/CSV).
- `12_public data.md` — volledige shapes/voorbeelden voor public data (YAML/CSV).

Kernpunten (samenvatting):
- Bundels: `public_data.yaml` (exogeen), `operator_public.yaml` (Public), `operator_private.yaml` (Private).
- Alleen **YAML en CSV** zijn toegestaan; indien beide aanwezig voor dezelfde dataset, **CSV wint** (shadowing WARNING).
 - Seeds per pijplijn: `operator_public.yaml: meta.seed` en `operator_private.yaml: meta.seed`.

### 5.1 Normatieve input‑vereisten

- **GPU rentals schema (public).** `gpu_rentals` MUST exact de kolommen `gpu, vram_gb, provider, usd_hr` bevatten. Geen min/median/max of percentvelden in de bron.
- **Throughput coverage (public).** Voor iedere curated `catalog.models` MUST er ≥1 `(model,gpu)` entry in `throughput_tps` bestaan; ontbrekende entries zijn **ERROR**.
- **Curated lijsten (operator).** Alleen items uit `catalog.models` en `catalog.gpus` worden meegenomen; onbekende modellen/GPUs t.o.v. `throughput_tps` of `gpu_rentals` geven **WARNING** of **ERROR** afhankelijk van impact.
- **Currency & FX.** `fx.eur_usd_rate` komt uit `public_data.yaml`; `fx_buffer_pct` uit `operator_public.yaml` en/of `operator_private.yaml` (MUST consistent zijn indien in beide aanwezig). Engine MUST EUR als rekeneenheid gebruiken.

## 6. Globale Validator

De validator MUST:

- **Schema‑checks** uitvoeren op alle topniveaus en types (inclusief numerieke grenzen waar relevant). Bij fouten → **ERROR** en non‑zero exit.
- **Referentiële volledigheid** afdwingen: elke curated `model` heeft ten minste één TPS‑entry en wordt door ten minste één curated GPU ondersteund.
- **Providerprijzen** controleren op `usd_hr > 0` en plausibiliteit (SHOULD `usd_hr < 50`).
- **Financiële parameters** controleren: vaste kosten ≥ 0; marketing % tussen 0 en 100; loan velden aanwezig en valide.
- **Policy‑consistentie**: `pricing_policy.public_tap.target_margin_pct` tussen 0 en 95; `round_increment_eur_per_1k` > 0.
- **Warnings** produceren voor ontbrekende floors/caps en voor modellen die enkel op 1 GPU draaien (risicoconcentratie).
- **Determinisme‑echo**: seed MUST gelogd worden in `run_summary.json` en `run_summary.md`.
- **CSV‑schema’s.** Indien CSV is geleverd, MUST kolomnamen exact matchen aan `10_inputs.md §7`; MUST unieke `(model,gpu)` in `throughput_tps.csv`; MUST numerieke domeinen afdwingen; anders **ERROR**.
- **Precedentie‑log.** Bij shadowing YAML→CSV MUST een duidelijke **WARNING** gelogd worden met bestandsnamen (zie `10_inputs.md §5`).

## 7. Kosten en prijsbepaling per model (Public Tap)

### 7.1 Grondkosten

Voor elk model m en GPU g:

- `eur_hr(g) = min_provider_over(g, usd_hr) * (1 + fx_buffer_pct/100) / eur_usd_rate` (MUST).
- `tokens_per_hour(m,g) = throughput_tps[m][g] * 3600` (MUST).
- `cost_per_1M_tokens(m,g) = eur_hr(g) / (tokens_per_hour(m,g)/1_000_000)` (MUST).
- De engine MUST de GPU g* kiezen die `cost_per_1M_tokens(m,g)` minimaliseert; ties: SHOULD kiezen op laagste `eur_hr`, daarna alfabetisch `gpu`.

#### 7.1.1 Gekozen provider logging (Public)

- Omdat `eur_hr(g)` is afgeleid van providerprijzen, MUST de engine per model de **provider** loggen die de minimale `eur_hr` (en dus `cost_per_1M`) levert voor de gekozen g*.
- Output MUST bevatten: `public_vendor_choice.csv` met kolommen `model,gpu,provider,usd_hr,eur_hr_effective,cost_eur_per_1M`.

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

### 7.4 Accounting & Credits (ADR‑aligned)

- **Meet-eenheid.** De Public Tap MUST afrekenen in gecombineerde tokens (input + output). De rapportage MAY aggregeren op 1M tokens voor pricing‑overzichten; de engine rekent intern in €/1k tokens en converteert consistent.
- **Prepaid credits.** Credits MUST **non‑refundable** zijn, met **geldigheid 12 maanden**. MUST duidelijk in ToS/voorwaarden en in het rapport vermeld staan.
- **Saldo‑zichtbaarheid.** De UI/UX MUST het actuele creditsaldo zichtbaar maken (dashboard of API‑simulatie in de lokale UI). Bij saldo **0** MUST serviceconsumptie stoppen (conceptueel; in simulatie geen negatieve saldo’s toestaan).
- **Credit packs.** De UI SHOULD standaard packs tonen van **€50/€200/€500** en de bijbehorende geschatte tokens berekenen op basis van de actuele blended €/1k. Packs MAY aangepast worden via config.

## 8. Private Tap (GPU‑uren)

- `provider_eur_hr_med(g)` = mediaan over providers van `eur_hr(g)` (MUST).
- `sell_eur_hr(g) = provider_eur_hr_med(g) * (1 + default_markup_over_provider_cost_pct/100)` (MUST).
- Management fee per klant per maand MUST worden opgeteld in voorbeelden/rapportage.
- Output MUST tabel‑ en grafiekvormen voor GPU‑uren (per uur kosten/verkoop/marge) leveren.

### 8.1 Aanvullende aanbod‑opties (ADR‑aligned)

- **GPU‑schaalopties.** Offering SHOULD ondersteunen: 1×, 2×, 4×, 8× GPU‑schaal (contractueel/operationeel; simulatie mag dit parametriseren).
- **Prepaid uurblokken.** Verkoop en planning SHOULD in prepaid **GPU‑uurblokken** plaatsvinden; engine berekent marges conform `sell_eur_hr` en blokgrootte.
- **OpenAI‑compatible gateway (optioneel).** Offering MAY een compatibele gateway aanbieden voor gemakkelijke integratie; dit is buiten de scope van de financiële berekening.
- **Basistarief (ops/SLA).** Naast `management_fee_eur_per_month` MAY een **basisfee per klant** of per GPU‑klasse worden geconfigureerd. Indien gezet, MUST deze in marge‑/voorbeeldtabellen worden opgenomen. Een uitbreidingspad is:
  - `prepaid_policy.private_tap.base_fee_eur_per_month` (algemeen), of
  - `prepaid_policy.private_tap.base_fee_by_gpu_class: { A100: 250, H100: 400, ... }`.

### 8.2 Vendor‑selectie (Private) en aanbevelingen

- **Prijsstelling.** `sell_eur_hr(g)` blijft gebaseerd op de **mediaan** providerprijs (MUST) voor conservatieve quotering.
- **Aanbevolen leverancier.** Voor operationele inkoop MUST de engine een **aanbevolen provider per GPU‑klasse** berekenen:
  - Default: provider met minimale `eur_hr`.
  - Optioneel: **vendor‑score** met gewichten (operator) via `operator_private.yaml → prepaid_policy.private_tap.vendor_weights` (MAY): `{ cost: 0.7, availability: 0.2, reputation: 0.1 }` (gewichten som = 1.0). Beschikbaarheid/reputatie kunnen voorlopig op 1.0 worden gezet of uit een later CSV komen.
- **Output.** `private_vendor_recommendation.csv` met `gpu,provider,usd_hr,eur_hr_effective,score`.

## 9. Simulatie & determinisme

- Alle stochastiek (funnel, CAC, budgetvariatie) MUST via een **seeded RNG** lopen. Bij dezelfde invoer + seed MUST alle outputs bit‑voor‑bit gelijk zijn.
- Scenario’s `worst/base/best` MUST door multiplicatoren (en seed) worden aangestuurd; ranges MAY later worden toegevoegd.
- Engine MUST alle gebruikte seeds, parameters en input‑hashes opnemen in `outputs/run_summary.{json,md}`.

### 9.1 Simulatiepijplijnen (Public vs Private) en consolidatie

- **Twee pijplijnen.** De engine MUST twee **gescheiden** simulaties uitvoeren:
  - `PublicTapSim` — per‑model grondkosten → prijsafleiding → marges → scenario’s (budget/cvr/cac) en creditsconsumptie.
  - `PrivateTapSim` — providerprijzen → medianen → `sell_eur_hr` → contracteconomics (uurblokken, base/management fees) → scenario’s.
- **Consolidatie.** Na beide runs MUST een **consolidatie‑fase** draaien die:
  - KPI’s en cashflows combineert.
  - **Beslisondersteuning** synthese levert: aanbevolen Public modelset (in/out), aanbevolen GPU‑leverancier(s) per klasse, verwachte marges en gevoeligheden (p90 drift).
  - Resultaat schrijft naar `consolidated_summary.{md,json}` + `consolidated_kpis.csv`.
- **Determinisme.** Beide pijplijnen en de consolidatie MUST deterministisch zijn gegeven identieke inputs + seed.

## 10. Templates & rapportage

- D3 MUST een nieuwe template `financial_plan_v3.md` aanleveren die minimaal de volgende secties bevat:
  - Executive Summary (prepaid model, targets)
  - Inputs & Catalog (curated modellen/GPUs)
  - Public Tap per‑model economieën (inclusief gekozen GPU per model en afleiding van prijs)
  - Private Tap economics (GPU‑uren + management fee)
  - Scenarios (maand/jaar/60m), Break‑even, Loan schedule
  - VAT & compliance, Safeguards
  - Data Quality legend
- De template MUST een expliciet **ToS/credits‑snippet** opnemen: credits zijn **non‑refundable**, **12 maanden geldig**, en service **stopt bij saldo 0** (halt‑at‑zero). Dit moet zichtbaar zijn in de Public Tap sectie.
- Bestaande D2‑template onderdelen SHOULD worden hergebruikt en uitgebreid met een nieuwe subsectie “Prijsafleiding per model”.
- Tabellen MUST als CSV worden weggeschreven én als Markdown in het rapport worden geïnjecteerd.

## 11. UI (Vue) vereisten

- **Lokale bediening.** De UI MUST lokaal draaien (Vite). “Run simulation” MUST de engine starten via een lokale runner (CLI of HTTP‑bridge) zonder netwerkdependenties.
- **Functionaliteit.**
  - Parameters bewerken (form + YAML‑editor).
  - Beide invoerbestanden bewerken: `public_data.yaml` (readonly secties in UI, maar wel zichtbaar) en `operator_general.yaml` + `operator_public.yaml` + `operator_private.yaml` (volledig bewerkbaar). UI MUST de **effectieve merge** en validatiestatus tonen.
  - Pijplijnselectie: knoppen/toggles voor `Run Public`, `Run Private`, `Run Both (Consolidated)` (MUST aanwezig).
  - Seed kiezen/locken.
  - Run starten → voortgang en logs tonen.
  - Resultaten tonen: Markdown‑rapport (preview), grafieken, tabellen, en downloadknoppen.
  - Profielbeheer MAY toegevoegd worden (meerdere inputprofielen opslaan).
- **Robuustheid.** UI MUST validatiefouten prominent tonen en het uitvoeren blokkeren bij **ERROR**.

### 11.1 Public‑Tap UX‑uitbreidingen (ADR‑aligned)

- **Saldo‑weergave.** De UI MUST een zichtbare saldo‑indicator tonen (gesimuleerd o.b.v. gekozen packs en blended prijs) en mag consumptie‑simulatie in scenario’s tonen.
- **Halt‑at‑zero.** De UI SHOULD tonen dat service stopt bij saldo **0**; simulaties MUST niet onder 0 gaan.
- **Credit packs.** Presets **€50/€200/€500** SHOULD beschikbaar zijn met afgeleide tokens; aangepaste packs MAY worden ingevoerd.
- **ToS/beleid.** UI SHOULD een duidelijke verwijzing/snippet tonen: non‑refundable credits, 12 maanden geldigheid.

## 12. Outputs & bestandsstructuur

- MUST de volgende artefacten produceren onder `.003-draft/outputs/` (namen indicatief):
  - `model_price_per_1m_tokens.csv`
  - `public_tap_prices_per_model.csv` (model, gpu*, cost €/1M, sell €/1k, margin %)
  - `public_vendor_choice.csv` (model, gpu, provider, usd_hr, eur_hr_effective, cost €/1M)
  - `private_tap_economics.csv`
  - `private_vendor_recommendation.csv` (gpu, provider, usd_hr, eur_hr_effective, score)
  - `public_tap_scenarios.csv`
  - `break_even_targets.csv`
  - `loan_schedule.csv`
  - `vat_set_aside.csv`
  - `financial_plan_v3.md` (gerenderd)
  - `charts/*.png`
  - `run_summary.json` + `run_summary.md` (incl. seed en input‑hashes)
  - `consolidated_kpis.csv`, `consolidated_summary.{md,json}`

## 13. Acceptatiecriteria (MUST)

- Identieke invoer + seed → identieke outputs (hash‑gelijk) over meerdere runs en machines.
- Validator vangt ontbrekende TPS, ongeldige GPU‑schemas, ongeldige loan‑velden met **ERROR**.
- Voor elk curated model wordt een GPU gekozen en een prijs afgeleid die de target marge haalt bij basisaanname; negatieve marge op p90 levert **WARNING**, niet **ERROR**.
- Public‑ vs Private‑toewijzing is consistent met margeregels.
- UI toont bewerkingsformulieren, seed‑controle, run‑knop, validatiefeedback en alle outputs.

### 13.1 Public/Private Tap‑specifiek

- **Credits‑beleid.** Rapport en UI MUST expliciet tonen: non‑refundable credits, 12 maanden geldigheid.
- **Saldo‑zichtbaarheid.** Een saldo‑indicator MUST aanwezig zijn; scenario’s MUST geen negatief saldo produceren.
- **Credit‑packs.** Presets worden correct omgerekend naar tokens o.b.v. de blended €/1k; afwijking ≤ 0.5% na afronding (SHOULD).
- **Private‑opts.** Indien `base_fee` of `base_fee_by_gpu_class` is geconfigureerd, MUST deze fees in voorbeeldtabellen en marges worden verwerkt.
- **Split‑inputs handhaving.** Keys die niet zijn toegestaan in een van beide YAML’s (public vs operator) MUST een **ERROR** geven en de UI MUST dit duidelijk tonen met een fix‑suggestie.
- **CSV‑ondersteuning.** Indien CSV wordt gebruikt voor `throughput_tps` en/of `gpu_rentals`, MUST schema/precedentie‑regels uit `10_inputs.md §§5–7` worden afgedwongen; bij conflicten MUST CSV winnen met duidelijke WARNING.
- **Twee pijplijnen.** `Run Public`, `Run Private`, en `Run Both` MUST deterministisch dezelfde resultaten geven across runs; consolidatie‑artefacten MUST aanwezig en consistent zijn met de twee deelruns.

## 14. Verificatieplan (SHOULD)

- **Unit tests** voor validator (schema, referenties, grenzen).
- **Golden tests** voor prijsafleiding per model (kleine fixture met 2 modellen × 2 GPU’s).
- **Determinism test**: twee runs met dezelfde seed geven byte‑gelijke CSV/MD.
- **UI e2e smoke**: invullen → run → preview toont niet‑lege secties en grafieken.

## 15. Operationele en data‑eisen

- Geen netwerk tijdens engine‑run (MUST). Alle data komt uit public YAML/CSV en operator YAML/CSV.
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
