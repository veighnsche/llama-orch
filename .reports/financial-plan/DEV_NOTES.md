# DEV_NOTES — Financieel Plan (v1)

Status: locked v1 (assumptions confirmed by user)
Datum: 2025-09-26
Taal: Nederlands

Doel

- Dit document legt alle aannames, bronnen en scenario-parameters vast voor het deterministisch genereren van een Qredits-geschikt financieel plan (Markdown + CSV).
- We gebruiken GEEN gegevens uit de verboden map `/.business/financieelplan_NL_nl/` om self‑referential bias te voorkomen.

Bronnen (repo)

- Product & architectuur: `README.md` (status, SSE, metrics)
- Profiel Home Lab: `.docs/HOME_PROFILE.md`, `.docs/HOME_PROFILE_TARGET.md`
- Normatieve SPEC: `.specs/00_llama-orch.md`
- Metrics-contract: `.specs/metrics/otel-prom.md`
- Consumer/API gids: `CONSUMER_CAPABILITIES.md`
- Ondernemingsplan (draft, buiten verboden map): `/.business/ondernemersplan_NL_nl/.002-draft/` (o.a. `index.md`, `front-page.md`, `USP.md`, `services.md`, `naming.md`, `target-audiences.md`, `competitors.md`, `ADR-XXX-public-tap-pricing.md`, `ADR-XXX-public-tap-prepaid-credits.md`)
- Proof-bundle (voor later bundelen van artefacten): `libs/proof-bundle/` en `.specs/00_proof-bundle.md` (geheugen)

Aannames (geconsolideerd)

- Bedrijf/merk: Orchyra; tagline: “Private LLM Hosting in the Netherlands”.
- Startdatum: 2025-10-01; horizon: 36 maanden; valuta: EUR; rapportage: maandelijks.
- Administratie: factuur/opbouw (accrual); BTW-aangifte: kwartaal.
- Offerings: OSS toolkit (gratis), Public Tap (prepaid credits, niet-restitueerbaar, 12 mnd), Private Tap (dedicated GPU-uur + basisfee), optioneel Custom Toolkit Development.
- Public Tap pricing DRAFT: €1,20 per 1M tokens (input+output), packs: €50/~41M; €200/~166M; €500/~416M.
- Private Tap voorbeelden: A100 80GB €1,80/GPU-uur + €250/m; H100 80GB €3,80/GPU-uur + €400/m.
- Volumes (base): m12 → 150 Public Tap betaalde accounts; 20 Private nodes; seasonality jul/aug −15%.
- COGS/GM: Public Tap GM 55% (COGS 45%); Private cloud GPU-kosten indicatief €0,85–€1,50/GPU-uur @ 50% util;
  on‑prem stroom ~€151/m bij 600W 24/7 @ €0,35/kWh.
- PSP: 1,8% + €0,30; payouts 2 dagen; factuur Private 14 dagen.
- OPEX/m: founder €3.500 bruto; contractor 0,5 FTE €3.500 pro rata; wg‑lasten 25%; 8% vakantiegeld; marketing €500; SaaS €150; accounting €125; verzekeringen €75; huur €0; overig €100.
- CAPEX: Workstation €2.500, 36 mnd lineair (start 2025-10-01); upgrade 2026-07-01 €1.500, 36 mnd.
- Financiering: eigen inbreng €10.000; Qredits €35.000, 9,0%, 60 mnd, 6 mnd grace, annuïtair, 1% afsluitkosten.
- Werkkapitaal: DSO 2d PSP/14d factuur; DPO 7d infra/30d SaaS; deferred revenue: 20% jaarvooruit (Public Tap); BTW: kwartaal.
- Belastingen: eenmanszaak (IB) v1; BTW 21%; geen KOR; IB-aftrekken toepasbaar; inflatie 2,5%/jr.
- Scenario’s: base/best/worst + stress (−30% omzet, +20% OPEX, +30d DSO, +200 bps rente, +50% churn).

Determinisme & bewijs

- Generatie gebruikt een vaste dataset (`dataset.v1.json`) en vaste formules/formattering (geen randomness).
- Uitvoer (Markdown + CSV) wordt onder `/.docs/financial-plan/outputs/` geplaatst.
- Optioneel kan later een proof‑bundle worden aangemaakt (NDJSON/Markdown) via een extern script dat `LLORCH_RUN_ID` en `LLORCH_PROOF_DIR` respecteert.

Beperkingen & keuzes

- We mappen (voor nu) naar een **neutrale** key‑set. De daadwerkelijke `TEMPLATE_KEYS` in `/.business/financieelplan_NL_nl/` zijn niet gelezen (bewuste beperking).
- We leveren een voorbeeld‑mappingtabel ter referentie. Als je de echte keys elders aanlevert, maken we een 1‑op‑1 mapping file.

Outputplan (v1)

- Markdown secties: `60_pricing.md`, `70_breakeven.md` (simplified), `80_unit_economics.md` (baseline), `90_working_capital.md`.
- CSV: `dataset.csv` (kernparameters key→value), deterministic.

Disclaimers

- Indicatief; geen fiscale/fundingsadvisering; cijfers onder voorbehoud; prijspeil 2025; excl. BTW.
