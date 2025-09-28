# 4N — [SECTION_NAME] Testing Template

Status: Draft
Version: 0.1.0

Dit document is een template voor extra testsecties in de 4x‑reeks (40–49). Gebruik dit om specifieke subsystemen diepgaand te testen (bijv. RNG, Public Pricing, Private Clients, Consolidatie, CLI Contract, Performance, Fixtures).

## 1. Scope

- Beschrijf hier precies wat je test (submodule, pijplijn, service, of cross‑cutting concern).
- Link terug naar de relevante specs (bijv. `31_engine.md`, `22_sim_private_tap.md`).

## 2. Doelstellingen (MUST)

- Benoem meetbare doelen (acceptatiecriteria) als MUST/SHOULD/MAY.
- Voorbeeld:
  - MUST: deterministische uitkomsten gegeven seed S en inputs I.
  - MUST: pricing respecteert floors/caps/rounding uit `public_tap.yaml`.
  - SHOULD: performance binnen N seconden voor fixture X.

## 3. Testmatrix

- Unit: functies/methoden met randgevallen en foutpaden.
- Contract: interfaces (CLI‑args, JSONL velden, CSV headers) en foutcodes.
- Golden: byte‑gelijke outputs en SHA‑256 hashes.
- Acceptance: formele checks (monotone groei, marge, capaciteit) indien toepasselijk.
- Integration/E2E: end‑to‑end pad met minimale fixture.

## 4. Fixtures

- Directory: `/.003-draft/tests/fixtures/[naam]/`
- Minimale samenstelling:
  - `inputs/` (simulation.yaml, operator/, variables/, facts/)
  - `expected_outputs/` (CSV/MD/JSON + `SHA256SUMS`)
- Notities: hoe random variabelen en seeds gezet worden om determinisme te waarborgen.

## 5. Assertions & Checks

- CSV‑schema’s: exacte kolommen, types, grenzen.
- Numeriek: tolerantiebeleid (indien nodig) of vaste afronding in writers.
- Logging: JSONL regels bevatten minimaal `{ ts, level, event }` en relevante velden.
- Acceptatie: expliciete checks met duidelijke foutboodschappen.

## 6. Commando’s

- Engine:
  - `make -C .003-draft/engine install`
  - `python -m d3_engine.cli --inputs .003-draft/tests/fixtures/[naam]/inputs --out .003-draft/tests/tmp/out --pipelines public,private --seed 424242`
- UI (indien relevant):
  - `pnpm -F orchyra-d3-sim-frontend dev` of test via vitest/playwright.

## 7. CI & Artefacten

- Voeg de tests toe aan de CI‑jobs (pytest/vitest/playwright).
- Upload output‑artefacten bij failures om regressies te analyseren.

## 8. Referenties

- `40_testing.md` — overkoepelend testregime
- `41_engine.md` — engine testing
- `49_ui.md` — UI testing
- Relevante pijplijnspecs (`22/23`) en architectuur (`31/32`).
