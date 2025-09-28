# 21 — Simulatie (Algemeen: tijdsbasis, RNG, allocaties)

Status: Draft
Version: 0.1.0

## 1. Scope

- Dit document is GEEN pijplijn; het definieert algemene sim‑regels die gelden voor `PublicTapSim` en `PrivateTapSim`.
- Cross‑refs:
  - Runplan: `inputs/simulation.yaml` (zie `15_simulation_constants.md`).
  - Variabelen/treatments: `inputs/variables/*.csv` (zie `16_simulation_variables.md` en `12_oprator_variables.md`).
  - Operator‑constanten: `inputs/operator/*.yaml` (zie `11_operator_constants.md`).
  - Facts: `inputs/facts/*` (zie `19_facts.md`).

## 2. Tijdsbasis & units (MUST)

- Tijdseenheid van rapportage: **maand** (MUST). Sommige grootheden (GPU‑uren) rekenen intern in **uur**.
- Valuta: **EUR** (MUST). USD → EUR conversie met facts `market_env.yaml → finance.eur_usd_fx_rate.value` en operator `meta.fx_buffer_pct` per pijplijn.
- Tokens: tel als gecombineerde tokens (input + output) voor Public Tap.

## 3. Variabelen: treatments en structuur (MUST)

- `fixed`: gebruik `default` exact.
- `low_to_high`: rooster `min..max` met `step` (inclusief eindpunten); cartesisch product binnen dezelfde scope.
- `random`: uniform sample in `[min,max]` met step‑kwantisatie (offset `min`). Discrete variabelen: gelijke kans over `notes.values`.
- Scopes: `general`, `public_tap`, `private_tap` (geen cross‑scope overschrijving).
- Zie `16_simulation_variables.md` voor volledige regels.

## 4. Run‑expansie (grid → replicates → MC) (MUST)

1) Bouw per scope een basisset uit `fixed` en een grid over `low_to_high` (cartesisch product).
2) Voor elke grid‑combinatie voer `run.random_runs_per_simulation` replicates uit; trek alle `random` variabelen opnieuw.
3) Binnen elke replicate voer `stochastic.simulations_per_run` Monte Carlo iteraties uit (funnel, CAC, vraag, etc.).

## 5. RNG & determinisme (MUST)

- Seed‑resolutie: `stochastic.random_seed` → `run.random_seed` → `operator/<tap>.yaml: meta.seed` → anders **ERROR** (zie `15_simulation_constants.md §4`).
- RNG‑streams: `RNG_PARAMS(scope)` voor variabelen en `RNG_SIM(scope)` voor MC; substreams via stabiele hashing met indices (`grid_index`, `replicate_index`, `mc_index`).
- Identieke inputs + seed → byte‑gelijke outputs en identieke `variable_draws.csv` transcript.

## 6. Allocaties & renormalisatie (MUST)

- Allocaties moeten per set sommen naar `1.0` (bijv. `acquisition.channel_allocation.*`).
- Engine MAY renormaliseren met **WARNING**; indien `run.fail_on_warning: true`, escaleren naar **ERROR**.
- Vendor‑gewichten volgen dezelfde regels (som=1.0).

## 7. Rounding, clipping & guardrails (MUST)

- Valuta wordt afgerond op 2 decimalen voor rapportage; intern met hogere precisie.
- Percentages binnen 0..100; fracties 0..1.
- Non‑negativiteit: prijzen, kosten, budgetten, credit‑saldo’s ≥ 0.
- Floors/caps: Public prijsafleiding respecteert `min_floor_eur_per_1k` en `max_cap_eur_per_1k` indien gezet.

## 8. FX & conversies (MUST)

- Gebruik `inputs/facts/market_env.yaml → finance.eur_usd_fx_rate.value`.
- Pijplijnspecifieke buffer `meta.fx_buffer_pct` uit `inputs/operator/public_tap.yaml` of `inputs/operator/private_tap.yaml`.
- Geen netwerkgebruik; alle waarden komen uit lokale files.

## 9. Artefacten & logging (SHOULD)

- `variable_draws.csv`: `scope,variable_id,path,grid_index,replicate_index,draw_value`.
- `run_summary.{json,md}`: seeds, input‑hashes, overlay‑beslissingen, shadowing‑warnings, gridgrootte, replicates, MC‑iteraties.

## 10. Fouten & waarschuwingen

- Schema/keys buiten scope → **ERROR**.
- Allocatie som ≠ 1.0 → **WARNING** (renormalisatie) of **ERROR** bij `fail_on_warning`.
- Seed ontbreekt na resolutie → **ERROR**.

## 11. Zie ook

- `20_simulations.md` (overzicht/runner), `22_sim_private_tap.md`, `23_sim_public_tap.md`.
- `15_simulation_constants.md`, `16_simulation_variables.md`.
- `10_inputs.md`, `11_operator_constants.md`, `12_oprator_variables.md`, `19_facts.md`.
