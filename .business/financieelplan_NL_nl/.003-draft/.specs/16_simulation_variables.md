# 16 — Simulation Variables (Treatments & RNG)

Status: Draft
Version: 0.1.0

## 1. Scope

- Definieert hoe variabelen uit `inputs/variables/*.csv` worden toegepast tijdens simulaties.
- De kolomschema’s en allowed paths staan in `12_oprator_variables.md`.

## 2. Behandelings‑types (MUST)

- **fixed**: gebruik `default` exact.
- **low_to_high** (grid): construeer roosterwaarden `min, min+step, …, max` (inclusief eindpunten). De engine vormt de cartesische combinatie over alle `low_to_high` variabelen binnen dezelfde scope.
- **random**: sample uniform in `[min, max]` en kwantiseer naar step‑tik (offset `min`). Voor `discrete` variabelen: trek met gelijke kans uit `notes.values`.

## 3. Run‑expansie en volgorde (MUST)

1) **Bouw basisset** per scope uit alle `fixed` waarden en het grid van `low_to_high` (cartesisch product).
2) **Random replicates**: voor elke basisset voer `run.random_runs_per_simulation` replicates uit. In iedere replicate worden alle `random` variabelen opnieuw getrokken volgens §2.
3) **Monte Carlo**: binnen elke replicate voer `stochastic.simulations_per_run` iteraties uit voor funnel/uitkomststochastiek (CVR/CAC/etc.).

Opmerking (replicates): Replicates gelden uitsluitend voor variabelen met `treatment=random`. Als er géén `random` variabelen zijn, zal een waarde >1 voor `run.random_runs_per_simulation` identieke resultaten produceren; aanbevolen is dan `run.random_runs_per_simulation=1` voor performance.

De drie niveaus zijn genest: `grid combo` → `random replicate` → `MC iteraties`.

## 4. Seed‑architectuur & determinisme (MUST)

Seed resolutie (zie ook `15_simulation_constants.md §4`):

1) `stochastic.random_seed` (indien gezet)
2) `run.random_seed` (indien gezet)
3) Pijplijnseed `inputs/operator/<tap>.yaml: meta.seed`
4) Geen seed → **ERROR**

RNG‑streams (aanbevolen indeling):

- `RNG_PARAMS(scope)` voor `random` variabelen in die scope. Substream per `variable_id` via een stabiele hash: `H(seed, "params", scope, variable_id, grid_index, replicate_index)`.
- `RNG_SIM(scope)` voor Monte Carlo binnen de replicate: `H(seed, "sim", scope, grid_index, replicate_index, mc_index)`.

Determinisme‑eis: identieke inputs + seed → byte‑gelijke outputs en identieke `variable_draws.csv` transcript.

## 5. Constraints (MUST)

- `numeric` grenzen en step‑rooster afdwingen (zie `12_oprator_variables.md`).
- Allocaties (bijv. `acquisition.channel_allocation.*`) MUST sommen naar `1.0` per scope. De engine MAY renormaliseren met **WARNING** of, indien `run.fail_on_warning: true`, escaleren naar **ERROR**.
- `vendor_weights.*` MUST sommen naar `1.0` (zelfde beleid als allocaties).

## 6. Scope‑isolatie (MUST)

- Variabelen zijn geïsoleerd per scope: `general`, `public_tap`, `private_tap`. Er is geen cross‑scope overschrijving.
- Overlap van `variable_id` namen tussen scopes is toegestaan maar niet aangeraden (SHOULD vermijden). Log als `variable_id_collision` met scope.

## 7. Logging & Artefacten (SHOULD)

- `variable_draws.csv` per run met kolommen: `scope,variable_id,path,grid_index,replicate_index,draw_value`.
- `run_summary.{json,md}` MUST opnemen: grid‑grootte, aantal replicates, aantal MC‑iteraties, en seeds per scope.

## 8. Fouten & waarschuwingen

- Onbekend `path` of niet‑toegestane root → **ERROR**.
- `default` buiten [min,max] of niet op step‑rooster → **ERROR**.
- `discrete` zonder `values` in `notes` → **ERROR**.
- Allocaties/gewichten som ≠ 1.0 → **WARNING** met renormalisatie, of **ERROR** als `fail_on_warning`.

