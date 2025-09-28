# 41 — Engine Testing (Python)

Status: Draft
Version: 0.1.0

## 1. Scope

- Teststrategie voor de Python‑engine (`/.003-draft/engine/src/d3_engine`).
- Dekt unit, contract, golden/determinism, acceptance, performance en concurrency.

## 2. Principes (MUST)

- Deterministisch: identieke inputs + seeds → identieke outputs (byte‑gelijk).
- Pure berekeningen testen los van I/O (pipelines/services als pure functies).
- Duidelijke foutcodes: `0=OK`, `2=VALIDATION_ERROR`, `3=RUNTIME_ERROR`.

## 3. Unit Test Matrix (MUST)

- `core/loader.py`: YAML/CSV load, overlay (CSV > YAML), shadowing WARNING.
- `core/validator.py`: kolomschema’s, paden (allowed roots), referenties (curated lijsten), targets.
- `core/rng.py`: seed‑hiërarchie, substreams, step‑kwantisatie, stabiliteit bij parallelisatie.
- `services/fx_service.py`: EUR/USD + buffer randen (0, klein, groot).
- `services/acquisition.py`: CAC=0/negatief → 0 klanten.
- `services/tps_heuristics.py`: monotone scaling over VRAM/klasse, documented heuristiek.
- `services/autoscaling.py`: `instances_needed` bij 0/∞ tps en extreme util.
- `pipelines/public/*`: cost €/1M, pricing (floors, caps, rounding), demand (budget0, groei, churn, tokens/conv), capacity.
- `pipelines/private/*`: median provider EUR/hr, pricing met markup/fees, clients (budget→uren/churn).

## 4. Contract Tests — CLI (MUST)

- `python -m d3_engine.cli --inputs <dir> --out <dir> --pipelines public,private --seed 424242` → exit 0.
- Onbekende flags/ontbrekende vereisten → non‑zero.
- JSONL progress: `{ ts, level, event, ... }` minimaal `ts,level,event` aanwezig.

## 5. Golden & Determinism (MUST)

- Fixtures met vaste seeds in `/.003-draft/tests/fixtures/minimal_001/` produceren:
  - Byte‑gelijke artefacten: CSV/MD/JSON + `SHA256SUMS`.
  - Identieke `variable_draws.csv` (indien random variabelen gebruikt).
- Hash‑vergelijkingen in tests (SHA‑256). Kleine floating deltas → formatter/rounding policy in writer.

## 6. Acceptance Tests (MUST)

- Public: monotone `active_customers_m` over `targets.horizon_months` en (optioneel) minimale MoM groei.
- Private: `margin_pct(gpu_class) ≥ targets.private_margin_threshold_pct`.
- Capaciteit: `instances_needed(m,g*) ≤ autoscaling.max_instances_per_model`.
- Falen → **ERROR** of **WARNING** volgens `run.fail_on_warning`.

## 7. Performance & Concurrency (SHOULD)

- Minimal fixture E2E ≤ N seconden, RAM < M MB (vul N/M in wanneer meetbaar).
- `run.max_concurrency` variaties MUST geen result drift veroorzaken (identieke hashes).

## 8. Error Handling (MUST)

- Onbekende kolommen/paden → **ERROR**.
- Allocaties/gewichten som ≠ 1.0 → **WARNING** met renormalisatie, of **ERROR** als `fail_on_warning`.
- Ontbrekende seeds → **ERROR**.

## 9. Fixtures (SHOULD)

```
.003-draft/tests/fixtures/
  minimal_001/
    inputs/
      simulation.yaml
      operator/
      variables/
      facts/
    expected_outputs/
      SHA256SUMS
      public_tap_prices_per_model.csv
      ...
  stress_001/
    ...
```

## 10. Coverage & CI (SHOULD)

- Coverage ≥ 80% voor services/pure pipeline modules.
- CI jobs: pytest (unit + golden), cache wheels, upload artefacten bij failure.

## 11. Voorbeeld pytest (indicatief)

```python
import json
from pathlib import Path
import hashlib
import subprocess

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def test_cli_minimal(tmp_path: Path):
    inputs = Path('.003-draft/tests/fixtures/minimal_001/inputs').resolve()
    out = tmp_path / 'out'
    out.mkdir()
    proc = subprocess.run([
        'python','-m','d3_engine.cli',
        '--inputs', str(inputs),
        '--out', str(out),
        '--pipelines','public,private',
        '--seed','424242',
    ], capture_output=True, text=True)
    assert proc.returncode == 0
    assert (out / 'run_summary.json').exists()
    for line in proc.stdout.splitlines():
        rec = json.loads(line)
        assert {'ts','level','event'} <= set(rec)
```

## 12. Referenties

- `40_testing.md` — overkoepelend testregime
- `30_project_structure.md`, `31_engine.md` — structuur & CLI contract
- `22_sim_private_tap.md`, `23_sim_public_tap.md` — pijplijnregels
- `16_simulation_variables.md` — grid/replicates/MC & RNG
