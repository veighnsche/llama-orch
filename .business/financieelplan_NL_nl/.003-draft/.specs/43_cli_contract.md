# 43 — CLI Contract

Status: Draft
Version: 0.1.0

## 1. Scope

- Commandline interface, flags, exit codes, stdout JSONL progress en run summaries.

## 2. Contract (MUST)

```bash
python -m d3_engine.cli \
  --inputs .003-draft/inputs \
  --out .003-draft/outputs \
  --pipelines public,private \
  --seed 424242 \
  [--fail-on-warning] [--max-concurrency 4]
```

- Exit codes: `0=OK`, `2=VALIDATION_ERROR`, `3=RUNTIME_ERROR`.
- `--pipelines` accepteert subset van `public,private`.
- Stdout JSONL: minimaal `{ ts, level, event }` met events uit `21_engine_flow.md`.
- Output files: `run_summary.{json,md}` + CSV’s per gekozen pipelines.

## 3. Tests

- `cli_contract.feature`: run met minimale inputs → exit 0, `run_summary.json` aanwezig, JSONL keys aanwezig.
- Onbekende flags/ontbrekende vereisten → non‑zero.

## 4. Referenties

- `21_engine_flow.md`, `31_engine.md`, `40_testing.md`, `41_engine.md`.
