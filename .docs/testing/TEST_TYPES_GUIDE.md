# Test Types Guide — Centralized Guidelines and Proof Bundles

This guide centralizes the why, when, and how for every test type used in this monorepo and standardizes the proof bundle artifacts each test must produce.

It complements:
- `.docs/testing/TESTING_POLICY.md` (policy and CI gates)
- `.docs/testing/spec-derived-test-catalog.md` and `.docs/testing/test-case-discovery-method.md` (Spec→Contract→Tests→Code, RFC-2119, stable IDs)

Follow the Spec→Contract→Tests→Code workflow described in `README_LLM.md` and AGENTS.md.

## Global conventions

- Proof bundles are crate-local: `<crate>/.proof_bundle/<type>/<run_id>/...`
- `run_id` format: `YYYYMMDD-HHMMSS-<git_sha8>` or provide explicitly via env `LLORCH_RUN_ID`. If `git` is unavailable, fall back to epoch seconds.
- Redaction: never include secrets (tokens, headers). Provide `*_redacted.*` artifacts when logging upstream errors or HTTP traces.
- Determinism: prefer fixed seeds. Respect `*_TEST_SEED` envs where present. Record all RNG seeds used in `seeds.txt`.
- Time: avoid wall-clock sleeps; use mocked time where possible.
- File formats: prefer NDJSON (`.ndjson` or `.jsonl`) for streams; JSON for structured configs; CSV for timing tables; Markdown for human summaries.
- Link tests and artifacts to spec IDs (e.g., ORCH-####) in bundle `test_report.md`.

## Test types overview

- Unit tests
  - Purpose: Validate a single module or function with fast, deterministic cases.
  - Scope: One crate/module at a time; no external process/network.
  - Artifacts: `retry_timeline.jsonl` (if applicable), `redacted_errors.*`, `seeds.txt`, `test_report.md`.
  - Template: `.proof_bundle/templates/unit/README.md`.
  - Guide: `.docs/testing/types/unit.md`.

- Integration tests
  - Purpose: Exercise interactions across modules in a crate using local stubs.
  - Scope: Crate boundary; use in-memory or file-based test doubles only.
  - Artifacts: `retry_timeline.jsonl`, `streaming_transcript.ndjson`, `redacted_errors.*`, `seeds.txt`, `test_report.md`.
  - Template: `.proof_bundle/templates/integration/README.md`.
  - Guide: `.docs/testing/types/integration.md`.

- Contract tests
  - Purpose: Verify public API shapes, pacts, and error mapping against the spec.
  - Scope: Contracts in `contracts/**`, server frames, SSE schema snapshots.
  - Artifacts: `contract_fixtures.md`, `sse_fixtures.ndjson`, `error_mapping_table.md`, optional `drift_report.md`, `test_report.md`.
  - Template: `.proof_bundle/templates/contract/README.md`.
  - Guide: `.docs/testing/types/contract.md`.

- BDD (Cucumber) tests
  - Purpose: Validate user journeys end-to-end through public interfaces.
  - Scope: Harness under `test-harness/bdd/` with `bdd-runner`.
  - Artifacts: `bdd_transcript.ndjson`, `http_traces_redacted.ndjson` (if applicable), `test_report.md`, optional `world_logs/`.
  - Template: `.proof_bundle/templates/bdd/README.md`.
  - Guide: `.docs/testing/types/bdd.md`.

- Determinism suite
  - Purpose: Ensure identical inputs produce identical token streams across replicas/engines.
  - Scope: `test-harness/determinism-suite/` running against real model engines.
  - Artifacts: `pairs/`, `diffs/`, `run_config.json`, optional `timing.csv`, `test_report.md`.
  - Template: `.proof_bundle/templates/determinism/README.md`.
  - Guide: `.docs/testing/types/determinism.md`.

- Home Profile Smoke
  - Purpose: Minimal acceptance gate on the reference workstation.
  - Scope: Scripts and probes on `.docs/HOME_PROFILE_TARGET.md` hardware.
  - Artifacts: `environment.md`, `smoke_report.md`, `logs_redacted/`.
  - Template: `.proof_bundle/templates/home-profile-smoke/README.md`.
  - Guide: `.docs/testing/types/smoke.md`.

- E2E Haiku (GPU)
  - Purpose: Live GPU end-to-end streaming token test with metrics capture.
  - Scope: `test-harness/e2e-haiku/` with `REQUIRE_REAL_LLAMA=1`.
  - Artifacts: `gpu_env.json`, `sse_transcript.ndjson`, `metrics_snapshot.json`, `run_log_redacted.md`, `test_report.md`.
  - Template: `.proof_bundle/templates/e2e-haiku/README.md`.
  - Guide: `.docs/testing/types/e2e-haiku.md`.

## Proof bundle paths and naming

- Default base path: crate root
  - Example: `orchestrator-core/.proof_bundle/integration/20250926-131530-a1b2c3d4/`
- Recommended environment variables set by runners
  - `LLORCH_RUN_ID`: if set, use as the run directory under the test type
  - `LLORCH_PROOF_DIR`: override base directory (defaults to `CARGO_MANIFEST_DIR/.proof_bundle`)

## Redaction and privacy

- Redact credentials and PII in all logs. Provide parallel `*_redacted.*` artifacts when raw logs are needed for debugging but cannot be committed.
- If an artifact cannot be sufficiently redacted, store a structured summary instead and link to an internal secure location in `test_report.md`.

## Seeds and determinism

- Always record seeds in `seeds.txt` if randomness is used.
- Prefer deterministic tests; when true randomness is necessary, isolate it and record the seed and policy.

## CI and release gates

Tests of all types must keep the repo green. GPU-gated suites may run in dedicated jobs, but must pass before a release, as per `TESTING_POLICY.md`.

## How to choose the right test type

- Start at the smallest scope that can catch the bug or prove the requirement.
- Use unit for algorithmic invariants and edge conditions.
- Use integration for crate-level behavior with internal adapters/stubs.
- Use contract to lock external shapes and mapping tables.
- Use BDD for user journey semantics and cross-service flows.
- Use determinism and E2E Haiku to validate real-engine behavior under the home profile.
- Use Smoke to guard the reference environment readiness.

## See also

- `.docs/testing/types/*.md` for deep-dives per type
- `.proof_bundle/templates/*/README.md` for artifact checklists
- `.docs/testing/TESTING_POLICY.md` for policy and CI
