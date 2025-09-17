# Workflow — Home Lab Edition

This document explains how we build and validate llama-orch for the home profile. It replaces all legacy “enterprise” handbooks; every reference here assumes a single workstation host and the developer box that drives it.

---

## 0. Guiding Principles

1. **Spec is law.** Start with `.specs/00_llama-orch.md` and overlays such as `.specs/00_home_profile.md`.
2. **Contract-first.** Update OpenAPI, config schema, and metrics contracts before writing code.
3. **Tests before code.** Pact/BDD/provider/determinism suites must describe new behaviour before implementation.
4. **Determinism by default.** Same `{prompt, params, seed}` on the same replica MUST produce identical tokens.
5. **Fail fast.** Reject impossible requests up front; surface typed errors and backpressure.
6. **Short sessions.** No long chats; maintain TTLs and KV limits.
7. **Real hardware.** Every release must pass on the reference workstation (RTX 3090 + 3060) described in `.docs/HOME_PROFILE_TARGET.md`.

---

## 1. Repository Orientation

```
/ Cargo.toml                  # workspace
/.specs/                      # normative specs (core + components)
/.docs/                       # guides, workflow, testing catalog, archived TODOs
/contracts/openapi/           # data/control plane OpenAPI
/contracts/config-schema/     # Rust schema → JSON schema
/orchestrator-core/           # queue + placement primitives
/orchestratord/               # HTTP handlers and state
/pool-managerd/               # replica registry, drain/reload
/worker-adapters/             # adapters (mock, llama.cpp, vLLM, TGI, Triton)
/test-harness/                # BDD, determinism, chaos, Haiku
/cli/                         # CLI and consumer tests
/tools/                       # regen utilities (spec extract, OpenAPI client)
```

Key docs to read first:
- `.docs/HOME_PROFILE.md` — what we promise to users.
- `.specs/00_llama-orch.md` — normative requirements.
- `.specs/20-orchestratord.md` — HTTP behaviour for handlers.
- `.docs/PROCESS.md` — end-to-end change process.

---

## 2. Product Stages (Home Profile)

We track progress through the following stages. A stage is complete only when specs, contracts, tests, code, and docs align.

1. **Stage 0 — Contract Freeze**: OpenAPI + config schema validated, regenerated deterministically.
2. **Stage 1 — Consumer Contracts**: CLI pact tests and snapshots define expected behaviour.
3. **Stage 2 — Provider Verification**: orchestrator handlers satisfy pact files.
4. **Stage 3 — Core Properties**: queue invariants proven via property tests.
5. **Stage 4 — Determinism**: byte-exact streams across replicas per engine.
6. **Stage 5 — Observability**: metrics contract satisfied, `/metrics` endpoint available, logs structured.
7. **Stage 6 — Admission→SSE Vertical**: POST/GET/cancel handlers wired to worker adapters with determinism knobs.
8. **Stage 7 — Catalog & Reloads**: drain/reload, catalog state transitions, artifact persistence.
9. **Stage 8 — Capability Discovery**: `GET /v1/replicasets` or `GET /v1/capabilities` returns usable limits for the CLI.
10. **Stage 9 — Home Scheduling Heuristics**: mixed-GPU placement, queue metadata, optional priority weighting.
11. **Stage 10 — Budgets & Sessions**: advisory/enforced budgets, session introspection, SSE metrics exposure.
12. **Stage 11 — Tooling Policy**: outbound HTTP policy hook, audit logging.
13. **Stage 12 — BDD Coverage**: journeys cover admission, streaming, cancel, catalog, artifacts, mixed GPUs.
14. **Stage 13 — Dashboards & Alerts**: minimal Grafana dashboards + alert thresholds for the workstation.
15. **Stage 14 — Startup Self-tests**: preload, minimal decode, cancel, telemetry checks on boot.
16. **Stage 15 — Haiku Anti-cheat**: end-to-end Haiku test on real hardware with `REQUIRE_REAL_LLAMA=1`.

Stages 16–17 (nightly chaos, release prep) will be defined when we approach release.

---

## 3. Stage Deliverables (Snapshot)

### Stage 6: Admission → SSE Vertical
- `POST /v1/tasks` stores tasks in the queue and returns admission metadata.
- `GET /v1/tasks/{id}/stream` streams SSE through the adapter with correlation IDs and optional budgets.
- `POST /v1/tasks/{id}/cancel` stops queued/active jobs deterministically.
- Determinism flags wired end-to-end; metrics updated for tokens and latency.

### Stage 7: Catalog, Drain, Reload
- Catalog CRUD persists to local storage with verification warnings.
- `POST /v1/pools/{id}/drain` flips draining state; `POST /v1/pools/{id}/reload` swaps model or rolls back.
- Pool health exposes readiness, draining, last error, GPU metrics.

### Stage 8: Capability Discovery
- Capability endpoint returns engine versions, max context, concurrency hints, supported workloads.
- CLI uses this data to schedule agents safely across GPUs.

### Stage 10: Budgets & Sessions
- Session registry tracks TTL, turns, KV usage, budgets.
- SSE `metrics` frame includes remaining budget when enforcement is active.

### Stage 15: Haiku Anti-cheat
- Orchestrator must complete the Haiku test using a real worker over LAN/tunnel, emitting metrics/logs for audit.

---

## 4. Developer Workflow Checklist

1. Read the relevant spec (`.specs/**`) and overlay (`.specs/00_home_profile.md`).
2. Update contracts (`contracts/openapi`, `contracts/config-schema`, metrics spec).
3. Regenerate artifacts: `cargo xtask regen-openapi`, `cargo xtask regen-schema`, `cargo run -p tools-spec-extract --quiet`.
4. Add/adjust tests (pact, provider, BDD, property, unit) referencing requirement IDs.
5. Implement code changes.
6. Run the dev loop: `cargo xtask dev:loop`.
7. Update `TODO.md` with what changed and why.

---

## 5. Testing Catalogue (Pointers)

- BDD features live under `test-harness/bdd/tests/features/`; see `.docs/testing/spec-derived-test-catalog.md` for mappings.
- Determinism suite: `test-harness/determinism-suite/`.
- Metrics lint: `test-harness/metrics-contract/` + `ci/metrics.lint.json`.
- Reference environment smoke test: documented in `.docs/HOME_PROFILE_TARGET.md` (scripts to follow).

---

## 6. Release Gate Summary

A release candidate must satisfy:
- All stages ≤15 complete (tracked in README).
- Specs and contracts diff-clean.
- BDD, determinism, Haiku, and metrics suites green on the reference workstation.
- CLI pact/provider snapshots updated.
- Docs and TODO tracker reflect the change.

---

The home lab is the product. Keep the docs current, keep the tests honest, and keep the workstation happy.
