# CHECKLIST — Rust MVP (llama-orch)

Status: Active
Date: 2025-09-30
Scope: Rust crates only under `bin/` and `libs/` (no Vue, no .business). Grounded to `.specs/00_mvp.md`.

Legend: [ ] todo · [~] in progress · [x] done · (MVP) required for MVP · (DEF) deferred (post‑MVP)

---

## 0) Cross‑Repo MVP Gates

- [ ] (MVP) Tests green minimal set: `cargo test --workspace -- --nocapture` (focus crates below)
- [ ] (MVP) Lints: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] (MVP) Contracts trimmed to MVP: `contracts/openapi/{data.yaml,control.yaml}` consistent with `.specs/00_mvp.md`
- [ ] (MVP) Disable non‑MVP features via Cargo features where possible (metrics/artifacts/mock)
- [ ] (MVP) Loopback default bind; Minimal Auth seam guarded behind config

---

## 1) bin/orchestratord

Reality check: Crate looks pre‑mature relative to new MVP. Must re‑scope to a thin server for single adapter (llama.cpp HTTP) and single queue.

- [ ] (MVP) Thin HTTP server (Axum) with minimal endpoints
  - [ ] `POST /v1/tasks` → basic guardrails (ctx), enqueue to single bounded FIFO
  - [ ] `GET /v1/tasks/:id/stream` → SSE `started`→`token*`→`end` (include `queue_position`; `predicted_start_ms` optional)
  - [ ] `POST /v1/tasks/:id/cancel` → race‑free cancel (no tokens after cancel)
  - [ ] `GET /v1/pools/:id/health` → liveness + readiness from adapter
- [ ] (MVP) SSE encoder ensures event ordering; stream `error` semantics per MVP
- [ ] (MVP) Minimal middleware: `X-Correlation-Id` on all responses
- [ ] (MVP) Config: loopback default; Minimal Auth seam for non‑loopback (Bearer token)
- [ ] (MVP) Queue/backpressure: return 429 with headers and JSON `{ policy_label, retriable, retry_after_ms }`
- [ ] (MVP) Wire llama.cpp adapter call path (no pool registry; no provisioners)
- [ ] (MVP) Logging fields: `job_id`, `engine`, `pool_id`, `tokens_out`, timings (decode optional)
- [ ] (MVP) Tests: happy path (admission→stream→end), cancel, 429 queue‑full
- [ ] (DEF) Drain/reload APIs and artifact registry
- [ ] (DEF) Metrics exporter; rich SSE `metrics` frames
- [ ] (DEF) Budgets, sessions, capabilities snapshot richness

Alignment with crate TODO (`bin/orchestratord/TODO.md`): many items are non‑MVP (artifacts, drain/reload, budgets, capabilities). Keep API/Data plane essentials and SSE only; defer sections 2,3,4,5,6 as needed.

---

## 2) libs/orchestrator-core

Use only what MVP needs: single in‑memory queue, backpressure policy, cancel, queue_position. No multi‑pool placement for MVP.

- [ ] (MVP) In‑memory bounded FIFO queue API with cancel support
- [ ] (MVP) Backpressure policy helper → 429 advisory values
- [ ] (MVP) Queue metrics hooks (optional for MVP runtime; useful for logs/testing)
- [ ] (MVP) Unit/property tests: FIFO order, cancel, queue bounds
- [ ] (DEF) VRAM‑aware deterministic placement, compatibility predicate, performance scoring
- [ ] (DEF) Budgets and `predicted_start_ms`

Note: `libs/orchestrator-core/TODO.md` and `CHECKLIST.md` include broad placement/compat requirements; mark as deferred for MVP.

---

## 3) libs/worker-adapters/llamacpp-http

- [ ] (MVP) Implement minimal mapping to llama.cpp HTTP server
  - [ ] `health`, `props` (slots minimal), `submit` (streaming), `cancel`
  - [ ] Stream mapping preserves SSE order; translate deltas to `{ t, i }`
- [ ] (MVP) Basic error mapping and redaction (no secrets in logs)
- [ ] (MVP) Integration test with a stubbed server emitting `started/token/end`
- [ ] (DEF) Extended sampling param mapping; metrics richness

---

## 4) libs/worker-adapters/http-util

Observation: Extensive test/production checklists; likely “overtested” for MVP. Keep minimal path stable.

- [ ] (MVP) Provide `make_client` with sane defaults (timeouts, TLS verify on)
- [ ] (MVP) Provide simple retry/backoff wrapper for idempotent calls
- [ ] (MVP) Provide streaming helpers sufficient for SSE/line decode with ordering and indices
- [ ] (MVP) Redact `Authorization` and `X-API-Key` in logs
- [ ] (DEF) HTTP/2 preference, proxy/TLS knobs, mock clock, advanced limits, deep metrics, proofs

Cross‑reference: `CHECKLIST_TESTING.md` shows ordering/indices covered; production streaming items are largely TODO. Acceptable for MVP if llama.cpp adapter relies only on essentials.

---

## 5) libs/adapter-host

For MVP, can be bypassed (or used minimally) — orchestrator may call adapter crate directly. If kept:

- [ ] (MVP) Facade for `submit/stream/cancel/health` with basic retries and correlation IDs
- [ ] (DEF) Registry lifecycle, breaker, capability cache, narration logs

Reference: `libs/adapter-host/TODO.md` — most items are post‑MVP.

---

## 6) libs/pool-managerd

Out of scope for MVP if orchestrator points to an existing llama.cpp endpoint.

- [ ] (DEF) Preload gating with model/engine provisioners
- [ ] (DEF) Health/version reporting and device masks
- [ ] (DEF) Supervisor/backoff/drain

Reference: `libs/pool-managerd/TODO.md` — all high items are deferred for MVP.

---

## 7) libs/catalog-core

Out of scope for MVP (no catalog/provisioners). Keep crate parked.

- [ ] (DEF) Atomic index writes, locks, `exists/locate`, GC/evict, trust_state
- [ ] (DEF) Unit/concurrency/interop tests

References: `libs/catalog-core/CHECKLIST.md`, `TODO.md` — defer entirely.

---

## 8) libs/provisioners/* (engine/model)

Out of scope for MVP.

- [ ] (DEF) Define `prepare()` API returning `{ bin, args, env, ports }` (if revisited post‑MVP)
- [ ] (DEF) Replace direct downloads with model‑provisioner; add policy gates
- [ ] (DEF) Mode implementations and logs/metrics

Reference: `libs/provisioners/engine-provisioner/TODO.md` — all deferred.

---

## 9) libs/proof-bundle

Useful today for tests; not runtime. Considered “heavily tested” which is fine.

- [x] (MVP) Basic API works (`ProofBundle::for_type`, write markdown/json/ndjson, seeds) — verified in `src/lib.rs` tests
- [ ] (DEF) Spec PB‑1xxx full compliance (headers, sanitize_name, CI header checker integration)
- [ ] (DEF) Benches, concurrency notes, extended docs

Reference: `libs/proof-bundle/CHECKLIST_PRODUCTION.md` — many items remain; non‑blocking for MVP.

---

## 10) libs/auth-min

- [ ] (MVP) Provide token parse/validate and timing‑safe compare; expose identity breadcrumb helpers
- [ ] (MVP) Orchestrator integration for non‑loopback binds
- [ ] (DEF) TRUST_PROXY_AUTH path and docs

Note: Align with `.specs/11_min_auth_hooks.md`.

---

## 11) libs/observability/narration-core

- [ ] (DEF) Optional narration hook types/helpers; not required for MVP

---

## 12) libs/worker-adapters/{mock, openai-http, vllm-http, tgi-http, triton}

- [ ] (DEF) Entirely deferred for MVP; ensure not pulled by default features

---

## 13) Integration Threads (MVP focus)

- [ ] Wire `orchestratord` → `llamacpp-http` adapter path end‑to‑end (admission→stream→cancel)
- [ ] Minimal provider test ensures OpenAPI examples align to endpoints
- [ ] Determinism smoke: fixed seed set on single replica → byte‑exact tokens

---

## 14) Risk Log (MVP)

- **[orchestratord maturity]** Current crate has broader ambitions (artifacts, drain/reload, metrics). Must prune to avoid scope creep.
- **[http-util maturity]** Production checklist incomplete; rely on minimal path and avoid advanced knobs until post‑MVP.
- **[proof-bundle]** Valuable for tests; avoid blocking runtime. Keep as dev‑only dep where used.

---

## 15) Next Commands (suggested)

```bash
# Lints & tests
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace -- --nocapture

# Focus orchestratord tests once endpoints stubbed
cargo test -p orchestratord -- --nocapture
```
