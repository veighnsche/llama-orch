# MVP SPEC — llama-orch (Minimal Viable Product)

Status: Accepted (MVP scope)
Date: 2025-09-30
Owners: @llama-orch-maintainers
Profile: Home lab, single host, NVIDIA GPU

This document narrows the full repository specs to the minimal shippable slice. It supersedes non‑MVP requirements for the first public cut. Where possible, requirement IDs reference the originating specs for continuity.

---

## 0) Goals & Non‑Goals

- **[MVP-001] Goals.**
  - Single-host orchestrator serving streaming LLM tokens from a llama.cpp server via a thin adapter.
  - Bounded FIFO admission with backpressure and cancellation.
  - Deterministic token streams on a single replica given fixed seed and parameters.
  - Loopback-first posture with a minimal auth seam when exposing beyond localhost.

- **[MVP-002] Non‑Goals (deferred).**
  - vLLM/TGI/Triton/OpenAI adapters; multi-engine catalogs and provisioners.
  - Multi-pool, pin overrides, session affinity across replicas, KV migration.
  - Engine/model provisioners; engine catalog writes; schema-driven reload flows.
  - Metrics contract completeness; rich SSE metrics payloads; full determinism suite corpus.
  - Mixed-GPU placement heuristics and multi-GPU orchestration beyond single GPU.

---

## 1) Platform & Binding

- **[MVP-010] Platform.** Linux host with NVIDIA drivers + CUDA runtime installed. Inference MUST run on NVIDIA GPUs; CPU inference is out of scope. (origin: `ORCH-1101..1102`)
- **[MVP-011] Binding.** Server MUST bind loopback by default (`127.0.0.1:8080`). Exposing non-loopback MUST require a shared Bearer token per Minimal Auth. (origin: `.specs/11_min_auth_hooks.md`)

---

## 2) Configuration (Minimal)

- **[MVP-020] Config file.** Single file (YAML/TOML) with:
  - `bind_addr` (default `127.0.0.1:8080`).
  - `auth.token` (string, required for non-loopback), `auth.optional` (bool; loopback may skip), `auth.trust_proxy_auth` (bool; default false). (origin: `OC-CONFIG-6030..6033`)
  - `adapter.engine = "llamacpp"` and `adapter.endpoint` (URL to llama.cpp HTTP server). No provisioner fields required in MVP.
- **[MVP-021] Strictness.** Unknown fields MUST be rejected (strict mode). (origin: `OC-CONFIG-6001`)

---

## 3) Data Plane — Admission & SSE

- **[MVP-030] Task submit.** `POST /v1/tasks` accepts a minimal Task request with prompt + sampling params + optional `seed`. Admission performs basic guardrails: context length check against a configured limit. (origin: `OC-CTRL-2010` subset)
- **[MVP-031] Queueing.** Single bounded FIFO queue (single priority: `interactive`). On full queue, reply `429` with `Retry-After` and body `{ policy_label, retriable, retry_after_ms }`. (origin: `OC-CTRL-2011`)
- **[MVP-032] Cancel.** `POST /v1/tasks/:id/cancel` MUST be race-free; no tokens after cancel. (origin: `OC-CTRL-2012`)
- **[MVP-033] Streaming.** `GET /v1/tasks/:id/stream` MUST emit SSE events in order: `started`, `token*`, `end`, and `error` on failure. (origin: `OC-CTRL-2020..2029`)
  - `started` → `{ queue_position: int, predicted_start_ms?: int }` (predicted start is OPTIONAL in MVP).
  - `token` → `{ t: string, i: int }`.
  - `end` → `{ tokens_out: int, decode_time_ms?: int }` (decode time OPTIONAL in MVP).
  - `error` → `{ code: string, retriable: boolean, retry_after_ms?: number, message?: string }`.

---

## 4) Scheduling & Execution (MVP simplifications)

- **[MVP-040] Replica model.** Single adapter target (llama.cpp HTTP) and a single GPU/replica. No cross-pool placement, no pin overrides. (origin subset of `OC-CORE-1010..1013`)
- **[MVP-041] Ready/health.** A simple health probe MUST exist: `GET /v1/pools/:id/health` returning liveness and a basic readiness flag from the adapter’s `/health` (or equivalent). (origin: `OC-CTRL-2001` minimal)
- **[MVP-042] Determinism.** With identical `{prompt, parameters, seed, engine_version, model_digest}`, streams from the same replica MUST be byte-identical. (origin: `ORCH-3025` family)

---

## 5) Adapter — llama.cpp HTTP only

- **[MVP-050] Coverage.** Implement only llama.cpp HTTP adapter mapping: health, props (slots), submit (SSE), cancel. (origin: `OC-ADAPT-5001`)
- **[MVP-051] Streaming order.** Adapter MUST preserve `started → token* → end`, mapping llama.cpp SSE deltas to `token`. (origin: `.specs/35-worker-adapters.md`)
- **[MVP-052] Version capture.** Adapter SHOULD capture `engine_version` and `model_digest` when available for logs. (origin: `OC-ADAPT-5011`)
- **[MVP-053] HTTP client.** Timeouts and retries SHOULD be bounded; secrets MUST be redacted from logs. (origin: adapter HTTP util spec)

---

## 6) Auth (Minimal seam)

- **[MVP-060] Bearer token.** When bound to non-loopback, requests MUST present `Authorization: Bearer <token>`. (origin: `AUTH-1001..1004`)
- **[MVP-061] Errors.** Use 401/403 JSON errors per Minimal Auth with timing‑safe compare. (origin: `AUTH-1007`)

---

## 7) Observability (MVP)

- **[MVP-070] Logs.** JSON Lines with correlation IDs; include standard fields: `job_id`, `engine`, `pool_id`, `replica_id`, `tokens_in`, `tokens_out` when known. Metrics emission is OPTIONAL in MVP.
- **[MVP-071] Queue hints.** `queue_position` MUST be surfaced in `started`. `predicted_start_ms` is OPTIONAL.

---

## 8) Testing & Proof (MVP)

- **[MVP-080] BDD smoke.** Admission → stream → cancel happy path, plus queue‑full `429`. Minimal features only. (origin: `.specs/72-bdd-harness.md`)
- **[MVP-081] Determinism smoke.** Small fixed seed set (e.g., 8 seeds) on a single replica verifies byte‑exactness. Full corpus deferred. (origin: `.specs/70-determinism-suite.md` subset)

---

## 9) Out‑of‑Scope Now (explicit deferrals)

- Multi‑engine adapters (vLLM, TGI, Triton, OpenAI).
- Engine/model provisioners; engine catalog; catalog CRUD.
- Multi‑pool scheduling, pin overrides, VRAM‑aware placement, heterogeneous device masks.
- Budgets (token/time/cost) and KV pressure controls surfaced to clients.
- Rich metrics contract and histogram buckets; SSE `metrics` frames.
- Session TTLs/turn caps and affinity beyond MVP single‑shot flows.

---

## 10) Minimal Cross‑Walk to Existing Specs

- Control/Data plane: `.specs/20-orchestratord.md` (subset: `OC-CTRL-2001`, `2010..2012`, `2020..2029`).
- Core invariants: `.specs/10-orchestrator-core.md` (subset: ready gating, single queue FIFO).
- Adapter: `.specs/40-worker-adapters-llamacpp-http.md` and `.specs/35-worker-adapters.md` (subset).
- Auth: `.specs/11_min_auth_hooks.md` (subset; loopback‑first default elevated for MVP).
- Config: `.specs/60-config-schema.md` (subset; adapter endpoint only).
- Determinism: `.specs/70-determinism-suite.md` (smoke subset).

---

## 11) Acceptance for MVP

- Start orchestrator bound to loopback with llama.cpp endpoint configured.
- Submit a task; receive `started` then `token` stream, then `end`.
- Cancel a running task; observe no further tokens.
- Fill the queue; receive `429` with backoff body.
- Determinism smoke passes on single replica with fixed seeds.
