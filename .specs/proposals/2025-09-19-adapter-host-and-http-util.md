# Proposal: Adapter Host (in‑process) and Shared HTTP Util for Adapters

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025‑09‑19

## 0) Motivation

Centralize cross‑adapter concerns (registry, retries/timeouts, capability caching, narration) without adding a network hop, and define shared HTTP client contracts so all adapters behave consistently and efficiently. Keep `worker-adapters/adapter-api` as the canonical trait boundary.

## 1) Scope

In scope:
- New library crate `adapter-host` (in‑process) consumed by `orchestratord`.
- New library crate `worker-adapters/http-util` shared by adapters.
- Shared HTTP client contracts: connection pooling, HTTP/2, timeouts, retries with jitter, streaming decode helpers.
- Capability snapshot cache and exposure via orchestrator endpoints.
- Circuit breaker/cancellation policy centralization.
- Human narration logging wrappers around adapter calls.

Out of scope:
- Changing the `WorkerAdapter` trait surface.
- Introducing a separate daemon (no extra hop).

## 2) Normative Requirements (RFC‑2119)

IDs use ORCH‑36xx (adapter host & HTTP util).

### Adapter Host (in‑process)
- [ORCH‑3600] `adapter-host` MUST provide a registry keyed by `(pool_id, replica_id)` → `Box<dyn WorkerAdapter>` with lifecycle hooks for bind/rebind on preload/reload events.
- [ORCH‑3601] `adapter-host` MUST expose a facade with `submit(pool_id, job)`, `cancel(pool_id, task_id)`, `health(pool_id)`, `props(pool_id)` and MUST propagate cancellation promptly.
- [ORCH‑3602] `adapter-host` MUST implement capped, jittered retries and a circuit breaker for adapter calls, and MUST map errors to `WorkerError` taxonomy consistently.
- [ORCH‑3603] `adapter-host` SHOULD maintain a capability snapshot cache (ctx_max, batching, features) per pool/adapter and expose it for orchestrator `/v1/capabilities`.
- [ORCH‑3604] `adapter-host` MUST emit human‑readable narration with taxonomy fields (actor=adapter-host, action=submit|cancel, target=engine, plus ids) alongside structured logs.

### Shared HTTP Util (adapters)
- [ORCH‑3610] `worker-adapters/http-util` MUST provide a shared reqwest::Client builder with: keep‑alive enabled, HTTP/2 preferred, TLS verification on, reasonable pool sizes, and per‑request timeouts.
- [ORCH‑3611] The util MUST provide retry/backoff helpers with jitter and idempotency gating, and streaming decode utilities that minimize allocations for token deltas.
- [ORCH‑3612] Adapters SHOULD adopt the shared client and helpers; per‑adapter overrides MAY be provided via config for special cases.
- [ORCH‑3613] Secrets (API keys, tokens) MUST be redacted by default logging.

### Trait boundary remains canonical
- [ORCH‑3620] `worker-adapters/adapter-api` remains the canonical trait boundary; no network hop is introduced. `orchestratord` MUST depend only on the trait and the host facade.

## 3) Design Overview

- `adapter-host` (lib crate):
  - Holds adapter instances per pool/replica; manages bind/rebind lifecycle on reload/drain.
  - Provides a simple facade that `orchestratord` calls; handles retries, timeouts, circuit breakers, and cancellation routing.
  - Caches capability snapshots and surfaces them to orchestrator endpoints.
  - Wraps calls with narration logs and metrics (submit latency, retries, breaker trips).
- `worker-adapters/http-util` (lib crate):
  - Builds a tuned reqwest::Client and exposes helpers for retry/backoff and streaming decode (zero‑copy where possible).
  - Ensures consistent timeouts and error mapping across adapters.

## 4) Changes by Crate

- `orchestratord`: replace direct adapter bindings with `adapter-host` facade; keep `adapter-api` trait dependency.
- `worker-adapters/*`: adopt the shared HTTP util crate; remove adapter‑local client boilerplate; keep trait implementation local.
- No changes to `adapter-api` trait or types.

## 5) Migration Plan

Phase A
- Create `adapter-host` with registry + facade (submit/cancel/health/props) and basic narration wrappers.
- Create `worker-adapters/http-util` with shared client/retry + streaming decode helpers.
- Switch llamacpp adapter to use http‑util; wire orchestrator to use adapter-host for submit/cancel.

Phase B
- Extend host with capability snapshot cache and circuit breaker.
- Port remaining adapters to http‑util.

Phase C
- Integrate narration coverage into BDD (informational); add metrics for retries/breaker.

## 6) CI & Testing

- Unit tests for host: registry lifecycle, cancel routing, retry/backoff, breaker state transitions, capability cache consistency.
- Unit/behavior tests for http‑util: timeout enforcement, retry jitter, streaming decode correctness.
- Orchestrator integration tests keep using the host facade; BDD remains cross‑crate only.

## 7) Risks & Mitigations

- Central host adds a new code path → Keep host minimal with no network hop; comprehensive unit tests.
- HTTP/2 flakiness on some stacks → auto‑fallback to HTTP/1.1; configurable.
- Over‑retry causing latency spikes → cap attempts; circuit breaker with cool‑down.

## 8) Acceptance Criteria

- Orchestrator uses `adapter-host` for adapter calls; adapters adopt `http-util` client.
- Shared timeouts/retries in effect; narration and metrics emitted around adapter calls.
- BDD continues to pass; determinism suite unaffected (trait unchanged).

## 9) Refinement Opportunities

- Host‑level micro‑batch hints surfaced to orchestrator for SSE coalescing under load.
- Capability schema standardization and transport.
- Per‑engine policy hooks (e.g., token budget advisories) without changing trait.

## 10) Mapping to Repo Reality (Anchors)

- `worker-adapters/adapter-api` — canonical trait (unchanged).
- New `adapter-host/` — registry + facade.
- New `worker-adapters/http-util/` — shared HTTP client & streaming decoder.
- `orchestratord` — swaps to host facade for submit/cancel; `/v1/capabilities` backed by host cache.
