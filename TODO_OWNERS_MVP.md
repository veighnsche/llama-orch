# TODO — MVP Ownership Split (Rust)

Status: Active
Date: 2025-09-30
Scope: Derived from `CHECKLIST_RUST_MVP.md` and `.specs/00_mvp.md`.

---

## Owner A — Orchestratord + Core

- [x] Implement `POST /v1/tasks` with ctx guardrails; enqueue to single bounded FIFO
- [x] Implement `GET /v1/tasks/:id/stream` SSE (started → token* → end, error) with `queue_position`
- [x] Implement `POST /v1/tasks/:id/cancel` (race-free; no tokens after cancel)
- [x] Implement `GET /v1/pools/:id/health` (liveness + readiness via adapter)
- [x] Add middleware to ensure `X-Correlation-Id` on all responses (incl. errors)
- [x] 429 backpressure with `Retry-After`, `X-Backoff-Ms`, body `{ policy_label, retriable, retry_after_ms }`
- [x] Orchestrator-core: in-memory bounded FIFO + cancel + `queue_position` helper
- [x] Wire queue into admission/streaming; logs include `job_id`, `engine`, `pool_id`, `tokens_out`
- [x] Default loopback bind; integrate auth-min for non-loopback (Bearer, timing-safe compare)
- [x] Adjust Cargo features to disable non-MVP defaults (metrics/artifacts/mock) where possible
- [x] Tests: admission → stream → end happy path; cancel path; queue-full 429

---

## Owner B — Llama.cpp Adapter + HTTP Util + Wiring

- [x] Llama.cpp adapter: minimal `health`, `props` (slots), `submit` (stream), `cancel`
- [x] Ensure streaming order and token indices (started → token* → end)
- [x] Basic error mapping and secret redaction in logs
- [x] http-util: `make_client` with sane defaults (timeouts; TLS verify on)
- [x] http-util: simple retry/backoff wrapper for idempotent calls
- [x] http-util: streaming helpers for SSE/line decode preserving order and indices
- [x] Adapter + http-util: integration test against stub server emitting `started/token/end`
- [x] Wire orchestrator to llama.cpp adapter end-to-end (admission → stream → cancel)
- [x] Determinism smoke: fixed seed on single replica → byte-exact tokens
- [x] Ensure non-MVP adapters are not pulled by default features
