# TODO — MVP Ownership Split (Rust)

Status: Active
Date: 2025-09-30
Scope: Derived from `CHECKLIST_RUST_MVP.md` and `.specs/00_mvp.md`.

---

## Owner A — Orchestratord + Core

- [ ] Implement `POST /v1/tasks` with ctx guardrails; enqueue to single bounded FIFO
- [ ] Implement `GET /v1/tasks/:id/stream` SSE (started → token* → end, error) with `queue_position`
- [ ] Implement `POST /v1/tasks/:id/cancel` (race-free; no tokens after cancel)
- [ ] Implement `GET /v1/pools/:id/health` (liveness + readiness via adapter)
- [ ] Add middleware to ensure `X-Correlation-Id` on all responses (incl. errors)
- [ ] 429 backpressure with `Retry-After`, `X-Backoff-Ms`, body `{ policy_label, retriable, retry_after_ms }`
- [ ] Orchestrator-core: in-memory bounded FIFO + cancel + `queue_position` helper
- [ ] Wire queue into admission/streaming; logs include `job_id`, `engine`, `pool_id`, `tokens_out`
- [ ] Default loopback bind; integrate auth-min for non-loopback (Bearer, timing-safe compare)
- [ ] Adjust Cargo features to disable non-MVP defaults (metrics/artifacts/mock) where possible
- [ ] Tests: admission → stream → end happy path; cancel path; queue-full 429

---

## Owner B — Llama.cpp Adapter + HTTP Util + Wiring

- [ ] Llama.cpp adapter: minimal `health`, `props` (slots), `submit` (stream), `cancel`
- [ ] Ensure streaming order and token indices (started → token* → end)
- [ ] Basic error mapping and secret redaction in logs
- [ ] http-util: `make_client` with sane defaults (timeouts; TLS verify on)
- [ ] http-util: simple retry/backoff wrapper for idempotent calls
- [ ] http-util: streaming helpers for SSE/line decode preserving order and indices
- [ ] Adapter + http-util: integration test against stub server emitting `started/token/end`
- [ ] Wire orchestrator to llama.cpp adapter end-to-end (admission → stream → cancel)
- [ ] Determinism smoke: fixed seed on single replica → byte-exact tokens
- [ ] Ensure non-MVP adapters are not pulled by default features
