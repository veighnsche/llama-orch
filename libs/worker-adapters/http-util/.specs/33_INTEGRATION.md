# http-util — Integration Tests (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `worker-adapters/http-util/`

## 0) Scope

Validate client construction, retry/backoff (jitter bounds), HTTP/2 keep‑alive, streaming decode conformance, and header redaction against a local stub server.

### Policy Alignment

- Follow `/.docs/testing/TESTING_POLICY.md`: tests are the contract; no skipped tests; use only public surfaces.
- Determinism: retries with jitter MUST have a seeded mode in tests (e.g., `HTTP_UTIL_TEST_SEED`).
- No real network dependencies; use local stubs (e.g., `wiremock`).

## 1) Test Matrix (normative)

- [HTU-INT-3301] Client Initialization
  - GIVEN default `Config`
  - WHEN `build_client` is called
  - THEN the client MUST have connect timeout ≈ 5s (±epsilon), request timeout ≈ 30s (configurable), HTTP/2 enabled when stub supports ALPN, TLS verification ON, and a connection pool for reuse.

- [HTU-INT-3302] Retry/Backoff Jitter Bounds
  - GIVEN a stub that returns transient 503 or timeouts
  - WHEN `with_retries` wraps an idempotent call
  - THEN attempts MUST respect the default policy (base 100ms, multiplier 2.0, cap 2s, max 4 attempts) and delays MUST be random within `[0, min(cap, base*multiplier^n)]`.

- [HTU-INT-3303] Streaming Decode Conformance
  - GIVEN a stub that emits `started`, `token*`, optional `metrics`, `end`
  - WHEN `stream_decode` processes the body
  - THEN ordering MUST remain `started → token* → end` and token indices MUST be strictly increasing from 0.

- [HTU-INT-3304] Secret Redaction
  - GIVEN responses with `Authorization`/`X-API-Key` in error context
  - WHEN errors are logged/formatted
  - THEN headers and token patterns MUST be redacted; no raw secrets may appear in output.

- [HTU-INT-3305] HTTP/2 Keep‑Alive
  - GIVEN many sequential requests to the same origin
  - WHEN using the built client
  - THEN the connection MUST be reused (no reconnect on each request) when server supports HTTP/2 keep‑alive.

## 2) Traceability

- Contracts: `./00_http_util.md`
- Root specs: `/.specs/35-worker-adapters.md`, `/.specs/20-orchestratord.md`
- Code: `worker-adapters/http-util/src/`

### Execution & CI

- Run: `cargo test -p worker-adapters-http-util -- --nocapture`
- CI: local stub server (wiremock) with scenarios for 429/5xx/timeouts and streaming.

### Proof Bundle Artifacts

- Retry timeline logs with seed disclosure
- Streaming transcript samples (started/token*/metrics?/end)
- Redacted error log snapshots (no secrets)

## Refinement Opportunities

- Add chaos profiles: injected latency/jitter/disconnect to stress buffering and decode.
- Provide benchmarks for decode hot path allocations.
