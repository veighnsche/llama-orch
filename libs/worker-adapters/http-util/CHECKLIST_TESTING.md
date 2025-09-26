# CHECKLIST_TESTING — http-util

Crate: `libs/worker-adapters/http-util/`
Specs to reflect:
- `./.specs/30_TESTING.md`
- `./.specs/31_UNIT.md`
- `./.specs/33_INTEGRATION.md`
- `./.specs/40_ERROR_MESSAGING.md`
- Normative: `./.specs/00_http_util.md`

Status Key: [ ] todo • [~] in progress • [x] done

## Unit Tests (see 31_UNIT.md)
- [ ] HTU-UNIT-3101 — Builder defaults
  - [ ] Connect timeout ≈ 5s (±epsilon)
  - [ ] Request timeout ≈ 30s default, overrideable per request
  - [ ] HTTP/2 preferred when supported; TLS verify ON; pool reuse
- [ ] HTU-UNIT-3102 — Retry/backoff policy
  - [ ] Given base, multiplier, cap, attempts → delays within `[0, min(cap, base*multiplier^n)]`
  - [ ] Full jitter distribution shows variance
  - [ ] Deterministic mode via seeded RNG (`HTTP_UTIL_TEST_SEED`) yields exact expected delays
  - [ ] Non-retriable codes (400/401/403/404/422) do not retry
- [ ] HTU-UNIT-3103 — Redaction helpers
  - [ ] `Authorization` header redacted
  - [ ] `X-API-Key` redacted
  - [ ] Bearer-like token patterns redacted in messages
  - [ ] Snapshot of redacted output (no secrets)
- [ ] HTU-UNIT-3104 — Streaming decode helpers
  - [ ] Ordering `started → token* → end` preserved
  - [ ] Token indices strictly increasing from 0
  - [ ] Low-allocation path (buffer reuse) — assert minimal allocations where feasible or via benchmarks later

## Integration Tests (see 33_INTEGRATION.md)
- [ ] HTU-INT-3301 — Client Initialization
  - [ ] Built client exhibits defaults and HTTP/2 keep-alive when stub supports ALPN
- [ ] HTU-INT-3302 — Retry/Backoff Jitter Bounds
  - [ ] Stub returns 503/timeouts; attempts follow policy (base=100ms, mult=2.0, cap=2s, attempts=4)
  - [ ] Delays randomized within allowed bounds; deterministic with seed
- [ ] HTU-INT-3303 — Streaming Decode Conformance
  - [ ] Stub emits started/token*/metrics?/end; decoder preserves ordering and indices
- [ ] HTU-INT-3304 — Secret Redaction
  - [ ] Error contexts with `Authorization`/`X-API-Key` are redacted
- [ ] HTU-INT-3305 — HTTP/2 Keep‑Alive
  - [ ] Sequential requests reuse connection when server supports HTTP/2 keep‑alive

## Determinism & Time Control
- [ ] Tests accept seed via `HTTP_UTIL_TEST_SEED` (or explicit seeded RNG) for retry jitter
- [ ] Time abstraction in retry helper allows tokio time to be paused and advanced in tests

## CI & Isolation
- [ ] No external network; use local stub (wiremock)
- [ ] Avoid timing-based assertions on wall-clock; assert logical sequences and configured delays

## Proof Bundle Artifacts
- [ ] Retry timeline with seed disclosure
- [ ] Streaming transcript sample (started/token*/metrics?/end)
- [ ] Redacted error snapshots (no secrets)

## Cross-References & Traceability
- [ ] Link test IDs (HTU-UNIT/HTU-INT) in test names/docs
- [ ] Ensure coverage across the contracts in `./.specs/00_http_util.md` and error messaging in `./.specs/40_ERROR_MESSAGING.md`
