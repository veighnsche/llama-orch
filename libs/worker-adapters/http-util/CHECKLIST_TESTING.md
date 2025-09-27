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

- [~] HTU-UNIT-3101 — Builder defaults
  - [x] Connect timeout ≈ 5s (±epsilon)
  - [x] Request timeout ≈ 30s default, overrideable per request
  - [ ] HTTP/2 preferred when supported; TLS verify ON; pool reuse
- [~] HTU-UNIT-3102 — Retry/backoff policy
  - [x] Given base, multiplier, cap, attempts → delays within `[0, min(cap, base*multiplier^n)]`
  - [ ] Full jitter distribution shows variance
  - [ ] Deterministic mode via seeded RNG (`HTTP_UTIL_TEST_SEED`) yields exact expected delays
  - [ ] Non-retriable codes (400/401/403/404/422) do not retry
- [~] HTU-UNIT-3103 — Redaction helpers
  - [x] `Authorization` header redacted
  - [x] `X-API-Key` redacted
  - [ ] Bearer-like token patterns redacted in messages
  - [ ] Snapshot of redacted output (no secrets)
- [~] HTU-UNIT-3104 — Streaming decode helpers
  - [x] Ordering `started → token* → end` preserved
  - [x] Token indices strictly increasing from 0
  - [ ] Low-allocation path (buffer reuse) — assert minimal allocations where feasible or via benchmarks later

### Additional Unit Coverage

- [ ] Encoding & newlines: UTF-8 validation, CRLF vs LF normalization
- [ ] Decompression support: gzip/deflate/zstd (if enabled)
- [ ] Limits: max line/frame size → clear error; bounded buffers
- [ ] Error helpers: `Retry-After` parsing (seconds/date); status and IO classification
- [ ] Cancellation semantics surfaced distinctly from timeouts
- [ ] Redaction of query/body keys (known sensitive names)

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

### Additional Integration Coverage

- [ ] Proxy configuration honored (HTTP(S) proxy, NO_PROXY)
- [ ] TLS: custom root store; TLS verification ON by default
- [ ] Retry-After honored for backoff when provided by upstream
- [ ] Cancellation: cooperative cancel during streaming terminates cleanly
- [ ] Large stream and partial frames robustness under backpressure
- [ ] Unicode tokens and multi-byte boundaries across chunks

## Determinism & Time Control

- [x] Tests accept seed via `HTTP_UTIL_TEST_SEED` (or explicit seeded RNG) for retry jitter
- [ ] Time abstraction in retry helper allows tokio time to be paused and advanced in tests
- [ ] Avoid wall-clock sleeps; use mocked time to assert logical delay sequences

## CI & Isolation

- [ ] No external network; use local stub (wiremock)
- [ ] Avoid timing-based assertions on wall-clock; assert logical sequences and configured delays
- [ ] Coverage: optional threshold or report summary generated as artifact
- [ ] Doctests compile and pass (examples in rustdoc)
- [ ] MSRV job passes (if MSRV is declared)

## Proof Bundle Artifacts

- [x] Retry timeline with seed disclosure
- [x] Streaming transcript sample (started/token*/metrics?/end)
- [ ] Redacted error snapshots (no secrets)
- [ ] metadata.json (commit, crate version, rustc, OS)
- [ ] seeds.txt (seeds used)

## Cross-References & Traceability

- [ ] Link test IDs (HTU-UNIT/HTU-INT) in test names/docs
- [ ] Ensure coverage across the contracts in `./.specs/00_http_util.md` and error messaging in `./.specs/40_ERROR_MESSAGING.md`
- [ ] Verify adapter root norms in `/.specs/35-worker-adapters.md` are satisfied (shared HTTP util usage)

## Advanced: Property, Fuzz, and Concurrency

- [ ] Property tests for retry jitter distribution stability (seeded)
- [ ] Fuzz tests for streaming decoder (malformed/partial inputs) with sanitization
- [ ] Concurrency/race tests: parallel streaming sessions, cancellation overlap, and resource cleanup
