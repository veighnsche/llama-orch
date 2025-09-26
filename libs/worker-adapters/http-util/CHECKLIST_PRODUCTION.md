# CHECKLIST_PRODUCTION — http-util

Crate: `libs/worker-adapters/http-util/`
Specs to honor:

- `./.specs/00_http_util.md`
- `./.specs/10_contracts.md`
- `./.specs/40_ERROR_MESSAGING.md`
- Testing specs: `./.specs/30_TESTING.md`, `31_UNIT.md`, `33_INTEGRATION.md`
- Root norms: `/.specs/35-worker-adapters.md`, `/.specs/20-orchestratord.md`, `/.specs/metrics/otel-prom.md`

Status Key: [ ] todo • [~] in progress • [x] done

## A) Public API & Configuration

- [ ] Export `HttpClientConfig` and `make_client(cfg: &HttpClientConfig) -> reqwest::Client`
  - [ ] Defaults per 00_http_util.md: connect_timeout≈5s, request_timeout≈30s (overrideable), HTTP/2 preferred, TLS verify ON, connection pool reuse
  - [ ] Explicit proxy support (opt-in); insecure modes gated and documented as test-only
- [ ] Provide `RetryPolicy` (base, multiplier, cap, max_attempts)
- [ ] Provide `with_retries<T, E, F>(policy, op)` wrapper with typed errors
- [ ] Provide `streaming` module with decode helpers for newline/SSE-like formats
- [ ] Provide `redact` module with `headers()` and `line()` helpers
- [ ] Provide `auth` helper to inject `Authorization: Bearer <token>` (opt-in)
- [ ] Ensure all public items have rustdoc with examples

## B) Retry/Backoff & Error Classification

- [ ] Implement default policy: base=100ms, multiplier=2.0, cap=2s, max_attempts=4
- [ ] Full jitter: `delay_n = rand(0, min(cap, base*multiplier^n))`
- [ ] Error classes retriable: 5xx, 429, timeouts, connect errors
- [ ] Non-retriable: 400/401/403/404/422 (configurable extension), map immediately
- [ ] Injectable RNG support (default: thread_rng; tests: seeded RNG)
- [ ] Injectable time/sleep (default: tokio::time::sleep; tests: mock clock)
- [ ] Structured retry decision logs (debug-level) with redaction applied

## C) Streaming Decode

- [ ] Decoder preserves strict ordering `started → token* → end`; optional `metrics`
- [ ] Token indices strictly increasing from 0
- [ ] Low-allocation path: reusable buffers, avoid per-token heap allocations where possible
- [ ] Handle chunked transfer, partial frames, and graceful termination
- [ ] Map failures to error taxonomy guidance (e.g., `DECODE_TIMEOUT` during stream, `POOL_UNAVAILABLE` pre-start), leaving final mapping to adapters

## D) Redaction & Security

- [ ] Redact `Authorization`, `X-API-Key`, bearer/token-like patterns in headers and messages
- [ ] Ensure no secrets at info/error log levels (see 40_ERROR_MESSAGING.md)
- [ ] TLS verification ON by default; proxies explicit; insecure-only with explicit opt-in for tests

## E) Concurrency & Performance

- [ ] `reqwest::Client` usage is `Send + Sync`, documented for reuse/clone across requests
- [ ] Keep-alive reuse validated (HTTP/2 where server supports ALPN)
- [ ] Provide guidance on concurrent usage and resource tuning in README

## F) Observability & Metrics (Division of Responsibility)

- [ ] Document that adapters emit metrics; http-util only provides redaction/logging helpers
- [ ] Provide sample debug logs for retry timelines (with redaction)

## G) Documentation & Examples

- [ ] README documents responsibilities and non-goals
- [ ] `.specs/10_contracts.md` reflects Provided/Consumed Contracts, Data Exchange, Security, Testing Expectations
- [ ] Add minimal usage examples: client creation, retry wrapper, streaming decode, redaction
- [ ] Cross-link to `.specs/35-worker-adapters.md` (normative shared util)

## H) Quality Gates

- [ ] `cargo fmt --all -- --check` clean
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] `cargo test -p worker-adapters-http-util -- --nocapture` green
- [ ] No unsupported Cargo.toml keys (e.g., `autodocs`)
- [ ] License headers, package metadata (private crate)

## I) Integration Plan

- [ ] Migrate `libs/worker-adapters/llamacpp-http` to use http-util
  - [ ] Replace ad-hoc client/retry/stream decode with shared util
  - [ ] Add wiremock-backed integration test
- [ ] Add follow-up migrations: vLLM, TGI, Triton, OpenAI adapters

## J) Proof Bundle Artifacts (for PRs)

- [ ] Retry timeline logs with seed disclosed
- [ ] Streaming transcript sample (started/token*/metrics?/end)
- [ ] Redacted error snapshot (no secrets)

## K) Optional Enhancements

- [ ] Shared error-mapping helpers on top of 40_ERROR_MESSAGING.md
- [ ] Opt-in metrics hooks (counters and latency timers)
- [ ] Benchmarks for streaming allocation profile

## L) Networking & HTTP Semantics

- [ ] Support HTTP/2 ALPN negotiation; gracefully fallback to HTTP/1.1 when unsupported
- [ ] Connection pooling/keep-alive tuned (max idle, idle timeout) and documented
- [ ] Enforce sane request body size limits (configurable)
- [ ] Enforce header size limits (documented; rely on reqwest/hyper defaults or override)
- [ ] Handle 100-Continue, redirects policy (document defaults)
- [ ] Respect `Retry-After` header (seconds/date) for backoff hints when present
- [ ] Idempotency enforcement guidelines for adapters (safe methods/operations)

## M) TLS/CA & Security Hardening

- [ ] TLS verification ON by default; document how to trust additional roots
- [ ] rustls with modern ciphers; document MSRV and rustls/webpki versions
- [ ] System roots vs custom root store support (documented)
- [ ] Pinning support (optional) documented as advanced/test-only
- [ ] Disable insecure renegotiation; HTTP/2 prior knowledge disabled unless explicit

## N) Proxy, DNS & Environment

- [ ] Explicit proxy configuration support; document NO_PROXY semantics
- [ ] DNS resolution strategy documented; note platform resolver vs trust-dns (if used)
- [ ] Timeouts for connect, read, write documented; per-request override guidance
- [ ] No implicit env reads by the crate; adapters pass config explicitly

## O) Limits, Backpressure & Resource Use

- [ ] Streaming line/frame size cap with clear error when exceeded
- [ ] Bounded in-flight buffer sizes (document defaults)
- [ ] Backpressure guidance for adapters (yield strategy, cooperative cancellation)
- [ ] Memory footprint guidance for high-throughput (buffer reuse)

## P) Decompression & Encoding

- [ ] Support gzip/deflate/zstd (as supported by reqwest) with safe limits
- [ ] UTF-8 handling documented; invalid UTF-8 strategy for streaming frames
- [ ] Newline normalization policy for line-based decoders (\n vs \r\n)

## Q) Error Taxonomy & Mapping Aids

- [ ] Document upstream→`WorkerError` mapping guidelines (40_ERROR_MESSAGING.md)
- [ ] Provide helpers for parsing `Retry-After`, classifying status/io errors
- [ ] Ensure redaction applies to error display at info/error levels

## R) Cancellation & Timeouts

- [ ] Support request-level timeout overrides
- [ ] Cooperative cancellation: propagate tokio cancellation to request/stream
- [ ] Surface cancellation as a distinct error kind
- [ ] Decoder shutdown path: stop cleanly on cancel/end-of-stream

## S) Threading & Runtime

- [ ] Safe usage under tokio multi-threaded runtime; document Send + Sync guarantees
- [ ] Do not block reactor threads; avoid blocking ops in async paths
- [ ] Bounded tokio tasks/spawns; no unbounded background loops

## T) API Stability & Feature Flags

- [ ] Document feature flags (if any) for optional components (e.g., zstd, wiremock-only helpers)
- [ ] Pre-1.0 changes require synchronized updates across adapters (workspace policy)
- [ ] MSRV documented and tested in CI (if applicable)

## U) Documentation & Examples (Deep)

- [ ] Example: build client with custom timeouts and HTTP/2
- [ ] Example: retry wrapper with seeded jitter for tests
- [ ] Example: streaming decode from in-memory buffer and from a stubbed response
- [ ] Example: redact headers/messages before logging an error
- [ ] Example: cooperative cancellation of a streaming request

## V) CI & Release Gates

- [ ] CI executes fmt, clippy -D warnings, unit + integration tests for this crate
- [ ] No network egress in CI; wiremock/local stubs only
- [ ] Optional coverage or report summary uploaded as artifact
- [ ] Include artifact bundle into `.proof_bundle/` on CI runs for auditing

## W) Audit & Proof Bundle Requirements

- [ ] `.proof_bundle/metadata.json` populated (commit, crate version, rustc/cargo, OS)
- [ ] Retry timeline JSON with seed disclosure captured
- [ ] Streaming transcript (NDJSON) sample captured
- [ ] Redacted error logs sample (no secrets) captured
- [ ] Test run report (pass/fail, seed used, summaries) captured
- [ ] Seeds file containing any RNG seeds used in tests
- [ ] CI manifest (job id, workflow url) captured
