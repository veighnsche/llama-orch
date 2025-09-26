# worker-adapters — Integration Tests (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `worker-adapters/*`

## 0) Scope & Approach

End-to-end adapter streaming via the shared HTTP util against stubbed upstream servers. These tests validate deterministic streaming shape (`started → token* → end`, optional `metrics`), cancel semantics, error taxonomy mapping, and adoption of `worker-adapters/http-util` (timeouts, retries with capped+jittered backoff, HTTP/2 keep-alive, redaction).

## 1) Test Matrix (normative)

- [WA-INT-3001] Streaming Happy Path
  - GIVEN a stub upstream that emits a deterministic SSE/comet stream of token deltas
  - WHEN `submit` is called and the stream is consumed
  - THEN events MUST be observed in order: `started`, `token*`, optional `metrics`, `end` and no additional frames
  - AND token indexes MUST be strictly increasing starting at 0

- [WA-INT-3002] Cancel Semantics (race-free)
  - GIVEN an in-flight stream
  - WHEN `cancel` is issued
  - THEN no further `token` events MAY be observed after the cancel acknowledgment window
  - AND the adapter MUST release resources and close the upstream connection

- [WA-INT-3003] Error Taxonomy Mapping — Retryable
  - GIVEN upstream returns transient errors (e.g., HTTP 429 with `Retry-After`, 503, or network timeouts)
  - WHEN `submit` is executed
  - THEN the adapter MUST apply retries with capped+jittered backoff (via `http-util`), up to configured limits
  - AND on final failure, the mapped error MUST include `retriable=true` and advisory `retry_after_ms` when available

- [WA-INT-3004] Error Taxonomy Mapping — Non‑Retryable
  - GIVEN upstream returns permanent errors (e.g., 4xx invalid params)
  - WHEN `submit` is executed
  - THEN the adapter MUST fail without retry and map to the correct `WorkerError` code (`INVALID_PARAMS`, etc.) with `retriable=false`

- [WA-INT-3005] HTTP Util Adoption
  - GIVEN tests instrument the HTTP client
  - WHEN adapters construct clients
  - THEN they MUST use `worker-adapters/http-util` helpers (timeouts, retry policy with jitter, header redaction, HTTP/2 keep‑alive)
  - AND Authorization or sensitive headers MUST NOT appear in logs (redacted)

- [WA-INT-3006] Deterministic Streams (seeded)
  - GIVEN identical inputs including `seed` and sampler profile
  - WHEN two runs are executed against the same stub
  - THEN token streams MUST be byte‑identical and event ordering preserved

- [WA-INT-3007] Metrics & Logging Fields
  - WHEN a stream completes
  - THEN logs MUST include the standard fields aligned with `README_LLM.md` and `/.specs/metrics/otel-prom.md` (`job_id`, `session_id`, `engine`, `engine_version`, `tokens_out`, `decode_time_ms`)

- [WA-INT-3008] SSE Error Frame Compatibility (when surfaced via orchestrator)
  - GIVEN an upstream error during streaming
  - WHEN mapped through orchestrator to SSE `event:error`
  - THEN the error JSON SHOULD include `{ code, retriable, retry_after_ms? }` in alignment with `/.specs/proposals/2025-09-20-orch-spec-change-close-home-profile-gaps.md` (additive, optional)

## 2) Test Harness & Stubs

- Use a local stub server per adapter to simulate engine‑specific behaviors (status codes, SSE/delta quirks, slow start, mid‑stream fault, malformed frames).
- Provide fixtures to inject latency and fault profiles to cover backoff and jitter paths.

## 3) Traceability

- Contracts: `worker-adapters/adapter-api`, `worker-adapters/http-util/.specs/00_http_util.md`
- Root specs: `/.specs/35-worker-adapters.md`, `/.specs/20-orchestratord.md` (§SSE), `/.specs/metrics/otel-prom.md`
- Code under test: `worker-adapters/*/src/`

## Refinement Opportunities

- Add CDC tests that parse SSE transcripts from orchestrator end-to-end and assert compatibility with SDKs.
- Expand error mapping coverage across engines (e.g., vLLM, TGI) with canonical examples.
- Add optional heartbeat/keepalive compatibility tests when introduced at the data plane.
