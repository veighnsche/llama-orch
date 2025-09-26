# http-util — Error Messaging (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `worker-adapters/http-util/`

## 0) Purpose

Provide shared helpers to produce consistent, redacted error messages and to map upstream HTTP errors into the repository‑wide `WorkerError` taxonomy used by adapters.

## 1) Normative Requirements

- [HTU-ERR-4001] All error formatting/logging helpers MUST redact secrets in headers and messages, including but not limited to `Authorization`, `X-API-Key`, and common bearer token patterns.
- [HTU-ERR-4002] Mapping from upstream HTTP errors to `WorkerError` MUST be stable and cover common classes:
  - 400/422 invalid parameters → `INVALID_PARAMS` with `retriable=false`.
  - 401/403 auth issues (if applicable) → `UPSTREAM_AUTH` with `retriable=false` (adapter‑internal only; not exposed publicly).
  - 404 model/resource not found → `UPSTREAM_NOT_FOUND` with `retriable=false` (adapter‑internal only).
  - 409/425/429 backpressure/conflict → `ADMISSION_REJECT` or `QUEUE_FULL_DROP_LRU` analog with `retriable=true`, populate `retry_after_ms` when available.
  - 5xx transient upstream → `POOL_UNAVAILABLE` or `WORKER_RESET` with `retriable=true`.
  - IO/timeout → `DECODE_TIMEOUT` (during stream) or `POOL_UNAVAILABLE` (pre‑start) with `retriable=true`.
- [HTU-ERR-4003] Error envelopes returned to adapters MUST include: `{ code, message, retriable: bool, retry_after_ms?: number }`. Message MUST be safe (no secrets) and short.
- [HTU-ERR-4004] When propagating into SSE via orchestrator, adapters SHOULD preserve `retriable` and `retry_after_ms` hints when known.

## 2) Testing Expectations

- Unit: header redaction patterns, error mapping table coverage, presence of advisory fields (`retriable`, `retry_after_ms`).
- Integration: stub upstream returning representative error codes and payloads; verify mapped taxonomy and redaction in logs.

## 3) Traceability

- Root specs: `/.specs/20-orchestratord.md` (§4 Error Taxonomy, §3.3 Streaming Failure Semantics)
- Worker adapters root: `/.specs/35-worker-adapters.md`

## Refinement Opportunities

- Extend mapping table with engine‑specific idiosyncrasies (vLLM/TGI) while keeping shared codes stable.
- Provide localized/structured hint fields to aid remediation without leaking sensitive context.
