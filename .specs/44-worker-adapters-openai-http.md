# Worker Adapter SPEC — OpenAI HTTP (v1.0)

Status: Draft
Applies to: `worker-adapters/openai-http/`
Conformance language: RFC‑2119
Date: 2025-09-19

## 0) Scope & Versioning

Requirements are versioned as `OC-ADAPT-OAI-6xxx`.

This adapter integrates OpenAI-compatible HTTP endpoints with the `WorkerAdapter` trait. It performs streaming completion requests, maps responses to `TokenEvent` (`started → token* → end`, optional `metrics`), and enforces timeouts/retries while redacting secrets in logs.

## 1) API Mapping

- [OC-ADAPT-OAI-6000] Adapter MUST implement `health`, `props`, `submit`, `cancel`, `engine_version` per `worker-adapters/adapter-api`.
- [OC-ADAPT-OAI-6001] `submit(TaskRequest)` MUST map to OpenAI Chat/Completions or Responses API with streaming enabled and MUST translate token deltas to `token` events.
- [OC-ADAPT-OAI-6002] Adapter MUST bound request timeouts and implement capped, jittered retries for idempotent calls.
- [OC-ADAPT-OAI-6003] Adapter MUST NOT expose OpenAI endpoints publicly; all calls are internal.

## 1A) References & Cross-Cutting

- This adapter SHOULD use `worker-adapters/http-util` for HTTP client construction, retries (capped + jitter), HTTP/2 keep‑alive, and header redaction.
- Integration with orchestrator uses the in‑process facade described in `adapter-host/.specs/00_adapter_host.md`.
- Streaming MUST preserve `started → token* → end` ordering per `/.specs/35-worker-adapters.md`; apply redaction to logs consistently.

## 2) Determinism & Version Capture

- [OC-ADAPT-OAI-6010] Adapter MUST capture and report `engine_version` (model/version string from OpenAI) and SHOULD include `model_digest` if provided by the upstream API or via config.
- [OC-ADAPT-OAI-6011] Streams MUST preserve ordering (`started → token* → end`), and token boundaries SHOULD follow the upstream delta semantics.

## 3) Security & Policy

- [OC-ADAPT-OAI-6020] API keys MUST be redacted from logs and error messages; headers MUST not be logged at error level.
- [OC-ADAPT-OAI-6021] Network egress MUST be limited by policy; TLS verification MUST be on; proxies (if any) MUST be explicitly configured.

## 4) Observability

- [OC-ADAPT-OAI-6030] Adapter SHOULD log retry/backoff details and map upstream error codes to `WorkerError`.
- [OC-ADAPT-OAI-6031] Latency and token counts SHOULD be captured; logs MUST include standard fields from `README_LLM.md`.

## 5) Testing Ownership

- Unit/behavior tests MUST cover: streaming mapping, error taxonomy, timeouts/retries, secret redaction.
- Cross-crate streaming behavior remains in the BDD harness.

## 6) Traceability

- Code: `worker-adapters/openai-http/src/lib.rs`

## Refinement Opportunities

- Shared HTTP client/retry helper adoption (keep-alive, HTTP/2) from a common adapter util crate.
- Optional partial decoding optimizations to reduce per-token allocations.
- Support additional OpenAI APIs (Responses API) behind feature flags.
