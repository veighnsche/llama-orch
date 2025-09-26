# worker-adapters-http-util — worker-adapters-http-util (adapter)

## 1. Name & Purpose

Shared HTTP utilities for all adapter crates. This library provides a single, consistent place
for HTTP client construction (timeouts, HTTP/2), retry/backoff with jitter, streaming decode
helpers for token events, and safe redaction of secrets in logs. Adapters like
`llamacpp-http`, `vllm-http`, `tgi-http`, `triton`, and `openai-http` depend on this crate to
avoid duplicating cross-cutting concerns and to meet spec requirements consistently.

## 2. Why it exists (Spec traceability)

- ORCH-3054 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3054)
- ORCH-3055 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3055)
- ORCH-3056 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3056)
- ORCH-3057 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3057)
- ORCH-3058 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3058)

- Worker Adapters (normative shared util): [.specs/35-worker-adapters.md](../../../.specs/35-worker-adapters.md)

## 3. Public API surface

- `HttpClientConfig` — base timeouts, retry limits, backoff/jitter knobs.
- `make_client(&HttpClientConfig) -> reqwest::Client` — HTTP/2 preferred, rustls TLS, sane defaults.
- `retry(policy, op)` — classify errors (429/5xx/connect/timeouts) and apply exp. backoff + jitter.
- `streaming::decode_*` — low‑alloc helpers to parse newline/SSE‑ish token streams into events.
- `redact::{headers, line}` — redact Authorization and other secrets in diagnostics.
- `auth::with_bearer(client, token)` (opt‑in) — helper to inject Authorization header.

## 4. How it fits

- Supports engine‑specific adapter crates with a shared HTTP layer. Keeps adapter code focused on
  mapping engine APIs to the orchestrator worker contract while relying on a single place for
  transport and robustness concerns (timeouts, retries, streaming decode, redaction).

```mermaid
flowchart LR
  orch[Orchestrator]
  orch --> host[Adapter Host]
  host --> adapters[Adapters]
  adapters --> http_util[http-util (shared)]
  adapters --> engines[Engine APIs]
```

## 5. Build & Test

- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features
-- -D warnings`
- Tests for this crate: `cargo test -p worker-adapters-http-util -- --nocapture`
- Preferred integration test approach: use `wiremock` in an adapter crate to validate streaming
  and retry behavior with this util.

## 6. Contracts

- None

## 7. Config & Env

- The util itself does not read environment variables; adapters pass configuration explicitly.
- Typical adapter configuration:
  - Base URL(s) per engine instance
  - Timeouts and retry policy (max attempts, base/backoff cap, jitter)
  - Optional `Authorization` bearer token to inject (via helper)
- HTTP/2 is preferred when supported by the engine; falls back to HTTP/1.1.

## 8. Metrics & Logs

- Adapters are expected to emit request/streaming metrics; this util focuses on consistent logging
  and redaction. Helpers ensure Authorization and other secrets are not logged. Retry decisions and
  backoff intervals can be logged at `debug` with redaction applied.

## 9. Runbook (Dev)

- Regenerate artifacts: `cargo xtask regen-openapi && cargo xtask regen-schema`
- Rebuild docs: `cargo run -p tools-readme-index --quiet`
- Unit tests: `cargo test -p worker-adapters-http-util`

## 10. Status & Owners

- Status: alpha
- Owners: @llama-orch-maintainers

## 11. Changelog pointers

- None

## 12. Footnotes

- Spec: [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md)
- Requirements: [requirements/00_llama-orch.yaml](../../../requirements/00_llama-orch.yaml)

### Additional Details

- Responsibilities:
  - Provide a single, consistent HTTP client with timeouts and HTTP/2 preference
  - Implement a minimal, spec‑aligned retry policy with jitter
  - Offer streaming decode helpers for token event flows with low allocations
  - Redact secrets in diagnostic logs and traces
- Non‑goals:
  - No engine‑specific API mapping (belongs in each adapter)
  - No direct environment parsing (adapters own config → util APIs)

## What this crate is not

- Not a public API; do not expose engine endpoints directly.
