# Investigation — SDK/Client DX

Status: done · Date: 2025-09-15

## Scope

Assess generated client ergonomics, streaming helpers, retries, and examples.

## Findings

- Generated client stubs exist; need simple examples for enqueue/stream/cancel in Rust and curl.
- Add helpers for SSE parsing and correlation-id propagation.

## Recommendations

- Add `tools/openapi-client` examples and trybuild tests for enqueue/stream/cancel.
- Document retry policy using `retriable` and `retry_after_ms` when present.

## Proofs

- Examples planned under `tools/openapi-client/examples/` and tests under `tools/openapi-client/tests/ui/`.
