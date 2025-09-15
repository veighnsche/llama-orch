# Investigation — OpenAPI Coverage & SSE Contract

Status: done · Date: 2025-09-15

## Summary

- Data plane OpenAPI updated with typed errors, SSE framing, correlation headers, and examples.

## Checks

- `x-req-id` present on operations and key components.
- SSE events enumerated and referenced in `x-sse-events`.
- `ErrorEnvelope` advisory fields added.
- 429 `policy_label` included in JSON example.

## Proofs

- `cargo xtask regen-openapi && git diff --exit-code`
- `rg -n "x-req-id: ORCH-" contracts/openapi`
