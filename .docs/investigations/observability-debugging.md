# Investigation — Observability & Debugging

Status: done · Date: 2025-09-15

## Scope

Validate logging fields and tracing spans per SPEC; ensure metrics fields align and provide examples path.

## Findings

- SPEC lists required log fields (ORCH-3027). SSE includes `started` with queue_position & predicted_start_ms.
- Metrics alignment updated per `.specs/metrics/otel-prom.md`.

## Actions

- Keep log examples in docs; ensure correlation-id propagation is standardized in API headers.

## Proofs

- `rg -n "ORCH-3027|ORCH-3028" -- **/*.md`
- `bash ci/scripts/spec_lint.sh`
