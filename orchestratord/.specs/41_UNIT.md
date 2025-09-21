# orchestratord — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Handler functions, header parsing, auth seam guards (when enabled), basic SSE framing.

## Test Catalog

- Request Validation
  - Missing/invalid fields → 400 with JSON body aligned to contracts (no panics).
  - Over-budget inputs (ctx, max_tokens) rejected early with structured error (mapping to ORCH‑3016).

- Headers & Correlation
  - `X-Correlation-Id` passthrough or generation when missing; value appears in logs and responses.
  - Backpressure headers on 429: `Retry-After`, `X-Backoff-Ms` present and coherent.

- Minimal Auth seam (feature-gated)
  - Unauthorized requests → 401/403 with redacted headers in logs.
  - Loopback exemption path (when configured) allows local tools.

- SSE Framing Utilities
  - Emit `started`, `token`, `metrics`, `end` with correct event names and JSON serialization helpers.
  - Error case emits `event:error` with minimal JSON shape when applicable.

## Execution & Tooling

- Run: `cargo test -p orchestratord -- --nocapture`
- Keep unit tests pure: do not bind sockets; test handler functions and helpers in isolation.

## Traceability

- ORCH‑3406..3409: SSE error frame shape (helpers used by data plane tests).
- ORCH‑3330/3331: Error-class mapping table used by handlers.
- ORCH‑3016: No silent truncation on budgets.

## Refinement Opportunities

- Add unit tests for backpressure envelope construction (429 + headers).
