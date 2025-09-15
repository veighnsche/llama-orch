# UX/DX Pre‑Freeze Analysis — OrchQueue v1

Status: draft · Scope: Data Plane + Control Plane contracts and developer workflow

## 1) Background

Per `README_LLM.md` and Stage 0 gates, we must lock contracts with strong UX/DX by default. This document captures findings and concrete proposals prior to contract freeze.

## 2) Summary of findings

- Missing OpenAPI `x-examples` on key endpoints limits client ergonomics and CDC clarity.
- 429/backpressure body lacks policy label; `.specs/00_llama-orch.md §6.2` expects policy label surfaced to callers.
- Error envelopes do not expose `retriable`/`retry_after_ms` hints; clients must infer from headers.
- No explicit correlation ID is surfaced in responses; logs mention correlation, but API contract does not standardize a header.

## 3) Proposed changes (contract-facing)

- Add examples (`x-examples`) to:
  - `POST /v1/tasks` — success 202; 400 invalid params; 429 backpressure.
  - `GET /v1/tasks/{id}/stream` — SSE frames for `started`, `token`, `metrics`, `end`, `error`.
  - `POST /v1/tasks/{id}/cancel` — 204.
  - Session `GET/DELETE`.
- Enhance `ErrorEnvelope`:
  - Optional fields: `retriable: boolean`, `retry_after_ms: int64`.
  - Backpressure 429 body MUST include `policy_label: string` representing the full admission policy label.
    - Option A: extend `ErrorEnvelope` with optional `policy_label`.
    - Option B: introduce `BackpressureError` with `policy_label` and reuse for 429.
- Introduce standardized correlation ID:
  - Request header: `X-Correlation-Id` (optional; if absent, server generates).
  - Response header: `X-Correlation-Id` MUST echo the effective correlation id for all Data Plane responses and SSE streams.
  - Document logging field alignment.

## 4) Proposed changes (spec-facing)

- Amend `.specs/00_llama-orch.md §6.2` to normatively require `policy_label` in 429 JSON bodies.
- Define `Correlation ID` behavior under transport conventions.
- Define optional `retriable` and `retry_after_ms` as advisory fields in `ErrorEnvelope`.

## 5) Tests and proof plan

- CDC examples include the new fields where relevant.
- Pact consumer tests assert body fields on 429 and presence/echo of `X-Correlation-Id`.
- Provider verification updated accordingly.
- Add `tools/openapi-client` sample code showing correlation id propagation and retry logic using `retry_after_ms`.
- Include `x-examples` in OpenAPI; spec-extract + link checks must remain green.

## 6) Compatibility and migration

- All additions are backward compatible (new optional fields and headers). Clients may ignore them.
- 429 body change is additive (new field). If strict consumers parse unknown fields, this is compatible per JSON object semantics.

## 7) Open questions

- Whether to enforce `X-Correlation-Id` as required on streaming responses (SSE). Proposal: required in HTTP headers; no duplication inside SSE frames.
- Whether to keep both `Retry-After` header and `retry_after_ms` body. Proposal: keep both; header for HTTP semantics, body for SDKs.

## 8) References

- `.specs/00_llama-orch.md §6.2`
- `contracts/openapi/data.yaml`
- `README_LLM.md` — Golden Rules
