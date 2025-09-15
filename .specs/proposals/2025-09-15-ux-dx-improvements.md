# Proposal: UX/DX Improvements Prior to Contract Freeze

Status: draft · Date: 2025-09-15

## Problem

- Important UX/DX affordances are missing from the Data Plane and Control Plane contracts, making clients and tests more cumbersome:
  - No `x-examples` on key OpenAPI endpoints, making CDC and SDK generation less ergonomic.
  - 429/backpressure body lacks `policy_label` required by `.specs/00_llama-orch.md §6.2`.
  - `ErrorEnvelope` does not provide advisory `retriable` and `retry_after_ms` fields.
  - No standardized correlation ID behavior across requests/responses/SSE.

## Change Summary

- Add OpenAPI `x-examples` to data-plane endpoints: enqueue, stream (SSE frames), cancel, sessions.
- Extend 429 bodies with a policy label field.
- Extend `ErrorEnvelope` with optional advisory fields `retriable: boolean` and `retry_after_ms: int64`.
- Standardize `X-Correlation-Id` header handling (request optional, response echo mandatory) including SSE responses.

## Impacted Areas

- contracts/openapi/data.yaml
- `.specs/00_llama-orch.md` (normative updates to §6.2 and transport conventions)
- CDC tests (consumer pact examples, provider verification expectations)
- tools/openapi-client examples and generated clients

## Requirements (new/changed IDs)

- Changes are additive to existing requirements:
  - ORCH-2006 (ErrorEnvelope): update to include optional advisory fields.
  - ORCH-2007 (429 backpressure): update to require `policy_label` in JSON body in addition to HTTP headers.
  - New: ORCH-20XX (Correlation ID): define request/response header behavior and logging field alignment.

## Migration / Rollback

- Migration:
  - Clients may continue ignoring new optional fields; headers remain canonical.
  - SDKs can start leveraging `retry_after_ms` and `retriable` for improved retry logic.
- Rollback:
  - If needed, remove `x-examples` is non-breaking; optional fields can be ignored by servers and clients.

## Proof Plan

- OpenAPI regeneration is diff-clean on second run.
- CDC examples updated; consumer tests for 429 and correlation id; provider verification green.
- Link/spec lint scripts pass.

## References

- `.docs/investigations/ux-dx-pre-freeze.md`
- `.specs/00_llama-orch.md`
- `contracts/openapi/data.yaml`
