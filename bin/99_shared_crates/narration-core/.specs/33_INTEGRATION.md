# narration-core — Integration Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Orchestratord integration via init; capture adapter wiring for BDD.

## Test Catalog

- Init Wiring
  - GIVEN queen-rbee initializes narration with capture adapter
  - WHEN emitting events during admission/streaming
  - THEN capture receives entries with required fields (correlation id, job/session ids, engine, tokens metrics when present)

- Pretty vs JSON Toggle
  - GIVEN toggle on/off
  - WHEN formatting a narrative entry
  - THEN output format matches expectation with stable keys/order (for snapshotting)

- Redaction End-to-End
  - GIVEN events containing secrets in headers/params
  - WHEN captured
  - THEN secrets are redacted as per helpers; no leaks in logs

## Fixtures & Mocks

- Test capture adapter collecting events in-memory for assertions

## Execution

- `cargo test -p narration-core -- --nocapture`

## Traceability

- Field alignment with `README_LLM.md` and `/.specs/metrics/otel-prom.md`

## Refinement Opportunities

- Pretty vs JSON toggle behavior verification.
