# narration-core — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Narration emission helpers and redaction behavior; test capture adapter.

## Test Catalog

- Unit
  - Redaction helpers mask secrets in values and nested JSON
  - Formatting helpers produce stable shapes for story entries
  - Sampling controls honor rates/thresholds deterministically in tests

- Integration
  - Capture adapter wiring collects narration events from `rbees-orcd` init path
  - Pretty vs JSON toggle produces expected output formats

## Execution & Tooling

- `cargo test -p narration-core -- --nocapture`
- Use a test logger/capture adapter for assertions; avoid stdout reliance

## Traceability

- Align log fields with `README_LLM.md` and `/.specs/metrics/otel-prom.md`

## Refinement Opportunities

- Sampling controls and story snapshot fixtures.
