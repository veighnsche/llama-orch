# narration-core — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Redaction helpers; narration formatting helpers.

## Test Catalog

- Redaction Helpers
  - Mask tokens/keys in flat and nested JSON structures
  - Ensure regex-based secret patterns handle edge cases (prefix/suffix, URL-embedded secrets)

- Formatting Helpers
  - Produce stable narrative entries (keys, ordering, types) for snapshot-style assertions
  - Validate required fields presence when enabled by policy

## Execution

- `cargo test -p narration-core -- --nocapture`

## Traceability

- Field alignment with `README_LLM.md` and `/.specs/metrics/otel-prom.md`

## Refinement Opportunities

- Add edge cases for secret patterns.
