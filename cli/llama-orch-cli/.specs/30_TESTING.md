# llama-orch-cli — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- CLI command behaviors against mock/local orchestrator endpoints; error handling and exit codes.

## Test Catalog

- Unit
  - Argument parsing and validation; config/env resolution precedence
  - Output formatting helpers (plain, JSON) and redaction of sensitive values

- Integration
  - Commands: `capabilities`, `create`, `stream`, `cancel`, `pools` (list/control)
  - Verify request/response mapping and headers (correlation id)
  - Error handling: 400/401/403/429/5xx mapped to exit codes and stderr messages

- Contract
  - OpenAPI client usage for core operations; golden fixtures for common flows

## Execution & Tooling

- `cargo test -p llama-orch-cli -- --nocapture`
- Snapshot tests for help/usage and typical outputs (use `insta` or similar if included)

## Traceability

- Aligns with `contracts/openapi/*` and `orchestratord` provider verify

## Refinement Opportunities

- Add golden snapshots for typical command outputs.
