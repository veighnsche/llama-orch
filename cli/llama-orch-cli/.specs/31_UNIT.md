# llama-orch-cli — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Argument parsing, config/env resolution, output formatting helpers.

## Test Catalog

- Arg Parsing
  - Required/optional flags; mutually exclusive options; default values
  - `--help` and `--version` output snapshot

  - Precedence: CLI flags > env vars > config file; missing values produce helpful errors

- Output Formatting
  - Plain vs JSON output; redaction of tokens/keys in logs

- Error Handling Helpers
  - Map error kinds to exit codes and user-facing messages

## Execution

- `cargo test -p llama-orch-cli -- --nocapture`
- Use snapshot testing for help/usage and representative outputs when available

## Traceability

- Aligns with `orchestratord` control/data plane contracts and OpenAPI bindings

## Refinement Opportunities

- Snapshot tests for help/usage and error messages.
