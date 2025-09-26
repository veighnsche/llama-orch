# Proof-Bundle — Unit Tests (31_UNIT)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-26

## Scope
Small, deterministic tests verifying pure functions and minimal IO helpers of `proof-bundle`, without asserting on OS-specific behavior beyond Linux filesystem basics.

## What to Test
- Run ID resolution (PB-1002)
  - Honors `LLORCH_RUN_ID`
  - Fallback to timestamp (and optional `git_sha8` if present)
- Directory mapping (PB-1001, PB-1010)
  - `TestType` → `unit|integration|contract|bdd|determinism|home-profile-smoke|e2e-haiku`
- Header prelude generation helpers (PB-1012)
  - `write_markdown_with_header` prepends header; `write_markdown` enforces header
  - `append_ndjson_autogen_meta` idempotency
  - `write_json_with_meta` creates `<name>.json.meta`
- Seeds recorder (PB-1005/1006)
  - Appends lines; newline-terminated; multiple appends ok

## Example Cases
- PB-UT-001: `resolve_run_id` respects env and format
- PB-UT-002: `TestType::as_dir()` exact mapping
- PB-UT-003: header helpers prepend/write correctly, idempotent behavior
- PB-UT-004: seeds recorder appends expected format `seed=<N>`

## Artifacts
None (unit tests should avoid writing to the repo). Use `tempfile` when needed.

## Commands
- `cargo test -p proof-bundle --lib -- --nocapture`

## Links
- Spec: `/.specs/00_proof-bundle.md`
- Contracts: `/.specs/10_contracts.md`
- Types guide: `/.docs/testing/TEST_TYPES_GUIDE.md`
