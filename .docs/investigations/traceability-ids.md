# Investigation — SPEC IDs and Traceability

Status: done · Date: 2025-09-15

## Method

Scan `.specs/` and contracts for `ORCH-*` and `OC-*` coverage and uniqueness.

## Findings

- v3.2 IDs added (ORCH-3060..3094). No duplicates detected in scan.
- OpenAPI operations/components annotated via `x-req-id`.

## Proofs

- Commands:
  - `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
  - `rg -n "\b(ORCH|OC)-[A-Z0-9-]+\b" -- **/*.{md,rs}`
