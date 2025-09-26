# Contract Tests — Guide and Proof Bundle

## What
Lock API shapes and error mappings to the spec. Includes SSE frame schemas and pact verifications.

## Where
- `contracts/**`, provider verify tests under `orchestratord/tests/` and related crates.
- Proof bundles: `<crate>/.proof_bundle/contract/<run_id>/`.

## When
- Changing API types, SSE frame structure, or error taxonomy.

## Artifacts (see template)
- `contract_fixtures.md` — list versions and stable IDs
- `sse_fixtures.ndjson` — canonical sample frames
- `error_mapping_table.md` — upstream → internal mapping
- `drift_report.md` (optional)
- `test_report.md`

## Recommended recipe
- Generate canonical fixtures alongside tests. Version them and include spec IDs.
- Use the same NDJSON shapes the server emits.

## Links
- Template: `.proof_bundle/templates/contract/README.md`
- Index: `.docs/testing/TEST_TYPES_GUIDE.md`
