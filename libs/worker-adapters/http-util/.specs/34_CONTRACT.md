# http-util — Contract Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- If provider verification applies (e.g., common adapter framing), document contract fixtures here.

## Proof Bundle Outputs (MUST)

Contract tests MUST generate artifacts under `libs/worker-adapters/http-util/.proof_bundle/`:

- `contract_fixtures.md` — list and brief description of fixtures provided (e.g., SSE token frame shapes)
- `sse_fixtures.ndjson` — canonical sample frames for token streaming that adapters can reuse
- `error_mapping_table.md` — upstream → `WorkerError` example rows aligned with `./40_ERROR_MESSAGING.md`

## Refinement Opportunities

- Shared fixtures for SSE chunk decoding across adapters.
