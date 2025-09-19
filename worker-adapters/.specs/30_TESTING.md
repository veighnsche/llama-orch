# worker-adapters — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Crate-level behavior: trait conformance, streaming framing, retries/backoff via shared HTTP util, error taxonomy mapping.

## Delegation

- Cross-crate flows covered by root BDD; this crate validates per-adapter behavior locally.

## Refinement Opportunities

- Add stress tests for low-allocation streaming decoder paths.
