# BDD Harness — Root Overview

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

Cross-crate integration scenarios over the orchestrator HTTP boundary (admission → stream; cancel; health/capabilities; catalog flows). Avoid crate-internal behavior tests; those live inside each crate.

## Provided Contracts (summary)

- Gherkin features and step inventories verifying contract-level behavior across crates.
- Proof bundles: SSE transcripts, logs, metrics samples.

## Consumed Contracts (summary)

- Orchestrator OpenAPI (control/data).
- Adapter streaming shape (via orchestrator), pool-managerd readiness signals (via orchestrator health APIs).

## Non-Goals

- Unit/behavior tests for a single crate.
- Provider/build correctness beyond API contracts.

## Refinement Opportunities

- Tagging (`@admission`, `@catalog`, `@reload`) to filter suites.
- Capability aggregation scenarios once standardized.
- Narration coverage stat in CI (see narration spec).
