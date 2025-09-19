# test-harness-e2e-haiku — End-to-End Real Engine Suite (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

Run a minimal real-engine end-to-end scenario with real SSE streaming, using staged models and configured pools.

Out of scope: mocks/stubs; per-crate behavior.

## Contracts
- Orchestrator OpenAPI; adapter streaming; pool-managerd readiness.

## Refinement Opportunities
- Multi-replica runs and drain/reload coverage.
