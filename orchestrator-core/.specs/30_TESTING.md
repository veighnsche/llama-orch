# orchestrator-core — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

Umbrella for crate-local tests. This crate owns queue invariants, placement feasibility and tie-break determinism. Cross-crate flows are exercised by the root BDD harness.

## Owned Test Layers

- Unit: queue semantics, cancel, snapshot readouts.
- Property: invariants for FIFO-within-priority, Drop-LRU behavior.
- Integration: minimal placement flows using in-crate mocks.

## Delegated

- BDD: admission → stream/cancel via HTTP is validated by `test-harness/bdd`.

## Refinement Opportunities

- Add randomized stress/property tests for Drop-LRU.
- Provide golden cases for deterministic tie-breaks.
