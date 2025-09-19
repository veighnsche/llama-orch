# orchestrator-core — Determinism (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Deterministic tie-breakers in placement; reproducible decisions given identical inputs.

## Delegation

- Engine sampler/seed determinism and single-slot enforcement live in adapters and root determinism suite.

## Refinement Opportunities

- Formalize an `IncompatibleReason` taxonomy to aid deterministic NoCapacity explanations.
