# Wiring: test-harness-bdd ↔ pool-managerd

Status: Draft
Date: 2025-09-19

## Relationship
- Validates readiness transitions and drain/reload effects via orchestrator control plane.

## Expectations
- Health summaries reflect manager registry state; draining blocks new placements.

## Refinement Opportunities
- Synthetic failure injection for preload/restart/backoff paths.
