# orchestrator-core — Error Messaging (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Error Shapes

- Internal typed errors for queue and planning only; no HTTP envelopes here.

## Delegation

- HTTP error envelopes live in `orchestratord/.specs/20_contracts.md`.

## Refinement Opportunities

- Add an `IncompatibleReason` taxonomy reference for planners.
