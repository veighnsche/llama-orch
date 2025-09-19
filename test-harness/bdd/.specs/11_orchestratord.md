# Wiring: test-harness-bdd ↔ orchestratord

Status: Draft
Date: 2025-09-19

## Relationship
- This harness validates orchestrator HTTP behavior spanning multiple crates.

## Expectations
- Exercise OpenAPI control/data endpoints and SSE framing; include Minimal Auth positive/negative scenarios per `/.specs/11_min_auth_hooks.md`.
- Assert error envelopes, correlation IDs, and backpressure.

## Refinement Opportunities
- Expand to capability aggregation once schema stabilizes.
