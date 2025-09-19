# Wiring: test-harness-bdd ↔ worker-adapters

Status: Draft
Date: 2025-09-19

## Relationship
- Uses adapters only through orchestrator endpoints; validates stream framing and error mapping.

## Expectations
- Started/token/end ordering; metrics frames optional; cancel propagation. No crate-local behavior leakage; auth is exercised only via orchestrator endpoints.

## Refinement Opportunities
- Include OpenAI/vLLM/TGI variants when adapters mature.
