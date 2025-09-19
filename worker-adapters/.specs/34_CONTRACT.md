# worker-adapters — Contract Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Contract-level verification of streaming framing and error taxonomy mapping against shared interfaces.
- If an adapter integrates external APIs (e.g., OpenAI), keep provider verify in adapter-local tests.

## Alignment

- Names/labels for metrics align with `/.specs/metrics/otel-prom.md`.

## Refinement Opportunities

- Add reusable fixtures for SSE token streams and retry backoff timelines.
