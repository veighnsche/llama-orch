# orchestratord — Metrics (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Alignment

- Conform to `/.specs/metrics/otel-prom.md` names/labels. Follow bucket guidance in `/.specs/71-metrics-contract.md`.

## Emission Sites

- Admission counters, queue gauges, SSE latency histograms, tokens counters.

## Refinement Opportunities

- Document label budgets for admission-level counters.
