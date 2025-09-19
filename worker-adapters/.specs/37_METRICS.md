# worker-adapters — Metrics (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Alignment

- Follow `/.specs/metrics/otel-prom.md` for names/labels.
- If adapters emit metrics directly, keep cardinality within budgets; admission-level counters may omit `engine_version`.

## Refinement Opportunities

- Provide per-adapter examples once stable.
