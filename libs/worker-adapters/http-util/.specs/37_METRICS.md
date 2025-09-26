# http-util — Metrics (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Emission

- This util does not emit metrics directly.

## Proof Bundle Outputs (MUST)

Even though this crate does not emit metrics, tests MUST generate the following artifact under `libs/worker-adapters/http-util/.proof_bundle/`:

- `metrics_note.md` — a short note confirming that http‑util does not emit metrics and pointing to adapter‑level metrics responsibilities per `/.specs/metrics/otel-prom.md`.

## Refinement Opportunities

- Consider exposing retry counters via callback hooks.
