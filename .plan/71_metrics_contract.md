# Metrics Contract Plan

## Objectives
- Keep `/metrics` output in sync with `.specs/metrics/otel-prom.md`.
- Ensure minimal series exist (queue depth, tasks counters, tokens, GPU/VRAM).
- Provide optional gauges for model lifecycle (`Active|Retired`) and KV cache usage.

## Tasks
1. Audit `orchestratord/src/metrics.rs` for unused legacy metrics (`admission_share`, `deadlines_met_ratio`, preemption counters) and plan their removal.
2. Update `ci/metrics.lint.json` to match the trimmed contract.
3. Add integration test to verify `/metrics` includes queue depth, tokens in/out, GPU utilisation on deque operations.
4. Document metric usage in `.docs/HOME_PROFILE.md` and dashboards.

## Tests
- `test-harness/metrics-contract/tests/metrics_lint.rs`
- `cargo test -p orchestratord --test metrics_endpoint`

## Dashboards
- Create minimal Grafana dashboard showing queue depth, tasks rejected, GPU/VRAM usage.
- Provide instruction snippet under `.docs/workflow.md` Stage 13 when ready.
