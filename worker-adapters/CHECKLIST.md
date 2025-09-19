# Worker Adapters — Production Readiness Checklist (shared)

Use this as the central checklist. Each adapter crate should add a per-adapter CHECKLIST.md with overrides and engine-specific items.

- [ ] Implement `WorkerAdapter` trait
  - [ ] `health()` returns `live`, `ready`
  - [ ] `props()` exposes `slots_total/free` if known
  - [ ] `submit()` streams started→token*→end (metrics optional) and maps errors
  - [ ] `cancel()` best-effort
  - [ ] `engine_version()` stable semantic version
- [ ] Error taxonomy mapping to `WorkerError`
- [ ] Timeouts/retries with caps and jitter for networked adapters
- [ ] Determinism signals surfaced (engine_version; seed/profile when applicable)
- [ ] Metrics and logs per `.specs/metrics/otel-prom.md` and `README_LLM.md`
- [ ] Capability/health reporting consistent with `pool-managerd`
- [ ] Security: redact secrets, honor egress policy, do not log full prompts in prod
- [ ] Tests: unit/behavior for mapping, retries, error taxonomy
- [ ] Docs: README points to shared specs; per-adapter 00_*.md summarizes unique behavior
