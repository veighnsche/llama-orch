# Project Checklist — Scaffolding for Proposals and Specs

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

This checklist aggregates all work to implement recent proposals and root specs. It maps to:
- Testing ownership/scope: `.specs/proposals/2025-09-19-testing-ownership-and-scope.md` (ORCH-32xx)
- Performance streamlining: `.specs/proposals/2025-09-19-performance-streamlining.md` (ORCH-34xx)
- Human narration logging: `.specs/proposals/2025-09-19-human-narration-logging.md` (ORCH-33xx)
- Adapter host + HTTP util: `.specs/proposals/2025-09-19-adapter-host-and-http-util.md` (ORCH-36xx)
- Token streaming & cancel robustness: `.specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md`
- Engine/Model Provisioners & Engine Catalog: `.specs/50-engine-provisioner.md`, `.specs/55-model-provisioner.md`, `.specs/56-engine-catalog.md`
- Catalog-core EngineEntry promotion: `catalog-core/.specs/10_contracts.md`
- Spec alignment plan: `SPEC_CHANGES_NEEDED.md`

## 0. Workspace & Build System
- [x] Add new crates to workspace `Cargo.toml` members: `adapter-host`, `worker-adapters/http-util`, `observability/narration-core` (ORCH-3620, ORCH-3610)
- [x] Plan private crate `auth-min/` and add to workspace members (timing-safe compare, header parse, loopback bypass, proxy trust) (AUTH-1xxx)
- [x] Ensure features for engine-provisioner providers: `provider-llamacpp`, `provider-vllm`, `provider-tgi`, `provider-triton` (ORCH-3430)
- [x] Update `xtask`/CI scripts to include new crates in fmt/clippy/tests (README_LLM workflow)
- [x] Update `tools/readme-index` map to include new crates
- [ ] Add new crate to workspace members: `container-runtime` (root-level crate)
- [ ] Update `xtask`/CI to include `container-runtime` in fmt/clippy/tests

## 1. New Library Crates (Scaffolding)
- [x] `adapter-host/` (lib): registry, facade (submit/cancel/health/props), narration & metrics wrappers (ORCH-3600..3604)
- [x] `worker-adapters/http-util/` (lib): shared reqwest::Client, retry/backoff, streaming decode helpers, redaction (ORCH-3610..3613)
- [x] `observability/narration-core/` (lib): minimal narration helper + test capture (ORCH-3300..3308)
- [x] `auth-min/` (lib, private): minimal auth decisions shared by server/worker/CLI (timing-safe compare, token fp6, loopback bypass, TRUST_PROXY_AUTH gate) (AUTH-1xxx)
- [ ] `container-runtime/` (lib): detect Podman (preferred) → Docker (fallback), NVIDIA toolkit preflight, pull-by-digest, run/stop with device masks and port mapping; feature-gated backends

## 2. Orchestratord
- [x] Integrate `adapter-host` facade for adapter calls (ORCH-3620)
- [x] SSE encoder uses buffered writer; optional micro-batch flag (off by default) (ORCH-3400)
- [ ] Per-task CancellationToken end-to-end (admission → dispatch → SSE) (see `.specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md`)
- [ ] Cancel-on-disconnect for SSE (proposal ref: token streaming & cancel robustness)
- [ ] Bounded backpressure in streaming service (proposal ref)
- [ ] Optional SSE heartbeats; surface micro-batching flag in docs (proposal ref)
- [ ] Narration co-streaming policy: emit `narration` frames at start/cancel/end by default; NEVER per-token; optional periodic cadence (proposal ref)
- [ ] Adapter streaming decode adoption path via adapter-host (proposal ref)
- [ ] Placement prefilter by feasibility (ctx_max, VRAM, compute, quantization, extensions) (ORCH-3411)
- [x] PlacementDecision cache keyed by `(job_spec_hash, snapshot_version, policy)` + TTL (ORCH-3410)
- [ ] Route all placement decisions through `orchestrator-core::policy::decide` (ORCH-3960)
- [ ] Enforce auth/policy gates on `TaskRequest.placement`; pass sanitized overrides into core (ORCH-3961)
- [ ] Log `DecisionLog { filters_applied[], tie_breakers_applied[], pinned, fallback_used, candidates_considered }` and emit placement metrics (ORCH-3962)
- [ ] API override to pin model/engine to specific GPU/pool (placement override) (see placement preferences)
- [ ] Auto-provision engines at startup/first-use via engine-provisioner; conform to Arch/CachyOS package policy in UX/docs
- [ ] HTTP/2 preferred for SSE with fallback to HTTP/1.1 (ORCH-3450)
- [ ] SSE headers on live stream: `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no` (advisory) (proposal ref)
- [ ] Metrics pre-registration; throttle per-token histograms (ORCH-3460)
- [x] `/v1/capabilities` backed by host capability snapshot cache (ORCH-3603)
- [ ] `/v1/capabilities` includes `engine_version` per pool (capabilities enrichment)
- [x] Narration logs for admission/cancel; reconcile `decode_time_ms` (ORCH-3300, ORCH-3310) — placement/stream narration pending
- [x] Minimal Auth Hooks: refuse startup on non-loopback bind without `AUTH_TOKEN`; loopback bypass when `AUTH_OPTIONAL=true`; optional `TRUST_PROXY_AUTH` gate (AUTH-1002/1004/1006)
- [x] Minimal Auth Middleware: parse `Authorization: Bearer <token>`, timing-safe compare, add `identity=localhost|token:<fp6>` in logs; never log full tokens (AUTH-1001/1007/1008)
- [x] Minimal Auth Error Mapping: 40101 MISSING_TOKEN, 40102 BAD_TOKEN (scaffold), 40301 reserved; JSON envelope wired (aligning) — extend mapping later
- [x] Config wiring: support `AUTH_TOKEN`, `AUTH_OPTIONAL`, `TRUST_PROXY_AUTH` alongside `ORCHD_ADDR`; defaults retain current bind
- [x] Worker registration path requires token (server-side check) and logs `identity` breadcrumbs (AUTH-1003/1008)

## 3. Orchestrator-core
- [x] Property tests for queue invariants and policies (Reject/Drop-LRU) (ORCH-3250)
- [ ] Placement feasibility/tie-break determinism tests (ORCH-3251, ORCH-3252)

## 4. Pool-managerd
- [x] Registry: add `engine_version`, `engine_digest`, `engine_catalog_id`, `device_mask`, `slots_total/free`, perf hints (ORCH-3420)
- [ ] Enforce readiness after model present + engine ensured + health pass (ORCH-3421)
- [ ] Narration logs on preload/reload/drain and readiness flips (ORCH-3490)
- [ ] Optional metrics scrape (align with metrics-contract)
- [x] GPU-only Enforcement: preflight asserts CUDA/device availability; never spawn CPU inference; fail fast on insufficiency (GPUs only)
- [ ] Use `container-runtime` to start/stop engines in container mode; prefer rootless Podman, fallback Docker
- [ ] Health checks cover container-based engines before flipping `ready=true`
- [ ] Registration: include Bearer token to orchestrator; handle 401/403 with clear backoff and operator hints (AUTH-1003)

## 5. Provisioners
### Engine-provisioner
- [x] Feature-gate providers; llama.cpp provider first (ORCH-3430)
- [x] Fail fast if CUDA unavailable; NO CPU fallback (ORCH-3431, ORCH-3437) — ccache/CUDA hint caching later
- [ ] Emit `PreparedEngine` summary (version, build_ref, digest, flags, mode, binary_path, engine_catalog_id) (ORCH-3432)
- [ ] Write `EngineEntry` after successful ensure/build (ORCH-3441)
- [x] Respect Arch/CachyOS policy for installs when explicitly allowed (ORCH-3434)
- [ ] Implement container provider using `container-runtime` with NVIDIA toolkit preflight and digest pinning (home-profile)
- [ ] `PreparedEngine` includes identity metadata (engine_version, digest, image) and `engine_catalog_id` when available (home-profile)

### Model-provisioner
- [ ] Parallel staging; enable `HF_HUB_ENABLE_HF_TRANSFER=1` when using HF CLI (ORCH-3435)
- [ ] Atomic catalog registration to Active; digest verification when provided (ORCH-3436)

## 6. Catalog-core
- [ ] EngineEntry index support (separate file), atomic writes (Promoted) (ORCH-3440)
- [ ] Helpers for `exists(id|ref)` and `locate(ModelRef)` (Refinement)
- [ ] Tests for EngineEntry round-trip & incompatible schema rejection

## 7. Worker Adapters
- [x] `http-util` shared client and helpers added (redaction, bearer injection); adapter adoption pending (ORCH-3610..3612)
- [ ] Enforce timeouts/retries with jitter; consistent error taxonomy mapping (ORCH-3276, ORCH-3275)
- [ ] Streaming order `started → token* → end`; low-alloc token decode path (ORCH-3274)
- [ ] Determinism signals: `engine_version`, `sampler_profile_version` (when applicable), `model_digest` (ORCH-3277)
- [ ] Redact secrets in logs (ORCH-3278, ORCH-3613)
- [ ] Worker to orchestrator: include Bearer token when configured; handle 401/403 with actionable errors (AUTH-1003)
- [ ] Update adapter READMEs with High/Mid/Low behavior and links to specs (doc_style)

## 8. Test Harnesses
- [ ] BDD: ensure only cross-crate scenarios; step registry rejects unknown/ambiguous (ORCH-3279)
- [ ] BDD: scenarios for placement overrides (pin/prefer/avoid/mask with/without fallback) and priority dispatch
- [x] BDD harness builds: World derives compile under cucumber via manual Debug redaction (no `AppState: Debug` requirement)
- [ ] Determinism suite: seed corpus, record engine_version/engine_digest; byte-exact streams (ORCH-3280)
- [ ] E2E Haiku: REQUIRE_REAL_LLAMA gating; SSE transcripts; cleanup (ORCH-3281)
- [ ] Chaos: fault inject at adapters and pool-managerd supervision hooks (as designed)
- [ ] Metrics-contract: orchestrator scrape; optional pool-managerd scrape; linter green (ORCH-3282)
- [ ] BDD auth-min scenarios: loopback + AUTH_OPTIONAL=true (allow), loopback + AUTH_OPTIONAL=false (401), non-loopback w/o token (startup refusal), wrong token (401), correct token (200 + identity), worker registration w/o token (401)

## 9. Contracts & Metrics
- [ ] OpenAPI: add `PlacementMode`/`PlacementOverrides` and optional `TaskRequest.placement`; mirror in `contracts/api-types`; run regen tasks (ORCH-3980)
- [ ] Metrics contract: names/labels in `.specs/metrics/otel-prom.md` + `ci/metrics.lint.json` (ORCH-3460)
- [ ] Add placement metrics: `placement_decisions_total{outcome, pinned, fallback}` and `placement_candidates_considered`; optional `predicted_end_ms` histogram; ensure `ci/metrics.lint.json` passes
- [ ] Add ETag/If-Modified-Since and cursors to catalog endpoints (ORCH-3470)
- [ ] Error taxonomy: ensure new auth error codes (40101/40102/40301) are wired to provider verify tests and snapshots

## 10. CI & Tooling
- [ ] Add SSE emitter micro-bench; adapter streaming throughput tests (ORCH-6: CI benches)
- [ ] Expand CI matrix to run crate-local tests, BDD subset, determinism smoke, metrics lint (ORCH-3210..3213)
- [ ] Narration coverage stat in BDD (informational first) (ORCH-3307)
- [ ] Update `cargo xtask dev:loop` to include regen + new crates (README_LLM)
- [ ] Add auth-min check in CI: orchestrator refuses non-loopback w/o token; basic auth middleware unit tests

## 10.1 Verification Commands (run locally before PR)
- [ ] Format/lint: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] Workspace tests: `cargo test --workspace --all-features -- --nocapture`
- [ ] Orchestrator provider verify: `cargo test -p orchestratord --test provider_verify -- --nocapture`
- [ ] BDD smoke: `cargo test -p test-harness-bdd --test bdd -- --nocapture`
- [ ] Determinism suite: `cargo test -p test-harness-determinism-suite`
- [ ] Metrics lints: `cargo test -p test-harness-metrics-contract` (or `ci/scripts` linter)

## 11. Observability & Narration
- [ ] Add narration-core; integrate in orchestrator submit/cancel and placement (ORCH-3300..3306)
- [ ] Provisioners & pool-managerd adopt narration for preflight/build/fail-fast/spawn/readiness (ORCH-3490)
- [ ] Redaction helpers applied to secrets; reconcile `decode_time_ms` naming (ORCH-3310)

## 12. Docs & READMEs
- [ ] Propagate High/Mid/Low behavior sections across crate READMEs (doc_style)
- [x] README wiring diagrams reflect adapter-host + http-util + capability cache (keep up to date)
- [ ] Document Arch/CachyOS package policy in provisioners & README (env prefs)

## 16. CLI & Client (llama-orch-cli, consumer-tests)
- [x] CLI: support `--addr`, `--auth-token` flags and read `AUTH_TOKEN` env; default to loopback in examples; never print full tokens (mask; show fp6)
- [ ] CLI: include Authorization header when token configured; handle 401/403 with actionable messages
- [ ] consumer-tests: add cases for auth-min flows (happy path, missing/bad token) and identity fingerprint presence

## 17. Runtime Config Surfaces (Alignment)
- [ ] Normalize env/config across crates: `ORCHD_ADDR`, `AUTH_TOKEN`, `AUTH_OPTIONAL`, `TRUST_PROXY_AUTH`
- [ ] Precedence rules: CLI flag > env > config file; document in crate READMEs (runtime-facing only)
- [ ] Ensure secrets redaction in logs for all crates that may log env/config

## 13. Security & Policy
- [ ] No secrets in logs; redact API keys (adapters) by default (ORCH-3613)
- [ ] System package manager policy: Arch/CachyOS pacman/AUR when allowed (ORCH-3434)

## 13.1 Arch/CachyOS Ops (Runtime-facing tasks)
- [ ] Provisioners: when `allow_package_installs=true`, prefer pacman/AUR; otherwise emit actionable guidance with exact pacman/AUR commands
- [ ] Document in provisioners README the pacman/AUR commands (runtime docs only; not spec) and environment hints for CUDA toolchain
- [ ] Ensure CI readme mentions Arch/CachyOS prerequisites (driver/CUDA) and how to skip installs when offline

## 14. Proof Bundles & Artifacts
- [ ] Include EngineEntry snapshots, PreparedEngine metadata, SSE transcripts, determinism outputs, and metrics lints per `.docs/testing/` guidance

## 15. Acceptance Criteria (Roll-up)
- [ ] Perf: SSE CPU reduced ≥15–30%; adapter throughput improved; provision rebuilds faster with ccache/CUDA hints (ORCH-34xx)
- [ ] Tests: per-crate hardening in place; outer harnesses remain integration-only (ORCH-325x)
- [ ] Narration: emitted across key flows; coverage stat present; `decode_time_ms` consistent (ORCH-33xx)
- [ ] Engine catalog: EngineEntry created and referenced; registry fields surfaced and logged (ORCH-3440/3441)
- [ ] GPU-only: no code path performs CPU inference; CUDA/device checks fail fast with clear diagnostics
- [ ] Minimal Auth: startup refusal on non-loopback without token; loopback bypass honored with `AUTH_OPTIONAL=true`; correct token passes; logs carry `identity` without leaking full tokens

## 18. Scaffold Targets by Crate (File/Module-Level TODOs)
- **orchestrator-core/**
  - [ ] `src/policy.rs` — implement `decide`; helpers for feasibility, overrides, scoring/tie-breaks; unit/property tests
- **orchestratord/**
  - [ ] `src/app/auth_min.rs` — minimal auth middleware (header parse, timing-safe compare, identity breadcrumb)
  - [ ] `src/app/bootstrap.rs` — startup check: refuse non-loopback without `AUTH_TOKEN`
  - [ ] `src/app/router.rs` — wire middleware; ensure worker registration path checks token
  - [ ] `src/api/data.rs` SSE buffering; `services/streaming.rs` micro-batch flag
  - [ ] Admission/placement: call `orchestrator-core::policy::decide`; map `TaskRequest.placement` → `JobSpec.placement`; log DecisionLog; emit placement metrics
  - [ ] `src/state.rs` capability snapshot cache plumbing
- **pool-managerd/**
  - [ ] `src/preflight.rs` — CUDA/device checks (GPU-only gate)
  - [ ] `src/registry.rs` — add engine fields; token on registration
  - [ ] `src/observability.rs` — narration additions (fail-fast diagnostics)
- **provisioners/engine-provisioner/**
  - [ ] `src/providers/llamacpp/*.rs` — remove CPU fallback paths; add fail-fast diagnostics
  - [ ] `src/plan.rs` — prepared engine metadata; engine catalog ID return
- **worker-adapters/**
  - [ ] `http-util/` — helper to inject `Authorization` header when configured (opt-in)
  - [ ] Each adapter (`llamacpp-http`, `vllm-http`, `tgi-http`, `triton`, `openai-http`) — honor header injection and redact secrets in logs
- **cli/llama-orch-cli/**
  - [ ] `src/main.rs` — `--addr`, `--auth-token`; env read; identity masking (fp6 in logs)
- **test-harness/bdd/**
  - [ ] `src/steps/security.rs` — auth-min scenarios (loopback optional, non-loopback refusal, bad/correct token, worker registration)
  - [ ] `src/steps/pool_manager.rs` — GPU-only preflight expectations
- **tools/xtask/**
  - [x] Include new crates in fmt/clippy/tests; add auth-min CI stub in xtask

## 19. Rollout Plan & Feature Switches
- **Phase A (foundations)**: create `auth-min/` crate; wire orchestrator middleware and startup check; add CLI flags/env support; basic BDD and unit tests; no behavior change for loopback with AUTH_OPTIONAL=true.
- **Phase B (GPU-only)**: enforce CUDA/device checks in pool-managerd; remove CPU fallback in engine-provisioner providers; update adapters to assume GPU-only.
- **Phase C (performance)**: SSE buffering/micro-batch, placement prefilter/cache, metrics throttling; adopt http-util across adapters.
- **Feature toggles**: expose `AUTH_OPTIONAL` and `TRUST_PROXY_AUTH`; keep binding default unchanged for now; GPU-only has no off-switch.
