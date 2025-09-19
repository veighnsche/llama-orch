# Project Checklist — Scaffolding for Proposals and Specs

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

This checklist aggregates all work to implement recent proposals and root specs. It maps to:
- Testing ownership/scope: `.specs/proposals/2025-09-19-testing-ownership-and-scope.md` (ORCH-32xx)
- Performance streamlining: `.specs/proposals/2025-09-19-performance-streamlining.md` (ORCH-34xx)
- Human narration logging: `.specs/proposals/2025-09-19-human-narration-logging.md` (ORCH-33xx)
- Adapter host + HTTP util: `.specs/proposals/2025-09-19-adapter-host-and-http-util.md` (ORCH-36xx)
- Engine/Model Provisioners & Engine Catalog: `.specs/50-engine-provisioner.md`, `.specs/55-model-provisioner.md`, `.specs/56-engine-catalog.md`
- Catalog-core EngineEntry promotion: `catalog-core/.specs/10_contracts.md`

## 0. Workspace & Build System
- [x] Add new crates to workspace `Cargo.toml` members: `adapter-host`, `worker-adapters/http-util`, `observability/narration-core` (ORCH-3620, ORCH-3610)
- [ ] Ensure features for engine-provisioner providers: `provider-llamacpp`, `provider-vllm`, `provider-tgi`, `provider-triton` (ORCH-3430)
- [ ] Update `xtask`/CI scripts to include new crates in fmt/clippy/tests (README_LLM workflow)
- [x] Update `tools/readme-index` map to include new crates

## 1. New Library Crates (Scaffolding)
- [x] `adapter-host/` (lib): registry, facade (submit/cancel/health/props), narration & metrics wrappers (ORCH-3600..3604)
- [x] `worker-adapters/http-util/` (lib): shared reqwest::Client, retry/backoff, streaming decode helpers, redaction (ORCH-3610..3613)
- [x] `observability/narration-core/` (lib): minimal narration helper + test capture (ORCH-3300..3308)

## 2. Orchestratord
- [ ] Integrate `adapter-host` facade for adapter calls (ORCH-3620)
- [ ] SSE encoder uses buffered writer; optional micro-batch flag (off by default) (ORCH-3400)
- [ ] Placement prefilter by feasibility (ctx_max, VRAM, compute, quantization, extensions) (ORCH-3411)
- [ ] PlacementDecision cache keyed by `(job_spec_hash, snapshot_version, policy)` + TTL (ORCH-3410)
- [ ] HTTP/2 preferred for SSE with fallback to HTTP/1.1 (ORCH-3450)
- [ ] Metrics pre-registration; throttle per-token histograms (ORCH-3460)
- [ ] `/v1/capabilities` backed by host capability snapshot cache (ORCH-3603)
- [ ] Narration logs for admission/placement/stream/cancel; reconcile `decode_time_ms` (ORCH-3300, ORCH-3310)

## 3. Orchestrator-core
- [ ] Property tests for queue invariants and policies (Reject/Drop-LRU) (ORCH-3250)
- [ ] Placement feasibility/tie-break determinism tests (ORCH-3251, ORCH-3252)

## 4. Pool-managerd
- [ ] Registry: add `engine_version`, `engine_digest`, `engine_catalog_id`, `device_mask`, `slots_total/free`, perf hints (ORCH-3420)
- [ ] Enforce readiness after model present + engine ensured + health pass (ORCH-3421)
- [ ] Narration logs on preload/reload/drain and readiness flips (ORCH-3490)
- [ ] Optional metrics scrape (align with metrics-contract)

## 5. Provisioners
### Engine-provisioner
- [ ] Feature-gate providers; llama.cpp provider first (ORCH-3430)
- [ ] ccache & CUDA hint caching; CPU-only fallback on repeated failures (ORCH-3431, ORCH-3433)
- [ ] Emit `PreparedEngine` summary (version, build_ref, digest, flags, mode, binary_path, engine_catalog_id) (ORCH-3432)
- [ ] Write `EngineEntry` after successful ensure/build (ORCH-3441)
- [ ] Respect Arch/CachyOS policy for installs when explicitly allowed (ORCH-3434)

### Model-provisioner
- [ ] Parallel staging; enable `HF_HUB_ENABLE_HF_TRANSFER=1` when using HF CLI (ORCH-3435)
- [ ] Atomic catalog registration to Active; digest verification when provided (ORCH-3436)

## 6. Catalog-core
- [ ] EngineEntry index support (separate file), atomic writes (Promoted) (ORCH-3440)
- [ ] Helpers for `exists(id|ref)` and `locate(ModelRef)` (Refinement)
- [ ] Tests for EngineEntry round-trip & incompatible schema rejection

## 7. Worker Adapters
- [ ] Adopt `http-util` shared client and helpers (ORCH-3610..3612)
- [ ] Enforce timeouts/retries with jitter; consistent error taxonomy mapping (ORCH-3276, ORCH-3275)
- [ ] Streaming order `started → token* → end`; low-alloc token decode path (ORCH-3274)
- [ ] Determinism signals: `engine_version`, `sampler_profile_version` (when applicable), `model_digest` (ORCH-3277)
- [ ] Redact secrets in logs (ORCH-3278, ORCH-3613)
- [ ] Update adapter READMEs with High/Mid/Low behavior and links to specs (doc_style)

## 8. Test Harnesses
- [ ] BDD: ensure only cross-crate scenarios; step registry rejects unknown/ambiguous (ORCH-3279)
- [ ] Determinism suite: seed corpus, record engine_version/engine_digest; byte-exact streams (ORCH-3280)
- [ ] E2E Haiku: REQUIRE_REAL_LLAMA gating; SSE transcripts; cleanup (ORCH-3281)
- [ ] Chaos: fault inject at adapters and pool-managerd supervision hooks (as designed)
- [ ] Metrics-contract: orchestrator scrape; optional pool-managerd scrape; linter green (ORCH-3282)

## 9. Contracts & Metrics
- [ ] OpenAPI/data & control unchanged; align provider verify with envelope mapping (ORCH-3254)
- [ ] Metrics contract: names/labels in `.specs/metrics/otel-prom.md` + `ci/metrics.lint.json` (ORCH-3460)
- [ ] Add ETag/If-Modified-Since and cursors to catalog endpoints (ORCH-3470)

## 10. CI & Tooling
- [ ] Add SSE emitter micro-bench; adapter streaming throughput tests (ORCH-6: CI benches)
- [ ] Expand CI matrix to run crate-local tests, BDD subset, determinism smoke, metrics lint (ORCH-3210..3213)
- [ ] Narration coverage stat in BDD (informational first) (ORCH-3307)
- [ ] Update `cargo xtask dev:loop` to include regen + new crates (README_LLM)

## 11. Observability & Narration
- [ ] Add narration-core; integrate in orchestrator submit/cancel and placement (ORCH-3300..3306)
- [ ] Provisioners & pool-managerd adopt narration for preflight/build/fallback/spawn/readiness (ORCH-3490)
- [ ] Redaction helpers applied to secrets; reconcile `decode_time_ms` naming (ORCH-3310)

## 12. Docs & READMEs
- [ ] Propagate High/Mid/Low behavior sections across crate READMEs (doc_style)
- [x] README wiring diagrams reflect adapter-host + http-util + capability cache (keep up to date)
- [ ] Ensure each `.specs/*` doc contains a "Refinement Opportunities" section (prefs)
- [ ] Document Arch/CachyOS package policy in provisioners & README (env prefs)

## 13. Security & Policy
- [ ] No secrets in logs; redact API keys (adapters) by default (ORCH-3613)
- [ ] System package manager policy: Arch/CachyOS pacman/AUR when allowed (ORCH-3434)

## 14. Proof Bundles & Artifacts
- [ ] Include EngineEntry snapshots, PreparedEngine metadata, SSE transcripts, determinism outputs, and metrics lints per `.docs/testing/` guidance

## 15. Acceptance Criteria (Roll-up)
- [ ] Perf: SSE CPU reduced ≥15–30%; adapter throughput improved; provision rebuilds faster with ccache/CUDA hints (ORCH-34xx)
- [ ] Tests: per-crate hardening in place; outer harnesses remain integration-only (ORCH-325x)
- [ ] Narration: emitted across key flows; coverage stat present; `decode_time_ms` consistent (ORCH-33xx)
- [ ] Engine catalog: EngineEntry created and referenced; registry fields surfaced and logged (ORCH-3440/3441)
