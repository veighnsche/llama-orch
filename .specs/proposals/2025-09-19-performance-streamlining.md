# Proposal: Performance Streamlining Across Crate Boundaries (Repo‑Wide)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025‑09‑19

## 0) Motivation

Reduce end‑to‑end latency, CPU overhead, and cold‑start time by streamlining hot interfaces between crates: orchestrator ↔ adapters, orchestrator ↔ core, manager ↔ provisioners ↔ catalog. Preserve clarity of boundaries and improve determinism, observability, and reproducibility.

## 1) Scope

In scope:
- SSE and adapter hot paths (serialization, HTTP usage, retries/timeouts).
- Admission/placement linkage (snapshot caching, feasibility prefilters).
- Provisioning speed (build cache reuse, CUDA hinting, staged artifacts) and an engine catalog for reproducibility.
- Pool readiness/registry fields for placement and logs.
- Metrics/logging overhead reductions consistent with contracts.

Out of scope:
- Changing public OpenAPI shapes or the WorkerAdapter trait semantics (except optional internal optimizations).

## 2) Normative Requirements (RFC‑2119)

IDs use ORCH‑34xx (performance streamlining).

### Orchestrator ↔ Adapters (SSE & HTTP)
- [ORCH‑3400] The SSE encoder in `orchestratord` MUST use a buffered writer and avoid per‑token heap allocations (e.g., `to_writer` or equivalent preallocated buffer). An optional micro‑batch mode MAY coalesce tokens within a small latency budget, disabled by default.
- [ORCH‑3401] Adapters MUST use a shared, connection‑pooled HTTP client (keep‑alive, HTTP/2 where supported) and MUST bound request timeouts and retries with jittered backoff.
- [ORCH‑3402] Adapter streaming MUST preserve `started → token* → end` ordering; optional `metrics` frames MAY appear between tokens.

### Orchestrator ↔ Core (Admission/Placement)
- [ORCH‑3410] Orchestrator MUST compute a monotonic `snapshot_version` for pool topology and MAY cache `PlacementDecision` keyed by `(job_spec_hash, snapshot_version, policy)` with a short TTL.
- [ORCH‑3411] Orchestrator SHOULD prefilter ineligible pools (ctx_max, VRAM, compute capability, quantization, extensions) before scoring, passing only feasible pools to `orchestrator-core`.

### Pool‑managerd ↔ Provisioners ↔ Catalog
- [ORCH‑3420] Pool registry entries SHOULD include `engine_version`, `engine_digest` (when available), `device_mask`, `slots_total/free`, and perf hints `perf_tokens_per_s`, `first_token_ms`.
- [ORCH‑3421] Readiness MUST be `true` only after model artifacts present, engine ensured, and health checks pass.

### Engine‑provisioner (providers, speedups)
- [ORCH‑3430] Providers MUST be feature‑gated (e.g., `provider-llamacpp`, `provider-vllm`, `provider-tgi`, `provider-triton`) to reduce builds.
- [ORCH‑3431] Llama.cpp source provider SHOULD support ccache/host compiler hints and persist CUDA toolchain discovery to avoid repeated probing.
- [ORCH‑3432] Ensure flows MUST emit a `PreparedEngine` summary (engine name/version, build ref, digest, build flags, compute mode, binary path) consumable by pool‑managerd and logs.
- [ORCH‑3433] Providers MUST NOT fallback to CPU‑only. GPU is required; failures MUST be surfaced immediately with clear diagnostics and hints.
- [ORCH‑3437] GPU‑only policy: provisioning MUST validate CUDA/device availability up front and fail fast if insufficient; no CPU inference paths are allowed.
- [ORCH‑3434] Package installation (when explicitly allowed) MUST prefer system package managers; on Arch/CachyOS use `pacman`/AUR as appropriate.

### Model‑provisioner (staging)
- [ORCH‑3435] For multi‑file models (e.g., Transformers repos), staging SHOULD be parallelized; when using HF CLI, `HF_HUB_ENABLE_HF_TRANSFER=1` SHOULD be set.
- [ORCH‑3436] Catalog registration MUST be atomic and set lifecycle to `Active` on success only; digest verification SHOULD be performed when provided.

### Engine Catalog (reproducibility)
- [ORCH‑3440] `catalog-core` MUST support an `EngineEntry` index (separate from model entries): `{ id, engine, version, build_ref, digest?, build_flags?, artifacts[], created_ms }` with atomic writes.
- [ORCH‑3441] Engine‑provisioner MUST write an `EngineEntry` on successful build/pull and return the `engine_catalog_id` as part of `PreparedEngine`.

### SSE Transport & Metrics
- [ORCH‑3450] Orchestrator SHOULD enable HTTP/2 for SSE where stable; compression SHOULD be avoided for small token frames and MAY be enabled for large frames.
- [ORCH‑3460] Metrics MUST be pre‑registered (no hot‑path label string allocations) and per‑token updates SHOULD be throttled (e.g., every N tokens or T milliseconds). Histograms MAY be sampled.

### Catalog CRUD Optimizations
- [ORCH‑3470] Catalog list/read endpoints SHOULD support ETag/If‑Modified‑Since and cursor pagination.

### Placement Predictability
- [ORCH‑3480] Orchestrator SHOULD apply a soft TTL for session affinity to reduce unnecessary rescoring and improve cache warmth.

### Human Narration (tie‑in)
- [ORCH‑3490] Provisioners and orchestrator MUST emit human‑readable narration for key steps (preflight, CUDA fallback, build, spawn, placement decision), alongside structured fields, consistent with the narration spec (ORCH‑33xx).

## 3) Design Overview

- Orchestrator: buffer SSE writes; add placement cache keyed by snapshot version; prefilter pools using feasibility masks; pre‑register metrics and throttle per‑token updates.
- Adapters: shared HTTP client utility and retry/timeouts; retain trait surface; optimize hot token events internally.
- Engine‑provisioner: feature‑gated providers; ccache & CUDA hint caching; return `PreparedEngine`; write `EngineEntry`.
- Model‑provisioner: parallel staging; HF transfer acceleration; strict atomic catalog updates.
- Pool‑managerd: registry includes engine fields & perf hints; readiness remains three‑gate.
- Catalog‑core: separate engine index for reproducible builds.

## 4) Changes by Crate

- orchestrator-core: no API change; benefits from reduced candidate pools and fewer rescoring calls.
- orchestratord: SSE buffer, pool prefilter, placement cache, optional micro‑batch, metrics pre‑registration & throttling, narration for placement.
- worker‑adapters/*: shared HTTP client/retry utility, timeouts, narrations for adapter‑visible errors, no trait change required.
- pool‑managerd: extend registry with `engine_version`, `engine_digest`, perf hints; narrations for readiness flips.
- provisioners/engine‑provisioner: providers behind features; ccache/CUDA hints; `PreparedEngine`; write `EngineEntry`.
- provisioners/model‑provisioner: parallel staging; atomic catalog writes; digest handling.
- catalog‑core: add `EngineEntry` support with atomic writes.

## 5) Migration Plan

Phase 1 (Quick wins)
- Orchestrator SSE buffering and metrics throttling.
- Shared HTTP client utility for adapters; enable HTTP/2.
- Placement prefilter & short‑TTL cache.

Phase 2 (Provisioning & Catalog)
- Feature‑gate engine providers; add ccache/CUDA hint caching.
- Emit `PreparedEngine` and write `EngineEntry`.
- Pool‑managerd logs/registry updated to include engine fields.

Phase 3 (Model staging & CRUD optimizations)
- Parallel model staging / HF transfer env.
- Catalog ETag/cursor support.

Phase 4 (Refinements)
- Session affinity TTL; optional SSE micro‑batching; narration coverage extension.

## 6) CI & Benchmarks

- Add microbench for SSE emitter (tokens/sec at different sizes) and adapter streaming throughput tests.
- Compare provision times before/after ccache/CUDA hinting; record in report artifacts.
- Keep determinism suite green; add `engine_version`/`engine_digest` to proof bundles.

## 7) Risks & Mitigations

- Micro‑batching could alter client expectations → keep off by default; guard by config.
- HTTP/2 regressions on certain runtimes → negotiate and fallback to HTTP/1.1 automatically.
- Catalog engine index adds complexity → separate file keeps model paths unchanged.

## 8) Acceptance Criteria

- SSE throughput improved (≥15–30% CPU reduction in token hot path) in synthetic tests.
- Adapter throughput improved (connection reuse) with stable tail latencies.
- Provision latency reduced on rebuilds (ccache) and repeated CUDA paths (hint cache).
- Registry entries include engine fields; narrations present for key paths.
- Engine catalog entries created with atomic writes and referenced by pool/placement logs.

## 9) Refinement Opportunities

- Shared adapter streaming decoder with zero‑copy slices.
- Learned placement hints derived from perf telemetry.
- Automatic adaptive SSE coalescing under overload.
- OAuth‑style token redaction and log audits.

## 10) Mapping to Repo Reality (Anchors)

- `orchestratord/src/http/data.rs`, `services/streaming.rs` — SSE buffering, metrics throttling, narration.
- `orchestratord/src/state.rs`, `placement.rs` — placement cache & prefilters.
- `worker-adapters/adapter-api` & new `worker-adapters/http-util` — shared client & retries.
- `provisioners/engine-provisioner/src/providers/*` — feature‑gated providers, ccache/CUDA hints, `PreparedEngine`.
- `provisioners/model-provisioner/src/lib.rs` — parallel staging & catalog writes.
- `pool-managerd/src/registry.rs` — engine fields in registry; readiness narrations.
- `catalog-core` — new `EngineEntry` index & atomic write helpers.
