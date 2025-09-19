# Orchestrator SPEC — Home Lab Profile (ORCH-3xxx)

**Status:** Draft (home profile baseline)

This specification is the single source of truth for llama-orch in a home lab. It supersedes any earlier “enterprise” or “reduction” language; the requirements below are the only ones that matter. Requirement IDs remain in the `ORCH-3xxx` range so existing tests and tooling can continue to reference them.

---

## 0. Scope & Goals

* Provide deterministic, observable orchestration of NVIDIA-backed LLM workers on a single workstation with one or more GPUs. (ORCH-3000)
* Keep configuration lightweight: filesystem storage, no clustered control plane. (ORCH-3002)
* Optimise for multi-agent developer workflows: low queue latency, clear feedback, reliable catalog/artifact handling. (ORCH-3003)

---

## 1. Platform Assumptions

* Hosts are Linux, headless, with NVIDIA drivers + CUDA runtime installed. (ORCH-1101)
* Inference MUST run on NVIDIA GPUs. CPU-only or non-NVIDIA machines MAY run tooling but MUST NOT serve inference. (ORCH-1102)
* Mixed GPUs (e.g., RTX 3090 + 3060) are supported; placement rules must account for differing VRAM. (ORCH-1103)

---

## 2. Core Requirements

### 2.1 Process Model & Preload

* Workers MUST be one-model/one-device-mask processes. (ORCH-3001)
* Pools MUST preload models at startup and only report Ready after a successful preload. (ORCH-3002)
* Preload MUST fail fast on insufficient VRAM or host RAM. (ORCH-3003)

### 2.2 Admission & Queueing

* Each pool MUST expose a bounded FIFO queue with two priorities: `interactive` and `batch`. (ORCH-3004)
* Full-queue policy MUST be either `reject` or `drop-lru`; the chosen policy MUST be documented. (ORCH-3005)
* Simple per-token or per-request throttles SHOULD be configurable per API token before enqueue; defaults MAY be permissive. (ORCH-3006)
* Queue guardrails MUST reject requests that exceed model context or declared budgets before enqueue. (ORCH-3014)
* Backpressure MUST reply with HTTP 429, `Retry-After`, `X-Backoff-Ms`, and a JSON body containing `policy_label`, `retriable`, `retry_after_ms`. (ORCH-2007)

### 2.3 Placement & Scheduling

* Scheduler MUST only dispatch to Ready replicas and respect explicit device masks. (ORCH-3010 & ORCH-3011)
* Default placement heuristic MUST use least-loaded selection with VRAM awareness: prefer the GPU with the most free VRAM, then fewest active slots; tie-break deterministically (e.g., by `replica_id`). (ORCH-3012)
* Session affinity SHOULD keep a session on its previous replica when possible; failovers MUST surface `kv_warmth=false`. (ORCH-3009)

### 2.4 Sessions & Budgets

* Sessions are short-lived: default TTL ≤ 10 minutes, default max 8 turns. (ORCH-3021)
* KV cache usage MUST be bounded; metrics MUST expose KV pressure. (ORCH-3022)
* Cross-worker KV migration is disallowed; failover MUST surface `kv_migrated=false`. (ORCH-3023)
* Per-session token/time/cost budgets SHOULD be configurable; enforcement MUST occur before enqueue when enabled. (ORCH-3099)

### 2.5 Streaming & Determinism

* Each job MUST carry a unique `job_id`. (ORCH-3024)
* Cancellation MUST be race-free; no tokens may be emitted after cancel. (ORCH-3026)
* Determinism is the default: identical `{prompt, parameters, seed, sampler_profile_version, engine_version, model_digest}` on the same replica MUST yield identical token streams. (ORCH-3045)
* Replica sets MUST pin `engine_version` and `sampler_profile_version`; mixed versions are not allowed. (ORCH-3046)
* SSE framing MUST stay `started`, `token`, optional `metrics`, `end`, `error`. `started` MUST include `queue_position` and `predicted_start_ms` when available. (ORCH-2002 & ORCH-3029)

### 2.6 Catalog, Artifacts & Reloads

* Catalog APIs MUST persist model metadata locally and allow verification flows that warn (rather than fail) when signatures/SBOMs are missing. (ORCH-3037)
* Catalog storage layout MUST be documented. Filesystem directories are the default; a sqlite-backed index MAY be used. Both MUST preserve identical verification semantics and lifecycle behavior. (ORCH-3037)
* Lifecycle states are limited to `Active` and `Retired`; legacy `Draft`/`Canary` states are removed. Catalog state transitions MUST update pool readiness and related metrics consistently. (ORCH-3031, ORCH-3037)
* When checksums/digests are provided, verification MUST be performed and MUST fail pre‑admission on mismatch; otherwise verification MAY proceed with warnings. (ORCH-3037, ORCH-3097)
* Pool drain/reload MUST be atomic and reversible: reload success toggles Ready, failure rolls back. (ORCH-3031 & ORCH-3038)
* Artifact registry SHOULD expose `POST /v1/artifacts` + `GET /v1/artifacts/{id}` for plans/diffs/traces stored on local disk. (ORCH-3097 & ORCH-3098)

### 2.7 Security & Policy

* Home profile: there is no AuthN/AuthZ on the control/data planes; they are open locally. Future profiles MAY add auth behind features. (ORCH-3035)
* Minimal Auth seam (spec-only): when adopting Minimal Auth Hooks, behavior MUST align with `/.specs/11_min_auth_hooks.md` (AUTH-1001..AUTH-1008): optional loopback bypass, Bearer token on non-loopback binds, timing-safe comparisons, and identity breadcrumbs in logs. This document does not change current defaults; it references the seam for future adoption.
* Logs MUST NOT leak secrets or tokens (e.g., upstream adapter API keys). Redaction remains mandatory. (ORCH-3037)
* A lightweight policy hook MUST exist so outbound HTTP tooling can be allowed/denied per deployment. (ORCH-3080)

### 2.8 Observability

* Logs MUST include `job_id`, `session_id`, `engine`, `engine_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`, `decode_time_ms`. (ORCH-3027)
* Minimum Prometheus metrics: queue depth, tasks enqueued/started/canceled/rejected, tokens in/out, GPU util, VRAM used, optional KV cache ratio. (ORCH-3028)
* SSE `metrics` frames SHOULD include additive JSON (`queue_depth`, `on_time_probability`, `kv_warmth`, remaining budgets). (ORCH-3100)

#### 2.8.1 Human‑Readable Narration (ORCH-33xx)
* [ORCH-3300] Significant events/spans SHOULD attach a short narration string alongside structured fields under a consistent key (e.g., `human`).
* [ORCH-3302] Narration MUST NOT include secrets or PII; redaction helpers MUST be used by emitters.
* [ORCH-3303] Narration MUST work with both pretty console and JSON outputs; JSON remains the default in CI.
* [ORCH-3310] Canonicalize `decode_time_ms` as the field name; where `decode_ms` exists, implementations MUST preserve compatibility during migration.
* Proof bundles SHOULD include narration coverage excerpts and, for streams, SSE transcripts with correlation IDs.
* Emission points (normative, when events occur): admission decision, placement decision, stream start, stream end, cancel path. Emitters SHOULD include `identity` per Minimal Auth Hooks when available.

### 2.9 Capability Discovery

* The API MUST expose capability information via `GET /v1/capabilities` covering engine versions, max context, supported workloads, and declared concurrency. `GET /v1/replicasets` is removed pre‑1.0 and MUST NOT be served. (ORCH-3095)
* Capability payloads MUST include an API version compatible with OpenAPI `info.version`. (ORCH-3096)

### 2.10 Resilience & Recovery

* Driver or CUDA errors MUST mark pools Unready, trigger drains, and restart with exponential backoff. (ORCH-3038)
* VRAM OOM MUST be distinguished from host OOM; VRAM OOM SHOULD trigger capacity re-estimation. (ORCH-3039)
* Circuit breakers SHOULD shed load if SLOs are breached persistently. (ORCH-3040)

---

### 2.11 Model Selection & Auto-Fetch Policy

* The data-plane `TaskRequest.model_ref` selects the desired model artifact. (ORCH-3090)
* Supported schemes (engine-dependent) MUST include:
  * `hf:org/repo/path/to/file.gguf` — single GGUF file for llama.cpp.
  * `hf:org/repo` — full Transformers repo (config/tokenizer/safetensors) for vLLM/TGI.
  * `file:/abs/path` or `relative/path` — local file or directory (no download).
  * `https://…` — remote file/archive; engine-specific handling MAY download/extract.
  * `s3://bucket/key` — object storage (primarily Triton model repos).
  * `oci://registry/repo:tag` — OCI/NGC artifact (primarily Triton).
* If referenced artifacts are not present locally, the system SHOULD auto-fetch into its model cache when provisioning policy allows; otherwise it MUST reply with an advisory error (e.g., 503 `POOL_UNREADY`) rather than hang. (ORCH-3091)
* Auto-fetch MUST be governed by a deployment policy that can disable outbound network/tooling or require explicit allowlists. (ORCH-3080)
* Home profile (Arch/CachyOS) MAY offer opt-in package installs for required tooling (e.g., `python-huggingface-hub`) via `pacman`/AUR when `allow_package_installs=true`. (ORCH-3092)
* Pool preload MUST succeed only after the model is present and verified when checksums are provided; Ready MUST only be advertised post-preload. (ORCH-3002)

---

### 2.12 Engine Provisioning & Preflight

* The program provisions and manages engines automatically per pool; operators SHOULD NOT be required to pre-install or manually launch engines. (ORCH-3200)
* Provisioning modes (mirrors `contracts/config-schema`): (ORCH-3201)
  * `external` — orchestrator attaches to an already-running engine (no provisioning performed).
  * `source` — fetch from VCS and build (e.g., llama.cpp via git + cmake + make).
  * `container` — prefer container images for engines like vLLM/TGI/Triton.
  * `package` — install via system package manager when available.
  * `binary` — download/extract prebuilt binaries to an install dir.
* Preflight MUST check required tools and environment for the selected mode and either: (a) provision missing tools when policy allows, or (b) fail fast with actionable guidance. (ORCH-3202)
  * Typical tools: `git`, `cmake`, `make`, `gcc`; engine-specific: `nvcc` (CUDA), `huggingface-cli` (HF), `aws` (S3), `oras` (OCI). (ORCH-3203)
* Home profile Arch/CachyOS: when `allow_package_installs=true`, the system MAY install missing tools via `pacman`/AUR non-interactively when possible; otherwise it MUST instruct the operator. (ORCH-3204)
* Cache & install locations: (ORCH-3205)
  * Engine build/cache dir defaults to `~/.cache/llama-orch/<engine>` unless overridden by config.
  * Models cache defaults to `~/.cache/models` unless overridden by config.
* Outbound network/tooling MUST be policy-gated (same policy hook as §2.7 Security & Policy). Operators MUST be able to disable downloads globally. (ORCH-3206)
* Provisioning MUST produce a deterministic plan (steps) and logs suitable for inclusion in artifacts; a pool MUST only transition to Ready after successful engine provisioning and model preload. (ORCH-3207)
* GPU-only policy: inference MUST run on NVIDIA GPUs only. Provisioning and runtime MUST NOT fallback to CPU; when GPU/CUDA is unavailable, components MUST fail fast with actionable diagnostics. (ORCH-1102)

---

## 3. GPU Topology & Device Management

* Device masks are explicit; spillover is forbidden. (ORCH-3052)
* Cross-GPU tensor splits are opt-in via `tensor_split` ratios; defaults assume no split. Splits MUST respect the smallest GPU’s VRAM. (ORCH-3052)
* Driver resets/ECC events MUST be surfaced distinctly; restart storms MUST be limited with backoff. (ORCH-3053)

---

## 4. Engines & Adapters (Informative)

* llama.cpp adapter MUST implement health, properties, completion SSE, cancel, metrics. (ORCH-3054)
* vLLM adapter MUST capture engine version and expose OpenAI-compatible completion/embedding paths internally. (ORCH-3055)
* TGI adapter MUST support `/generate`, `/info`, and optional OpenAI-compatible path. (ORCH-3056)
* Triton/TensorRT-LLM adapter MUST expose health/metadata, infer (HTTP/gRPC), and metrics. (ORCH-3057)
* Mock adapter MUST remain for tests and support fault injection. (ORCH-3058)

---

## 5. Testing & Validation

* Startup self-tests MUST cover preload, minimal decode, cancel, metrics/log emission. (ORCH-3049)
* Determinism suite MUST verify byte-exact streams across replicas per engine (llama.cpp with `--parallel 1 --no-cont-batching`, others in single-slot mode). (ORCH-3050)
* Chaos/load tests SHOULD exercise drain/reload, driver resets, queue saturation, cancel. (ORCH-3051)
* Haiku E2E (anti-cheat) MUST run on a real GPU worker via OrchQueue v1 only; `REQUIRE_REAL_LLAMA=1` enforces this. (ORCH-3051)

---

## 6. Documentation & Process

* Spec changes MUST follow `.docs/PROCESS.md`. (ORCH-3030)
* Config schema is normative; unknown fields SHOULD be rejected unless explicitly marked experimental. (ORCH-3030)
* TODO tracker updates are mandatory after spec/contract/test/code changes. (ORCH-3044)


## Refinement Opportunities

* Placement heuristic research: incorporate measured decode time and KV pressure into "predicted_start_ms" modeling while preserving determinism.
* Catalog verification: optional signature/SBOM validation paths and caching policies for offline operation.
* Lifecycle flows: introduce `Retired` grace periods and background unload policies per engine.
* Auto-fetch policy: richer allowlisting (domains, protocols) and operator prompts in TTY to approve first‑time sources.
* Provisioning UX: smarter preflight on Arch/CachyOS (pacman/AUR) vs. source builds; improve failure messages with copy‑pasteable fixes.

---

> This spec intentionally omits any reference to multi-tenancy, fairness schedulers, or enterprise quotas. The home profile is the only supported deployment.
