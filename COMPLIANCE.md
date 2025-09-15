# COMPLIANCE — Requirements Coverage

### .specs/00_llama-orch.md
Total requirements: 66
- ORCH-1101 — **Inference hosts MUST have NVIDIA GPUs**; CPU‑only or non‑NVIDIA hosts **MAY** run control plane components (Controller/Scheduler) but **MUST NOT** serve i (section: 1) Platform Assumptions, level: must) — link: .specs/00_llama-orch.md#1-platform-assumptions
- ORCH-2001 — `POST /v1/tasks` [] (section: 6.2 Data Plane — OrchQueue v1, level: info) — link: .specs/00_llama-orch.md#62-data-plane-orchqueue-v1
- ORCH-2002 — The **SSE stream framing** (`started`, `token`, `metrics`, `end`, `error`) is part of the contract and MUST remain stable. [] (section: 3.16 API contracts & determinism, level: must) — link: .specs/00_llama-orch.md#316-api-contracts-determinism
- ORCH-2006 — Typed errors (authoritative):  [] (section: 6.2 Data Plane — OrchQueue v1, level: info) — link: .specs/00_llama-orch.md#62-data-plane-orchqueue-v1
- ORCH-2007 — **Backpressure headers** **SHOULD** be returned to callers. [] (section: 3.2 Queues & admission under load, level: should) — link: .specs/00_llama-orch.md#32-queues-admission-under-load
- ORCH-2008 — "engine": "llamacpp",             // "llamacpp" | "vllm" | "tgi" | "triton"  [] (section: 6.2 Data Plane — OrchQueue v1, level: info) — link: .specs/00_llama-orch.md#62-data-plane-orchqueue-v1
- ORCH-2101 — `POST /v1/pools/:id/drain` → { deadline\_ms } [] (section: 6.1 Control Plane, level: info) — link: .specs/00_llama-orch.md#61-control-plane
- ORCH-2102 — `POST /v1/pools/:id/reload` → { new\_model\_ref } [] (section: 6.1 Control Plane, level: info) — link: .specs/00_llama-orch.md#61-control-plane
- ORCH-2103 — `GET /v1/pools/:id/health` → { live, ready, draining, metrics } [] (section: 6.1 Control Plane, level: info) — link: .specs/00_llama-orch.md#61-control-plane
- ORCH-2104 — `GET /v1/replicasets` → list with load/SLO snapshots [] (section: 6.1 Control Plane, level: info) — link: .specs/00_llama-orch.md#61-control-plane
- ORCH-3001 — Workers **MUST** be one‑model/one‑device‑mask processes, pinned by config. [] (section: 3.1 Process model & preloading, level: must) — link: .specs/00_llama-orch.md#31-process-model-preloading
- ORCH-3002 — Pools **MUST** preload at `serve` start and delay `Ready` exposure until success. [] (section: 3.1 Process model & preloading, level: must) — link: .specs/00_llama-orch.md#31-process-model-preloading
- ORCH-3003 — Preload **MUST** fail fast if VRAM or host RAM headroom is insufficient; Pool remains **Unready** with a retry backoff. [] (section: 3.1 Process model & preloading, level: must) — link: .specs/00_llama-orch.md#31-process-model-preloading
- ORCH-3004 — Each Pool **MUST** have a bounded FIFO queue. [] (section: 3.2 Queues & admission under load, level: must) — link: .specs/00_llama-orch.md#32-queues-admission-under-load
- ORCH-3005 — Full queue policy **MUST** be one of: `reject`, `drop-lru` (oldest enqueued), or `shed-low-priority`. [] (section: 3.2 Queues & admission under load, level: must) — link: .specs/00_llama-orch.md#32-queues-admission-under-load
- ORCH-3006 — Per‑client **rate limits** and **burst buckets** **SHOULD** be enforced before enqueue. [] (section: 3.2 Queues & admission under load, level: should) — link: .specs/00_llama-orch.md#32-queues-admission-under-load
- ORCH-3007 — Pools with identical `{engine, model, quant, ctx, engine_version, sampler_profile_version}` **MUST** be treated as a replica set. [] (section: 3.3 Replicas, placement, and affinity, level: must) — link: .specs/00_llama-orch.md#33-replicas-placement-and-affinity
- ORCH-3008 — Scheduler **MUST** place on the **least‑loaded Ready** replica, respecting device masks and node/zone anti‑affinity if configured. [] (section: 3.3 Replicas, placement, and affinity, level: must) — link: .specs/00_llama-orch.md#33-replicas-placement-and-affinity
- ORCH-3009 — **Session affinity** **SHOULD** keep a session on its last good replica for KV reuse; on failure, **MAY** fail over with `kv_warmth=false` surfaced to telemetry (section: 3.3 Replicas, placement, and affinity, level: should) — link: .specs/00_llama-orch.md#33-replicas-placement-and-affinity
- ORCH-3010 — A Job **MUST NOT** be dispatched until the Pool is `Ready`. [] (section: 3.4 Placement, heterogeneity & readiness, level: must) — link: .specs/00_llama-orch.md#34-placement-heterogeneity-readiness
- ORCH-3011 — Placement **MUST** respect device masks; cross‑mask spillover **MUST NOT** occur. [] (section: 3.4 Placement, heterogeneity & readiness, level: must) — link: .specs/00_llama-orch.md#34-placement-heterogeneity-readiness
- ORCH-3012 — **Heterogeneous multi‑GPU splits** (across GPUs of different VRAM/compute) **MUST** be **opt‑in** with explicit per‑GPU ratios (e.g., `tensor_split: [0.67 (section: 3.4 Placement, heterogeneity & readiness, level: must) — link: .specs/00_llama-orch.md#34-placement-heterogeneity-readiness
- ORCH-3013 — **NUMA/PCIe topology hints** **SHOULD** influence placement when available. [] (section: 3.4 Placement, heterogeneity & readiness, level: should) — link: .specs/00_llama-orch.md#34-placement-heterogeneity-readiness
- ORCH-3014 — Requested context length **MUST** be ≤ model limit; otherwise **reject before enqueue** (`HTTP 400`). [] (section: 3.5 Capacity, NVIDIA‑only & guardrails, level: must) — link: .specs/00_llama-orch.md#35-capacity-nvidiaonly-guardrails
- ORCH-3015 — **Token budget** (prompt + generation) **MUST** be validated pre‑admission. [] (section: 3.5 Capacity, NVIDIA‑only & guardrails, level: must) — link: .specs/00_llama-orch.md#35-capacity-nvidiaonly-guardrails
- ORCH-3016 — Watchdog **MUST** abort stuck Jobs with configurable **wall** and **idle** timeouts. [] (section: 3.5 Capacity, NVIDIA‑only & guardrails, level: must) — link: .specs/00_llama-orch.md#35-capacity-nvidiaonly-guardrails
- ORCH-3017 — `cancel` **MUST** free the Worker slot and return a terminal state. [] (section: 3.5 Capacity, NVIDIA‑only & guardrails, level: must) — link: .specs/00_llama-orch.md#35-capacity-nvidiaonly-guardrails
- ORCH-3018 — **CPU spillover is disallowed** for inference: on GPU/Pool outage, Controller **MUST** fail fast (`POOL_UNAVAILABLE`) with retry hints. [] (section: 3.5 Capacity, NVIDIA‑only & guardrails, level: must) — link: .specs/00_llama-orch.md#35-capacity-nvidiaonly-guardrails
- ORCH-3019 — If an Engine supports **continuous batching**, Workers **SHOULD** expose total slot count and per‑slot state; the Scheduler **MUST** factor this into placemen (section: 3.6 Batching & throughput (engine‑agnostic), level: must) — link: .specs/00_llama-orch.md#36-batching-throughput-engineagnostic
- ORCH-3020 — **Speculative decoding**/**prefix caching** **MAY** be enabled; admission control **MUST** account for memory impact. [] (section: 3.6 Batching & throughput (engine‑agnostic), level: must) — link: .specs/00_llama-orch.md#36-batching-throughput-engineagnostic
- ORCH-3021 — Sessions are **short‑lived**; enforce TTL ≤ **10 minutes** and/or max **8 turns**. [] (section: 3.7 Sessions & KV (short‑lived), level: info) — link: .specs/00_llama-orch.md#37-sessions-kv-shortlived
- ORCH-3022 — KV cache residency **MUST** be bounded with LRU/LFU; expose pressure metrics. [] (section: 3.7 Sessions & KV (short‑lived), level: must) — link: .specs/00_llama-orch.md#37-sessions-kv-shortlived
- ORCH-3023 — **Cross‑Worker KV migration is disabled**; failover surfaces `kv_migrated=false`. [] (section: 3.7 Sessions & KV (short‑lived), level: info) — link: .specs/00_llama-orch.md#37-sessions-kv-shortlived
- ORCH-3024 — Each Job **MUST** carry a unique `job_id`. [] (section: 3.8 Cancellations, retries, idempotency, level: must) — link: .specs/00_llama-orch.md#38-cancellations-retries-idempotency
- ORCH-3025 — Client retries **SHOULD** target retryable error classes only. [] (section: 3.8 Cancellations, retries, idempotency, level: should) — link: .specs/00_llama-orch.md#38-cancellations-retries-idempotency
- ORCH-3026 — Cancellation **MUST** be race‑free (no post‑cancel tokens). [] (section: 3.8 Cancellations, retries, idempotency, level: must) — link: .specs/00_llama-orch.md#38-cancellations-retries-idempotency
- ORCH-3027 — Logs **MUST** include: `job_id`, `session_id`, `client_id`, `engine`, `pool_id`, `replica_id`, `model_id`, `quant`, `ctx`, `tensor_split`, `engine_version`, `sa (section: 3.9 Observability & telemetry, level: must) — link: .specs/00_llama-orch.md#39-observability-telemetry
- ORCH-3028 — Metrics **MUST** include: queue depth, reject/drop rates, p50/p95/p99 latency, GPU/VRAM/RAM utilization, KV pressure, preload outcomes, per‑priority SLO attai (section: 3.9 Observability & telemetry, level: must) — link: .specs/00_llama-orch.md#39-observability-telemetry
- ORCH-3029 — Admission logs and the `started` SSE event **MUST** include `queue_position` and `predicted_start_ms` when available. [] (section: 3.9 Observability & telemetry, level: must) — link: .specs/00_llama-orch.md#39-observability-telemetry
- ORCH-3030 — Config **MUST** be schema‑validated; unknown fields rejected (strict) or logged (compat). [] (section: 3.10 Config & lifecycle, level: must) — link: .specs/00_llama-orch.md#310-config-lifecycle
- ORCH-3031 — **Hot‑reload** **MAY** be supported but **MUST** be atomic and revertible; rolling changes use drain+replace. [] (section: 3.10 Config & lifecycle, level: must) — link: .specs/00_llama-orch.md#310-config-lifecycle
- ORCH-3032 — Workers **MUST** report `engine_version` and `model_digest` (or equivalent). [] (section: 3.11 Rollouts & versioning, level: must) — link: .specs/00_llama-orch.md#311-rollouts-versioning
- ORCH-3033 — **Canaries** **SHOULD** be supported via labels/percent splits; **Rollbacks** **MUST** be one action. [] (section: 3.11 Rollouts & versioning, level: must) — link: .specs/00_llama-orch.md#311-rollouts-versioning
- ORCH-3035 — **AuthN/AuthZ** **MUST** gate control/data paths; API keys acceptable day‑1. [] (section: 3.12 Security & tenancy, level: must) — link: .specs/00_llama-orch.md#312-security-tenancy
- ORCH-3036 — Per‑tenant quotas **MUST** bound concurrent Jobs and memory (KV + scratch). [] (section: 3.12 Security & tenancy, level: must) — link: .specs/00_llama-orch.md#312-security-tenancy
- ORCH-3037 — Model artifacts **MUST** be checksummed and verified before load. [] (section: 3.12 Security & tenancy, level: must) — link: .specs/00_llama-orch.md#312-security-tenancy
- ORCH-3038 — **Driver/CUDA errors** **MUST** transition the Pool to `Unready`, drain, and backoff‑restart. [] (section: 3.13 Resilience & recovery, level: must) — link: .specs/00_llama-orch.md#313-resilience-recovery
- ORCH-3039 — Distinguish **VRAM OOM** vs **host OOM**; VRAM OOM **SHOULD** trigger capacity re‑estimation. [] (section: 3.13 Resilience & recovery, level: should) — link: .specs/00_llama-orch.md#313-resilience-recovery
- ORCH-3040 — **Circuit breakers** **SHOULD** shed load on sustained SLO violations. [] (section: 3.13 Resilience & recovery, level: should) — link: .specs/00_llama-orch.md#313-resilience-recovery
- ORCH-3041 — Per‑priority SLOs **MUST** be defined and measured continuously; alerts on breach. [] (section: 3.14 Performance & SLOs, level: must) — link: .specs/00_llama-orch.md#314-performance-slos
- ORCH-3042 — Model storage **MUST** verify checksums; local caches **SHOULD** enforce quotas and eviction. [] (section: 3.15 Storage & integrity, level: must) — link: .specs/00_llama-orch.md#315-storage-integrity
- ORCH-3044 — Public APIs **MUST** be versioned. [] (section: 3.16 API contracts & determinism, level: must) — link: .specs/00_llama-orch.md#316-api-contracts-determinism
- ORCH-3045 — **Deterministic outputs are the default.** Same `{prompt, parameters, seed, sampler_profile_version, engine_version, model_digest}` within a replica set **MUST* (section: 3.16 API contracts & determinism, level: must) — link: .specs/00_llama-orch.md#316-api-contracts-determinism
- ORCH-3046 — Replica sets **MUST** pin `engine_version` and `sampler_profile_version`; replicas with differing values **MUST NOT** mix. [] (section: 3.16 API contracts & determinism, level: must) — link: .specs/00_llama-orch.md#316-api-contracts-determinism
- ORCH-3047 — Determinism **MUST NOT** be assumed across engine/model updates. [] (section: 3.16 API contracts & determinism, level: must) — link: .specs/00_llama-orch.md#316-api-contracts-determinism
- ORCH-3048 — Default plugin ABI **MUST** be **WASI** (pure, deterministic functions over explicit snapshots). Native in‑process trait **MAY** be offered as a high‑perf a (section: 3.17 Extensibility (policy plugins), level: must) — link: .specs/00_llama-orch.md#317-extensibility-policy-plugins
- ORCH-3049 — Startup self‑tests **MUST** cover: preload, minimal decode, cancel, telemetry emission. [] (section: 3.18 Testing & validation, level: must) — link: .specs/00_llama-orch.md#318-testing-validation
- ORCH-3050 — Determinism tests **MUST** verify byte‑exact streams across replicas with fixed seeds, run **per engine** with engine‑appropriate settings (e.g., llama.cpp  (section: 3.18 Testing & validation, level: must) — link: .specs/00_llama-orch.md#318-testing-validation
- ORCH-3051 — Chaos tests **SHOULD** be run for failover/idempotency; load tests **MUST** cover priority inversion. [] (section: 3.18 Testing & validation, level: must) — link: .specs/00_llama-orch.md#318-testing-validation
- ORCH-3052 — **Heterogeneous split policy**: when ratios are provided, **MUST** cap per‑GPU resident KV to avoid OOM on the smallest GPU; Scheduler **SHOULD** consider PCI (section: 4) NVIDIA Topology & Heterogeneity, level: must) — link: .specs/00_llama-orch.md#4-nvidia-topology-heterogeneity
- ORCH-3053 — **Driver resets** and ECC (where applicable) **MUST** be surfaced distinctly; the Pool Manager **MUST** bound restart storms with backoff and circuit breakers.  (section: 4) NVIDIA Topology & Heterogeneity, level: must) — link: .specs/00_llama-orch.md#4-nvidia-topology-heterogeneity
- ORCH-3054 — **Adapter contract**: MUST support health, properties (slots/commit), completion (SSE), cancel, and metrics scrape. This mapping is internal; the public API is  (section: 5.1 llama.cpp server (HTTP), level: must) — link: .specs/00_llama-orch.md#51-llamacpp-server-http
- ORCH-3055 — **Adapter contract**: same as above; engine version string MUST be captured. The adapter may call vLLM’s OpenAI server internally; this is not exposed publicl (section: 5.2 vLLM (OpenAI‑compatible server), level: must) — link: .specs/00_llama-orch.md#52-vllm-openaicompatible-server
- ORCH-3056 — **Adapter contract**: MUST implement the TGI custom path (`/generate`, `/info`, etc.). If an OpenAI‑compatible path is enabled, it remains internal only. [] (section: 5.3 Hugging Face Text‑Generation‑Inference (TGI), level: must) — link: .specs/00_llama-orch.md#53-hugging-face-textgenerationinference-tgi
- ORCH-3057 — **Adapter contract**: MUST support health/metadata, infer/streaming (if configured), and metrics. OpenAI‑compatible frontends are internal only. [] (section: 5.4 NVIDIA Triton / TensorRT‑LLM, level: must) — link: .specs/00_llama-orch.md#54-nvidia-triton-tensorrtllm
- ORCH-3058 — > All adapters **MUST** normalize detokenization templates and sampler profiles to keep determinism stable within a replica set. [] (section: 5.4 NVIDIA Triton / TensorRT‑LLM, level: must) — link: .specs/00_llama-orch.md#54-nvidia-triton-tensorrtllm

### .specs/10-orchestrator-core.md
Total requirements: 17
- OC-CORE-1001 — - [] Each Pool MUST expose a bounded FIFO queue per priority class. (section: 1) Queue & Admission, level: must) — link: .specs/10-orchestrator-core.md#1-queue-admission
- OC-CORE-1002 — - [] Admission MUST reject when the queue is full according to configured policy (reject/drop-lru/shed-low-priority). (section: 1) Queue & Admission, level: must) — link: .specs/10-orchestrator-core.md#1-queue-admission
- OC-CORE-1003 — - [] Enqueue MUST be O(1) amortized and MUST preserve request arrival order within the same priority. (section: 1) Queue & Admission, level: must) — link: .specs/10-orchestrator-core.md#1-queue-admission
- OC-CORE-1004 — - [] Dequeue MUST prefer higher priority and MUST be fair within a priority class. (section: 1) Queue & Admission, level: must) — link: .specs/10-orchestrator-core.md#1-queue-admission
- OC-CORE-1005 — - [] Cancellation MUST remove the task from the queue or mark the slot so it is not dispatched. (section: 1) Queue & Admission, level: must) — link: .specs/10-orchestrator-core.md#1-queue-admission
- OC-CORE-1010 — - [] Scheduler MUST dispatch only to Ready replicas. (section: 2) Scheduling & Placement, level: must) — link: .specs/10-orchestrator-core.md#2-scheduling-placement
- OC-CORE-1011 — - [] Placement MUST respect device masks; cross‑mask spillover MUST NOT occur. (section: 2) Scheduling & Placement, level: must) — link: .specs/10-orchestrator-core.md#2-scheduling-placement
- OC-CORE-1012 — - [] Least‑loaded placement MUST be used across replicas of the same replica set. (section: 2) Scheduling & Placement, level: must) — link: .specs/10-orchestrator-core.md#2-scheduling-placement
- OC-CORE-1013 — - [] Session affinity SHOULD keep a session on its last good replica when possible. (section: 2) Scheduling & Placement, level: should) — link: .specs/10-orchestrator-core.md#2-scheduling-placement
- OC-CORE-1020 — - [] Context length MUST be ≤ model limit; otherwise reject before enqueue. (section: 3) Capacity & Guardrails, level: must) — link: .specs/10-orchestrator-core.md#3-capacity-guardrails
- OC-CORE-1021 — - [] Token budget (prompt + generation) MUST be validated pre‑admission. (section: 3) Capacity & Guardrails, level: must) — link: .specs/10-orchestrator-core.md#3-capacity-guardrails
- OC-CORE-1022 — - [] Watchdog MUST abort stuck Jobs with configurable wall/idle timeouts. (section: 3) Capacity & Guardrails, level: must) — link: .specs/10-orchestrator-core.md#3-capacity-guardrails
- OC-CORE-1030 — - [] Within a replica set, identical {prompt, parameters, seed, sampler_profile_version, engine_version, model_digest} MUST yield identical token streams. (section: 4) Determinism, level: must) — link: .specs/10-orchestrator-core.md#4-determinism
- OC-CORE-1031 — - [] Replica sets MUST pin engine_version and sampler_profile_version; mixed replicas MUST NOT share a set. (section: 4) Determinism, level: must) — link: .specs/10-orchestrator-core.md#4-determinism
- OC-CORE-1032 — - [] Determinism MUST NOT be assumed across engine/model updates. (section: 4) Determinism, level: must) — link: .specs/10-orchestrator-core.md#4-determinism
- OC-CORE-1040 — - [] Logs MUST include job_id, session_id, engine, pool_id, replica_id, model_id, quant, ctx, kv_warmth, queue_time_ms, decode_time_ms. (section: 5) Observability, level: must) — link: .specs/10-orchestrator-core.md#5-observability
- OC-CORE-1041 — - [] Metrics MUST include queue depth, reject/drop rates, p50/p95/p99 latency, GPU/VRAM/RAM utilization, KV pressure, preload outcomes. (section: 5) Observability, level: must) — link: .specs/10-orchestrator-core.md#5-observability

### .specs/20-orchestratord.md
Total requirements: 16
- OC-CTRL-2001 — - [] `GET /v1/pools/:id/health` MUST return liveness, readiness, draining, and metrics snapshot fields. (section: 1) Control Plane, level: must) — link: .specs/20-orchestratord.md#1-control-plane
- OC-CTRL-2002 — - [] `POST /v1/pools/:id/drain` MUST accept a JSON body with `deadline_ms` and MUST begin draining. (section: 1) Control Plane, level: must) — link: .specs/20-orchestratord.md#1-control-plane
- OC-CTRL-2003 — - [] `POST /v1/pools/:id/reload` MUST atomically switch model references or fail and roll back. (section: 1) Control Plane, level: must) — link: .specs/20-orchestratord.md#1-control-plane
- OC-CTRL-2004 — - [] `GET /v1/replicasets` MUST enumerate replica sets with load/SLO snapshots. (section: 1) Control Plane, level: must) — link: .specs/20-orchestratord.md#1-control-plane
- OC-CTRL-2010 — - [] `POST /v1/tasks` MUST perform admission checks (ctx, token budget) before enqueue. (section: 2) Data Plane — OrchQueue v1, level: must) — link: .specs/20-orchestratord.md#2-data-plane-orchqueue-v1
- OC-CTRL-2011 — - [] On queue full, server MUST reply `429` and include `Retry-After` and `X-Backoff-Ms`. A JSON body MUST include the full policy label. (section: 2) Data Plane — OrchQueue v1, level: must) — link: .specs/20-orchestratord.md#2-data-plane-orchqueue-v1
- OC-CTRL-2012 — - [] `POST /v1/tasks/:id/cancel` MUST be race‑free; no tokens may be emitted after cancel. (section: 2) Data Plane — OrchQueue v1, level: must) — link: .specs/20-orchestratord.md#2-data-plane-orchqueue-v1
- OC-CTRL-2020 — - [] `GET /v1/tasks/:id/stream` MUST emit events `started`, `token`, `metrics`, `end`, `error`. (section: 3) SSE Framing, level: must) — link: .specs/20-orchestratord.md#3-sse-framing
- OC-CTRL-2021 — - [] `started` MUST include `queue_position` and `predicted_start_ms` when available. (section: 3) SSE Framing, level: must) — link: .specs/20-orchestratord.md#3-sse-framing
- OC-CTRL-2022 — - [] Event payloads MUST be well‑formed JSON; ordering MUST be per stream. (section: 3) SSE Framing, level: must) — link: .specs/20-orchestratord.md#3-sse-framing
- OC-CTRL-2030 — - [] Errors MUST include a stable `code` field: `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNREADY`, `POOL_UNAVAILABLE`, `REPLICA_EXHAUS (section: 4) Error Taxonomy, level: must) — link: .specs/20-orchestratord.md#4-error-taxonomy
- OC-CTRL-2031 — - [] Errors SHOULD include the `engine` and `pool_id` when applicable. (section: 4) Error Taxonomy, level: should) — link: .specs/20-orchestratord.md#4-error-taxonomy
- OC-CTRL-2040 — - [] Control and data plane MUST be gated by AuthN/AuthZ; API keys acceptable day‑1. (section: 5) Security, level: must) — link: .specs/20-orchestratord.md#5-security
- OC-CTRL-2041 — - [] Logs MUST NOT leak secrets or API keys. (section: 5) Security, level: must) — link: .specs/20-orchestratord.md#5-security
- OC-CTRL-2050 — - [] Admission logs and `started` MUST include `queue_position` and `predicted_start_ms` when available. (section: 6) Observability, level: must) — link: .specs/20-orchestratord.md#6-observability
- OC-CTRL-2051 — - [] Metrics MUST include queue depth, reject/drop rates, latency percentiles, and error counts by class. (section: 6) Observability, level: must) — link: .specs/20-orchestratord.md#6-observability

### .specs/30-pool-managerd.md
Total requirements: 9
- OC-POOL-3001 — - [] Workers MUST preload at `serve` start and MUST NOT expose Ready until success. (section: 1) Preload & Ready Lifecycle, level: must) — link: .specs/30-pool-managerd.md#1-preload-ready-lifecycle
- OC-POOL-3002 — - [] Preload MUST fail fast if VRAM/host RAM insufficient; Pool remains Unready with retry backoff. (section: 1) Preload & Ready Lifecycle, level: must) — link: .specs/30-pool-managerd.md#1-preload-ready-lifecycle
- OC-POOL-3003 — - [] Readiness endpoints MUST reflect preload state and last error cause. (section: 1) Preload & Ready Lifecycle, level: must) — link: .specs/30-pool-managerd.md#1-preload-ready-lifecycle
- OC-POOL-3010 — - [] Driver/CUDA errors MUST transition Pool to Unready, drain, and backoff‑restart. (section: 2) Restart/Backoff & Guardrails, level: must) — link: .specs/30-pool-managerd.md#2-restart-backoff-guardrails
- OC-POOL-3011 — - [] Restart storms MUST be bounded by exponential backoff and circuit breaker. (section: 2) Restart/Backoff & Guardrails, level: must) — link: .specs/30-pool-managerd.md#2-restart-backoff-guardrails
- OC-POOL-3012 — - [] CPU inference spillover is disallowed; controller MUST fail fast. (section: 2) Restart/Backoff & Guardrails, level: must) — link: .specs/30-pool-managerd.md#2-restart-backoff-guardrails
- OC-POOL-3020 — - [] Placement MUST respect device masks; no cross‑mask spillover. (section: 3) Device Masks & Placement Affinity, level: must) — link: .specs/30-pool-managerd.md#3-device-masks-placement-affinity
- OC-POOL-3021 — - [] Heterogeneous split ratios MUST be explicit and capped for smallest GPU. (section: 3) Device Masks & Placement Affinity, level: must) — link: .specs/30-pool-managerd.md#3-device-masks-placement-affinity
- OC-POOL-3030 — - [] Emit preload outcomes, VRAM/RAM utilization, driver_reset events, and restart counters. (section: 4) Observability, level: info) — link: .specs/30-pool-managerd.md#4-observability

### .specs/40-worker-adapters-llamacpp-http.md
Total requirements: 4
- OC-ADAPT-5001 — - [] Adapter MUST implement health, properties (slots/commit), completion (SSE), cancel, metrics scrape for llama.cpp native API. (section: 1) API Mapping, level: must) — link: .specs/40-worker-adapters-llamacpp-http.md#1-api-mapping
- OC-ADAPT-5002 — - [] When using OpenAI‑compatible endpoints internally, they MUST NOT be exposed publicly. (section: 1) API Mapping, level: must) — link: .specs/40-worker-adapters-llamacpp-http.md#1-api-mapping
- OC-ADAPT-5010 — - [] Adapter MUST normalize detokenization templates and sampler profiles for determinism within a replica set. (section: 2) Determinism & Version Capture, level: must) — link: .specs/40-worker-adapters-llamacpp-http.md#2-determinism-version-capture
- OC-ADAPT-5011 — - [] Adapter MUST capture and report engine_version and model_digest. (section: 2) Determinism & Version Capture, level: must) — link: .specs/40-worker-adapters-llamacpp-http.md#2-determinism-version-capture

### .specs/41-worker-adapters-vllm-http.md
Total requirements: 3
- OC-ADAPT-5020 — - [] Adapter MUST implement health/properties/completion/cancel/metrics against vLLM OpenAI‑compatible server. (section: 1) API Mapping, level: must) — link: .specs/41-worker-adapters-vllm-http.md#1-api-mapping
- OC-ADAPT-5021 — - [] OpenAI‑compatible endpoints MUST remain internal; public surface is OrchQueue v1. (section: 1) API Mapping, level: must) — link: .specs/41-worker-adapters-vllm-http.md#1-api-mapping
- OC-ADAPT-5030 — - [] Adapter MUST report engine_version and sampler/profile versions as applicable. (section: 2) Determinism & Version Capture, level: must) — link: .specs/41-worker-adapters-vllm-http.md#2-determinism-version-capture

### .specs/42-worker-adapters-tgi-http.md
Total requirements: 2
- OC-ADAPT-5040 — - [] Adapter MUST implement TGI custom endpoints (`/generate`, `/info`, `/health`) and optionally OpenAI‑compat internally; not public. (section: 1) API Mapping, level: must) — link: .specs/42-worker-adapters-tgi-http.md#1-api-mapping
- OC-ADAPT-5050 — - [] Adapter MUST capture engine_version and model info as available. (section: 2) Determinism & Version Capture, level: must) — link: .specs/42-worker-adapters-tgi-http.md#2-determinism-version-capture

### .specs/43-worker-adapters-triton.md
Total requirements: 2
- OC-ADAPT-5060 — - [] Adapter MUST support Triton HTTP/GRPC infer, health/metadata/stats, and optional OpenAI‑compat frontends internally. (section: 1) API Mapping, level: must) — link: .specs/43-worker-adapters-triton.md#1-api-mapping
- OC-ADAPT-5070 — - [] Adapter MUST report engine_version/trtllm_version where applicable. (section: 2) Determinism & Version Capture, level: must) — link: .specs/43-worker-adapters-triton.md#2-determinism-version-capture

### .specs/50-plugins-policy-host.md
Total requirements: 5
- OC-POLICY-4001 — - [] Default plugin ABI MUST be WASI; functions MUST be pure/deterministic over explicit snapshots. (section: 1) ABI & Determinism, level: must) — link: .specs/50-plugins-policy-host.md#1-abi-determinism
- OC-POLICY-4002 — - [] ABI versioning MUST be explicit; incompatible changes MUST bump MAJOR. (section: 1) ABI & Determinism, level: must) — link: .specs/50-plugins-policy-host.md#1-abi-determinism
- OC-POLICY-4010 — - [] Plugins MUST run in a sandbox with no filesystem/network unless explicitly granted. (section: 2) Sandboxing & Safety, level: must) — link: .specs/50-plugins-policy-host.md#2-sandboxing-safety
- OC-POLICY-4011 — - [] Host MUST bound CPU time/memory per invocation and abort on overuse. (section: 2) Sandboxing & Safety, level: must) — link: .specs/50-plugins-policy-host.md#2-sandboxing-safety
- OC-POLICY-4020 — - [] Host MUST log plugin id/version, decision outcome, and latency. (section: 3) Telemetry, level: must) — link: .specs/50-plugins-policy-host.md#3-telemetry

### .specs/51-plugins-policy-sdk.md
Total requirements: 3
- OC-POLICY-SDK-4101 — - [] Public SDK functions MUST be semver‑stable within a MAJOR. (section: 1) Stability & Compatibility, level: must) — link: .specs/51-plugins-policy-sdk.md#1-stability-compatibility
- OC-POLICY-SDK-4102 — - [] Breaking changes MUST be accompanied by a migration note and version bump. (section: 1) Stability & Compatibility, level: must) — link: .specs/51-plugins-policy-sdk.md#1-stability-compatibility
- OC-POLICY-SDK-4110 — - [] SDK MUST NOT perform network or filesystem I/O by default. (section: 2) Safety, level: must) — link: .specs/51-plugins-policy-sdk.md#2-safety

### .specs/60-config-schema.md
Total requirements: 3
- OC-CONFIG-6001 — - [] Config MUST be strictly validated; unknown fields rejected (strict) or logged (compat) per mode. (section: 1) Validation, level: must) — link: .specs/60-config-schema.md#1-validation
- OC-CONFIG-6002 — - [] Examples in tests MUST validate without errors. (section: 1) Validation, level: must) — link: .specs/60-config-schema.md#1-validation
- OC-CONFIG-6010 — - [] Schema generation MUST be deterministic and idempotent across runs. (section: 2) Generation, level: must) — link: .specs/60-config-schema.md#2-generation

### .specs/70-determinism-suite.md
Total requirements: 3
- OC-TEST-7001 — - [] Suite MUST verify byte‑exact token streams across replicas with fixed seeds. (section: 1) Semantics, level: must) — link: .specs/70-determinism-suite.md#1-semantics
- OC-TEST-7002 — - [] Engine‑specific settings MUST be applied (e.g., single‑slot modes) for determinism. (section: 1) Semantics, level: must) — link: .specs/70-determinism-suite.md#1-semantics
- OC-TEST-7003 — - [] Seeds corpus MUST contain at least 64 seeds and MUST be stable. (section: 1) Semantics, level: must) — link: .specs/70-determinism-suite.md#1-semantics

### .specs/71-metrics-contract.md
Total requirements: 2
- OC-METRICS-7101 — - [] Metric names and required labels MUST conform to `ci/metrics.lint.json`. (section: 1) Names/Labels, level: must) — link: .specs/71-metrics-contract.md#1-names-labels
- OC-METRICS-7102 — - [] Label cardinality budgets MUST be documented and enforced. (section: 1) Names/Labels, level: must) — link: .specs/71-metrics-contract.md#1-names-labels
