# COMPLIANCE — Requirements Coverage

### .specs/00_home_profile.md
Total requirements: 0

### .specs/00_llama-orch.md
Total requirements: 52
- ORCH-1101 — Hosts are Linux, headless, with NVIDIA drivers + CUDA runtime installed. () (section: 1. Platform Assumptions, level: info) — link: .specs/00_llama-orch.md#1-platform-assumptions
- ORCH-1102 — Inference MUST run on NVIDIA GPUs. CPU-only or non-NVIDIA machines MAY run tooling but MUST NOT serve inference. () (section: 1. Platform Assumptions, level: must) — link: .specs/00_llama-orch.md#1-platform-assumptions
- ORCH-1103 — Mixed GPUs (e.g., RTX 3090 + 3060) are supported; placement rules must account for differing VRAM. () (section: 1. Platform Assumptions, level: must) — link: .specs/00_llama-orch.md#1-platform-assumptions
- ORCH-2002 — SSE framing MUST stay `started`, `token`, optional `metrics`, `end`, `error`. `started` MUST include `queue_position` and `predicted_start_ms` when available. ( (section: 2.5 Streaming & Determinism, level: must) — link: .specs/00_llama-orch.md#25-streaming-determinism
- ORCH-2007 — Backpressure MUST reply with HTTP 429, `Retry-After`, `X-Backoff-Ms`, and a JSON body containing `policy_label`, `retriable`, `retry_after_ms`. () (section: 2.2 Admission & Queueing, level: must) — link: .specs/00_llama-orch.md#22-admission-queueing
- ORCH-3000 — Provide deterministic, observable orchestration of NVIDIA-backed LLM workers on a single workstation with one or more GPUs. () (section: 0. Scope & Goals, level: info) — link: .specs/00_llama-orch.md#0-scope-goals
- ORCH-3001 — Workers MUST be one-model/one-device-mask processes. () (section: 2.1 Process Model & Preload, level: must) — link: .specs/00_llama-orch.md#21-process-model-preload
- ORCH-3002 — Keep configuration lightweight: filesystem storage, single API token, no clustered control plane. () (section: 0. Scope & Goals, level: info) — link: .specs/00_llama-orch.md#0-scope-goals
- ORCH-3003 — Optimise for multi-agent developer workflows: low queue latency, clear feedback, reliable catalog/artifact handling. () (section: 0. Scope & Goals, level: info) — link: .specs/00_llama-orch.md#0-scope-goals
- ORCH-3004 — Each pool MUST expose a bounded FIFO queue with two priorities: `interactive` and `batch`. () (section: 2.2 Admission & Queueing, level: must) — link: .specs/00_llama-orch.md#22-admission-queueing
- ORCH-3005 — Full-queue policy MUST be either `reject` or `drop-lru`; the chosen policy MUST be documented. () (section: 2.2 Admission & Queueing, level: must) — link: .specs/00_llama-orch.md#22-admission-queueing
- ORCH-3006 — Simple per-token or per-request throttles SHOULD be configurable per API token before enqueue; defaults MAY be permissive. () (section: 2.2 Admission & Queueing, level: should) — link: .specs/00_llama-orch.md#22-admission-queueing
- ORCH-3009 — Session affinity SHOULD keep a session on its previous replica when possible; failovers MUST surface `kv_warmth=false`. () (section: 2.3 Placement & Scheduling, level: must) — link: .specs/00_llama-orch.md#23-placement-scheduling
- ORCH-3010 — Scheduler MUST only dispatch to Ready replicas and respect explicit device masks. ( & ORCH-3011) (section: 2.3 Placement & Scheduling, level: must) — link: .specs/00_llama-orch.md#23-placement-scheduling
- ORCH-3011 — Scheduler MUST only dispatch to Ready replicas and respect explicit device masks. (ORCH-3010 & ) (section: 2.3 Placement & Scheduling, level: must) — link: .specs/00_llama-orch.md#23-placement-scheduling
- ORCH-3012 — Default placement heuristic MUST prefer the GPU with the most free VRAM or available slots; heuristics MAY be extended but MUST remain deterministic. () (section: 2.3 Placement & Scheduling, level: must) — link: .specs/00_llama-orch.md#23-placement-scheduling
- ORCH-3014 — Queue guardrails MUST reject requests that exceed model context or declared budgets before enqueue. () (section: 2.2 Admission & Queueing, level: must) — link: .specs/00_llama-orch.md#22-admission-queueing
- ORCH-3021 — Sessions are short-lived: default TTL ≤ 10 minutes, default max 8 turns. () (section: 2.4 Sessions & Budgets, level: info) — link: .specs/00_llama-orch.md#24-sessions-budgets
- ORCH-3022 — KV cache usage MUST be bounded; metrics MUST expose KV pressure. () (section: 2.4 Sessions & Budgets, level: must) — link: .specs/00_llama-orch.md#24-sessions-budgets
- ORCH-3023 — Cross-worker KV migration is disallowed; failover MUST surface `kv_migrated=false`. () (section: 2.4 Sessions & Budgets, level: must) — link: .specs/00_llama-orch.md#24-sessions-budgets
- ORCH-3024 — Each job MUST carry a unique `job_id`. () (section: 2.5 Streaming & Determinism, level: must) — link: .specs/00_llama-orch.md#25-streaming-determinism
- ORCH-3026 — Cancellation MUST be race-free; no tokens may be emitted after cancel. () (section: 2.5 Streaming & Determinism, level: must) — link: .specs/00_llama-orch.md#25-streaming-determinism
- ORCH-3027 — Logs MUST include `job_id`, `session_id`, `engine`, `engine_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out` (section: 2.8 Observability, level: must) — link: .specs/00_llama-orch.md#28-observability
- ORCH-3028 — Minimum Prometheus metrics: queue depth, tasks enqueued/started/canceled/rejected, tokens in/out, GPU util, VRAM used, optional KV cache ratio. () (section: 2.8 Observability, level: info) — link: .specs/00_llama-orch.md#28-observability
- ORCH-3029 — SSE framing MUST stay `started`, `token`, optional `metrics`, `end`, `error`. `started` MUST include `queue_position` and `predicted_start_ms` when available. ( (section: 2.5 Streaming & Determinism, level: must) — link: .specs/00_llama-orch.md#25-streaming-determinism
- ORCH-3030 — Spec changes MUST follow `.docs/PROCESS.md`. () (section: 6. Documentation & Process, level: must) — link: .specs/00_llama-orch.md#6-documentation-process
- ORCH-3031 — Pool drain/reload MUST be atomic and reversible: reload success toggles Ready, failure rolls back. ( & ORCH-3038) (section: 2.6 Catalog, Artifacts & Reloads, level: must) — link: .specs/00_llama-orch.md#26-catalog-artifacts-reloads
- ORCH-3035 — Control and data planes MUST be gated by a shared API token; optional mTLS/OIDC MAY be added later. () (section: 2.7 Security & Policy, level: must) — link: .specs/00_llama-orch.md#27-security-policy
- ORCH-3037 — Catalog APIs MUST persist model metadata locally and allow verification flows that warn (rather than fail) when signatures/SBOMs are missing. () (section: 2.6 Catalog, Artifacts & Reloads, level: must) — link: .specs/00_llama-orch.md#26-catalog-artifacts-reloads
- ORCH-3038 — Pool drain/reload MUST be atomic and reversible: reload success toggles Ready, failure rolls back. (ORCH-3031 & ) (section: 2.6 Catalog, Artifacts & Reloads, level: must) — link: .specs/00_llama-orch.md#26-catalog-artifacts-reloads
- ORCH-3039 — VRAM OOM MUST be distinguished from host OOM; VRAM OOM SHOULD trigger capacity re-estimation. () (section: 2.10 Resilience & Recovery, level: must) — link: .specs/00_llama-orch.md#210-resilience-recovery
- ORCH-3040 — Circuit breakers SHOULD shed load if SLOs are breached persistently. () (section: 2.10 Resilience & Recovery, level: should) — link: .specs/00_llama-orch.md#210-resilience-recovery
- ORCH-3044 — TODO tracker updates are mandatory after spec/contract/test/code changes. () (section: 6. Documentation & Process, level: info) — link: .specs/00_llama-orch.md#6-documentation-process
- ORCH-3045 — Determinism is the default: identical `{prompt, parameters, seed, sampler_profile_version, engine_version, model_digest}` on the same replica MUST yield identic (section: 2.5 Streaming & Determinism, level: must) — link: .specs/00_llama-orch.md#25-streaming-determinism
- ORCH-3046 — Replica sets MUST pin `engine_version` and `sampler_profile_version`; mixed versions are not allowed. () (section: 2.5 Streaming & Determinism, level: must) — link: .specs/00_llama-orch.md#25-streaming-determinism
- ORCH-3049 — Startup self-tests MUST cover preload, minimal decode, cancel, metrics/log emission. () (section: 5. Testing & Validation, level: must) — link: .specs/00_llama-orch.md#5-testing-validation
- ORCH-3050 — Determinism suite MUST verify byte-exact streams across replicas per engine (llama.cpp with `--parallel 1 --no-cont-batching`, others in single-slot mode). () (section: 5. Testing & Validation, level: must) — link: .specs/00_llama-orch.md#5-testing-validation
- ORCH-3051 — Chaos/load tests SHOULD exercise drain/reload, driver resets, queue saturation, cancel. () (section: 5. Testing & Validation, level: should) — link: .specs/00_llama-orch.md#5-testing-validation
- ORCH-3052 — Device masks are explicit; spillover is forbidden. () (section: 3. GPU Topology & Device Management, level: info) — link: .specs/00_llama-orch.md#3-gpu-topology-device-management
- ORCH-3053 — Driver resets/ECC events MUST be surfaced distinctly; restart storms MUST be limited with backoff. () (section: 3. GPU Topology & Device Management, level: must) — link: .specs/00_llama-orch.md#3-gpu-topology-device-management
- ORCH-3054 — llama.cpp adapter MUST implement health, properties, completion SSE, cancel, metrics. () (section: 4. Engines & Adapters (Informative), level: must) — link: .specs/00_llama-orch.md#4-engines-adapters-informative
- ORCH-3055 — vLLM adapter MUST capture engine version and expose OpenAI-compatible completion/embedding paths internally. () (section: 4. Engines & Adapters (Informative), level: must) — link: .specs/00_llama-orch.md#4-engines-adapters-informative
- ORCH-3056 — TGI adapter MUST support `/generate`, `/info`, and optional OpenAI-compatible path. () (section: 4. Engines & Adapters (Informative), level: must) — link: .specs/00_llama-orch.md#4-engines-adapters-informative
- ORCH-3057 — Triton/TensorRT-LLM adapter MUST expose health/metadata, infer (HTTP/gRPC), and metrics. () (section: 4. Engines & Adapters (Informative), level: must) — link: .specs/00_llama-orch.md#4-engines-adapters-informative
- ORCH-3058 — Mock adapter MUST remain for tests and support fault injection. () (section: 4. Engines & Adapters (Informative), level: must) — link: .specs/00_llama-orch.md#4-engines-adapters-informative
- ORCH-3080 — A lightweight policy hook MUST exist so outbound HTTP tooling can be allowed/denied per deployment. () (section: 2.7 Security & Policy, level: must) — link: .specs/00_llama-orch.md#27-security-policy
- ORCH-3095 — The API MUST expose capability information (either via enriched `GET /v1/replicasets` or `GET /v1/capabilities`) covering engine versions, max context, supporte (section: 2.9 Capability Discovery, level: must) — link: .specs/00_llama-orch.md#29-capability-discovery
- ORCH-3096 — Capability payloads MUST include an API version compatible with OpenAPI `info.version`. () (section: 2.9 Capability Discovery, level: must) — link: .specs/00_llama-orch.md#29-capability-discovery
- ORCH-3097 — Artifact registry SHOULD expose `POST /v1/artifacts` + `GET /v1/artifacts/{id}` for plans/diffs/traces stored on local disk. ( & ORCH-3098) (section: 2.6 Catalog, Artifacts & Reloads, level: should) — link: .specs/00_llama-orch.md#26-catalog-artifacts-reloads
- ORCH-3098 — Artifact registry SHOULD expose `POST /v1/artifacts` + `GET /v1/artifacts/{id}` for plans/diffs/traces stored on local disk. (ORCH-3097 & ) (section: 2.6 Catalog, Artifacts & Reloads, level: should) — link: .specs/00_llama-orch.md#26-catalog-artifacts-reloads
- ORCH-3099 — Per-session token/time/cost budgets SHOULD be configurable; enforcement MUST occur before enqueue when enabled. () (section: 2.4 Sessions & Budgets, level: must) — link: .specs/00_llama-orch.md#24-sessions-budgets
- ORCH-3100 — SSE `metrics` frames SHOULD include additive JSON (`queue_depth`, `on_time_probability`, `kv_warmth`, remaining budgets). () (section: 2.8 Observability, level: should) — link: .specs/00_llama-orch.md#28-observability

### .specs/10-orchestrator-core.md
Total requirements: 19
- OC-CORE-1001 — - [] Each Pool MUST expose a bounded FIFO queue per priority class. (section: 1) Queue & Admission, level: must) — link: .specs/10-orchestrator-core.md#1-queue-admission
- OC-CORE-1002 — - [] Admission MUST reject when the queue is full according to configured policy (reject/drop-lru/shed-low-priority). (section: 1) Queue & Admission, level: must) — link: .specs/10-orchestrator-core.md#1-queue-admission
- OC-CORE-1003 — - [] Enqueue MUST be O(1) amortized and MUST preserve request arrival order within the same priority. (section: 1) Queue & Admission, level: must) — link: .specs/10-orchestrator-core.md#1-queue-admission
- OC-CORE-1004 — - [] Dequeue MUST prefer higher priority and MUST respect FIFO order within a priority class. (section: 1) Queue & Admission, level: must) — link: .specs/10-orchestrator-core.md#1-queue-admission
- OC-CORE-1005 — - [] Cancellation MUST remove the task from the queue or mark the slot so it is not dispatched. (section: 1) Queue & Admission, level: must) — link: .specs/10-orchestrator-core.md#1-queue-admission
- OC-CORE-1010 — - [] Scheduler MUST dispatch only to Ready replicas. (section: 2) Scheduling & Placement, level: must) — link: .specs/10-orchestrator-core.md#2-scheduling-placement
- OC-CORE-1011 — - [] Placement MUST respect device masks; cross‑mask spillover MUST NOT occur. (section: 2) Scheduling & Placement, level: must) — link: .specs/10-orchestrator-core.md#2-scheduling-placement
- OC-CORE-1012 — - [] Least‑loaded placement MUST be used across replicas of the same replica set. (section: 2) Scheduling & Placement, level: must) — link: .specs/10-orchestrator-core.md#2-scheduling-placement
- OC-CORE-1013 — - [] Session affinity SHOULD keep a session on its last good replica when possible. (section: 2) Scheduling & Placement, level: should) — link: .specs/10-orchestrator-core.md#2-scheduling-placement
- OC-CORE-1020 — - [] Context length MUST be ≤ model limit; otherwise reject before enqueue. (section: 3) Capacity & Guardrails, level: must) — link: .specs/10-orchestrator-core.md#3-capacity-guardrails
- OC-CORE-1021 — - [] Token budget (prompt + generation) MUST be validated pre‑admission. (section: 3) Capacity & Guardrails, level: must) — link: .specs/10-orchestrator-core.md#3-capacity-guardrails
- OC-CORE-1022 — - [] Watchdog MUST abort stuck Jobs with configurable wall/idle timeouts. (section: 3) Capacity & Guardrails, level: must) — link: .specs/10-orchestrator-core.md#3-capacity-guardrails
- OC-CORE-1023 — - [] When per‑session budgets (token/time/cost) are configured, admission and/or scheduling MUST enforce remaining budget and reject infeasible requests with  (section: 3) Capacity & Guardrails, level: must) — link: .specs/10-orchestrator-core.md#3-capacity-guardrails
- OC-CORE-1024 — - [] Budget accounting SHOULD be surfaced to clients via SSE `metrics` frames and/or response headers. (section: 3) Capacity & Guardrails, level: should) — link: .specs/10-orchestrator-core.md#3-capacity-guardrails
- OC-CORE-1030 — - [] Within a replica set, identical {prompt, parameters, seed, sampler_profile_version, engine_version, model_digest} MUST yield identical token streams. (section: 4) Determinism, level: must) — link: .specs/10-orchestrator-core.md#4-determinism
- OC-CORE-1031 — - [] Replica sets MUST pin engine_version and sampler_profile_version; mixed replicas MUST NOT share a set. (section: 4) Determinism, level: must) — link: .specs/10-orchestrator-core.md#4-determinism
- OC-CORE-1032 — - [] Determinism MUST NOT be assumed across engine/model updates. (section: 4) Determinism, level: must) — link: .specs/10-orchestrator-core.md#4-determinism
- OC-CORE-1040 — - [] Logs MUST include job_id, session_id, engine, pool_id, replica_id, model_id, quant, ctx, kv_warmth, queue_time_ms, decode_time_ms. (section: 5) Observability, level: must) — link: .specs/10-orchestrator-core.md#5-observability
- OC-CORE-1041 — - [] Metrics MUST include queue depth, reject/drop counts, first-token/decode latency, GPU/VRAM utilization, KV pressure, preload outcomes. (section: 5) Observability, level: must) — link: .specs/10-orchestrator-core.md#5-observability

### .specs/20-orchestratord.md
Total requirements: 23
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
- OC-CTRL-2023 — - [] The `metrics` SSE frames SHOULD include fields helpful for client-side planning under load, such as `on_time_probability` (number), `queue_depth` (int), an (section: 11) SSE Metrics – Scheduling Signals, level: should) — link: .specs/20-orchestratord.md#11-sse-metrics-scheduling-signals
- OC-CTRL-2030 — - [] Errors MUST include a stable `code` field: `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNREADY`, `POOL_UNAVAILABLE`, `REPLICA_EXHAUS (section: 4) Error Taxonomy, level: must) — link: .specs/20-orchestratord.md#4-error-taxonomy
- OC-CTRL-2031 — - [] Errors SHOULD include the `engine` and `pool_id` when applicable. (section: 4) Error Taxonomy, level: should) — link: .specs/20-orchestratord.md#4-error-taxonomy
- OC-CTRL-2040 — - [] Control and data plane MUST be gated by AuthN/AuthZ; API keys acceptable day‑1. (section: 5) Security, level: must) — link: .specs/20-orchestratord.md#5-security
- OC-CTRL-2041 — - [] Logs MUST NOT leak secrets or API keys. (section: 5) Security, level: must) — link: .specs/20-orchestratord.md#5-security
- OC-CTRL-2050 — - [] Admission logs and `started` MUST include `queue_position` and `predicted_start_ms` when available. (section: 6) Observability, level: must) — link: .specs/20-orchestratord.md#6-observability
- OC-CTRL-2051 — - [] Metrics MUST include queue depth, reject/drop rates, latency percentiles, and error counts by class. (section: 6) Observability, level: must) — link: .specs/20-orchestratord.md#6-observability
- OC-CTRL-2052 — - [] Responses that include headers SHOULD include `X-Correlation-Id` (except `204 No Content`) to enable client trace stitching. (section: 6) Observability, level: should) — link: .specs/20-orchestratord.md#6-observability
- OC-CTRL-2060 — - [] The server MUST expose capability information required for client scheduling and compatibility. This MAY be achieved by either: (section: 8) Capabilities & Discovery, level: must) — link: .specs/20-orchestratord.md#8-capabilities-discovery
- OC-CTRL-2061 — - [] Capability payloads MUST include an API version field compatible with OpenAPI `info.version`, enabling the CLI to pin a compatible range. (section: 8) Capabilities & Discovery, level: must) — link: .specs/20-orchestratord.md#8-capabilities-discovery
- OC-CTRL-2065 — - [] The server SHOULD provide `POST /v1/artifacts` to persist structured artifacts (plans, summaries, diffs, traces) with content-addressed IDs and tags. Reque (section: 9) Artifact Registry (Optional, Recommended), level: must) — link: .specs/20-orchestratord.md#9-artifact-registry-optional-recommended
- OC-CTRL-2066 — - [] The server SHOULD provide `GET /v1/artifacts/{id}` to retrieve artifacts by ID, including metadata (tags, lineage, timestamps). Authorization MUST be enfor (section: 9) Artifact Registry (Optional, Recommended), level: must) — link: .specs/20-orchestratord.md#9-artifact-registry-optional-recommended
- OC-CTRL-2068 — - [] Per-session budgets (token/time/cost) SHOULD be supported and enforced at admission or scheduling time. When budgets are active, the server SHOULD surface  (section: 10) Budgets & Guardrails, level: should) — link: .specs/20-orchestratord.md#10-budgets-guardrails

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
Total requirements: 9
- OC-POLICY-4001 — - [] Default plugin ABI MUST be WASI; functions MUST be pure/deterministic over explicit snapshots. (section: 1) ABI & Determinism, level: must) — link: .specs/50-plugins-policy-host.md#1-abi-determinism
- OC-POLICY-4002 — - [] ABI versioning MUST be explicit; incompatible changes MUST bump MAJOR. (section: 1) ABI & Determinism, level: must) — link: .specs/50-plugins-policy-host.md#1-abi-determinism
- OC-POLICY-4010 — - [] Plugins MUST run in a sandbox with no filesystem/network unless explicitly granted. (section: 2) Sandboxing & Safety, level: must) — link: .specs/50-plugins-policy-host.md#2-sandboxing-safety
- OC-POLICY-4011 — - [] Host MUST bound CPU time/memory per invocation and abort on overuse. (section: 2) Sandboxing & Safety, level: must) — link: .specs/50-plugins-policy-host.md#2-sandboxing-safety
- OC-POLICY-4020 — - [] Host MUST log plugin id/version, decision outcome, and latency. (section: 3) Telemetry, level: must) — link: .specs/50-plugins-policy-host.md#3-telemetry
- OC-POLICY-4030 — - [] The policy host SHOULD expose a mediated HTTP fetch/search tool to clients (e.g., CLI agents), enforcing allowlists/denylists of domains and MIME types. (section: 4) Tooling Proxy: HTTP Fetch/Search (Client Docs Access), level: should) — link: .specs/50-plugins-policy-host.md#4-tooling-proxy-http-fetch-search-client-docs-access
- OC-POLICY-4031 — - [] The tool MUST redact secrets and PII as configured and MUST bound response size and rate. (section: 4) Tooling Proxy: HTTP Fetch/Search (Client Docs Access), level: must) — link: .specs/50-plugins-policy-host.md#4-tooling-proxy-http-fetch-search-client-docs-access
- OC-POLICY-4032 — - [] The tool MUST emit audit logs that include request URL (redacted), policy decision, and byte counts. (section: 4) Tooling Proxy: HTTP Fetch/Search (Client Docs Access), level: must) — link: .specs/50-plugins-policy-host.md#4-tooling-proxy-http-fetch-search-client-docs-access
- OC-POLICY-4033 — - [] The ABI for invoking tools MUST be stable and versioned; tool invocation SHOULD be deterministic with respect to inputs and policy snapshot. (section: 4) Tooling Proxy: HTTP Fetch/Search (Client Docs Access), level: must) — link: .specs/50-plugins-policy-host.md#4-tooling-proxy-http-fetch-search-client-docs-access

### .specs/51-plugins-policy-sdk.md
Total requirements: 5
- OC-POLICY-SDK-4101 — - [] Public SDK functions MUST be semver‑stable within a MAJOR. (section: 1) Stability & Compatibility, level: must) — link: .specs/51-plugins-policy-sdk.md#1-stability-compatibility
- OC-POLICY-SDK-4102 — - [] Breaking changes MUST be accompanied by a migration note and version bump. (section: 1) Stability & Compatibility, level: must) — link: .specs/51-plugins-policy-sdk.md#1-stability-compatibility
- OC-POLICY-SDK-4110 — - [] SDK MUST NOT perform network or filesystem I/O by default. (section: 2) Safety, level: must) — link: .specs/51-plugins-policy-sdk.md#2-safety
- OC-POLICY-SDK-4111 — - [] SDK SHOULD provide typed helpers for invoking host-exposed tools (e.g., `fetch`, `search`) that accept policy hints (allowed domains, max bytes) and propag (section: 2) Safety, level: should) — link: .specs/51-plugins-policy-sdk.md#2-safety
- OC-POLICY-SDK-4112 — - [] SDK MUST redact sensitive inputs/outputs according to configuration and provide hooks for additional user redaction. (section: 2) Safety, level: must) — link: .specs/51-plugins-policy-sdk.md#2-safety

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
Total requirements: 4
- OC-METRICS-7101 — - [] Metric names and required labels MUST conform to `ci/metrics.lint.json`. (section: 1) Names/Labels, level: must) — link: .specs/71-metrics-contract.md#1-names-labels
- OC-METRICS-7102 — - [] Label cardinality budgets MUST be documented and enforced. (section: 1) Names/Labels, level: must) — link: .specs/71-metrics-contract.md#1-names-labels
- OC-METRICS-7110 — - [] The `metrics` SSE event payloads SHOULD include additive fields helpful for client-side planning under load. Example fields (non-exhaustive, non-breaking i (section: 3) SSE Metrics Signals (Client Planning), level: should) — link: .specs/71-metrics-contract.md#3-sse-metrics-signals-client-planning
- OC-METRICS-7111 — - [] When per-session budgets (token/time/cost) are enabled, budget remaining SHOULD be surfaced either in `metrics` events or as response headers to allow clie (section: 3) SSE Metrics Signals (Client Planning), level: should) — link: .specs/71-metrics-contract.md#3-sse-metrics-signals-client-planning
