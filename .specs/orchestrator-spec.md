# Orchestrator SPEC — NVIDIA‑Only Inference, Multi‑Engine (v3.1)

**Status:** Stable (candidate)

**Date:** 2025‑09‑15

**Applies to:** Linux hosts with **NVIDIA CUDA‑capable GPUs** (inference only), multi‑engine worker pools (llama.cpp, vLLM, TGI, Triton/TensorRT‑LLM), multi‑tenant orchestration

**Conformance language:** RFC‑2119 (MUST/SHOULD/MAY)

---

## 0) Scope & Goals

* Provide deterministic, safe, high‑throughput orchestration of **NVIDIA‑backed** LLM workers across one or more GPUs per host.
* Optimize for **low tail latency** under bursty/agentic workloads while preserving **fairness** and **resource isolation**.
* Be resilient to **model hot‑swaps**, **driver resets**, and **misconfiguration**.
* Be evolvable: policy surfaces and config schema accept new engines/capabilities without breaking existing deployments.
* Public API: The orchestrator‑native, queue‑centric API (**OrchQueue v1**) is the authoritative public API.

**Non‑goals:** Implementing GPU drivers, CUDA/NCCL kernels, or engine runtimes; supporting non‑NVIDIA accelerators for inference.

---

## 1) Platform Assumptions

* Hosts are **Linux, headless**, with NVIDIA drivers + CUDA runtime installed.
* **Inference hosts MUST have NVIDIA GPUs**; CPU‑only or non‑NVIDIA hosts **MAY** run control plane components (Controller/Scheduler) but **MUST NOT** serve inference traffic. [ORCH-1101]
* Heterogeneous GPUs (e.g., different VRAM and SM counts) **ARE SUPPORTED**; see §3.4 and §3.6 for placement and split rules.
* No OpenAI‑compatible public APIs are exposed by the orchestrator; those are internal to adapters. OrchQueue v1 is the sole public surface.

---

## 2) Glossary (additions)

* **Engine:** A specific serving stack (llama.cpp server, vLLM OpenAI‑compatible server, HF Text‑Generation‑Inference (TGI), NVIDIA Triton/TensorRT‑LLM).
* **Adapter:** A library that maps an Engine’s native API to the orchestrator’s Worker contract (see §6.1–§6.4).

---

## 3) Hard Requirements

### 3.1 Process model & preloading

* Workers **MUST** be one‑model/one‑device‑mask processes, pinned by config. [ORCH-3001]
* Pools **MUST** preload at `serve` start and delay `Ready` exposure until success. [ORCH-3002]
* Preload **MUST** fail fast if VRAM or host RAM headroom is insufficient; Pool remains **Unready** with a retry backoff. [ORCH-3003]

### 3.2 Queues & admission under load

* Each Pool **MUST** have a bounded FIFO queue. [ORCH-3004]
* Full queue policy **MUST** be one of: `reject`, `drop-lru` (oldest enqueued), or `shed-low-priority`. [ORCH-3005]
* Per‑client **rate limits** and **burst buckets** **SHOULD** be enforced before enqueue. [ORCH-3006]
* **Backpressure headers** **SHOULD** be returned to callers. [ORCH-2007]

### 3.3 Replicas, placement, and affinity

* Pools with identical `{engine, model, quant, ctx, engine_version, sampler_profile_version}` **MUST** be treated as a replica set. [ORCH-3007]
* Scheduler **MUST** place on the **least‑loaded Ready** replica, respecting device masks and node/zone anti‑affinity if configured. [ORCH-3008]
* **Session affinity** **SHOULD** keep a session on its last good replica for KV reuse; on failure, **MAY** fail over with `kv_warmth=false` surfaced to telemetry. [ORCH-3009]

### 3.4 Placement, heterogeneity & readiness

* A Job **MUST NOT** be dispatched until the Pool is `Ready`. [ORCH-3010]
* Placement **MUST** respect device masks; cross‑mask spillover **MUST NOT** occur. [ORCH-3011]
* **Heterogeneous multi‑GPU splits** (across GPUs of different VRAM/compute) **MUST** be **opt‑in** with explicit per‑GPU ratios (e.g., `tensor_split: [0.67, 0.33]`). Default is **no cross‑GPU split** unless the pool is declared homogeneous or split ratios are configured. [ORCH-3012]
* **NUMA/PCIe topology hints** **SHOULD** influence placement when available. [ORCH-3013]

### 3.5 Capacity, NVIDIA‑only & guardrails

* Requested context length **MUST** be ≤ model limit; otherwise **reject before enqueue** (`HTTP 400`). [ORCH-3014]
* **Token budget** (prompt + generation) **MUST** be validated pre‑admission. [ORCH-3015]
* Watchdog **MUST** abort stuck Jobs with configurable **wall** and **idle** timeouts. [ORCH-3016]
* `cancel` **MUST** free the Worker slot and return a terminal state. [ORCH-3017]
* **CPU spillover is disallowed** for inference: on GPU/Pool outage, Controller **MUST** fail fast (`POOL_UNAVAILABLE`) with retry hints. [ORCH-3018]

### 3.6 Batching & throughput (engine‑agnostic)

* If an Engine supports **continuous batching**, Workers **SHOULD** expose total slot count and per‑slot state; the Scheduler **MUST** factor this into placement. [ORCH-3019]
* **Speculative decoding**/**prefix caching** **MAY** be enabled; admission control **MUST** account for memory impact. [ORCH-3020]

### 3.7 Sessions & KV (short‑lived)

* Sessions are **short‑lived**; enforce TTL ≤ **10 minutes** and/or max **8 turns**. [ORCH-3021]
* KV cache residency **MUST** be bounded with LRU/LFU; expose pressure metrics. [ORCH-3022]
* **Cross‑Worker KV migration is disabled**; failover surfaces `kv_migrated=false`. [ORCH-3023]

### 3.8 Cancellations, retries, idempotency

* Each Job **MUST** carry a unique `job_id`. [ORCH-3024]
* Client retries **SHOULD** target retryable error classes only. [ORCH-3025]
* Cancellation **MUST** be race‑free (no post‑cancel tokens). [ORCH-3026]

### 3.9 Observability & telemetry

* Logs **MUST** include: `job_id`, `session_id`, `client_id`, `engine`, `pool_id`, `replica_id`, `model_id`, `quant`, `ctx`, `tensor_split`, `engine_version`, `sampler_profile_version`, `kv_warmth`, `queue_time_ms`, `decode_time_ms`, `tokens_in/out`, `eviction_events`, `oom_events`, `driver_reset_events`. [ORCH-3027]
* Metrics **MUST** include: queue depth, reject/drop rates, p50/p95/p99 latency, GPU/VRAM/RAM utilization, KV pressure, preload outcomes, per‑priority SLO attainment. Metric labels **MUST** include `engine` and engine‑specific version labels (e.g., `engine_version`, `trtllm_version` where applicable). Counters MUST include: `tasks_enqueued_total`, `tasks_started_total`, `tasks_canceled_total`, `tasks_rejected_total{reason=...}`. Engine‑native `/metrics` **SHOULD** be enabled where available. [ORCH-3028]
* Admission logs and the `started` SSE event **MUST** include `queue_position` and `predicted_start_ms` when available. [ORCH-3029]

### 3.10 Config & lifecycle

* Config **MUST** be schema‑validated; unknown fields rejected (strict) or logged (compat). [ORCH-3030]
* **Hot‑reload** **MAY** be supported but **MUST** be atomic and revertible; rolling changes use drain+replace. [ORCH-3031]

### 3.11 Rollouts & versioning

* Workers **MUST** report `engine_version` and `model_digest` (or equivalent). [ORCH-3032]
* **Canaries** **SHOULD** be supported via labels/percent splits; **Rollbacks** **MUST** be one action. [ORCH-3033]

### 3.12 Security & tenancy

* **AuthN/AuthZ** **MUST** gate control/data paths; API keys acceptable day‑1. [ORCH-3035]
* Per‑tenant quotas **MUST** bound concurrent Jobs and memory (KV + scratch). [ORCH-3036]
* Model artifacts **MUST** be checksummed and verified before load. [ORCH-3037]

### 3.13 Resilience & recovery

* **Driver/CUDA errors** **MUST** transition the Pool to `Unready`, drain, and backoff‑restart. [ORCH-3038]
* Distinguish **VRAM OOM** vs **host OOM**; VRAM OOM **SHOULD** trigger capacity re‑estimation. [ORCH-3039]
* **Circuit breakers** **SHOULD** shed load on sustained SLO violations. [ORCH-3040]

### 3.14 Performance & SLOs

* Per‑priority SLOs **MUST** be defined and measured continuously; alerts on breach. [ORCH-3041]

### 3.15 Storage & integrity

* Model storage **MUST** verify checksums; local caches **SHOULD** enforce quotas and eviction. [ORCH-3042]

### 3.16 API contracts & determinism

* Public APIs **MUST** be versioned. [ORCH-3044]
* **Deterministic outputs are the default.** Same `{prompt, parameters, seed, sampler_profile_version, engine_version, model_digest}` within a replica set **MUST** yield identical token streams. Determinism is asserted **per engine**; do not compare across engines or across differing engine versions. [ORCH-3045]
* Replica sets **MUST** pin `engine_version` and `sampler_profile_version`; replicas with differing values **MUST NOT** mix. [ORCH-3046]
* Determinism **MUST NOT** be assumed across engine/model updates. [ORCH-3047]
* The **SSE stream framing** (`started`, `token`, `metrics`, `end`, `error`) is part of the contract and MUST remain stable. [ORCH-2002]

### 3.17 Extensibility (policy plugins)

* Default plugin ABI **MUST** be **WASI** (pure, deterministic functions over explicit snapshots). Native in‑process trait **MAY** be offered as a high‑perf alternative. [ORCH-3048]

### 3.18 Testing & validation

* Startup self‑tests **MUST** cover: preload, minimal decode, cancel, telemetry emission. [ORCH-3049]
* Determinism tests **MUST** verify byte‑exact streams across replicas with fixed seeds, run **per engine** with engine‑appropriate settings (e.g., llama.cpp with `--parallel 1` and `--no-cont-batching`; others in single‑slot/single‑request mode). [ORCH-3050]
* Chaos tests **SHOULD** be run for failover/idempotency; load tests **MUST** cover priority inversion. [ORCH-3051]

---

## 4) NVIDIA Topology & Heterogeneity

* **Device masks** are explicit; **cross‑mask spillover is forbidden**.
* **Default**: no cross‑GPU splits unless pool is homogeneous **or** explicit `tensor_split` ratios are set.
* **Heterogeneous split policy**: when ratios are provided, **MUST** cap per‑GPU resident KV to avoid OOM on the smallest GPU; Scheduler **SHOULD** consider PCIe/NVLink and NUMA. [ORCH-3052]
* **Driver resets** and ECC (where applicable) **MUST** be surfaced distinctly; the Pool Manager **MUST** bound restart storms with backoff and circuit breakers. [ORCH-3053]

---

## 5) Engines & Worker Adapters (internal adapter notes)

### 5.1 llama.cpp server (HTTP)

* **Native**: `/health`, `/metrics` (when started with metrics), `/slots`, `/props`, `/completion`, `/tokenize`, `/detokenize`, `/embedding`.
* **OpenAI‑compatible**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`.
* **Adapter contract**: MUST support health, properties (slots/commit), completion (SSE), cancel, and metrics scrape. This mapping is internal; the public API is OrchQueue v1. [ORCH-3054]

### 5.2 vLLM (OpenAI‑compatible server)

* **OpenAI‑compatible**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`; API key support recommended.
* **Adapter contract**: same as above; engine version string MUST be captured. The adapter may call vLLM’s OpenAI server internally; this is not exposed publicly. [ORCH-3055]

### 5.3 Hugging Face Text‑Generation‑Inference (TGI)

* **Custom API**: `/generate` (sync/stream), `/health`, `/info`; **and** optional **OpenAI Messages API** for compatibility.
* **Adapter contract**: MUST implement the TGI custom path (`/generate`, `/info`, etc.). If an OpenAI‑compatible path is enabled, it remains internal only. [ORCH-3056]

### 5.4 NVIDIA Triton / TensorRT‑LLM

* **Triton HTTP/GRPC**: model health/metadata/stats; infer via `/v2/models/{model}/infer` (HTTP) or corresponding gRPC; supports dynamic load/unload.
* **OpenAI‑compatible frontends**: supported via `trtllm-serve` (TensorRT‑LLM) or Triton OpenAI‑compatible frontend; label as BETA where applicable.
* **Adapter contract**: MUST support health/metadata, infer/streaming (if configured), and metrics. OpenAI‑compatible frontends are internal only. [ORCH-3057]

> All adapters **MUST** normalize detokenization templates and sampler profiles to keep determinism stable within a replica set. [ORCH-3058]

---

## 6) API Surface (overlay)

### 6.1 Control Plane

* `POST /v1/pools/:id/drain` → { deadline\_ms }
* `POST /v1/pools/:id/reload` → { new\_model\_ref }
* `GET /v1/pools/:id/health` → { live, ready, draining, metrics }
* `GET /v1/replicasets` → list with load/SLO snapshots

### 6.2 Data Plane — OrchQueue v1

`POST /v1/tasks`

```json
{
  "task_id": "uuid",
  "session_id": "uuid",
  "workload": "completion",         // "completion" | "embedding" | "rerank"
  "model_ref": "sha256:... or catalog name",
  "engine": "llamacpp",             // "llamacpp" | "vllm" | "tgi" | "triton"
  "ctx": 8192,
  "priority": "interactive",        // or "batch"
  "seed": 123456789,                // if omitted, injected as hash(task_id)
  "determinism": "strict",          // "strict" | "best_effort"
  "sampler_profile_version": "v1",
  "prompt": "...",                  // or "inputs": [...]
  "max_tokens": 64,
  "deadline_ms": 30000,
  "expected_tokens": 64,            // placement hint
  "kv_hint": "reuse"                // "reuse" | "cold"
}
```

Responses

```json
202 Accepted:
{
  "task_id": "uuid",
  "queue_position": 3,
  "predicted_start_ms": 420,
  "backoff_ms": 0
}
```

`429` with Retry-After and X-Backoff-Ms when queue full (include policy: "reject" | "drop-lru" | "shed-low-priority" in JSON body).
`400` for invalid ctx/token budget; `401/403` for auth; typed errors listed below.

`GET /v1/tasks/:id/stream` (SSE)

Events: `started`, `token`({"t":"...","i":N}), `metrics` (periodic), `end`({"tokens_out":N,"decode_ms":...}), `error`. Framing is deterministic and stable.

`POST /v1/tasks/:id/cancel` → `204 No Content`

`GET /v1/sessions/:id` → `{ ttl_ms_remaining, turns, kv_bytes, kv_warmth }`

`DELETE /v1/sessions/:id` → force KV eviction

Typed errors (authoritative):
`ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNREADY`, `POOL_UNAVAILABLE`, `REPLICA_EXHAUSTED`, `DECODE_TIMEOUT`, `WORKER_RESET`, `INTERNAL`.

Backpressure headers:
`Retry-After: <seconds>`, `X-Backoff-Ms: <ms>`, optional `X-Queue-Position`, `X-Queue-ETA-Ms`.

Keep Control Plane endpoints as-is (drain, reload, health, replicasets).
 Error envelopes **MUST** include the `engine` context where applicable.

---

## 7) SLOs (example defaults)

* `interactive`: p95 ≤ 900 ms to first token, p95 ≤ 2.5 s for 64 tokens, error rate ≤ 0.5%.
* `batch`: throughput ≥ 80% of theoretical tokens/sec, error rate ≤ 1%.

---

## 8) Configuration Schema (excerpt)

```yaml
pools:
  - id: llama3-8b-q4-gpu0
    engine: llamacpp
    model: sha256:...
    quant: q4_k_m
    ctx: 8192
    devices: [0]
    tensor_split: null
    preload: true
    require_same_engine_version: true
    sampler_profile_version: "v1"
    queue: { capacity: 256, full_policy: reject }
    admission:
      priorities:
        - { name: interactive, queue_capacity: 128, rate_limit_rps: 20 }
        - { name: batch,       queue_capacity: 128, rate_limit_rps: 5 }
    timeouts: { wall_ms: 60000, idle_ms: 5000 }

  - id: llama3-8b-vllm-gpu1
    engine: vllm
    model: repo:NousResearch/Meta-Llama-3-8B-Instruct
    ctx: 8192
    devices: [1]
    preload: true
    queue: { capacity: 256, full_policy: reject }

  - id: mixtral-tgi-gpu01
    engine: tgi
    model: repo:mistralai/Mixtral-8x7B-Instruct
    devices: [0,1]
    tensor_split: [0.67, 0.33]   # explicit heterogeneous split (opt-in)
    preload: true
    queue: { capacity: 256, full_policy: drop-lru }
```

**Job (deterministic by default):**

```json
{
  "job_id": "...",
  "session_id": "...",
  "engine": "vllm",
  "model": "repo:NousResearch/Meta-Llama-3-8B-Instruct",
  "ctx": 8192,
  "priority": "interactive",
  "prompt": "...",
  "max_tokens": 64,
  "seed": 123456789,
  "determinism": "strict",
  "sampler_profile_version": "v1"
}
```

---

## 9) Compliance Checklist (Quick Audit)

* [ ] NVIDIA‑only inference enforcement
* [ ] Preload gating of `Ready`
* [ ] Bounded FIFO + full policy
* [ ] Least‑loaded placement with device masks
* [ ] Heterogeneous split requires explicit ratios
* [ ] Session TTL/max‑turns; no KV migration
* [ ] Token/context pre‑admission checks
* [ ] Watchdog + cancel semantics
* [ ] Structured telemetry fields
* [ ] Strict schema validation
* [ ] Canary + rollback
* [ ] Multi‑tenancy quotas
* [ ] Circuit breakers & drains
* [ ] Graceful shutdown
* [ ] Determinism suite across replicas

---

## 10) Assumptions & Non‑Goals

* Determinism is guaranteed **within** a replica set that pins `engine_version`, `sampler_profile_version`, and artifacts. Cross‑version determinism is out of scope.
* CPU inference fallback is **not supported**; controller fails fast on GPU unavailability.
* Client automation/CLIs are **out of scope** of this SPEC.

---

## 11) References (engine contracts)

* llama.cpp server API (native + OpenAI‑compatible)
* vLLM OpenAI‑compatible server (Completions/Chat/Embeddings)
* Hugging Face Text‑Generation‑Inference (custom API + OpenAI Messages API)
* NVIDIA Triton Inference Server (HTTP/GRPC) and OpenAI‑compatible frontend
* TensorRT‑LLM `trtllm-serve` (OpenAI‑compatible)
