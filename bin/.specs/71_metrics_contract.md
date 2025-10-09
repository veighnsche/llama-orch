# Metrics Contract (OBS-71xx)

Status: Draft
Version: 0.1.0
Conformance language: RFC-2119 (MUST/SHOULD/MAY)
Parent spec: `.specs/00_llama-orch.md`

---

## 0. Scope

This document defines the canonical Prometheus metrics emitted by Llama-Orch components and the SSE `metrics` event used during inference streams. It standardizes names, labels, units, and aggregation guidance.

Conventions:
- Counters use `_total` and are monotonic.
- Durations use `_ms` in milliseconds unless noted.
- Sizes use `_bytes`.
- Labels are stable and low-cardinality.

---

## 1. Orchestrator Metrics (orchd_*)

- `orchd_queue_depth{priority}`
  - Gauge. Current queue depth per priority (`interactive|batch`).

- `orchd_tasks_enqueued_total{outcome}`
  - Counter. Task submissions classified by `accepted|rejected`.

- `orchd_tasks_dispatched_total{worker_id, outcome}`
  - Counter. Dispatch attempts to workers by `success|failure`.

- `orchd_scheduling_latency_ms{algorithm}`
  - Histogram. Time from dequeue decision start to dispatch, labeled by scheduling algorithm (`least-loaded|most-vram-free|round-robin`).

- `orchd_worker_connections_active{worker_id}`
  - Gauge. Active HTTP/SSE connections from orchestrator to a worker.

---

## 2. Pool-Manager Metrics (pool_mgr_*)

- `pool_mgr_gpus_total`
  - Gauge. Number of GPUs discovered.

- `pool_mgr_gpu_vram_total_bytes{gpu_id}`
  - Gauge. Total VRAM per GPU.

- `pool_mgr_gpu_vram_allocated_bytes{gpu_id}`
  - Gauge. VRAM currently allocated by workers per GPU.

- `pool_mgr_workers_total{status}`
  - Gauge. Workers by status (`starting|ready|busy|draining|failed`).

- `pool_mgr_worker_starts_total{outcome}`
  - Counter. Worker start attempts by `success|failure`.

- `pool_mgr_worker_stops_total{reason}`
  - Counter. Worker stop events (e.g., `evict|drain|crash`).

- `pool_mgr_model_downloads_total{outcome}`
  - Counter. Model download attempts by `success|failure`.

Notes:
- Heartbeat frequency defaults to 15s; metrics emission SHOULD align with heartbeat/state refresh cadence to avoid excessive cardinality.

---

## 3. Worker Metrics (worker_*)

- `worker_vram_bytes{worker_id}`
  - Gauge. VRAM used by the worker process.

- `worker_requests_total{outcome}`
  - Counter. `/execute` requests by `success|error|cancelled|timeout`.

- `worker_tokens_in_total`
  - Counter. Total input tokens processed.

- `worker_tokens_generated_total`
  - Counter. Total output tokens generated.

- `worker_inference_duration_ms`
  - Histogram. End-to-end latency for `/execute` in milliseconds.

- `worker_uptime_seconds`
  - Counter. Process uptime.

---

## 4. SSE `metrics` Event

Workers MAY emit periodic `metrics` events in the SSE stream to surface real-time progress.

Event type: `metrics`

Minimal JSON schema:
```json
{
  "job_id": "job-xyz",
  "tokens_out": 123,
  "decode_time_ms": 456,
  "vram_bytes": 16000000000
}
```

Requirements:
- Emission cadence SHOULD be bounded (e.g., every 250-1000ms) to avoid flooding.
- Orchestrator MUST relay `metrics` events unchanged to the client, MAY add metadata in separate fields if needed.
- On cancellation, workers MUST stop emitting `metrics` and emit `error` with code `CANCELLED`.

---

## 5. Units & Cardinality

- Time: milliseconds (`_ms`) unless Prometheus ecosystem mandates seconds; conversions happen at scrape/query-time if needed.
- Sizes: bytes (`_bytes`).
- Labels: `worker_id`, `gpu_id`, `status`, `algorithm`, `outcome` MUST remain low-cardinality.

---

## 6. Traceability

- Orchestrator spec: `bin/queen-rbee/.specs/00_queen-rbee.md` (§9)
- Pool-managerd spec: `bin/pool-managerd/.specs/00_pool-managerd.md` (§8)
- Worker-orcd spec: `bin/worker-orcd/.specs/00_worker-orcd.md` (§9)

---

Version: 0.1.0
Last Updated: 2025-10-03
Status: Living document
