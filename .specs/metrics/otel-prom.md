# Metrics Contract — OTEL/Prometheus

Status: pre-code · Scope: OrchQueue v1 data plane & control plane

## Conventions

- Prometheus metrics exported by orchestrator and workers must include labels:
  - `engine` (llamacpp|vllm|tgi|triton)
  - engine-specific version labels: `engine_version`, and when applicable `trtllm_version`
  - `pool_id`, `replica_id` (where applicable), `model_id` or `model_digest`
- Metrics are contracts: names and label sets are stable. Changes are breaking.
- Links to requirements use ORCH-IDs when available in SPEC (none assigned yet in current SPEC).
- Exception: Admission-level counters (e.g., `tasks_rejected_total`) MAY omit `engine_version` to limit cardinality and because rejections can occur before engine assignment. This exception is normative and aligns with `ci/metrics.lint.json`.

## Counters

- tasks_enqueued_total
  - type: counter
  - labels: engine, engine_version, pool_id, replica_id, priority
  - unit: 1
  - description: Number of tasks accepted into the queue.
- tasks_started_total
  - type: counter
  - labels: engine, engine_version, pool_id, replica_id, priority
  - unit: 1
  - description: Number of tasks that started decoding on a worker slot.
- tasks_canceled_total
  - type: counter
  - labels: engine, engine_version, pool_id, replica_id, reason
  - unit: 1
  - description: Number of tasks canceled (user or watchdog). 
- tasks_rejected_total
  - type: counter
  - labels: reason (ADMISSION_REJECT|QUEUE_FULL_DROP_LRU|INVALID_PARAMS|POOL_UNREADY|POOL_UNAVAILABLE|REPLICA_EXHAUSTED), engine
  - unit: 1
  - description: Number of rejected tasks with the typed reason.
  - note: `engine_version` is intentionally omitted per Conventions exception (admission-level, pre-engine; cardinality control).

## Gauges

- queue_depth
  - type: gauge
  - labels: engine, engine_version, pool_id, priority
  - unit: 1
  - description: Current queue length per pool/priority.
- kv_cache_usage_ratio
  - type: gauge
  - labels: engine, engine_version, pool_id, replica_id
  - unit: ratio
  - description: Ratio of KV cache used vs capacity.
- gpu_utilization
  - type: gauge
  - labels: engine, engine_version, pool_id, replica_id, device
  - unit: percent
  - description: GPU utilization percent.
- vram_used_bytes
  - type: gauge
  - labels: engine, engine_version, pool_id, replica_id, device
  - unit: bytes
  - description: VRAM used by the worker process(es).

## v3.2 Metrics (Catalog/Lifecycle + Advanced Scheduling)

- catalog_verifications_total
  - type: counter
  - labels: result (pass|fail), reason, engine
  - unit: 1
  - description: Number of model artifact verification attempts and outcomes.

- model_state
  - type: gauge
  - labels: model_id, state (Draft|Active|Canary|Deprecated|Retired)
  - unit: 1
  - description: Current lifecycle state per model.

- admission_share
  - type: gauge
  - labels: tenant, priority
  - unit: ratio
  - description: EWMA of observed admission share for fairness verification.

- deadlines_met_ratio
  - type: gauge
  - labels: priority
  - unit: ratio
  - description: Ratio of jobs meeting deadlines over a rolling window.

- preemptions_total
  - type: counter
  - labels: mode (soft|hard), engine
  - unit: 1
  - description: Number of preemption events.

- resumptions_total
  - type: counter
  - labels: engine
  - unit: 1
  - description: Number of resumed jobs after preemption.

## Summaries/Histograms

- latency_first_token_ms
  - type: histogram
  - labels: engine, engine_version, pool_id, priority
  - unit: milliseconds
  - description: Time to first token since job admission.
- latency_decode_ms
  - type: histogram
  - labels: engine, engine_version, pool_id, priority
  - unit: milliseconds
  - description: Decode time for the full stream per job.
- tokens_in_total
  - type: counter
  - labels: engine, engine_version, pool_id, replica_id
  - unit: tokens
  - description: Total prompt tokens processed.
- tokens_out_total
  - type: counter
  - labels: engine, engine_version, pool_id, replica_id
  - unit: tokens
  - description: Total output tokens generated.

## Backpressure

- admission_backpressure_events_total
  - type: counter
  - labels: engine, policy (reject|drop-lru|shed-low-priority)
  - unit: 1
  - description: Number of backpressure decisions taken.

## Linter Expectations

- All metrics above must be present in code emission sites (no runtime logic here).
- Labels must include `engine` and engine-specific version labels as described.
