# Metrics Contract — OTEL/Prometheus

Status: pre-code · Scope: OrchQueue v1 data plane & control plane

## Conventions

- Prometheus metrics exported by orchestrator and workers must include labels:
  - `engine` (llamacpp|vllm|tgi|triton)
  - engine-specific version labels: `engine_version`, and when applicable `trtllm_version`
  - `pool_id`, `replica_id` (where applicable), `model_id` or `model_digest`
- Metrics are contracts: names and label sets are stable. Changes are breaking.
- Links to requirements use ORCH-IDs when available in SPEC (none assigned yet in current SPEC).

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
