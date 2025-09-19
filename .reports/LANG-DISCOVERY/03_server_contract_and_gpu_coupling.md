# 03 — Server Contract and GPU Coupling

## Execution Primitives (intended)

| Primitive | Inputs (summary) | Outputs | Errors | Source Doc |
|---|---|---|---|---|
| run_llm (enqueue → stream) | `TaskRequest { task_id, session_id, engine, model_ref, ctx, priority, seed, max_tokens, ... }` | SSE events: `started`, `token*`, optional `metrics*`, `end`; artifacts persisted | `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNREADY`, `POOL_UNAVAILABLE`, `REPLICA_EXHAUSTED`, `DECODE_TIMEOUT`, `INTERNAL` | .docs/HOME_PROFILE.md:21–27; .specs/20-orchestratord.md:20–31,50–55,82–86 |
| run_tool (policy-gated egress) | Not a first-class primitive; outbound HTTP tooling controlled via policy hooks | — | Deny by policy | .specs/00_llama-orch.md:76–77,116–117,137–138 |
| apply_patch / artifact write | `POST /v1/artifacts { content, tags }` | `201 { id }` (content-addressed) | 4xx/5xx | .specs/20-orchestratord.md:82–86 |
| fetch_facts | `GET /v1/artifacts/{id}` | `{content,...}` | 404 | .specs/20-orchestratord.md:82–86 |
| reserve_pool / placement | Internal: policy decides pool based on snapshots and overrides | `Assigned{pool_id}` or `NoCapacity` | `NoCapacity` reasons | .specs/proposals/2025-09-19-centralized-placement-and-priority-policy.md:35–73,97–153 |
| preload_model | `POST /v1/pools/:id/reload` | Ready or rollback with last_error | 409 on conflict | .specs/20-orchestratord.md:13–19; 15 |
| capabilities | `GET /v1/capabilities` | snapshot inc. `api_version`, ctx_max, concurrency, features | — | .specs/20-orchestratord.md:76–81 |

Notes:
- SSE transport MUST be buffered, HTTP/2-preferred, optional micro-batch (off by default). .specs/20-orchestratord.md:32–37
- Cancel path MUST be race-free; proposal adds cancel-on-disconnect and bounded backpressure. .specs/20-orchestratord.md:24–25; proposals/token-streaming…:17–24,41–58

## Resource Model & Scheduling

- Pools/replicas
  - One model per worker; explicit device masks; preload before Ready. .specs/00_llama-orch.md:27–33; 145–147
  - Bounded FIFO queues per priority; backpressure 429 with advisories. .specs/00_llama-orch.md:35–40
- Model selection
  - `TaskRequest.model_ref` selects artifact; schemes include `hf:`, `file:`, `https:`, `s3:`, `oci:`. Auto-fetch per policy. .specs/00_llama-orch.md:105–119
- Placement & fairness
  - Least-loaded with VRAM awareness; deterministic tie-breakers; session affinity SHOULD apply. .specs/00_llama-orch.md:41–46
  - Centralized overrides: pin/prefer/avoid/mask with fallback semantics. .specs/proposals/2025-09-19-centralized-placement-and-priority-policy.md:51–63
- Preloading & batching
  - Pools preload at startup and after reload; batching semantics are engine-specific (not a language primitive). .specs/00_llama-orch.md:29–31; adapter specs 40–44
- Telemetry (capacity, costs)
  - Minimal metrics include queue depth, tokens, GPU/VRAM, KV cache ratio. .specs/metrics/otel-prom.md:41–61,76–98

## LLM Controls (server-validated)

- Decoding params live in adapter/engine mapping; server must validate context length and token budgets pre-admission. .specs/10-orchestrator-core.md:66–71
- Determinism controls: `seed`, `sampler_profile_version`, engine/model digests; replica sets pin versions. .specs/10-orchestrator-core.md:74–76; .specs/00_llama-orch.md:58–60
- Streaming: JSON payload shapes validated by OpenAPI; ordering must be preserved. .specs/20-orchestratord.md:26–45
- Sessions: TTL and turn caps; KV cache bounded; no cross-worker migration. .specs/00_llama-orch.md:47–53

## Observability Expectations

- Logs fields: `job_id`, `session_id`, `engine`, `engine_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`, `decode_time_ms`. .specs/00_llama-orch.md:80–83
- Metrics series and labels: counters/gauges/histograms per metrics spec; linter enforces shape. .specs/metrics/otel-prom.md:5–14,15–61,76–98,107–111
- SSE metrics frames SHOULD include additive JSON: `queue_depth`, `on_time_probability`, `kv_warmth`, budgets. .specs/00_llama-orch.md:82–83; .specs/20-orchestratord.md:91–94
- Capability payload includes API version; OpenAPI `info.version` compatible. .specs/00_llama-orch.md:94–96; .specs/20-orchestratord.md:76–81

## Gaps / Inconsistencies

- `decode_time_ms` vs `decode_ms` naming mismatch between root spec and SSE examples. .specs/00_llama-orch.md:88–89 vs .specs/20-orchestratord.md:43
- Cancel-on-disconnect and bounded backpressure are in proposals; not yet normative in root/server specs. .specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md:17–24,41–58
- Centralized placement overrides (`TaskRequest.placement`) proposed but not reflected in OpenAPI yet. .specs/proposals/2025-09-19-centralized-placement-and-priority-policy.md:159–185
- Metric bucket guidance for SSE latencies is optional and not finalized. .specs/metrics/otel-prom.md:112–118

## Top 5 Opportunities

- Promote cancel-on-disconnect, bounded channel backpressure, and heartbeat fields to normative in `/.specs/20-orchestratord.md`. .specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md:80–107
- Finalize centralized placement overrides in OpenAPI and api-types; add examples. .specs/proposals/2025-09-19-centralized-placement-and-priority-policy.md:159–186
- Align `decode_time_ms` vs `decode_ms` naming and update examples/tests. .specs/00_llama-orch.md:88–89; .specs/20-orchestratord.md:43
- Extend capability discovery payload and examples; ensure API version pinning guidance is present. .specs/20-orchestratord.md:76–81
- Add metrics series for placement decisions and candidates considered; document labels. .specs/proposals/2025-09-19-centralized-placement-and-priority-policy.md:249–256
