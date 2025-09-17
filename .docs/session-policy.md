# Session Policy (Home Profile)

Status: normative · Scope: OrchQueue v1

- **TTL**: default ≤ 10 minutes; configurable per deployment. Sessions older than TTL MUST be evicted before new work is accepted.
- **Turns**: default maximum 8 turns; exceeding the limit MUST trigger eviction and a typed error on new requests.
- **KV Migration**: disabled across workers; failover MUST surface `kv_warmth=false` and reset budgets. See ORCH-3023.
- **Budgets**: token/time/cost budgets are optional but when enabled MUST be accounted for before enqueue. Remaining budget SHOULD appear in `X-Budget-*` headers and SSE `metrics` frames.
- **Inspection**: `GET /v1/sessions/{id}` MUST return `ttl_ms_remaining`, `turns`, `kv_bytes`, `kv_warmth`, and any active budget counters.
- **Eviction**: `DELETE /v1/sessions/{id}` MUST remove session metadata and any cached KV; it SHOULD succeed even if the session is already gone.
- **Metrics**: expose gauges/counters for active sessions, KV usage, and budget rejections to help tune agent behaviour.
