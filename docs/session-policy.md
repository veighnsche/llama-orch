# Session Policy

Status: pre-code · Scope: OrchQueue v1

- TTL: sessions are short-lived; TTL ≤ 10 minutes.
- Turns: maximum 8 turns per session.
- KV migration: disabled across workers; failover surfaces `kv_warmth=false`.
- Metrics to emit per session: `turns`, `kv_bytes`, `kv_warmth`, `ttl_ms_remaining`.
