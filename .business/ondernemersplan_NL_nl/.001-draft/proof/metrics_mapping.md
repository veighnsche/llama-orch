# Metrics Mapping (Concept)

| Metric key | Betekenis | Bron | Opmerkingen |
|---|---|---|---|
| latency_ms | TTFB/median latency | Metrics | Per use‑case vastleggen |
| uptime_pct | Beschikbaarheid | Monitor | Per SLA‑pakket |
| queue_position | Wachtpositie | SSE/Admission | Log in frames |
| tokens_in | In tokens | Adapter | Kosten/1K tokens berekenen |
| tokens_out | Uit tokens | Adapter | Idem |
| decode_time_ms | Engine decode tijd | Adapter | Performance analyse |

Opmerking: stem namen af op .specs/metrics/otel‑prom.md (repo) en dashboards.
