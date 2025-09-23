# Dynamiek (SSG‑compatibel)

- Live stats: client fetch naar /api/status (Pages Function) met JSON (uptime_pct, p50_latency_ms, sla_clients_count)
  - Bronopties: Cloudflare KV/R2 of handmatig geüpdatete JSON (interim)
  - Fallback SSR: render ‘laatst bijgewerkt’ waarden vanaf build JSON
- Download counts (optioneel): client fetch + KV increment
- Status badges: eenvoudige kleur (groen/oranje/rood) op thresholds (metrics_mapping.md)
