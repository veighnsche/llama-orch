# Componenten & Datacontracten

- TierCard: `{ name: 'Essential'|'Advanced'|'Enterprise', price: number, features: string[], rt: string }`
- KPIBadge: `{ label: string, value: string, status: 'ok'|'warn'|'err' }`
- StatusWidget: `{ uptime_pct: number, p50_latency_ms: number, updated_at: string, sla_clients_count?: number }`
- ContactForm: `{ name, email, company, message }` → POST `/api/contact`
- DownloadGate (optioneel): `{ email, consent }` → POST `/api/download?doc=dpiA|onprem`

## Status API (JSON)
```json
{
  "uptime_pct": 99.7,
  "p50_latency_ms": 280,
  "sla_clients_count": 6,
  "updated_at": "2025-01-05T10:00:00Z"
}
```
