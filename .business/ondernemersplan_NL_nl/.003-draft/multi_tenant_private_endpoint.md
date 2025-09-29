# Multi‑Tenant Private Endpoint (Draft 3)

This note captures the product and business direction: no public tap; no 1:1 GPU-per-customer private tap. Instead, a multi‑tenant platform where each customer gets a private API endpoint (namespaced) backed by shared GPU servers to raise utilization and margins. Accessing the private endpoint also grants free access to a hosted chat front‑end for convenience.

## Why this direction
- **Avoids public-tap pitfalls**: price race, abuse surface, weak contribution to opex.
- **Avoids 1:1 GPU razor‑thin margins**: rentals + idle time crush unit economics.
- **Leans into strengths**: orchestration, placement, queuing, and agentic workflows across GPU pools.

## Product definition
- **Private namespaced endpoint**
  - Example: `https://api.llama-orch.dev/tenants/{tenant_id}/v1/chat` (SSE streaming) and related endpoints.
  - The endpoint runs the llama‑orch agentic AI (tools, plans, function calls), with sensible defaults per tenant.
- **Included chat front‑end**
  - A hosted web UI bound to the tenant’s endpoint for human testing and collaboration.
  - SSO to the same tenant identity; role‑based access (Owner, Developer, Analyst, Read‑only).
- **Multi‑tenant compute**
  - Shared GPU pools with admission→dispatch→SSE queueing.
  - Automatic placement across GPUs with policy controls; optional pinning/overrides per tenant when needed.

## Pricing and quotas (anti‑loss guardrails)
- **Subscription per endpoint (per tenant)**
  - Tiers include bundled monthly quota (tokens or GPU‑minutes) and SLOs.
  - Example tiers (illustrative):
    - Starter: 50M tokens/month, soft cap, community support.
    - Growth: 250M tokens/month, priority queue, email support.
    - Scale: 1B tokens/month, higher priority, account manager.
- **Overage and burst**
  - Overage priced per 1k tokens or per GPU‑minute with a floor ensuring positive gross margin.
  - Optional “burst credits” to absorb short spikes; auto‑throttle at hard caps.
- **Rate limits**
  - Per‑tenant TPS caps and concurrency limits to protect fairness.
- **Fair Use Policy (FUP)**
  - Guardrails against scraping, non‑human traffic, and automated hoarding.
- **Billing**
  - Pre‑authorized card or deposit before high limits. Monthly invoice with metered overage.

## Technical architecture (high level)
- **Orchestration pipeline**
  - Stage 6: Admission → Dispatch → SSE. `orchestrator-core/` enforces queue invariants; `orchestratord/` hosts HTTP/SSE.
- **Placement & pools**
  - Default: automatic engine→GPU placement optimizing queue and cost.
  - Overrides: per‑tenant pinning to model/engine/GPU/pool when required (SLA or compliance).
  - GPU is required; fail fast if capacity is unavailable (no CPU fallback).
- **Isolation**
  - Per‑tenant tokens, rate limits, and budgets.
  - Namespaced API keys, separate logs and traces.
- **Observability**
  - Metrics per tenant: `tokens_in`, `tokens_out`, `queue_position`, `predicted_start_ms`, `decode_time_ms`, errors.
  - Dashboards and alerting for saturation, timeouts, and SLO breaches.
- **Frontend**
  - Hosted chat bound to the tenant’s endpoint; uses the same SSE contracts and auth.

## Security, privacy, compliance
- **Data isolation**: tenant data is logically isolated; no cross‑tenant leakage.
- **Retention**: configurable retention; default no training on customer prompts/outputs.
- **Compliance**: EU region option; DPA terms; content policy and takedowns.

## SLO/SLA (illustrative)
- **Availability**: 99.5% monthly for paid tiers; maintenance windows announced.
- **Latency targets**: p50/p95 end‑to‑first‑token and tokens/sec ranges per model family.
- **Support**: response times by tier; incident comms via status page and mail.

## Economics & capacity
- **Shared GPU efficiency**
  - Higher utilization across tenants increases margin vs 1:1 GPU.
  - Planner sizes pools by forecasted tenant demand; autoscaler handles bursts.
- **Quotas protect downside**
  - Hard caps prevent runaway costs; soft caps + throttling smooth experience.
- **Placement policy levers**
  - Batch sizing, speculative decoding, KV cache reuse to improve tokens/sec.
  - Vendor selection by EUR/hr and availability; ability to shift pools when costs change.

## Risks and mitigations
- **Noisy neighbors / contention**
  - Mitigation: priority tiers, per‑tenant concurrency, isolation queues.
- **Abuse/fraud**
  - Mitigation: verified billing, velocity heuristics, device/IP reputation, manual review.
- **Price competition**
  - Mitigation: focus on agentic features, latency, developer UX, and vertical solutions rather than raw token price.
- **Vendor volatility**
  - Mitigation: multi‑provider pools, rapid rebalancing, internal price floors.

## Go‑to‑market notes
- **Positioning**: "Your private AI endpoint, ready in minutes—shared GPU efficiency, enterprise‑grade controls."
- **Upgrade path**: SDK → private endpoint → higher tiers → optional model pinning and enterprise add‑ons.
- **Bundle**: hosted chat, logs, analytics, and team workspaces.

## MVP scope (Draft 3)
- Namespaced endpoints with API keys.
- Shared pool placement with default policy and per‑tenant overrides.
- Subscription tiers with quotas and basic overage.
- Hosted chat UI bound to tenant; simple org roles.
- Per‑tenant metrics dashboard; CSV export.

## Next steps
- Write the minimal contracts and config for: tiers, quotas, rate limits, and placement overrides.
- Implement quota/rate enforcement in `orchestrator-core/` and surface limits/errors in `orchestratord/`.
- Add tenant‑bound chat UI under `frontend/` with SSE and API key auth.
- Define pricing floors and overage tables; connect billing provider.
- Prepare a private‑first launch plan; keep public endpoints disabled or allow‑listed only.
