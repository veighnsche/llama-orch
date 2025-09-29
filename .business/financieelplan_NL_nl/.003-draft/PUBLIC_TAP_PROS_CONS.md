# Public Tap (Shared Inference API) — Pros and Cons (Draft 3 Context)

This document evaluates the trade-offs of operating a multi-tenant, pay-per-token “Public Tap” (shared inference API), grounded in the evidence and codepaths of draft 3 under `.business/financieelplan_NL_nl/.003-draft/`.

## Executive Summary
- **Bottom line**: With draft 3’s assumptions, a Public Tap is a weak profit engine. It is valuable as a top-of-funnel (TOFU) acquisition and R&D surface, but unlikely to cover fixed opex unless you (a) raise prices, (b) radically improve throughput/cost, or (c) scale to very large volumes. If focus and runway are scarce, **go Private-first** and treat Public as optional/limited.

## Quick Numbers from Draft 3
- **Observed unit economics (example row)**: `outputs/public_tap_prices_per_model.csv`
  - `cost_eur_per_1M ≈ 2.942760942760943` on `RTX 3090`
  - `sell_eur_per_1k ≈ 0.007` → implied `€7 / 1M`
  - `margin_pct ≈ 57.96%` (because price greatly exceeds modeled cost at that floor)
- **Total contribution is small**: `outputs/consolidated_summary.json`
  - Public revenue (total): `€3,932.28`
  - Public costs (total): `€1,653.11`
  - Public margin (total): `€2,279.17`
- **Overall P&L still negative**: `outputs/pnl_by_month.csv` shows negative EBITDA across months, driven by fixed opex (`inputs/operator/general.yaml` ≈ €4,275/month).
- **Drivers in code**:
  - GPU baselines: `inputs/facts/gpu_baselines.yaml` (e.g., RTX 3090 = 150 tps)
  - Provider cost basis: `inputs/operator/curated_gpu.csv`
  - Pricing policy: `engine/src/pipelines/public/pricing.py` and `public/artifacts.py`
  - KPIs: `engine/src/aggregate/kpis.py`

## Pros — Why keep a Public Tap
- **Top-of-funnel growth**
  - Low-friction onboarding. Self-serve API with per-token pricing helps developers try before larger commitments.
  - Natural discovery vector; pairs well with SDKs and examples.
- **Lead generation for Private Tap**
  - Convert active Public accounts to higher-ARPU private deployments with SLAs, VPC isolation, and custom models.
- **Product/UX validation**
  - Real usage informs model choices, prompt tooling, SSE/streaming UX, and error handling.
  - A/B tests and feature flags are easier on a shared plane.
- **Telemetry and benchmarking**
  - Cross-model performance data by prompt length, temperature, and batching feeds capacity planning and pricing calibration.
- **Operational readiness**
  - Exercising queuing, autoscaling, tracing, and on-call playbooks in production-like conditions strengthens the platform.
- **Brand surface and credibility**
  - Public status page, docs, and quickstart improve visibility; supports content and community marketing.
- **Marginal revenue (non-zero)**
  - Even thin margins can offset a slice of GPU time if utilization is high and overheads are controlled.

## Cons — Why avoid or limit a Public Tap
- **Economics under price pressure**
  - Hyperscalers set very low per-token benchmarks you cannot match on rentals + modest TPS. Competing on price alone is a losing game.
  - Thin margins amplify billing/currency fees, fraud loss, and support costs.
  - Volatility in demand → underutilization → higher effective cost per 1M.
- **Abuse and fraud risk**
  - Key scraping, card testing, bot farms; mitigation requires KYC, velocity controls, fingerprinting, and manual review.
  - Content moderation and abuse handling impose real costs and liabilities.
- **Operational complexity**
  - Multi-tenant fairness, rate limits, quota tiers, preemption, and anti-hoarding.
  - SSE longevity, backpressure, retries, and head-of-line blocking.
  - Capacity management and batching across heterogeneous models/GPUs.
- **Support and SLO/SLAs**
  - Ticket load, incident response, and reliability expectations rise quickly with external consumers.
  - Public incidents damage brand disproportionately relative to revenue.
- **Compliance and legal**
  - GDPR/PII handling, content policies, takedowns, and export controls.
  - Retention policies vs. developer expectations around training/telemetry.
- **Strategic distraction**
  - Building a public-grade API, billing, and abuse stack steals cycles from higher-ARPU private deals.
  - Easy to fall into a “race to the bottom” while starving the differentiators.

## When a Public Tap Makes Sense
- **As a TOFU channel with guardrails**
  - Strict free-tier limits, verified billing before heavy use, generous but sustainable price floor (e.g., €0.003–€0.010 per 1k for 7B-class), and clear upgrade path to Private Tap.
- **If you unlock better unit costs**
  - Measured TPS materially above baselines (batching improvements, KV cache reuse, speculative decoding), or negotiated GPU rates, or owned hardware with good utilization.
- **If you need a production R&D loop**
  - A controlled set of models/features is exposed for experimentation, feature validation, and telemetry.

## When to Avoid or Pause Public Tap
- **Runway at risk**
  - If it cannot materially contribute to covering fixed opex and diverts resources, pause.
- **No competitive angle beyond price**
  - Without a differentiator (dx, vertical focus, latency/SLA, or unique models), public comps will anchor you into loss-making tiers.
- **Abuse overwhelms ops**
  - If fraud/abuse outpaces your ability to police it, shut down or revert to allow-listed usage only.

## Mitigations if You Keep It
- **Pricing guardrails**
  - Enforce `min_floor_eur_per_1k` and use round-up increments to protect margin; publish price component rationale.
- **Strong auth and billing**
  - Pre-authorization, spend caps, velocity-based throttles, refundable deposits for high limits, and 3DS/SCA where possible.
- **Tiered quotas and fair usage**
  - Per-key and per-org limits; surge controls; regional gating; ban VPN/proxy ranges known for abuse.
- **Abuse and moderation pipeline**
  - Prompt/content heuristics, rate-based triggers, manual review queue, and retrospective clawbacks.
- **Observability and transparency**
  - Per-tenant metrics, anomaly alerts, and a public status page with partial outage communication playbooks.

## Decision Framework
- **Break-even yardsticks**
  - Required price for break-even at forecast volume:
    ```text
    price_per_1k = (cost_eur_per_1M / 1000) / (1 - target_margin)
    ```
  - Required monthly tokens for covering opex at chosen price:
    ```text
    tokens_needed_per_month = opex_per_month / ( (price_per_1k - cost_per_1k) / 1000 )
    ```
- **Conversion yardsticks**
  - Target Public→Private conversion rate and expected time-to-convert. If Public does not feed Private, reconsider.

## Draft 3–Specific Observations to Inform the Decision
- **Unit cost and price**: `RTX 3090` baseline yields `cost_eur_per_1M ≈ €2.94`; selling at `€0.007/1k` gives healthy gross margin on paper, but absolute revenue is tiny.
- **Scale of revenue**: Public total revenue over the whole plan is only ~`€3.9k` (`outputs/consolidated_summary.json`), dwarfed by fixed opex.
- **Overall profitability**: P&L remains negative each month (`outputs/pnl_by_month.csv`), so Public Tap—while not loss-making per token at the current floor—does not move the needle.

## Recommendation (Draft 3)
- **Private-first** focus. Keep Public only if it serves a clear acquisition pipeline with hard controls and minimal engineering overhead.
- If you keep it:
  - Set a sustainable floor (e.g., `€0.003–€0.010` per 1k for 7B-class); use round-up increments.
  - Gate high-usage behind verified billing and deposits.
  - Narrow the exposed model set and regions; invest in abuse prevention before growth.
- If you drop it:
  - Preserve internal endpoints for demos and benchmarks, but remove public signups and docs until unit economics or differentiation improves.

## Pointers to Relevant Files
- Pricing outputs: `outputs/public_tap_prices_per_model.csv`
- Consolidated totals: `outputs/consolidated_summary.json`
- P&L: `outputs/pnl_by_month.csv`
- GPU baselines: `inputs/facts/gpu_baselines.yaml`
- Provider pricing: `inputs/operator/curated_gpu.csv`
- Pricing logic: `engine/src/pipelines/public/pricing.py`, `engine/src/pipelines/public/artifacts.py`
- KPIs: `engine/src/aggregate/kpis.py`

---

If you want, I can implement a “Public-lite” mode (strict floor, quotas, billing gates) or remove it from the v3 plan and re-run the outputs to compare runway and KPIs.
