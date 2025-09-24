# CLOUD\_PROFILE — Multi-Tenant & Dedicated Cloud Deployments (Draft)

**Status:** Draft (design-only)
**Relates to:** ORCH Home Profile (`.docs/00_home_profile.md`)
**Owner:** Orchestrator / DevOps
**Intent:** Define requirements to run `llama-orch` as (a) a shared, metered Agentic API and (b) a dedicated, per-customer API on cloud GPUs. This doc is additive; it does **not** change Home behavior.

> RFC-2119 keywords **MUST/SHOULD/MAY** apply.

---

## 0. Goals & Non-Goals

* **Goals**

  * \[CLOUD-100] Provide a **multi-tenant** (shared) agentic API with quotas, budgets, and metering.
  * \[CLOUD-101] Provide a **single-tenant/dedicated** deployment preset (“Level 2”) using the same core.
  * \[CLOUD-102] Keep DevOps **open-source**; keep metering/billing pluggable (private repo OK).
  * \[CLOUD-103] Preserve **determinism**, **observability**, and **fail-fast** posture from Home.

* **Non-Goals**

  * \[CLOUD-110] No enterprise RBAC/SSO in v1 (API keys only).
  * \[CLOUD-111] No CPU inference fallback; **NVIDIA-GPU only**.

---

## 1. Deployment Envelopes

* **Shared (Multi-Tenant)**

  * \[CLOUD-200] Public HTTPS endpoint behind gateway/WAF.
  * \[CLOUD-201] Per-API-key quotas (RPM/TPM/concurrency) enforced *pre-admission*.
  * \[CLOUD-202] Usage metering events emitted for each job/end/reject.

* **Dedicated (Single-Tenant)**

  * \[CLOUD-210] Private endpoint per customer (own VPC/project/namespace).
  * \[CLOUD-211] Same quotas interface **MAY** be disabled; metering remains available.
  * \[CLOUD-212] Models preloaded to hit cold-start SLOs (see §7).

---

## 2. Platform Assumptions

* \[CLOUD-300] Hosts are cloud VMs or k8s nodes with **NVIDIA GPUs** and drivers.
* \[CLOUD-301] Supported engines include `llama.cpp`, `vLLM`, `TGI`, `Triton` (per core spec).
* \[CLOUD-302] Artifact/model cache persists on node-local NVMe or attached SSD; cold fetch allowed per policy.

---

## 3. Security, Auth, Tenancy

* \[CLOUD-400] **Auth required** on non-loopback binds: static **API keys** via `Authorization: Bearer`.
* \[CLOUD-401] Keys map to **tenants**; each key has a **Policy** object with budgets/quotas.
* \[CLOUD-402] Multi-tenant logs and metrics **MUST** carry `tenant_id` (from key).
* \[CLOUD-403] No PII in logs; redact tokens/secrets.
* \[CLOUD-404] Network: public LB → gateway → orch data-plane; control-plane scoped to ops networks only.

**Error semantics**

* \[CLOUD-420] Policy violation pre-enqueue: `HTTP 429` with `Retry-After`, body `{ code, message, policy_label, retriable, retry_after_ms }`.
* \[CLOUD-421] Invalid request: `HTTP 400` with `code=INVALID_PARAMS`.
* \[CLOUD-422] Pool unavailable: `HTTP 503` with `code=POOL_UNAVAILABLE`.

---

## 4. Quotas, Budgets, Backpressure

* \[CLOUD-500] **Per-key quotas** (enforced pre-admission):

  * Requests/min (RPM)
  * Tokens/min (TPM) (expected or historical window)
  * Concurrent jobs
  * Optional time or cost budgets

* \[CLOUD-501] Queue full policy MUST be `reject` or `drop-lru` (documented per pool).

* \[CLOUD-502] Backpressure MUST return 429 + `Retry-After` and include a **deterministic** policy label.

* \[CLOUD-503] Idempotency: clients MAY pass `Idempotency-Key`; server SHOULD dedupe for short windows.

---

## 5. Metering & Billing Interface

*(DevOps remains OSS; metering service can be private.)*

* \[CLOUD-600] Orchestrator MUST emit **usage events** at **stream end** and on **reject**:

  ```
  {
    "ts": "2025-09-24T10:00:00Z",
    "tenant_id": "acc_123",
    "api_key_hash": "sha256:…",
    "job_id": "uuid",
    "model_ref": "hf:org/repo",
    "engine": "vllm",
    "tokens_in": 123,
    "tokens_out": 456,
    "decode_ms": 9876,
    "duration_ms": 10023,
    "exit": "ok|cancel|error|reject",
    "error_code": null|"QUEUE_FULL_DROP_LRU"|…,
    "region": "eu-west-1"
  }
  ```
* \[CLOUD-601] Transport MUST support **JSONL file** and **HTTP POST** (batch) to `USAGE_URL`.
* \[CLOUD-602] Emission MUST be **non-blocking** (buffered; drop oldest with counter if sink stalls).
* \[CLOUD-603] Prometheus labels MUST NOT explode cardinality (no raw `job_id` as label).

---

## 6. Observability

* \[CLOUD-700] **Prometheus** endpoints for: queue depth, enqueued/started/rejected/canceled, tokens in/out, GPU util/VRAM, KV pressure where applicable.
* \[CLOUD-701] **Tracing (OTel)** SHOULD cover admission → placement → stream start/end with `job_id` correlation (kept out of labels).
* \[CLOUD-702] Provide a default **Grafana** dashboard JSON (tokens/sec, waiters, p50/p95 latency, GPU util, errors).

---

## 7. SLOs & Rollouts

* **Shared multi-tenant baseline**

  * \[CLOUD-800] Availability: 99.5% monthly for data-plane.
  * \[CLOUD-801] p95 admission wait < 10s under nominal load; document saturation behavior.
  * \[CLOUD-802] Cold-start: model preload policy to avoid first-request timeouts.

* **Dedicated baseline**

  * \[CLOUD-810] Preload required models before marking pool Ready.
  * \[CLOUD-811] Change windows & blue/green deploys; **instant rollback** on SLO breach.

* **Rollouts**

  * \[CLOUD-820] Version pinning for engine and sampler profiles; mixed versions **forbidden** in a replica set.
  * \[CLOUD-821] Canary percentage & automatic abort on error rate/latency regression.

---

## 8. Scheduling & Placement

* \[CLOUD-900] Scheduler MUST be **VRAM-aware**; fail fast on model not fitting.
* \[CLOUD-901] Default heuristic: most free VRAM → fewest active slots → deterministic tie-break.
* \[CLOUD-902] Session affinity SHOULD be honored when capacity allows; on failover expose `kv_warmth=false`.

---

## 9. Config Schema (Cloud Deltas)

Extend config with a **Tenant Policy Map**:

```yaml
tenants:
  - id: acc_123
    name: "Acme"
    api_keys:
      - "sk_live_…"
    quotas:
      rpm: 120
      tpm: 120000
      concurrent: 4
      time_budget_ms: 600000
      cost_budget_eur: null
usage_sink:
  mode: "http"   # "jsonl"|"http"
  jsonl_path: "/var/log/orch/usage.jsonl"
  http_url: "https://metering.internal/usage"
  api_key: "mtr_…"
gateway:
  enable_openai_compat: true
```

* \[CLOUD-1000] If `gateway.enable_openai_compat=true`, the OpenAI shim MUST translate to native requests and error envelopes losslessly.

---

## 10. DevOps Profiles

* **Kubernetes (preferred)**

  * \[CLOUD-1100] Provide Helm chart with values for **shared** and **dedicated** presets.
  * \[CLOUD-1101] NVIDIA device plugin + node labels/taints for GPU pools.
  * \[CLOUD-1102] Autoscaling via **HPA/KEDA** using custom metrics (tokens/sec, queued).
  * \[CLOUD-1103] Optional gateway (OpenAI-compat) as sidecar or separate Deployment.

* **Terraform (IaaC)**

  * \[CLOUD-1110] Modules for at least one cloud (e.g., AWS or RunPod) to provision GPU nodes, storage, and LB.
  * \[CLOUD-1111] Output kubeconfig and values files for Helm apply.
  * \[CLOUD-1112] Dedicated preset spins isolated namespace/VPC per customer.

* **Docker Compose (dev)**

  * \[CLOUD-1120] Single-node stack with mocked usage sink for local validation.

---

## 11. Data Protection & Retention

* \[CLOUD-1200] Logs and usage events MUST be **purpose-bound**; default retention ≤ 30 days (configurable).
* \[CLOUD-1201] Artifacts/models cache retention documented; operator controls clean-up.
* \[CLOUD-1202] Subprocessor list & data-flow diagrams maintained in `/legal/`.

---

## 12. Incident Management

* \[CLOUD-1300] Severities S0–S3 (see incident SOP).
* \[CLOUD-1301] Customer-visible status page for shared profile; dedicated may opt-in.
* \[CLOUD-1302] Post-mortems with actionable remediations; update runbooks and quotas if capacity-linked.

---

## 13. Compatibility & Migration

* \[CLOUD-1400] API and SSE events remain identical to Home; Cloud adds auth/quotas only.
* \[CLOUD-1401] Config keys **MUST** have sensible defaults so Home configs run unchanged unless Cloud-only features are enabled.

---

## 14. Validation & Tests

* \[CLOUD-1500] **Smoke**: provision → preload → minimal decode → cancel → reject path (429).
* \[CLOUD-1501] **Load**: saturation to trigger backpressure, verify `Retry-After` and no post-cancel tokens.
* \[CLOUD-1502] **Determinism**: identical seeds/versions → byte-identical streams on same replica.
* \[CLOUD-1503] **Metering**: golden usage events produced for ok/cancel/error/reject.
* \[CLOUD-1504] **Autoscaling** (k8s): scale-out on tokens/sec threshold; scale-in with cooldown.

---

## 15. Open Items (to resolve when implementing)

* \[OPEN-1] Choose first cloud targets (e.g., AWS + RunPod).
* \[OPEN-2] Define minimal `usage_event.schema.json` and publish in `/contracts/`.
* \[OPEN-3] Provide Grafana dashboard JSON and HPA/KEDA sample rules.
* \[OPEN-4] Decide on OpenAI gateway sidecar vs. integrated router.
* \[OPEN-5] Document model preload matrices per GPU class (VRAM → model sizes).

---

### Appendix A — Error Envelope (Cloud)

```json
{
  "code": "ADMISSION_REJECT | QUEUE_FULL_DROP_LRU | INVALID_PARAMS | POOL_UNREADY | POOL_UNAVAILABLE | REPLICA_EXHAUSTED | DEADLINE_UNMET | INTERNAL",
  "message": "string",
  "policy_label": "queue.reject.rpm" ,
  "retriable": true,
  "retry_after_ms": 1200,
  "engine": "vllm"
}
```

### Appendix B — SSE Contract (unchanged)

Events: `started`, `token`, `metrics`, `end`, `error`
`started` includes `queue_position` and `predicted_start_ms`.
`metrics` additive (tokens budget, queue depth, kv warmth, on\_time\_probability).
`cancel` guarantees no tokens after acknowledgment.

---

## Practical notes

* **You, as one person:** Keep **Home** as your reference runtime (fast iteration). Cloud = presets + DevOps + quotas + metering shim. Most changes are **config + small hooks**, not deep rewrites.
* **Now:** Park this file, finish the website to win Qredits. When ready, convert OPEN items into tickets and let IDE-AI scaffold DevOps/metering repos.
