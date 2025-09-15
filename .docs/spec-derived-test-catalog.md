# Spec‑Derived Test Catalog (Traceable)

Status: generated v2025-09-15
Source of Truth: `.specs/*.md` (see IDs) and `README_LLM.md` rules.

This catalog enumerates all testable requirements with proposed test artifacts. Each entry references a requirement ID and includes a short description, test type(s), and suggested test locations. Keep this file diff‑clean across regenerations; update via proposals when specs change.

---

## ORCH — Top‑Level Orchestrator

- ORCH-1101 — Inference hosts MUST have NVIDIA GPUs; CPU‑only hosts MUST NOT serve inference.
  - Types: BDD e2e (env gate), integration (startup guard)
  - Tests: `test-harness/bdd/` (security/platform), `orchestratord/tests/`

- ORCH-3001 — One‑model/one‑device‑mask worker processes pinned by config.
  - Types: integration, BDD
  - Tests: `pool-managerd/tests/`, `test-harness/bdd/src/steps/pool_manager.rs`

- ORCH-3002 — Pools preload at serve; Ready only after success.
  - Types: integration, BDD
  - Tests: `pool-managerd/tests/`, BDD lifecycle

- ORCH-3003 — Preload fails fast on insufficient memory; Unready + backoff.
  - Types: chaos/integration, BDD
  - Tests: `pool-managerd/tests/`

- ORCH-3004 — Bounded FIFO queue per Pool.
  - Types: property, integration
  - Tests: `orchestrator-core/tests/props_queue.rs`

- ORCH-3005 — Full queue policy is reject/drop-lru/shed-low-priority.
  - Types: BDD, integration
  - Tests: `orchestratord/tests/`, BDD scheduling

- ORCH-3006 — Per-client rate limits/burst buckets enforced before enqueue.
  - Types: integration, BDD
  - Tests: `orchestratord/tests/`

- ORCH-2007 — Backpressure headers MUST be returned.
  - Types: CDC/OpenAPI provider, integration
  - Tests: `orchestratord/tests/provider_verify.rs`

- ORCH-3007 — Replica set grouping rule.
  - Types: property, integration
  - Tests: `orchestrator-core/tests/`

- ORCH-3008 — Least‑loaded Ready placement; masks respected.
  - Types: property, integration, BDD
  - Tests: `orchestrator-core/tests/`, BDD scheduling

- ORCH-3009 — Session affinity SHOULD keep last good replica; failover MAY surface `kv_warmth=false`.
  - Types: BDD, integration
  - Tests: BDD scheduling

- ORCH-3010 — Do not dispatch until Ready.
  - Types: integration
  - Tests: `orchestrator-core/tests/`

- ORCH-3011 — Respect device masks; no spillover.
  - Types: property/integration
  - Tests: `orchestrator-core/tests/`, `pool-managerd/tests/`

- ORCH-3012 — Heterogeneous multi‑GPU splits opt‑in with ratios.
  - Types: integration/BDD
  - Tests: `pool-managerd/tests/`

- ORCH-3013 — Topology hints SHOULD influence placement.
  - Types: property/integration
  - Tests: `orchestrator-core/tests/`

- ORCH-3014 — Reject ctx > model limit before enqueue (400).
  - Types: provider tests, BDD guardrails
  - Tests: `orchestratord/tests/`, BDD core_guardrails

- ORCH-3015 — Validate token budget pre‑admission.
  - Types: provider tests, BDD guardrails

- ORCH-3016 — Watchdog aborts stuck jobs with wall/idle timeouts.
  - Types: integration/chaos

- ORCH-3017 — Cancel frees slot; terminal state.
  - Types: provider/integration

- ORCH-3018 — No CPU spillover; fail fast on outage.
  - Types: integration/BDD

- ORCH-3019 — Continuous batching exposure affects placement.
  - Types: integration/property

- ORCH-3020 — Speculative/prefix caching admissions account for memory.
  - Types: integration/property

- ORCH-3021 — Sessions short‑lived; TTL/turns configurable; MAY be delegated to policy.
  - Types: BDD lifecycle/policy

- ORCH-3022 — KV cache bounded; pressure metrics exposed.
  - Types: metrics, BDD

- ORCH-3023 — No cross‑Worker KV migration; failover flag.
  - Types: BDD

- ORCH-3024 — Unique job_id per Job.
  - Types: provider

- ORCH-3025 — Retries target retryable error classes.
  - Types: provider/semantics

- ORCH-3026 — Race‑free cancellation.
  - Types: provider

- ORCH-3027 — Logs fields coverage.
  - Types: observability contract
  - Tests: metrics/logs BDD + linters

- ORCH-3028 — Metrics names/labels and counters coverage.
  - Types: metrics contract
  - Tests: `test-harness/metrics-contract/` per `ci/metrics.lint.json`

- ORCH-3029 — Admission logs + SSE started include queue_position and predicted_start_ms.
  - Types: provider/observability

- ORCH-3030 — Config schema strict validation.
  - Types: config-schema tests

- ORCH-3031 — Hot‑reload atomic + revertible.
  - Types: integration

- ORCH-3032 — Workers report engine_version and model_digest.
  - Types: adapter/provider

- ORCH-3033 — Canaries supported; rollback one action.
  - Types: control‑plane tests

- ORCH-3035 — AuthN/AuthZ gates.
  - Types: security tests

- ORCH-3036 — Per‑tenant quotas bound concurrent jobs and memory.
  - Types: BDD scheduling/quotas

- ORCH-3037 — Model artifacts checksummed & verified before load.
  - Types: pool manager/integration

- ORCH-3038 — Driver/CUDA errors → Unready + drain + backoff‑restart.
  - Types: chaos/integration

- ORCH-3039 — Distinguish VRAM vs host OOM.
  - Types: observability/integration

- ORCH-3040 — Circuit breakers shed on sustained SLO violations.
  - Types: reliability tests

- ORCH-3041 — Define & measure per‑priority SLOs + alerts.
  - Types: observability/perf

- ORCH-3042 — Storage checksums; cache quotas/eviction.
  - Types: integration

- ORCH-3044 — Public APIs versioned.
  - Types: contracts

- ORCH-3045 — Determinism within replica set with fixed seeds etc.
  - Types: determinism suite

- ORCH-3046 — Pin engine_version/sampler_profile_version; no mixing.
  - Types: determinism/config

- ORCH-3047 — No cross‑version determinism.
  - Types: determinism

- ORCH-2002 — SSE stream framing stable.
  - Types: provider/SSE tests

- ORCH-3048 — Default plugin ABI WASI and deterministic.
  - Types: policy host tests

- ORCH-3049 — Startup self‑tests coverage.
  - Types: integration smoke

- ORCH-3050 — Determinism tests per engine with specific settings.
  - Types: determinism suite

- ORCH-3051 — Chaos tests and priority inversion coverage.
  - Types: chaos/load

- ORCH-3052 — Heterogeneous split policy caps KV; topology hints.
  - Types: placement/integration

- ORCH-3053 — Driver resets surfaced; restart storms bounded.
  - Types: pool manager/observability

- ORCH-3054..3058 — Adapter contracts and normalization.
  - Types: adapter integration/CDC

- ORCH-3060..3094 — v3.2 catalog/lifecycle/scheduling glue.
  - Types: control‑plane, BDD lifecycle, BDD scheduling, metrics

---

## OC‑CORE — Core

- OC-CORE-1001..1005 — Queue invariants
  - Types: property tests
  - Tests: `orchestrator-core/tests/props_queue.rs`

- OC-CORE-1010..1013 — Scheduling & placement
  - Types: property, BDD

- OC-CORE-1020..1022 — Capacity & guardrails
  - Types: provider/BDD

- OC-CORE-1030..1032 — Determinism
  - Types: determinism suite

- OC-CORE-1040..1041 — Observability
  - Types: metrics/logs

---

## OC‑CTRL — Orchestratord

- OC-CTRL-2001..2004 — Control plane endpoints
  - Types: OpenAPI provider verification

- OC-CTRL-2010..2012 — Data plane admission/cancel semantics
  - Types: provider tests

- OC-CTRL-2020..2022 — SSE framing and payloads
  - Types: SSE tests

- OC-CTRL-2030..2031 — Error taxonomy/logging context
  - Types: provider tests

- OC-CTRL-2040..2041 — Security
  - Types: security tests

- OC-CTRL-2050..2051 — Observability/metrics
  - Types: metrics tests

---

## OC‑POOL — Pool Managerd

- OC-POOL-3001..3003 — Preload/Ready lifecycle
- OC-POOL-3010..3012 — Restart/backoff/CPU spillover disallowance
- OC-POOL-3020..3021 — Device masks/heterogeneous splits
- OC-POOL-3030 — Observability counters

Types: integration/chaos; Tests: `pool-managerd/tests/`

---

## OC‑ADAPT — Worker Adapters

- OC-ADAPT-5001..5011 — llama.cpp
- OC-ADAPT-5020..5030 — vLLM
- OC-ADAPT-5040..5050 — TGI
- OC-ADAPT-5060..5070 — Triton/TRT‑LLM

Types: adapter integration/CDC; Tests: under `worker-adapters/*/tests/` and BDD adapter features.

---

## OC‑POLICY — Policy Host & SDK

- OC-POLICY-4001..4020 — ABI, safety, telemetry
- OC-POLICY-SDK-4101..4110 — SDK stability and safety

Types: unit/integration for host; SDK tests; BDD policy features.

---

## OC‑CONFIG — Config Schema

- OC-CONFIG-6001..6010 — Validation and generation

Types: schema tests; Tests: `contracts/config-schema/tests/`.

---

## OC‑TEST — Determinism Suite

- OC-TEST-7001..7003 — Byte‑exact streams, engine settings, seed corpus

Types: determinism suite; Tests: `test-harness/determinism-suite/tests/`.

---

## OC‑METRICS — Metrics Contract

- OC-METRICS-7101..7102 — Names/labels and budgets

Types: metrics linter/tests; Tests: `test-harness/metrics-contract/`.

---

# Notes

- For each bullet above, ensure the corresponding `requirements/*.yaml` file links requirement → tests → code path. Keep this catalog synchronized via the regeneration loop.
