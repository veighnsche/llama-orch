# Spec Combination Matrix — Pairwise and 3‑wise Coverage

Status: v2025-09-15
Source of Truth: `.specs/*.md` (IDs). See method: `.docs/test-case-discovery-method.md`.

## Factors (with domains and references)

- Engine (OC-ADAPT): `llamacpp | vllm | tgi | triton` → OC-ADAPT-5001..5070
- Queue Full Policy: `reject | drop-lru | shed-low-priority` → ORCH-3005
- Fairness: `wfq_on/off` → ORCH-3075; Observed share → ORCH-3076
- Quotas: `quotas_on/off` → ORCH-3077
- Deadlines: `feasible | infeasible` → ORCH-3079
- Preemption: `off | soft | hard(capability)` → ORCH-3085/3086, metrics ORCH-3087
- Lifecycle: `Active | Deprecated | Retired` → ORCH-3069..3073
- Trust Policy: `strict_signed | strict_unsigned` → ORCH-3060..3065, ORCH-3093
- Auth: `apikey_present | apikey_missing` → OC-CTRL-2040
- Host/Placement: `Ready | Unready`, `mask_respected` → ORCH-3010/3011, OC-POOL-3001..3021
- SSE Started Fields: `present | missing` → ORCH-3029, OC-CTRL-2021
- Determinism Context: `same_version | cross_version` → ORCH-3045/3047

## Constraints (prune invalid or undefined states)

- Hard preemption only when adapter proves `interruptible_decode` → ORCH-3086
- CPU‑only hosts cannot serve inference → ORCH-1101
- No cross‑mask spillover → ORCH-3011, OC-POOL-3020
- OpenAI‑compatible endpoints are internal only → OC-ADAPT-5002/5021
- No determinism guarantee across engine/model updates → ORCH-3047
- Deprecated blocks new sessions unless `override=true`; Retired unloads → ORCH-3070

## Pairwise selection (prioritized set)

- (Engine × Queue Full Policy)
  - Validate reject/drop-lru/shed policies against each engine. IDs: ORCH-3005, OC-ADAPT-5xxx.
- (Engine × Determinism Context)
  - Same version strict determinism, and cross-version nondeterminism expectation. IDs: ORCH-3045/3047.
- (Engine × Preemption)
  - soft across all engines; hard only where capability proven. IDs: ORCH-3085/3086.
- (Fairness × Quotas)
  - WFQ active with and without quotas; observe share and enforcement. IDs: ORCH-3075/3076/3077.
- (Deadlines × Preemption)
  - Feasible vs infeasible with soft/hard preemption behaviors and metrics. IDs: ORCH-3079/3085/3087.
- (Lifecycle × Data Plane Admission)
  - Deprecated blocks new sessions; Retired unloads; MODEL_DEPRECATED error. IDs: ORCH-3069..3073, ORCH-3093.
- (Trust Policy × Control Plane Ingest)
  - strict_unsigned rejects with UNTRUSTED_ARTIFACT error. IDs: ORCH-3060..3065, ORCH-3093.
- (Placement × Device Masks)
  - Ready only after preload; no spillover across masks. IDs: ORCH-3010/3011, OC-POOL-3001..3021.
- (SSE Started Fields × Backpressure/Admission)
  - started includes queue_position and predicted_start_ms. IDs: ORCH-3029, OC-CTRL-2021.
- (Auth × Data Plane)
  - apikey_missing rejected per security requirements. IDs: OC-CTRL-2040.

## 3‑wise high‑risk triads

- (WFQ Fairness × Deadlines EDF × Preemption)
  - Ensure urgent tasks meet deadlines without starving others; metrics exported. IDs: ORCH-3075/3076/3079/3085/3087.
- (Lifecycle Deprecated/Retired × Admission × Typed Errors)
  - New sessions blocked with MODEL_DEPRECATED; pools drain/unload; model_state gauge. IDs: ORCH-3069..3073, ORCH-3093.
- (Trust strict × unsigned artifact × Control Plane ingest)
  - Ingest/load refusal with UNTRUSTED_ARTIFACT and verification metrics. IDs: ORCH-3060..3065, ORCH-3093.
- (Engine Capability × Hard Preemption × SSE/Error surfacing)
  - Hard preemption only when interruptible_decode; preempted flag and resumable state surfaced. IDs: ORCH-3086/3087.
- (Heterogeneous Split × Placement × Device Masks)
  - Explicit ratios honored; no cross‑mask spillover. IDs: ORCH-3011/3012, OC-POOL-3021.

## Proposed test artifacts per combo

- BDD Scenarios: `test-harness/bdd/`
  - Scheduling: WFQ, quotas, deadlines, preemption, session affinity, SSE fields.
  - Lifecycle: Deprecated/Retired transitions and effects.
  - Catalog/Trust: strict policy acceptance/rejection cases.
- Provider/CDC: `orchestratord/tests/provider_verify.rs`, OpenAPI artifacts.
- Property Tests: `orchestrator-core/tests/props_queue.rs` (fairness and placement invariants).
- Metrics Contract: `test-harness/metrics-contract/` with `ci/metrics.lint.json`.

## Traceability

Each combo references the base requirement IDs. Link combo → concrete test case names in `.docs/spec-derived-test-catalog.md` under "Cross‑Spec Interaction Tests".
