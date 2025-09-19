# Proposal: Centralized Placement, Priority, and Consumer Overrides Policy

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## 0) Motivation

Today, the decision logic that determines "which engine/pool/GPU should serve this task, in what order, and under what constraints" is split across:
- `orchestrator-core` (queue invariants, feasibility + scoring hooks)
- `orchestratord` (admission endpoints, minimal adapter selection stub, lease accounting)
- `pool-managerd` (device masks, VRAM/health/capacity signals, supervision, drain/reload)

We want a single, robust source of truth for placement and priority handling that:
- Centralizes feasibility, scoring, tie-breakers, and override semantics.
- Supports both automatic placement and explicit consumer/operator selection (pin/prefer/avoid/mask).
- Preserves determinism and auditability with structured decision reasons.
- Improves observability with metrics and logs for decision quality and fairness.

This proposal defines a consolidated policy surface that remains within the current architectural boundaries (no runtime coupling changes), while making the decision logic “closer together” in one module/crate with stable inputs/outputs.

## 1) Scope

In scope:
- Centralized placement policy: feasibility predicate, scoring, tie-breakers, and decision logging.
- Priority-aware queueing semantics and their integration in dispatch order.
- Consumer overrides: pin/prefer/avoid/require-device-mask with strict vs. prefer semantics.
- Minimal contract extensions (`TaskRequest.placement`) and core `JobSpec.placement`.

Out of scope:
- Process supervision and restart/backoff (owned by `pool-managerd`).
- Engine provisioning and model staging (owned by `engine-provisioner`/`model-provisioner`).
- HTTP surface and session budgets/cancel (owned by `orchestratord`).

## 2) Normative Requirements (RFC-2119)

IDs: ORCH-39xx (centralized placement & priority policy)

- [ORCH-3900] The workspace MUST expose a single policy entry-point for placement decisions:
  - `fn decide(input: PlacementInput) -> PlacementDecision` where `PlacementInput` aggregates pool snapshots and a `JobSpec` (see §3).
  - The policy MUST be pure (no network/process I/O) and deterministic given the same inputs.

- [ORCH-3901] Feasibility MUST be checked before scoring. Pools that fail minimum requirements MUST be excluded:
  - Device mask compatibility and optional `require_device_mask` filters.
  - Model requirements (ctx, quantization, extensions) vs. pool/replica capabilities.
  - Resource availability (e.g., VRAM free vs. estimated KV bytes when provided).

- [ORCH-3902] Scoring SHOULD minimize `predicted_end_ms = admission_latency + first_token_ms + decode_ms(tokens_out, perf_tokens_per_s)` when hints exist; otherwise, apply deterministic tie-breakers:
  - Tie-break order MUST be: (1) KV/session affinity; (2) least loaded; (3) highest residual VRAM; (4) higher `perf_tokens_per_s`; (5) stable lexicographic `pool_id`.

- [ORCH-3903] Consumer overrides MUST be supported with explicit semantics:
  - `placement.mode = pin` MUST restrict candidates to `pin_pool_id` only; if infeasible/unready → `NoCapacity { reason: "pinned_pool_infeasible" }`.
  - `placement.mode = prefer` MUST restrict scoring to `prefer_pools` when feasible; if none feasible and `allow_fallback=true` → fall back to full candidate set; if `allow_fallback=false` → `NoCapacity { reason: "no_preferred_feasible" }`.
  - `placement.avoid_pools` MUST be excluded from candidates.
  - `placement.require_device_mask` MUST filter candidates to pools matching the mask exactly.

- [ORCH-3904] Priority MUST influence queueing and dispatch:
  - Enqueue MUST respect bounded FIFO per priority.
  - Dequeue MUST prefer higher priority and maintain FIFO within class.
  - With `DropLru`, the oldest `Batch` item SHOULD be dropped first.

- [ORCH-3905] Determinism: given identical `{prompt, parameters, seed, sampler_profile_version, engine_version, model_digest}` and identical policy inputs, the chosen pool MUST be identical. Mixed engine_version/sampler_profile_version replicas MUST NOT share a set.

- [ORCH-3906] Observability: the policy MUST emit structured decision reasons:
  - `DecisionLog { selected_pool_id, candidates_considered, filters_applied[], tie_breakers_applied[], pinned: bool, fallback_used: bool }`.
  - Logs MUST include fields aligned with `README_LLM.md` and `.specs/metrics/otel-prom.md`.

- [ORCH-3907] API contracts:
  - `contracts/openapi/data.yaml` MUST add optional `TaskRequest.placement` as described in §4.
  - `contracts/api-types` MUST mirror the schema with strong typing.

- [ORCH-3908] Backward compatibility: When `TaskRequest.placement` is absent, policy MUST behave exactly as current automatic placement.
 
### 2A) Crate-Specific Obligations (summary)

- orchestrator-core
  - [ORCH-3950] MUST expose `policy::decide(input: PlacementInput) -> PlacementDecision` and associated types in this proposal (§3).
  - [ORCH-3951] MUST emit `DecisionLog` with filters and tie-breakers applied, and ensure the function is pure/deterministic.
  - [ORCH-3952] MUST provide unit/property tests covering feasibility filters, tie-break ordering, and override semantics (pin/prefer/avoid/mask/fallback).

- orchestratord
  - [ORCH-3960] MUST route all placement decisions through `orchestrator-core::policy::decide` (no ad-hoc selection in the app layer).
  - [ORCH-3961] MUST pass through consumer overrides from `TaskRequest.placement`, after enforcing any authorization/policy gates.
  - [ORCH-3962] MUST log `DecisionLog` fields and emit metrics counters/gauges proposed in §7.

- pool-managerd
  - [ORCH-3970] MUST publish snapshots with `vram_total_bytes`, `vram_free_bytes`, `compute_capability`, `device_mask`, and `draining` flags consistently.
  - [ORCH-3971] MUST maintain non-negative lease accounting and a freshness/heartbeat policy so snapshots used for placement are recent.
  - [ORCH-3972] SHOULD expose optional perf hints (`perf_tokens_per_s`, `first_token_ms`) where measurable to improve scoring.

- contracts (openapi + api-types)
  - [ORCH-3980] MUST add `TaskRequest.placement` per §4 and mirror in `contracts/api-types` with safe defaults (Auto + allow_fallback=true), with examples.

- worker-adapters (and http-util)
  - [ORCH-3990] SHOULD surface capacity/props and, when available, perf hints that can be incorporated into pool snapshots (no breaking changes required).

## 3) Data Types & Semantics

```rust
// Canonical inputs
pub struct PoolSnapshot {
    pub id: String,
    pub engine: String,
    pub slots_total: i32,
    pub slots_free: i32,
    pub vram_total_bytes: i64,
    pub vram_free_bytes: i64,
    pub compute_capability: Option<String>,
    pub perf_tokens_per_s: Option<f64>,
    pub first_token_ms: Option<f64>,
    pub device_mask: Option<String>,
    pub draining: bool,
}

pub enum Priority { Interactive, Batch }

pub struct PlacementOverrides {
    pub mode: PlacementMode, // Pin | Prefer | Auto
    pub pin_pool_id: Option<String>,
    pub prefer_pools: Option<Vec<String>>,
    pub avoid_pools: Option<Vec<String>>,
    pub require_device_mask: Option<String>,
    pub allow_fallback: bool, // default true
}

pub enum PlacementMode { Pin, Prefer, Auto }

pub struct JobSpec {
    pub priority: Priority,
    pub expected_tokens: Option<i32>,
    pub engine: String,
    pub model_id: String,
    pub required_ctx: i32,
    pub est_kv_bytes: Option<i64>,
    pub placement: PlacementOverrides, // defaults: Auto + allow_fallback=true
}

pub struct PlacementInput { pub pools: Vec<PoolSnapshot>, pub job: JobSpec }

pub enum PlacementDecision {
    Assigned { pool_id: String, reason: DecisionLog },
    NoCapacity { reason: String, considered: usize },
}

pub struct DecisionLog {
    pub selected_pool_id: Option<String>,
    pub candidates_considered: usize,
    pub filters_applied: Vec<String>,
    pub tie_breakers_applied: Vec<String>,
    pub pinned: bool,
    pub fallback_used: bool,
}
```

Notes:
- `PoolSnapshot` fields align with `orchestrator-core/.specs/00_orchestrator_core.md` and wiring with `pool-managerd` expectations, adding `device_mask` and `draining` for richer feasibility.
- `JobSpec.placement` is new; defaulting keeps current behavior intact.

## 4) API & Contracts Changes

`contracts/openapi/data.yaml` additions:

```yaml
components:
  schemas:
    PlacementMode:
      type: string
      enum: [pin, prefer, auto]
    PlacementOverrides:
      type: object
      properties:
        mode: { $ref: '#/components/schemas/PlacementMode' }
        pin_pool_id: { type: string }
        prefer_pools: { type: array, items: { type: string } }
        avoid_pools: { type: array, items: { type: string } }
        require_device_mask: { type: string }
        allow_fallback: { type: boolean, default: true }
    TaskRequest:
      properties:
        placement: { $ref: '#/components/schemas/PlacementOverrides' }
```

`contracts/api-types` must mirror this with strong typing and safe defaults (Auto + allow_fallback=true).

## 5) Architectural Impact
 
### 5.1 New Crates?

- No new crates are REQUIRED by this proposal. The centralization goal is achieved by implementing a `policy` module inside `orchestrator-core` and routing all decisions through it.
- Optional (packaging choice): introduce a tiny `placement-policy` crate, owned by `orchestrator-core`, if we want to decouple policy iteration from other core churn. If chosen, it would:
  - Expose the same stable function `decide(PlacementInput) -> PlacementDecision` and types (§3).
  - Contain only pure logic, tests, and documentation. No runtime I/O or new dependencies.
  - Be consumed solely by `orchestratord` (indirectly via `orchestrator-core`) to compute placements.

### 5.2 Crate-by-crate improvements (what each must do better)

- orchestrator-core
  - Provide `policy::decide` and unify all feasibility, overrides, scoring, and tie-breakers in one place.
  - Add property tests and decision-reason logging helpers.
  - Optionally provide a converter or trait to standardize `PoolSnapshot` ingestion from `pool-managerd` snapshots.

- orchestratord
  - Always call the centralized policy; remove/minimize any in-crate placement stubs.
  - Validate and gate `placement` overrides (auth/policy) before passing to core; include decision reasons in structured logs and metrics.
  - Update provider verification and BDD tests to cover overrides and priority dispatch.

- pool-managerd
  - Ensure registry exposes VRAM fields, device masks, and `draining`; document and enforce snapshot freshness.
  - Keep non-negative leases and readiness gates; optionally enrich with perf hints.

- contracts (openapi + api-types)
  - Add `PlacementOverrides` schemas and defaults; update examples; run regen tasks.

- worker-adapters/http-util
  - No mandatory changes. Prefer consistent capacity/health reporting; expose perf hints when available to improve ETA scoring.

- Keep boundaries intact (recommended):
  - Centralize the policy logic inside `orchestrator-core` under a dedicated `policy` module. `orchestratord` continues to call into core with `PlacementInput`; `pool-managerd` continues to publish snapshots. This avoids new crates and minimizes churn while achieving centralization.

- Alternative (optional):
  - Introduce a small `placement-policy` crate owned by `orchestrator-core` to reduce churn in the core crate when policy evolves. Public API remains identical to §2/§3; consumers still treat it as a pure function. This is a packaging choice, not an architectural change.

Compatibility:
- No runtime coupling changes. `pool-managerd` remains the source of device/VRAM/health; `orchestratord` remains HTTP + session + control-plane; `worker-adapters` remain engine I/O.
- This proposal consolidates decision code paths and adds explicit override semantics without breaking the architecture.

## 6) Implementation Plan

Phase A — Spec & Contracts (this PR)
- Update `orchestrator-core/.specs/00_orchestrator_core.md` to declare the `policy` entry-point, data shapes, and ORCH-39xx requirements.
- Update root `/.specs/10-orchestrator-core.md` to reference `JobSpec.placement` and tie-breakers.
- Update `contracts/openapi/data.yaml` and `contracts/api-types` to add `TaskRequest.placement`.

Phase B — Tests & Harness
- Add unit/property tests in `orchestrator-core` for feasibility filters and tie-breakers.
- Extend BDD to cover pin/prefer/avoid/fallback behavior and priority queues.
- Determinism suite: verify constant decision under identical inputs.

Phase C — Code Wiring
- Implement `policy::decide` in `orchestrator-core`; remove/minimize any placement stubs in `orchestratord` and funnel all decisions through core.
- Ensure `pool-managerd` snapshots include `vram_total_bytes`, `vram_free_bytes`, `device_mask`, `draining` consistently.
- Add structured `DecisionLog` to logs; include fields in SSE `started` context if desired (optional, non-breaking).

Phase D — Observability & Docs
- Metrics: counters for `placement_decisions_total{outcome, pinned, fallback}`; histograms for `predicted_end_ms` if available.
- Logs: include `decision_reason`, `pinned`, `fallback_used`, and candidate counts.
- Update `.docs/testing/` proof-bundle checklists to include decision logs and metrics snapshots.

## 7) Observability

- Metrics (Prometheus-compatible) — align with `.specs/metrics/otel-prom.md`:
  - `placement_decisions_total{outcome="assigned|no_capacity", pinned, fallback}`
  - `placement_candidates_considered` (gauge or histogram)
  - Optional: `predicted_end_ms` histogram when hints exist
- Logs — align with `README_LLM.md` fields and add `decision_reason`, `pinned`, `fallback_used`.

## 8) Security

- No secrets handled by policy. Ensure redaction remains in HTTP/adapter layers.

## 9) Risks & Mitigations

- Risk: Override misuse can starve pools.
  - Mitigation: add policy gates (auth/role/limits) at `orchestratord` before passing overrides into core; document configs.
- Risk: Divergent pool snapshot quality (missing VRAM/perf hints) reduces decision quality.
  - Mitigation: define minimal snapshot fields as MUST (already present); add health checks to ensure freshness.
- Risk: Churn in core while policy evolves.
  - Mitigation: confine logic to `policy` module or opt for a small `placement-policy` crate.

## 10) Acceptance Criteria

- `TaskRequest.placement` supported end-to-end; omitted fields preserve current behavior.
- All placement decisions flow through a single policy entry-point in `orchestrator-core`.
- Unit/BDD/determinism tests added for pin/prefer/avoid/fallback and priority dispatch.
- Decision reasons are logged; metrics emitted and pass lints.
- No cross-mask spillover; device masks and draining are respected.

## 11) Mapping to Repo Reality (Anchors)

- `/.specs/10-orchestrator-core.md` — canonical `ModelRequirements`, device-mask rules, scheduling invariants.
- `orchestrator-core/.specs/00_orchestrator_core.md` — placement inputs/outputs and tie-breakers.
- `orchestrator-core/.specs/11_pool_managerd.md` — snapshot expectations.
- `pool-managerd/.specs/00_pool_managerd.md` — readiness gates and VRAM/mask signals.
- `orchestratord/.specs/00_orchestratord.md` — admission/backpressure, control-plane, leases.
- `contracts/openapi/data.yaml` — `Priority`, `TaskRequest`; add `placement`.

## 12) Refinement Opportunities

- Learn online performance hints and integrate into scoring while keeping deterministic fallbacks.
- Add per-tenant fairness weights (future, out-of-scope for home profile) behind a feature flag.
- Expose “why not selected” details per candidate for debugging (bounded cardinality).
- Add budget-aware scoring inputs (e.g., expected tokens) to improve ETA.
- Introduce policy gates and rate limits for consumer overrides at `orchestratord`.

## 13) Checklist Updates (for CHECKLIST.md and SPEC_CHECKLIST.md)

### 13.1 CHECKLIST.md additions

- 2. Orchestratord
  - [ ] Route all placement decisions through `orchestrator-core::policy::decide` (remove/minimize any in-crate selection stubs) (ORCH-3960)
  - [ ] Enforce auth/policy gates on `TaskRequest.placement` overrides; pass sanitized overrides into core (ORCH-3961)
  - [ ] Log `DecisionLog { filters_applied[], tie_breakers_applied[], pinned, fallback_used, candidates_considered }` and emit placement metrics (ORCH-3962)
  - [ ] Provider verify/BDD updated to cover pin, prefer(+fallback), avoid, require_device_mask, and priority dispatch

- 3. Orchestrator-core
  - [ ] Implement `policy::decide(PlacementInput) -> PlacementDecision` (pure, deterministic), unify feasibility/overrides/scoring/tie-breakers (ORCH-3950)
  - [ ] Emit `DecisionLog` and add unit/property tests covering feasibility filters, override semantics, tie-break order (ORCH-3951, ORCH-3952)

- 4. Pool-managerd
  - [ ] Registry/snapshots include `vram_total_bytes`, `vram_free_bytes`, `compute_capability`, `device_mask`, `draining` (ORCH-3970)
  - [ ] Maintain non-negative lease accounting and snapshot freshness/heartbeat for placement (ORCH-3971)
  - [ ] (Optional) Expose perf hints `perf_tokens_per_s`, `first_token_ms` when measurable (ORCH-3972)

- 8. Test Harnesses
  - [ ] BDD: scenarios for placement overrides (pin/prefer/avoid/mask with/without fallback) and priority dispatch
  - [ ] Determinism suite: identical inputs yield identical `Assigned.pool_id`; add decision transcript checks

- 9. Contracts & Metrics
  - [ ] OpenAPI: add `PlacementMode` and `PlacementOverrides`; add optional `TaskRequest.placement` with examples; mirror in `contracts/api-types`; run regen tasks
  - [ ] Metrics: add `placement_decisions_total{outcome="assigned|no_capacity", pinned, fallback}` and `placement_candidates_considered`; optional `predicted_end_ms` histogram when hints exist; ensure `ci/metrics.lint.json` passes

- 10. CI & Tooling
  - [ ] Ensure `cargo xtask regen-openapi`, `regen-schema`, and `tools-spec-extract` run in dev loop; add new unit/BDD tests to CI

- 11. Observability & Narration
  - [ ] Include `DecisionLog` fields in structured logs at admission/placement time; keep redaction rules intact

- 12. Docs & READMEs
  - [ ] Document override semantics in orchestrator README and examples (pin vs prefer vs auto; fallback)

- 15. Acceptance Criteria (Roll-up)
  - [ ] All placements flow through centralized policy; behavior unchanged when overrides absent
  - [ ] BDD and determinism tests for overrides and priority are green; metrics lints green
  - [ ] Logs include `DecisionLog`; metrics series present and correctly labeled

- 18. Scaffold Targets by Crate (File/Module-Level TODOs)
  - **orchestrator-core/**
    - [ ] `src/policy.rs` — implement `decide`; helpers for feasibility, overrides, scoring/tie-breaks; unit/property tests
  - **orchestratord/**
    - [ ] Call `policy::decide` from admission/placement path; map `TaskRequest.placement` → `JobSpec.placement`; log DecisionLog; emit metrics
  - **pool-managerd/**
    - [ ] `src/registry.rs` — ensure snapshot fields and freshness; optional perf hints plumbing

### 13.2 SPEC_CHECKLIST.md additions

- 0) Approvals
  - [ ] Set `/.specs/proposals/2025-09-19-centralized-placement-and-priority-policy.md` to Accepted

- 1) Root `.specs/` edits
  - [ ] `/.specs/10-orchestrator-core.md`: add `JobSpec.placement` semantics; centralize tie-breaker mapping; clarify feasibility inputs; reference policy entry-point
  - [ ] `/.specs/20-orchestratord.md`: document that placement is delegated to `orchestrator-core::policy::decide`; add DecisionLog observability expectations
  - [ ] `/.specs/metrics/otel-prom.md`: add `placement_decisions_total` and `placement_candidates_considered` names/labels; optional `predicted_end_ms` histogram guidance

- 2) Crate `.specs/` edits
  - [ ] `orchestrator-core/.specs/00_orchestrator_core.md`: define `policy::decide` entry-point, data shapes (`PlacementOverrides`), determinism note
  - [ ] `orchestrator-core/.specs/11_pool_managerd.md`: standardize `PoolSnapshot` fields (VRAM/mask/draining) and remove duplication once canonicalized
  - [ ] `orchestratord/.specs/00_orchestratord.md`: pass-through overrides, DecisionLog logging, placement metrics
  - [ ] `pool-managerd/.specs/00_pool_managerd.md`: make snapshot fields and freshness normative; reference perf hints as optional
  - [ ] `worker-adapters/.specs/00_worker_adapters.md`: no changes required; optionally note perf hints surfacing via props/health if available

- 3) Contracts
  - [ ] `contracts/openapi/data.yaml`: add `PlacementMode`, `PlacementOverrides`, and `TaskRequest.placement`; update examples
  - [ ] `contracts/api-types`: mirror strong types and defaults; update tests and regen artifacts

- 6) Verification & proof bundles
  - [ ] Update `.docs/testing/` to require DecisionLog snapshots/metrics excerpts in proof bundles alongside SSE transcripts

- 7) Post‑spec follow‑ups
  - [ ] Propagate override semantics and examples to orchestratord README and CLI docs (if applicable)
