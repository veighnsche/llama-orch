# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [x] Lifecycle: simulate pool unload semantics (Retired) via structured logs; model_state gauge exported
- [ ] Lifecycle: watchdog abort path (planning)
- [ ] Scheduling/Quotas: WFQ share assertions product-side; per-tenant quotas enforcement; session affinity
- [x] Config schema: validation pass/strict-mode rejection; regen idempotence
- [ ] Policy host/SDK: minimal assertions (WASI ABI, sandboxing, versioning) wired to stubs
- [x] Catalog: model create/get/verify stubs; strict trust policy UNTRUSTED_ARTIFACT
- [x] Observability logs: started/admission logs include queue ETA; no secrets/API keys

## Progress Log (what changed)

- 2025-09-16 — BDD wiring complete for admission, SSE, sessions, backpressure, error taxonomy, control plane, security, adapters, lifecycle gating (Deprecated/Retired), and guardrails pre-admission checks.
  - Product: `orchestratord/src/http/handlers.rs` (429 advisory headers/body; error taxonomy sentinels; guardrails; lifecycle gate; SSE metrics frame; `/v1/models/state` control);
    `orchestratord/src/state.rs` (add `ModelState`); `orchestratord/src/lib.rs` (route for lifecycle); `orchestratord/src/backpressure.rs` (headers/body helper).
  - BDD: `world.rs` dispatcher; data_plane/control_plane/security/observability/core_guardrails/lifecycle/adapters steps now drive in-memory handlers and adapter crates; traceability coverage file added.
  - Tests: `cargo test --workspace` green; traceability test reports all catalog IDs referenced.

## What’s left to wire (exhaustive, per steps)

- Determinism (`test-harness/bdd/src/steps/determinism.rs`)
  - [x] Then token streams are byte-exact across replicas (step compares two replicas)
  - [x] Then determinism is not assumed across engine or model updates (negative case)

- Scheduling (`steps/scheduling.rs`)
  - [ ] Then observed share approximates weights (needs scheduler plumbing or simulation)
  - [ ] Given quotas configured / Then beyond-quota rejected (admission hook or policy)
  - [ ] Then session affinity keeps last good replica (requires placement/session store)

- Config (`steps/config.rs`)
  - [x] Then schema validation passes / rejects unknown / outputs identical (generator + validator)

- Catalog (`steps/catalog.rs`)
  - [x] Then model is created / manifest signatures+sbom present / verification starts / UNTRUSTED_ARTIFACT

- Policy Host/SDK (`steps/policy_host.rs`, `steps/policy_sdk.rs`)
  - [ ] All THEN assertions (WASI ABI, determinism, sandboxing, bounds, logs; SDK stability/no I/O)

- Observability (`steps/observability.rs`)
  - [x] Then logs include queue ETA; Then logs do not contain secrets (structured logs + assertions)

- Lifecycle (`steps/lifecycle.rs`)
  - [x] Then pools unload and archives retained (simulated via structured log)

- Deadlines & Preemption (`steps/deadlines_preemption.rs`)
  - [x] DEADLINE_UNMET sentinel and SSE on_time_probability asserted
  - [ ] Preemption behavior ordering and resumable state (planning)

## Next actions (implementation plan)

1) Determinism steps: implement in harness by exercising two replicas and comparing streams; add a negative case across versions.
2) Scheduling placeholders: add simple simulator in harness for weights/quotas; keep product changes scoped.
3) Config schema driver: call generator/validator and assert outcomes.
4) Observability logs: add minimal structured log hook in handlers for started/admission; redact secrets.
5) Lifecycle retire unload: add state flag + health reflection; assert via health route.
6) Policy/Catalog: scaffold minimal stubs and assertions to satisfy steps.
