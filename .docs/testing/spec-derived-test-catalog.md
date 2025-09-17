# Spec-Derived Test Catalog (Home Profile)

Status: generated 2025-09-18
Sources: `.specs/00_llama-orch.md`, `.specs/00_home_profile.md`, component specs.

Each entry maps requirement IDs to the primary tests that prove them. Update this file whenever specs or tests change so that traceability remains intact.

| Requirement IDs | Description | Test Artifacts |
|-----------------|-------------|----------------|
| ORCH-3001, ORCH-3002, ORCH-3003 | Workers preload per device mask; preload failure handling | `orchestrator-core/tests/props_queue.rs`, `orchestratord/tests/preload_tests.rs` (todo) |
| ORCH-3004, ORCH-3005, ORCH-3006 | Bounded queue, policy (reject/drop-lru), optional throttles | `orchestrator-core/tests/props_queue.rs`, `test-harness/bdd/tests/features/data_plane/admission_queue.feature` |
| ORCH-3009, ORCH-3010, ORCH-3011 | Ready-only placement, device masks, session affinity | `pool-managerd/tests/registry.rs`, `test-harness/bdd/tests/features/control_plane/pool_drain_reload.feature` |
| ORCH-3021..3023, ORCH-3099 | Session TTL, turns, KV warmth, budgets | `test-harness/bdd/tests/features/data_plane/sessions.feature`, `test-harness/bdd/tests/features/data_plane/budgets.feature` |
| ORCH-3024..3029, ORCH-3045..3046 | Deterministic SSE streaming, cancel semantics | `orchestratord/tests/provider_verify.rs`, `test-harness/determinism-suite/tests/determinism.rs` |
| ORCH-3031, ORCH-3037, ORCH-3038 | Catalog persistence, drain/reload rollback, warning on unsigned artifacts | `test-harness/bdd/tests/features/control_plane/catalog.feature`, `test-harness/bdd/tests/features/control_plane/pool_drain_reload.feature` |
| ORCH-3035, ORCH-3080 | API token auth, tooling policy hook | `orchestratord/tests/auth.rs`, `test-harness/bdd/tests/features/tooling/policy.feature` (todo) |
| ORCH-3027, ORCH-3028, ORCH-3100 | Structured logs, metrics contract, SSE `metrics` frame | `test-harness/metrics-contract/tests/metrics_lint.rs`, `orchestratord/tests/provider_verify.rs` |
| ORCH-3095, ORCH-3096 | Capability discovery payload | `test-harness/bdd/tests/features/control_plane/capabilities.feature` |
| ORCH-3050, ORCH-3051 | Determinism + Haiku anti-cheat tests | `test-harness/determinism-suite/`, `test-harness/e2e-haiku/` |
| HME-001, HME-002 | Single-host deployment, loopback default | `test-harness/bdd/tests/features/setup/network_bind.feature` (todo) |
| HME-020..HME-022 | CLI integration (queue metadata, SSE metrics, artifacts) | `cli/consumer-tests/tests/orchqueue_pact.rs`, `test-harness/bdd/tests/features/data_plane/admission_queue.feature` |
| HME-040..HME-042 | Reference environment smoke, determinism per GPU, BDD coverage | `ci/reference-smoke/` (todo), `test-harness/determinism-suite/`, `test-harness/bdd/` |

_Update the table as implementation progresses. “todo” marks indicate gaps the team still needs to fill._
