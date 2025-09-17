# Compliance Matrix — Home Profile

This lightweight matrix tracks which requirements have proofs (tests/code/docs). Update it whenever a spec, test, or implementation changes.

| Requirement ID | Description | Proofs |
|----------------|-------------|--------|
| ORCH-3001..3006 | Queue setup, policies, throttles | `orchestrator-core/tests/props_queue.rs`, `test-harness/bdd/.../admission_queue.feature` |
| ORCH-3010..3012 | Placement to Ready replicas, device masks, least-loaded heuristic | `pool-managerd/tests/registry.rs`, `test-harness/bdd/.../pool_drain_reload.feature` |
| ORCH-3021..3023, ORCH-3099 | Sessions, KV warmth, budgets | `test-harness/bdd/.../sessions.feature`, `test-harness/bdd/.../budgets.feature` |
| ORCH-3024..3029, ORCH-3045..3046 | Deterministic SSE streams, cancel semantics | `orchestratord/tests/provider_verify.rs`, `test-harness/determinism-suite/` |
| ORCH-3031, ORCH-3037, ORCH-3038 | Catalog persistence, drain/reload behaviour | `test-harness/bdd/.../catalog.feature`, `test-harness/bdd/.../pool_drain_reload.feature` |
| ORCH-3095..3096 | Capability discovery endpoint | `test-harness/bdd/.../capabilities.feature` |
| HME-001..HME-022 | Home lab overlay (deployment envelope, CLI integration) | `.docs/HOME_PROFILE_TARGET.md`, CLI pact tests |

_Not yet implemented or proven requirements should be added to this table as work progresses._
