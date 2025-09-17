# Spec Combination Matrix — Home Profile

This matrix captures the major feature combinations we test explicitly. Use it to identify missing coverage when specs change.

| Area | Scenario | Specs / IDs | Tests |
|------|----------|-------------|-------|
| Admission | Single agent, empty queue | ORCH-3004..3006, ORCH-3029 | `test-harness/bdd/tests/features/data_plane/admission_queue.feature` |
| Admission | Queue saturation → 429 | ORCH-2007, ORCH-3005, ORCH-3006 | `test-harness/bdd/tests/features/data_plane/backpressure.feature`, provider verify 429 cases |
| Sessions | TTL expiry & eviction | ORCH-3021, ORCH-3023, HME-030 | `test-harness/bdd/tests/features/data_plane/sessions.feature` |
| SSE | Deterministic stream, metrics frame | ORCH-2002, ORCH-3100 | `test-harness/bdd/tests/features/data_plane/sse_stream.feature`, determinism suite |
| Catalog | Upload unsigned model (warn) | ORCH-3037 | `test-harness/bdd/tests/features/control_plane/catalog.feature` |
| Drain/Reload | Drain → reload success & rollback | ORCH-3031, ORCH-3038 | `test-harness/bdd/tests/features/control_plane/pool_drain_reload.feature` |
| Capability | CLI derives concurrency hints | ORCH-3095, ORCH-3096, HME-012 | `test-harness/bdd/tests/features/control_plane/capabilities.feature` |
| Artifacts | Store/retrieve plan | ORCH-3097, ORCH-3098, HME-022 | `test-harness/bdd/tests/features/control_plane/artifacts.feature` |
| Budgets | Token/time budget rejection | ORCH-3099, ORCH-2068 | `test-harness/bdd/tests/features/data_plane/budgets.feature` |
| Placement | Mixed GPU load balancing | ORCH-3012, HME-010 | `test-harness/bdd/tests/features/data_plane/mixed_gpu.feature` (todo) |
| Tooling Policy | Blocked outbound request | ORCH-3080 | `test-harness/bdd/tests/features/tooling/policy.feature` (todo) |

“todo” rows signal combinations we still need to codify. Update matrix rows alongside new tests.
