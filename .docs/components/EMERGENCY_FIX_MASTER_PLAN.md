# EMERGENCY FIX: Master Plan - Get to 90%+ Test Pass Rate

**Created by:** TEAM-116  
**Date:** 2025-10-19  
**Status:** 🚨 **CRITICAL - READY TO EXECUTE**

---

## Executive Summary

**Current State:** 69/300 tests passing (23%)  
**Target State:** 270+/300 tests passing (90%+)  
**Gap to Close:** 201 failing scenarios

### Failure Analysis
- **32 ambiguous steps** - Duplicate step definitions (QUICK FIX)
- **71 unimplemented steps** - Missing implementations (MAIN WORK)
- **185 timeouts** - Need service availability checks (MEDIUM FIX)
- **104 panics** - Need error handling (MEDIUM FIX)

---

## Work Breakdown - 6 Teams

### Team Assignment Strategy
- Each team gets **~4 hours of work**
- Work is parallelizable
- Clear deliverables
- Independent tasks

---

## TEAM-117: Fix Ambiguous Steps (4 hours)

**Deliverable:** Remove all 32 duplicate step definitions

**Tasks:**
1. Identify all duplicate step definitions (30 min)
2. Consolidate or rename duplicates (2 hours)
3. Update feature files if needed (1 hour)
4. Verify compilation (30 min)

**Files to Modify:**
- `test-harness/bdd/src/steps/error_handling.rs`
- `test-harness/bdd/src/steps/deadline_propagation.rs`
- `test-harness/bdd/src/steps/lifecycle.rs`
- `test-harness/bdd/src/steps/authentication.rs`
- Others as identified

**Success Criteria:**
- Zero "Step match is ambiguous" errors
- All tests compile
- No functionality lost

---

## TEAM-118: Implement Missing Steps (Batch 1) (4 hours)

**Deliverable:** Implement 18 missing step definitions

**Steps to Implement:**
1. `Then queen-rbee attempts SSH connection with 10s timeout`
2. `When rbee-hive reports worker "worker-001" with capabilities ["cuda:0"]`
3. `And the response contains 1 worker`
4. `And the exit code is 1`
5. `When rbee-hive spawns a worker process`
6. `Given rbee-keeper is configured to spawn queen-rbee`
7. `Given queen-rbee is already running as daemon at "http://localhost:8080"`
8. `And the exit code is 0`
9. `Given worker has 4 slots total`
10. `And validation fails`
11. `Then request is accepted`
12. `When I send request with node "workstation"`
13. `Given worker-001 is registered in queen-rbee with last_heartbeat=T0`
14. `When rbee-hive attempts to query catalog`
15. `Given worker-001 is processing request`
16. `Given 3 workers are running and registered in queen-rbee`
17. `Then worker stops accepting new requests`
18. `Then backup is created at "~/.rbee/backups/models-<timestamp>.db"`

**Files to Modify:**
- `test-harness/bdd/src/steps/error_handling.rs`
- `test-harness/bdd/src/steps/worker_registration.rs`
- `test-harness/bdd/src/steps/lifecycle.rs`
- `test-harness/bdd/src/steps/concurrency.rs`

**Success Criteria:**
- All 18 steps have real implementations
- No TODO markers
- Tests compile and run

---

## TEAM-119: Implement Missing Steps (Batch 2) (4 hours)

**Deliverable:** Implement 18 missing step definitions

**Steps to Implement:**
19. `Given device 0 has 2GB VRAM free`
20. `Given preflight starts with 8GB RAM available`
21. `Given GPU temperature is 85°C`
22. `Given system has 16 CPU cores`
23. `Given GPU has 8GB total VRAM`
24. `Given system bandwidth limit is 10 MB/s`
25. `Given disk I/O is at 90% capacity`
26. `When I send POST to "/v1/workers/spawn" without Authorization header`
27. `When I send GET to "/health" without Authorization header`
28. `When I send 1000 authenticated requests`
29. `And file permissions are "0644" (world-readable)`
30. `And file permissions are "0640" (group-readable)`
31. `Given systemd credential exists at "/run/credentials/queen-rbee/api_token"`
32. `When queen-rbee starts with config:`
33. `When queen-rbee starts and processes 100 requests`
34. `Then error message does not contain "secret-error-test-12345"`
35. `And log contains "API token reloaded"`
36. `And file contains:`

**Files to Modify:**
- `test-harness/bdd/src/steps/worker_preflight.rs`
- `test-harness/bdd/src/steps/authentication.rs`
- `test-harness/bdd/src/steps/secrets.rs`
- `test-harness/bdd/src/steps/configuration_management.rs`

**Success Criteria:**
- All 18 steps have real implementations
- No TODO markers
- Tests compile and run

---

## TEAM-120: Implement Missing Steps (Batch 3) (4 hours)

**Deliverable:** Implement 18 missing step definitions

**Steps to Implement:**
37. `When queen-rbee starts`
38. `When searching for unwrap() calls in non-test code`
39. `And rbee-hive continues running (does NOT crash)`
40. `Then error message does NOT contain password`
41. `Then error message does NOT contain raw token value`
42. `Then error message does NOT contain absolute file paths`
43. `Then error message does NOT contain internal IP addresses`
44. `Given rbee-hive is running with 1 worker`
45. `And log entry includes correlation_id`
46. `And audit entry includes token fingerprint (not raw token)`
47. `And hash chain is valid (each hash matches previous entry)`
48. `And entry contains "timestamp" field (ISO 8601)`
49. `Then queen-rbee logs warning "audit log disk space low"`
50. `When deadline is exceeded`
51. `Given worker is processing inference request`
52. `Then the response status is 200`
53. `Given pool-managerd is running`
54. `Given pool-managerd is running with GPU workers`

**Files to Modify:**
- `test-harness/bdd/src/steps/error_handling.rs`
- `test-harness/bdd/src/steps/audit_logging.rs`
- `test-harness/bdd/src/steps/deadline_propagation.rs`
- `test-harness/bdd/src/steps/integration.rs`

**Success Criteria:**
- All 18 steps have real implementations
- No TODO markers
- Tests compile and run

---

## TEAM-121: Implement Missing Steps (Batch 4) + Fix Timeouts (4 hours)

**Deliverable:** Implement 17 missing steps + add service availability checks

**Steps to Implement:**
55. `Given model provisioner is downloading "hf:meta-llama/Llama-3.2-1B"`
56. `Given pool-managerd performs health checks every 5 seconds`
57. `Given workers are running with different models`
58. `Given pool-managerd is running with narration enabled`
59. `Given pool-managerd is running with cute mode enabled`
60. `Given queen-rbee requests metrics from pool-managerd`
61. `And narration includes source_location field`
62. `Then config is reloaded without restart`
63. `And narration events contain "[REDACTED]" for sensitive fields`
64. `And worker returns to idle state`
65. `When registry database becomes unavailable`
66. `And rbee-hive detects worker crash`
67. `When 10 workers register simultaneously`
68. `When 3 clients request same model simultaneously`
69. `When queen-rbee is restarted`
70. `When rbee-hive is restarted`
71. `When inference runs for 10 minutes`

**Plus: Add Service Availability Checks**
- Add `check_service_available()` helper function
- Wrap integration tests with availability checks
- Skip gracefully with clear message when services unavailable
- Add `@requires_services` tag to integration scenarios

**Files to Modify:**
- `test-harness/bdd/src/steps/integration_scenarios.rs`
- `test-harness/bdd/src/steps/metrics_observability.rs`
- `test-harness/bdd/src/steps/configuration_management.rs`
- `test-harness/bdd/src/steps/world.rs` (add helper functions)

**Success Criteria:**
- All 17 steps implemented
- Service checks in place
- 185 timeout scenarios now skip gracefully
- Clear error messages

---

## TEAM-122: Fix Panics + Final Integration (4 hours)

**Deliverable:** Fix all 104 panic failures + verify final pass rate

**Tasks:**

### Part 1: Fix Panics (3 hours)
1. Identify all panicking steps (30 min)
2. Add proper error handling to each (2 hours)
3. Replace `unwrap()` with `?` or `expect()` with context (30 min)

**Common Panic Causes:**
- Unwrapping None values
- Index out of bounds
- Failed assertions without proper error messages
- Missing fields in test data

**Files to Review:**
- All step definition files
- Focus on steps that access `world` state
- Focus on steps that parse responses

### Part 2: Final Integration (1 hour)
1. Run full test suite
2. Verify 270+/300 passing (90%+)
3. Document remaining failures
4. Create summary report

**Success Criteria:**
- Zero panics in test runs
- 90%+ pass rate achieved
- Clear documentation of any remaining failures
- Completion report ready

---

## Coordination & Dependencies

### No Dependencies Between Teams
- Each team works independently
- No merge conflicts (different files/functions)
- Can work in parallel

### Integration Points
- TEAM-122 waits for all others to complete
- TEAM-122 does final verification
- All teams commit to their own branches

### Branch Strategy
```
main
├── fix/team-117-ambiguous-steps
├── fix/team-118-missing-batch-1
├── fix/team-119-missing-batch-2
├── fix/team-120-missing-batch-3
├── fix/team-121-missing-batch-4-timeouts
└── fix/team-122-panics-final
```

**Merge Order:**
1. TEAM-117 (ambiguous steps) - merge first
2. TEAM-118, 119, 120, 121 (parallel) - merge after 117
3. TEAM-122 (panics + final) - merge last

---

## Timeline

### Day 1 (8 hours)
- **Hour 0-4:** Teams 117, 118, 119 work in parallel
- **Hour 4-8:** Teams 120, 121 work in parallel
- **End of Day:** 5 teams complete, ready for integration

### Day 2 (4 hours)
- **Hour 0-3:** TEAM-122 fixes panics
- **Hour 3-4:** TEAM-122 final verification
- **End of Day:** 90%+ pass rate achieved

**Total:** 28 team-hours = 2 days with 6 teams

---

## Success Metrics

### Target Outcomes
- ✅ **Zero ambiguous steps**
- ✅ **Zero unimplemented steps**
- ✅ **Zero unexpected timeouts** (integration tests skip gracefully)
- ✅ **Zero panics**
- ✅ **270+/300 tests passing (90%+)**

### Acceptable Remaining Failures
- Complex integration scenarios requiring full infrastructure
- Edge cases that need specific hardware (GPU, etc.)
- Performance tests that need load generation

**Maximum acceptable failures:** 30/300 (10%)

---

## Risk Mitigation

### Risk 1: Teams finish at different speeds
**Mitigation:** TEAM-122 can help slower teams

### Risk 2: Merge conflicts
**Mitigation:** Different files per team, clear boundaries

### Risk 3: Tests still fail after implementation
**Mitigation:** TEAM-122 has buffer time for fixes

### Risk 4: Scope creep
**Mitigation:** Strict task lists, no additions

---

## Communication Protocol

### Daily Standups
- Morning: Each team reports progress
- Evening: Each team reports completion status

### Blockers
- Report immediately in shared channel
- TEAM-122 (integration team) helps unblock

### Completion Criteria
- All tasks in team's list complete
- Tests compile
- No new errors introduced
- Branch pushed and ready for review

---

## Deliverables

### Each Team Delivers
1. **Code:** All implementations complete
2. **Tests:** All tests passing for their scope
3. **Documentation:** Brief summary of changes
4. **Branch:** Clean, ready to merge

### Final Deliverable (TEAM-122)
1. **Test Report:** Final pass rate (target: 90%+)
2. **Summary Document:** What was fixed, what remains
3. **Recommendations:** Next steps for remaining failures

---

## Next Steps

**READ:** `START_HERE_EMERGENCY_FIX.md` for your team assignment

**TEAMS:**
- TEAM-117: Fix ambiguous steps
- TEAM-118: Missing steps batch 1
- TEAM-119: Missing steps batch 2
- TEAM-120: Missing steps batch 3
- TEAM-121: Missing steps batch 4 + timeouts
- TEAM-122: Fix panics + final integration

---

**Status:** ✅ **READY TO EXECUTE**  
**Estimated Completion:** 2 days with 6 teams  
**Expected Outcome:** 90%+ test pass rate
