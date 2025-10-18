# TEAM-109: Equal Work Units Distribution

**Date:** 2025-10-18  
**Total Files:** 227  
**Total Hours:** 99h  
**Work Units:** 10 units of ~10 hours each

---

## Work Distribution Strategy

**Target:** ~10 hours per unit (~23 files per unit)  
**Approach:** Mix high-priority and low-priority files in each unit

---

## Unit 1: Critical Entry Points + HTTP Security (10h)

**Files:** 21  
**Estimated Time:** 10 hours  
**Focus:** Authentication, secret loading, HTTP security

### Main Binaries (3 files - 2h)
1. ✅ `bin/queen-rbee/src/main.rs` (150 lines) - 45 min - 🔴 CRITICAL secret loading
2. ✅ `bin/rbee-hive/src/main.rs` (100 lines) - 30 min
3. ✅ `bin/llm-worker-rbee/src/main.rs` (300 lines) - 45 min - 🔴 CRITICAL secret loading

### Authentication Middleware (3 files - 2.5h)
4. ❌ `bin/rbee-hive/src/http/middleware/auth.rs` (150 lines) - 45 min - CRITICAL
5. ❌ `bin/queen-rbee/src/http/middleware/auth.rs` (150 lines) - 45 min - CRITICAL
6. ❌ `bin/llm-worker-rbee/src/http/middleware/auth.rs` (150 lines) - 45 min - CRITICAL

### HTTP Routes (3 files - 2.5h)
7. ❌ `bin/rbee-hive/src/http/routes.rs` (150 lines) - 45 min - CRITICAL
8. ❌ `bin/queen-rbee/src/http/routes.rs` (150 lines) - 45 min - CRITICAL
9. ❌ `bin/llm-worker-rbee/src/http/routes.rs` (150 lines) - 45 min - CRITICAL

### Shared Security Crates (12 files - 3h)
10. ❌ `bin/shared-crates/auth-min/src/lib.rs` (100 lines) - 20 min
11. ❌ `bin/shared-crates/auth-min/src/bearer.rs` (80 lines) - 15 min
12. ❌ `bin/shared-crates/auth-min/src/timing.rs` (80 lines) - 15 min
13-15. ❌ `bin/shared-crates/auth-min/*` (5 remaining files) - 1h
16. ❌ `bin/shared-crates/secrets-management/src/lib.rs` (100 lines) - 20 min
17. ❌ `bin/shared-crates/secrets-management/src/types/secret.rs` (150 lines) - 30 min
18. ❌ `bin/shared-crates/secrets-management/src/loaders/file.rs` (150 lines) - 30 min
19-21. ❌ `bin/shared-crates/secrets-management/*` (3 key files) - 45 min

**Deliverable:** All authentication and secret loading audited

---

## Unit 2: HTTP Handlers + Input Validation (10h)

**Files:** 24  
**Estimated Time:** 10 hours  
**Focus:** API handlers, input validation

### rbee-hive HTTP Handlers (5 files - 3h)
1. ❌ `bin/rbee-hive/src/http/workers.rs` (300 lines) - 1h - CRITICAL validation
2. ❌ `bin/rbee-hive/src/http/models.rs` (250 lines) - 1h - CRITICAL path traversal
3. ❌ `bin/rbee-hive/src/http/health.rs` (100 lines) - 30 min
4. ❌ `bin/rbee-hive/src/http/metrics.rs` (80 lines) - 20 min
5. ❌ `bin/rbee-hive/src/http/server.rs` (200 lines) - 45 min

### queen-rbee HTTP Handlers (4 files - 3h)
6. ❌ `bin/queen-rbee/src/http/beehives.rs` (250 lines) - 1h - CRITICAL validation
7. ❌ `bin/queen-rbee/src/http/workers.rs` (200 lines) - 45 min
8. ❌ `bin/queen-rbee/src/http/inference.rs` (300 lines) - 1h - CRITICAL validation
9. ❌ `bin/queen-rbee/src/http/health.rs` (100 lines) - 30 min

### llm-worker HTTP Handlers (3 files - 2.5h)
10. ❌ `bin/llm-worker-rbee/src/http/execute.rs` (400 lines) - 1.5h - CRITICAL validation
11. ❌ `bin/llm-worker-rbee/src/http/validation.rs` (200 lines) - 1h - CRITICAL
12. ❌ `bin/llm-worker-rbee/src/http/sse.rs` (250 lines) - 1h

### Input Validation Crate (12 files - 1.5h)
13. ❌ `bin/shared-crates/input-validation/src/lib.rs` (100 lines) - 15 min
14. ❌ `bin/shared-crates/input-validation/src/sanitize.rs` (150 lines) - 20 min
15. ❌ `bin/shared-crates/input-validation/src/validate.rs` (150 lines) - 20 min
16-24. ❌ `bin/shared-crates/input-validation/*` (9 remaining files) - 45 min

**Deliverable:** All HTTP handlers and input validation audited

---

## Unit 3: Core Logic + State Management (10h)

**Files:** 22  
**Estimated Time:** 10 hours  
**Focus:** Registries, state management, core logic

### rbee-hive Core (8 files - 4h)
1. ❌ `bin/rbee-hive/src/registry.rs` (400 lines) - 1.5h - CRITICAL state management
2. ❌ `bin/rbee-hive/src/worker_provisioner.rs` (250 lines) - 45 min
3. ❌ `bin/rbee-hive/src/monitor.rs` (200 lines) - 45 min
4. ❌ `bin/rbee-hive/src/download_tracker.rs` (200 lines) - 30 min
5. ❌ `bin/rbee-hive/src/metrics.rs` (150 lines) - 30 min
6. ❌ `bin/rbee-hive/src/timeout.rs` (100 lines) - 20 min
7. ❌ `bin/rbee-hive/src/cli.rs` (150 lines) - 30 min
8. ❌ `bin/rbee-hive/src/lib.rs` (100 lines) - 20 min

### queen-rbee Core (3 files - 2.5h)
9. ❌ `bin/queen-rbee/src/beehive_registry.rs` (300 lines) - 1h - CRITICAL state
10. ❌ `bin/queen-rbee/src/worker_registry.rs` (250 lines) - 45 min
11. ❌ `bin/queen-rbee/src/ssh.rs` (200 lines) - 45 min - CRITICAL command injection

### llm-worker Core (5 files - 2.5h)
12. ❌ `bin/llm-worker-rbee/src/lib.rs` (150 lines) - 30 min
13. ❌ `bin/llm-worker-rbee/src/device.rs` (200 lines) - 45 min
14. ❌ `bin/llm-worker-rbee/src/error.rs` (150 lines) - 30 min
15. ❌ `bin/llm-worker-rbee/src/narration.rs` (100 lines) - 30 min
16. ❌ `bin/llm-worker-rbee/src/token_output_stream.rs` (200 lines) - 45 min

### Shared Core Crates (6 files - 1h)
17. ❌ `bin/shared-crates/hive-core/src/lib.rs` (150 lines) - 20 min
18-22. ❌ `bin/shared-crates/hive-core/*` (5 remaining files) - 40 min

**Deliverable:** All core state management audited

---

## Unit 4: Commands + Provisioner (10h)

**Files:** 20  
**Estimated Time:** 10 hours  
**Focus:** CLI commands, model provisioning

### rbee-hive Commands (6 files - 3h)
1. ❌ `bin/rbee-hive/src/commands/daemon.rs` (200 lines) - 45 min - 🔴 CRITICAL secret loading
2. ❌ `bin/rbee-hive/src/commands/worker.rs` (180 lines) - 30 min
3. ❌ `bin/rbee-hive/src/commands/models.rs` (150 lines) - 30 min
4. ❌ `bin/rbee-hive/src/commands/detect.rs` (120 lines) - 30 min
5. ❌ `bin/rbee-hive/src/commands/status.rs` (100 lines) - 30 min
6. ❌ `bin/rbee-hive/src/commands/mod.rs` (50 lines) - 15 min

### rbee-hive Provisioner (5 files - 2.5h)
7. ❌ `bin/rbee-hive/src/provisioner/download.rs` (300 lines) - 1h - CRITICAL path validation
8. ❌ `bin/rbee-hive/src/provisioner/operations.rs` (250 lines) - 45 min
9. ❌ `bin/rbee-hive/src/provisioner/catalog.rs` (200 lines) - 30 min
10. ❌ `bin/rbee-hive/src/provisioner/types.rs` (100 lines) - 15 min
11. ❌ `bin/rbee-hive/src/provisioner/mod.rs` (50 lines) - 15 min

### HTTP Module Files (3 files - 1h)
12. ❌ `bin/rbee-hive/src/http/mod.rs` (50 lines) - 15 min
13. ❌ `bin/queen-rbee/src/http/mod.rs` (50 lines) - 15 min
14. ❌ `bin/llm-worker-rbee/src/http/mod.rs` (50 lines) - 15 min

### Middleware Module Files (3 files - 0.5h)
15. ❌ `bin/rbee-hive/src/http/middleware/mod.rs` (30 lines) - 10 min
16. ❌ `bin/queen-rbee/src/http/middleware/mod.rs` (30 lines) - 10 min
17. ❌ `bin/llm-worker-rbee/src/http/middleware/mod.rs` (30 lines) - 10 min

### Shared Crates (3 files - 3h)
18. ❌ `bin/shared-crates/model-catalog/*` (6 files) - 2h
19. ❌ `bin/shared-crates/gpu-info/*` (5 files) - 1h

**Deliverable:** All commands and provisioning audited

---

## Unit 5: Backend Inference (10h)

**Files:** 21  
**Estimated Time:** 10 hours  
**Focus:** Inference engine, model loading

### Backend Core (5 files - 3.5h)
1. ❌ `bin/llm-worker-rbee/src/backend/inference.rs` (500 lines) - 2h - CRITICAL
2. ❌ `bin/llm-worker-rbee/src/backend/sampling.rs` (300 lines) - 1h
3. ❌ `bin/llm-worker-rbee/src/backend/tokenizer_loader.rs` (200 lines) - 45 min
4. ❌ `bin/llm-worker-rbee/src/backend/gguf_tokenizer.rs` (250 lines) - 1h
5. ❌ `bin/llm-worker-rbee/src/backend/mod.rs` (100 lines) - 20 min

### Model Implementations (9 files - 5.5h)
6. ❌ `bin/llm-worker-rbee/src/backend/models/mod.rs` (100 lines) - 20 min
7. ❌ `bin/llm-worker-rbee/src/backend/models/llama.rs` (300 lines) - 1h
8. ❌ `bin/llm-worker-rbee/src/backend/models/quantized_llama.rs` (250 lines) - 45 min
9. ❌ `bin/llm-worker-rbee/src/backend/models/mistral.rs` (200 lines) - 30 min
10. ❌ `bin/llm-worker-rbee/src/backend/models/phi.rs` (200 lines) - 30 min
11. ❌ `bin/llm-worker-rbee/src/backend/models/quantized_phi.rs` (200 lines) - 30 min
12. ❌ `bin/llm-worker-rbee/src/backend/models/qwen.rs` (200 lines) - 30 min
13. ❌ `bin/llm-worker-rbee/src/backend/models/quantized_qwen.rs` (200 lines) - 30 min

### Binaries (3 files - 1.5h)
14. ❌ `bin/llm-worker-rbee/src/bin/cpu.rs` (150 lines) - 30 min
15. ❌ `bin/llm-worker-rbee/src/bin/cuda.rs` (200 lines) - 45 min
16. ❌ `bin/llm-worker-rbee/src/bin/metal.rs` (150 lines) - 30 min

### Common Modules (5 files - 2.5h)
17. ❌ `bin/llm-worker-rbee/src/common/startup.rs` (200 lines) - 45 min
18. ❌ `bin/llm-worker-rbee/src/common/sampling_config.rs` (150 lines) - 30 min
19. ❌ `bin/llm-worker-rbee/src/common/error.rs` (150 lines) - 30 min
20. ❌ `bin/llm-worker-rbee/src/common/inference_result.rs` (100 lines) - 30 min
21. ❌ `bin/llm-worker-rbee/src/common/mod.rs` (50 lines) - 15 min

**Deliverable:** All inference engine code audited

---

## Unit 6: HTTP Remaining + Preflight (10h)

**Files:** 23  
**Estimated Time:** 10 hours  
**Focus:** Remaining HTTP handlers, preflight checks

### llm-worker HTTP Remaining (5 files - 2.5h)
1. ❌ `bin/llm-worker-rbee/src/http/server.rs` (200 lines) - 45 min
2. ❌ `bin/llm-worker-rbee/src/http/health.rs` (100 lines) - 30 min
3. ❌ `bin/llm-worker-rbee/src/http/ready.rs` (150 lines) - 30 min
4. ❌ `bin/llm-worker-rbee/src/http/loading.rs` (200 lines) - 45 min
5. ❌ `bin/llm-worker-rbee/src/http/backend.rs` (150 lines) - 30 min
6. ❌ `bin/llm-worker-rbee/src/http/narration_channel.rs` (150 lines) - 30 min

### Preflight (1 file - 0.5h)
7. ❌ `bin/queen-rbee/src/preflight/rbee_hive.rs` (150 lines) - 30 min

### Secrets Management Remaining (15 files - 4h)
8-22. ❌ `bin/shared-crates/secrets-management/*` (15 remaining files) - 4h

### JWT Guardian (5 files - 2h)
23-27. ❌ `bin/shared-crates/jwt-guardian/*` (5 files) - 2h

**Deliverable:** HTTP layer complete, JWT audited

---

## Unit 7: Audit Logging + Deadlines (10h)

**Files:** 21  
**Estimated Time:** 10 hours  
**Focus:** Audit logging, deadline propagation

### Audit Logging (16 files - 6h)
1-16. ❌ `bin/shared-crates/audit-logging/*` (16 files) - 6h
   - **Focus:** Hash chain implementation, tamper detection
   - **Critical:** Cryptographic correctness

### Deadline Propagation (5 files - 2.5h)
17-21. ❌ `bin/shared-crates/deadline-propagation/*` (5 files) - 2.5h
   - **Focus:** Timeout handling, propagation

### Resource Limits (5 files - 2.5h)
22-26. ❌ `bin/shared-crates/resource-limits/*` (5 files) - 2.5h
   - **Focus:** CPU/memory/VRAM limits

**Deliverable:** All operational shared crates audited

---

## Unit 8: Narration Core (Part 1) (10h)

**Files:** 25  
**Estimated Time:** 10 hours  
**Focus:** Narration-core integration (TEAM-100's work)

### Narration Core (25 files - 10h)
1-25. ❌ `bin/shared-crates/narration-core/*` (25 files) - 10h
   - **Focus:** Integration points, API usage
   - **Note:** TEAM-100 integrated this - verify correctness
   - **Critical:** Ensure no performance impact

**Deliverable:** Narration-core fully audited

---

## Unit 9: Tests + Integration (10h)

**Files:** 24  
**Estimated Time:** 10 hours  
**Focus:** All test files

### llm-worker Tests (5 files - 2.5h)
1. ❌ `bin/llm-worker-rbee/tests/team_009_smoke.rs` (200 lines) - 30 min
2. ❌ `bin/llm-worker-rbee/tests/team_011_integration.rs` (300 lines) - 45 min
3. ❌ `bin/llm-worker-rbee/tests/team_013_cuda_integration.rs` (250 lines) - 45 min
4. ❌ `bin/llm-worker-rbee/tests/multi_model_support.rs` (200 lines) - 30 min
5. ❌ `bin/llm-worker-rbee/tests/test_question_mark_tokenization.rs` (100 lines) - 15 min

### rbee-hive Tests (1 file - 0.5h)
6. ❌ `bin/rbee-hive/tests/model_provisioner_integration.rs` (200 lines) - 30 min

### Shared Crate Tests (~18 files - 7h)
7-24. ❌ All test files in shared crates - 7h
   - auth-min tests
   - secrets-management tests
   - input-validation tests
   - audit-logging tests
   - Other shared crate tests

**Deliverable:** All tests audited (unwrap/expect OK in tests)

---

## Unit 10: Cleanup + Final Files (9h)

**Files:** 26  
**Estimated Time:** 9 hours  
**Focus:** Remaining miscellaneous files

### Remaining Shared Crate Files (26 files - 9h)
1-26. ❌ All remaining files not covered in Units 1-9
   - Documentation files
   - Example files
   - Build configuration
   - Utility modules
   - Helper functions

**Deliverable:** 100% file coverage, all 227 files audited

---

## Summary: Equal Distribution

| Unit | Files | Hours | Focus Area |
|------|-------|-------|------------|
| **Unit 1** | 21 | 10h | Critical Entry Points + HTTP Security |
| **Unit 2** | 24 | 10h | HTTP Handlers + Input Validation |
| **Unit 3** | 22 | 10h | Core Logic + State Management |
| **Unit 4** | 20 | 10h | Commands + Provisioner |
| **Unit 5** | 21 | 10h | Backend Inference |
| **Unit 6** | 23 | 10h | HTTP Remaining + Preflight |
| **Unit 7** | 21 | 10h | Audit Logging + Deadlines |
| **Unit 8** | 25 | 10h | Narration Core |
| **Unit 9** | 24 | 10h | Tests + Integration |
| **Unit 10** | 26 | 9h | Cleanup + Final Files |
| **TOTAL** | **227** | **99h** | - |

**Average:** 22.7 files per unit, 9.9 hours per unit

**Fairness:** ✅ Each unit has roughly equal work (±1 hour variance)

---

## Assignment Strategy

### Option A: Parallel Teams (10 teams, 1 day each)
- Team 1 → Unit 1
- Team 2 → Unit 2
- ...
- Team 10 → Unit 10
- **Total Time:** 1 day (with 10 teams working in parallel)

### Option B: Sequential (1 team, 10 days)
- Day 1 → Unit 1
- Day 2 → Unit 2
- ...
- Day 10 → Unit 10
- **Total Time:** 10 working days

### Option C: 2 Teams (5 days each)
- Team A: Units 1, 3, 5, 7, 9 (5 days)
- Team B: Units 2, 4, 6, 8, 10 (5 days)
- **Total Time:** 5 days (with 2 teams working in parallel)

---

## Critical Path

**Must complete in order:**
1. Unit 1 (Security) → Fix critical issues
2. Unit 2 (Validation) → Verify input handling
3. Units 3-10 → Can be done in any order

**P0 Critical Files Distributed Across Units:**
- Unit 1: 6 critical files (main.rs + auth middleware)
- Unit 2: 5 critical files (HTTP handlers)
- Unit 3: 3 critical files (registries + SSH)
- Unit 4: 2 critical files (daemon.rs + provisioner)
- Unit 5: 1 critical file (inference.rs)

**Total P0 Files:** 17 files across first 5 units

---

**Created by:** TEAM-109  
**Date:** 2025-10-18  
**Status:** Equal distribution complete - ready for assignment

**This is fair work distribution.**
