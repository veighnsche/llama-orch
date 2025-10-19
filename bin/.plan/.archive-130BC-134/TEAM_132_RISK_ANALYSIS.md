# TEAM-132: Risk Analysis

**Binary:** queen-rbee  
**Date:** 2025-10-19

---

## RISK MATRIX

| Risk Category | Severity | Likelihood | Impact | Mitigation |
|--------------|----------|------------|--------|------------|
| **HTTP API Breaking Changes** | High | Low | High | No API changes, pure refactor |
| **Worker Callback Failures** | Medium | Low | High | Isolated endpoints, comprehensive tests |
| **SSH Command Injection** | Medium | Medium | Medium | Fix during extraction (Phase 2) |
| **Test Failures** | Low | Medium | Medium | Test after each phase |
| **Circular Dependencies** | Low | Low | High | Clean hierarchy verified |
| **Integration Issues** | Medium | Low | High | Integration tests, BDD suite |
| **Performance Regression** | Low | Low | Medium | Benchmark before/after |
| **Timeline Overrun** | Low | Medium | Low | 33% buffer time included |

---

## DETAILED RISK ASSESSMENT

### 1. HTTP API Breaking Changes

**Risk Level:** 🔴 HIGH  
**Likelihood:** 🟢 Low  
**Impact:** 🔴 High

**Description:**
Changes to HTTP endpoints would break external consumers (rbee-keeper CLI, rbee-hive callbacks).

**Analysis:**
- queen-rbee exposes public HTTP API
- External consumers: rbee-keeper (CLI), rbee-hive (callbacks), clients (SDKs)
- Any endpoint path/type changes = breaking change
- Migration involves moving code, not changing APIs

**Current API Surface:**
```
GET  /health
POST /v2/registry/beehives/add
GET  /v2/registry/beehives/list
POST /v2/registry/beehives/remove
GET  /v2/workers/list
GET  /v2/workers/health
POST /v2/workers/shutdown
POST /v2/workers/register
POST /v2/workers/ready
POST /v2/tasks
POST /v1/inference
```

**Mitigation:**
- ✅ **No API changes planned** - Pure code organization refactor
- ✅ **Keep all endpoint paths identical**
- ✅ **Keep all request/response types identical**
- ✅ **Integration tests before/after migration**
- ✅ **Contract tests with wiremock**
- ✅ **Documentation of API stability commitment**

**Verification:**
```bash
# Before migration
curl http://localhost:8080/health > before.json

# After migration
curl http://localhost:8080/health > after.json

# Compare
diff before.json after.json  # Should be identical
```

**Residual Risk:** 🟢 Very Low (with mitigations applied)

---

### 2. Worker Callback Failures

**Risk Level:** 🟡 MEDIUM  
**Likelihood:** 🟢 Low  
**Impact:** 🔴 High

**Description:**
rbee-hive → queen-rbee callbacks (`/v2/workers/register`, `/v2/workers/ready`) fail after migration, breaking worker lifecycle.

**Analysis:**
- **TEAM-084:** Worker registration callback
- **TEAM-124:** Worker ready notification callback
- Critical for worker state management
- Failure = workers never transition to `Idle` state
- Failure = inference requests hang waiting for ready workers

**Current Flow:**
```
rbee-hive spawns worker
    ↓
rbee-hive → queen-rbee POST /v2/workers/register
    ↓
Worker startup completes
    ↓
rbee-hive → queen-rbee POST /v2/workers/ready
    ↓
queen-rbee updates worker state to Idle
```

**Mitigation:**
- ✅ **Callback endpoints isolated** in `http/workers.rs` (Phase 3)
- ✅ **No logic changes** - pure code movement
- ✅ **Test with actual rbee-hive integration** before declaring success
- ✅ **Verify callback URLs** are correctly configured
- ✅ **Log all callback attempts** (success and failure)
- ✅ **Add callback metrics** (duration, status codes)

**Verification Test:**
```bash
# Start queen-rbee
./queen-rbee --port 8080 &

# Start rbee-hive (triggers callbacks)
./rbee-hive daemon --addr 127.0.0.1:9200

# Spawn worker (triggers both callbacks)
curl -X POST http://localhost:9200/v1/workers/spawn \
  -d '{"model_ref":"hf:tinyllama","backend":"cpu","device":0}'

# Verify worker registered
curl http://localhost:8080/v2/workers/list | jq

# Verify worker ready
# Should see worker in "idle" state after model loads
```

**Rollback Procedure:**
```bash
# If callbacks fail
git revert <migration-commit>
cargo build --release --bin queen-rbee
systemctl restart queen-rbee
```

**Residual Risk:** 🟢 Low (callbacks well-isolated, testable)

---

### 3. SSH Command Injection Vulnerability

**Risk Level:** 🟡 MEDIUM  
**Likelihood:** 🟡 MEDIUM  
**Impact:** 🟡 MEDIUM

**Description:**
Security vulnerability in `ssh.rs:79` allows command injection via unsanitized user input.

**Analysis:**
- **TEAM-109 Audit:** Command injection vulnerability documented
- **Location:** `bin/queen-rbee/src/ssh.rs:79-81`
- **Vulnerable Code:**
  ```rust
  .arg(format!("{}@{}", user, host))
  .arg(command)  // ⚠️ UNSAFE: command is user-provided string
  ```
- **Attack Vector:** Malicious admin adds node with crafted command
- **Impact:** Arbitrary command execution on orchestrator host

**Attack Example:**
```bash
# Attacker adds node with malicious install_path
POST /v2/registry/beehives/add
{
  "node_name": "evil",
  "ssh_host": "localhost",
  "install_path": "/tmp/rbee && rm -rf /"  // Injection!
}

# Later, orchestrator executes:
ssh user@localhost "/tmp/rbee && rm -rf / daemon"
```

**Mitigation:**
- 🔴 **FIX REQUIRED** during Phase 2 (queen-rbee-remote extraction)
- ✅ **Use shellwords crate** to parse and validate commands
- ✅ **Whitelist allowed characters** in commands
- ✅ **Use `--` argument separator** to force command boundaries
- ✅ **Pass command as separate args** instead of single string
- ✅ **Add command injection test** to catch regressions

**Fixed Code:**
```rust
use shellwords;

// Validate command before execution
let sanitized = shellwords::split(command)
    .map_err(|e| anyhow::anyhow!("Invalid command: {}", e))?;

// Block dangerous patterns
for part in &sanitized {
    if part.contains("&&") || part.contains("||") || part.contains(";") {
        anyhow::bail!("Command injection detected");
    }
}

.arg(format!("{}@{}", user, host))
.arg("--")  // Force boundary
.args(&sanitized)  // Separate arguments
```

**Verification Test:**
```rust
#[tokio::test]
async fn test_command_injection_blocked() {
    let result = execute_remote_command(
        "localhost", 22, "user", None,
        "echo safe && rm -rf /"  // Malicious
    ).await;
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("injection"));
}
```

**Residual Risk:** 🟢 Low (after fix applied and tested)

---

### 4. Circular Dependencies

**Risk Level:** 🟢 LOW  
**Likelihood:** 🟢 Low  
**Impact:** 🔴 High

**Description:**
Crates depend on each other in a circular manner, preventing compilation.

**Analysis:**
Current dependency graph:
```
main.rs (binary)
    ├─> http-server
    │       └─> registry
    ├─> orchestrator
    │       ├─> registry
    │       └─> remote
    ├─> registry (leaf)
    └─> remote (leaf)
```

**No Circular Dependencies Detected:**
- ✅ `registry` has no dependencies on other queen-rbee crates
- ✅ `remote` has no dependencies on other queen-rbee crates
- ✅ `http-server` depends only on `registry`
- ✅ `orchestrator` depends on `registry` + `remote` (no cycles)
- ✅ Binary depends on all (top of hierarchy)

**Mitigation:**
- ✅ **Clear dependency hierarchy** designed upfront
- ✅ **Leaf crates first** (registry, remote)
- ✅ **Bottom-up extraction** prevents circular deps
- ✅ **Compile after each phase** catches issues early

**Verification:**
```bash
# Check dependency graph
cargo tree -p queen-rbee

# Should show:
# queen-rbee
# ├── queen-rbee-http-server
# │   └── queen-rbee-registry
# ├── queen-rbee-orchestrator
# │   ├── queen-rbee-registry
# │   └── queen-rbee-remote
# ├── queen-rbee-registry
# └── queen-rbee-remote
```

**Residual Risk:** 🟢 Very Low (verified no cycles)

---

### 5. Test Failures

**Risk Level:** 🟢 LOW  
**Likelihood:** 🟡 MEDIUM  
**Impact:** 🟡 MEDIUM

**Description:**
Tests fail after migration due to broken imports, missing mocks, or changed behavior.

**Analysis:**
Current test coverage:
- `beehive_registry::tests` - CRUD operations (1 test)
- `worker_registry::tests` - CRUD operations (1 test)
- `http/middleware/auth::tests` - Authentication (4 tests)
- `http/routes::tests` - Router creation (1 test)
- `http/health::tests` - Health endpoint (1 test)
- `ssh::tests` - Connection test (1 test, ignored)
- `preflight::tests` - Preflight checks (2 tests)

**Total:** 11 tests across 8 modules

**Migration Impact:**
- Import paths change: `crate::*` → `queen_rbee_*::*`
- Module structure changes: `http/types.rs` → `queen_rbee_http_server::types`
- Test dependencies may need updates

**Mitigation:**
- ✅ **Run tests after each phase** (5 checkpoints)
- ✅ **Keep test code with modules** during extraction
- ✅ **Update test imports incrementally**
- ✅ **Use wiremock for HTTP mocking** (already in dev-dependencies)
- ✅ **Add integration tests** for end-to-end flows
- ✅ **Document test patterns** for consistency

**Test Execution Plan:**
```bash
# Phase 1: Registry
cargo test -p queen-rbee-registry
cargo test --bin queen-rbee

# Phase 2: Remote
cargo test -p queen-rbee-remote
cargo test --bin queen-rbee

# Phase 3: HTTP Server
cargo test -p queen-rbee-http-server
cargo test --bin queen-rbee

# Phase 4: Orchestrator
cargo test -p queen-rbee-orchestrator
cargo test --workspace

# Phase 5: Final
cargo test --workspace --release
```

**Verification:**
- All tests pass in each crate individually
- All tests pass in workspace
- No test warnings or ignored tests (except SSH integration tests)

**Residual Risk:** 🟢 Low (tests are well-organized, incremental verification)

---

### 6. Integration Issues

**Risk Level:** 🟡 MEDIUM  
**Likelihood:** 🟢 Low  
**Impact:** 🔴 High

**Description:**
Crates integrate incorrectly, causing runtime failures despite passing unit tests.

**Analysis:**
Integration points:
1. **Binary ↔ All Crates:** Binary creates router, starts server
2. **HTTP Server ↔ Registry:** Endpoints query registries
3. **HTTP Server ↔ Orchestrator:** Inference endpoints delegate to orchestrator
4. **Orchestrator ↔ Registry:** Looks up nodes, registers workers
5. **Orchestrator ↔ Remote:** SSH to remote nodes
6. **rbee-hive → queen-rbee:** Callback notifications

**Failure Modes:**
- Type mismatches between crates
- Missing dependencies in Cargo.toml
- Incorrect trait implementations
- Async runtime incompatibilities

**Mitigation:**
- ✅ **Integration tests** after Phase 4 (orchestrator)
- ✅ **End-to-end test** with real rbee-hive + worker
- ✅ **BDD test suite** for critical flows
- ✅ **Contract tests** for HTTP APIs
- ✅ **Smoke test script** to verify all endpoints

**Integration Test Plan:**
```rust
// tests/integration_test.rs
#[tokio::test]
async fn test_full_inference_flow() {
    // 1. Start queen-rbee
    // 2. Add node to registry
    // 3. Trigger inference task
    // 4. Verify worker spawns
    // 5. Verify worker ready callback
    // 6. Verify inference executes
    // 7. Verify SSE stream
}
```

**Smoke Test Script:**
```bash
#!/bin/bash
# smoke_test.sh

# Start queen-rbee
./target/release/queen-rbee --port 8080 &
sleep 2

# Test health
curl -f http://localhost:8080/health || exit 1

# Test registry endpoints
curl -X POST http://localhost:8080/v2/registry/beehives/add \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"node_name":"test","ssh_host":"localhost",...}' || exit 1

curl http://localhost:8080/v2/registry/beehives/list \
  -H "Authorization: Bearer $TOKEN" || exit 1

# Test worker endpoints
curl http://localhost:8080/v2/workers/list \
  -H "Authorization: Bearer $TOKEN" || exit 1

echo "✅ Smoke test passed"
```

**Verification:**
- Integration tests pass
- Smoke test passes
- Manual E2E test passes (with real rbee-hive)
- BDD suite passes (if exists)

**Residual Risk:** 🟢 Low (comprehensive integration testing)

---

### 7. Performance Regression

**Risk Level:** 🟢 LOW  
**Likelihood:** 🟢 Low  
**Impact:** 🟡 MEDIUM

**Description:**
Migration causes performance degradation (slower inference, higher latency).

**Analysis:**
Performance-critical paths:
1. **Inference orchestration** - Must remain fast (~500ms overhead max)
2. **Worker ready checks** - Polling loops (every 2s)
3. **SSE streaming** - Pass-through, no buffering
4. **Registry queries** - In-memory (worker) and SQLite (beehive)

**Potential Causes:**
- Additional indirection through crate boundaries
- Async runtime overhead (unlikely with tokio)
- Compiler optimization differences
- Memory allocation patterns

**Mitigation:**
- ✅ **Benchmark before migration** (baseline)
- ✅ **Benchmark after migration** (compare)
- ✅ **Profile critical paths** (flamegraph)
- ✅ **Monitor in production** (metrics)
- ✅ **Optimize if regression >10%**

**Benchmark Plan:**
```bash
# Before migration
hyperfine --warmup 3 --runs 10 \
  'curl -X POST http://localhost:8080/v2/tasks -d @test_request.json'

# After migration
hyperfine --warmup 3 --runs 10 \
  'curl -X POST http://localhost:8080/v2/tasks -d @test_request.json'

# Compare results
```

**Acceptance Criteria:**
- Inference latency: <10% increase
- Registry query time: <5% increase
- SSE streaming: No measurable difference
- Memory usage: <10% increase

**Verification:**
```bash
# Profile critical path
cargo flamegraph --bin queen-rbee -- --port 8080
# Analyze flamegraph.svg for hotspots
```

**Residual Risk:** 🟢 Very Low (pure refactor, no algorithm changes)

---

### 8. Timeline Overrun

**Risk Level:** 🟢 LOW  
**Likelihood:** 🟡 MEDIUM  
**Impact:** 🟢 LOW

**Description:**
Migration takes longer than estimated 20 hours (2.5 days).

**Analysis:**
Estimated timeline:
- Phase 1: Registry (2h)
- Phase 2: Remote (3h)
- Phase 3: HTTP Server (4h)
- Phase 4: Orchestrator (5h)
- Phase 5: Binary Cleanup (1h)
- **Subtotal:** 15h
- **Buffer (33%):** 5h
- **Total:** 20h

**Overrun Risks:**
- Unexpected test failures (most likely)
- Complex import refactoring
- Integration issues
- Security fix takes longer than expected

**Mitigation:**
- ✅ **33% buffer time included** (5 hours)
- ✅ **Start with low-risk crate** (registry as pilot)
- ✅ **Stop if pilot takes >3 hours** (re-estimate)
- ✅ **Timebox each phase** (strict time limits)
- ✅ **Skip optional tasks** if behind schedule
- ✅ **Parallel work possible** (documentation during tests)

**Contingency Plan:**
```
If behind schedule after Phase 2:
  → Pause, reassess timeline
  → Consider simplified approach (fewer crates)
  → Update stakeholders

If behind schedule after Phase 4:
  → Complete Phase 5 (binary cleanup)
  → Defer documentation updates
  → Defer performance optimization
```

**Residual Risk:** 🟢 Very Low (buffer + phased approach + early warning)

---

## RISK MITIGATION SUMMARY

### Critical Mitigations (Must Do)

1. **🔴 Fix SSH Command Injection** (Phase 2)
   - Use shellwords crate
   - Add command validation
   - Add regression test

2. **🟡 Test All API Endpoints** (Phase 5)
   - Run smoke test script
   - Verify with real rbee-hive
   - Check worker callbacks

3. **🟡 Integration Testing** (Phase 4-5)
   - End-to-end test
   - BDD suite
   - Contract tests

### Recommended Mitigations (Should Do)

4. **🟢 Performance Benchmarking**
   - Baseline before migration
   - Compare after each phase
   - Profile critical paths

5. **🟢 Documentation Updates**
   - README with new structure
   - Architecture diagram
   - Migration notes

6. **🟢 CI/CD Updates**
   - Per-crate test jobs
   - Parallel test execution
   - Release workflow

### Optional Mitigations (Nice to Have)

7. **🟢 Shared Crate Opportunities**
   - Extract `rbee-http-types`
   - Move `BeehiveNode` to `hive-core`
   - Create `rbee-http-client` wrapper

8. **🟢 Monitoring**
   - Track build times
   - Monitor binary size
   - Track test execution time

---

## CONTINGENCY PLANS

### If Phase 1 (Registry) Fails
- **Impact:** Pilot failure, entire approach questioned
- **Action:**
  1. Revert changes (git reset)
  2. Analyze root cause
  3. Adjust approach (maybe merge registry into http-server?)
  4. Re-estimate timeline
  5. Decide: Continue or abort

### If Phase 2 (Remote) Security Fix Fails
- **Impact:** Cannot safely extract remote crate
- **Action:**
  1. Leave remote code in binary (skip Phase 2)
  2. Continue with Phase 3-4
  3. File security issue for future fix
  4. Adjust timeline (-3h)

### If Phase 3 (HTTP Server) Import Hell
- **Impact:** Too many broken imports, complex refactor
- **Action:**
  1. Pause Phase 3
  2. Create import mapping document
  3. Use semi-automated refactor script
  4. Add 2 hours to timeline
  5. Continue with mapping

### If Phase 4 (Orchestrator) Integration Fails
- **Impact:** Orchestrator doesn't integrate with http-server
- **Action:**
  1. Revert Phase 4
  2. Keep orchestrator in binary
  3. Complete Phase 5 with 3 crates only
  4. Document orchestrator split for future work

### If Tests Fail Catastrophically
- **Impact:** Many tests fail, unclear root cause
- **Action:**
  1. Stop migration
  2. Revert to last working state
  3. Fix tests one by one
  4. Resume migration with fixed tests
  5. Add 4 hours to timeline

### If Performance Regresses >20%
- **Impact:** Unacceptable performance degradation
- **Action:**
  1. Profile to find hotspots
  2. Optimize critical paths
  3. If no fix found: Revert migration
  4. Document findings
  5. Consider alternative approach

---

## ROLLBACK TRIGGERS

### Immediate Rollback (Stop Work)

- ❌ Security vulnerability introduced
- ❌ Data loss in registry
- ❌ Production system affected
- ❌ >50% of tests failing

### Phase Rollback (Revert Phase)

- ❌ Phase takes >2× estimated time
- ❌ Circular dependencies introduced
- ❌ Integration completely broken
- ❌ >20% performance regression

### Project Abort (Full Rollback)

- ❌ After Phase 3 and timeline >30 hours
- ❌ Fundamental architecture flaw discovered
- ❌ External dependency becomes blocker
- ❌ Stakeholder decides not to proceed

---

## MONITORING & VERIFICATION

### During Migration

**Per-Phase Checklist:**
- [ ] Crate compiles independently
- [ ] Crate tests pass
- [ ] Binary still builds
- [ ] Binary tests pass
- [ ] No new warnings
- [ ] Git commit with phase tag

**Red Flags:**
- ⚠️ More than 10 compiler errors
- ⚠️ More than 5 failing tests
- ⚠️ More than 20 warnings
- ⚠️ Phase takes 2× estimated time

### Post-Migration

**Acceptance Gates:**
- [ ] All crates compile
- [ ] All tests pass (11/11)
- [ ] Smoke test passes
- [ ] Integration test passes
- [ ] Performance <10% regression
- [ ] Binary size <10% increase
- [ ] Documentation updated
- [ ] CI/CD updated

**Success Metrics:**
- ✅ Incremental build time: <15s (vs 45-60s)
- ✅ Test iteration time: <8s per crate
- ✅ Full rebuild time: <40s (parallel)
- ✅ Binary size: Similar to before
- ✅ Zero production issues

---

## CONCLUSION

**Overall Risk Assessment:** 🟢 LOW-MEDIUM

**Key Strengths:**
- Well-structured existing code
- Clear module boundaries
- Good test coverage
- Small codebase (2,015 LOC)
- No external API consumers (only internal)

**Key Risks:**
- 🟡 HTTP API breaking changes (mitigated with no-change policy)
- 🟡 SSH command injection (must fix during Phase 2)
- 🟡 Integration failures (mitigated with comprehensive testing)

**Recommendation:** **🟢 PROCEED** with migration

**Confidence Level:** HIGH (85%)

The migration is low-risk with proper mitigations applied. The phased approach with pilot (registry) provides early validation. The 33% buffer time handles unexpected issues. Rollback procedures are well-defined.

**Next Steps:**
1. Review and approve this risk analysis
2. Begin Phase 1 (Registry extraction)
3. Monitor progress at each phase gate
4. Execute contingency plans if triggers hit

---

**Risk Analysis Complete**  
**Ready for Migration**
