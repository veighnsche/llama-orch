# TEAM-131: Risk Analysis

**Binary:** rbee-hive  
**Date:** 2025-10-19

---

## RISK MATRIX

| Risk Category | Severity | Likelihood | Impact | Mitigation |
|--------------|----------|------------|--------|------------|
| **Breaking Changes** | Medium | Low | Medium | Phased migration, continuous testing |
| **HTTP Coupling** | Medium | Medium | Medium | Careful API design, interface-only coupling |
| **Test Failures** | Low | Low | High | Run tests after each step |
| **Timeline Overrun** | Low | Medium | Low | Buffer time, start with low-risk crates |
| **Integration Issues** | Medium | Low | High | Integration tests, BDD suite |
| **Performance Regression** | Low | Low | Medium | Benchmark before/after |

---

## DETAILED RISK ASSESSMENT

### 1. Breaking Changes to Public APIs

**Risk Level:** 🟡 Medium  
**Likelihood:** Low  
**Impact:** Medium

**Description:**
Moving code to separate crates changes import paths. Any external consumers would break.

**Analysis:**
- rbee-hive is a binary, not a library
- No external consumers identified
- Only internal modules affected

**Mitigation:**
- Keep binary's public API unchanged
- Binary becomes thin wrapper around crates
- Verify no other binaries import from rbee-hive

**Residual Risk:** Very Low

---

### 2. HTTP Layer Coupling

**Risk Level:** 🟡 Medium  
**Likelihood:** Medium  
**Impact:** Medium

**Description:**
`http/workers.rs` has tight coupling with provisioner and registry. Splitting may reveal hidden dependencies.

**Evidence:**
```rust
// In http/workers.rs:
state.provisioner.download_model(&reference, &provider).await?;
state.model_catalog.register_model(&model_info).await?;
registry.register(worker).await;
```

**Mitigation:**
1. **Keep coupling at interface level:**
   - Pass dependencies via AppState
   - Use traits for mockability
   
2. **Extract gradually:**
   - Move registry first
   - Then provisioner
   - Finally HTTP server

3. **Test coverage:**
   - Integration tests for HTTP endpoints
   - Mock dependencies in tests

**Residual Risk:** Low

---

### 3. Test Failures During Migration

**Risk Level:** 🟢 Low  
**Likelihood:** Low  
**Impact:** High

**Description:**
Tests may fail due to broken imports or changed module paths.

**Current Test Coverage:**
- registry.rs: ~60% (18 tests)
- monitor.rs: ~40% (3 tests)
- shutdown.rs: ~30% (5 tests)
- resources.rs: ~50% (10 tests)

**Mitigation:**
1. **Run tests after EVERY step:**
   ```bash
   cargo test --all
   ```

2. **Fix imports incrementally:**
   - Update one module at a time
   - Verify tests pass before moving on

3. **Add integration tests:**
   - Test cross-crate boundaries
   - Verify no regressions

**Residual Risk:** Very Low

---

### 4. Timeline Overrun

**Risk Level:** 🟢 Low  
**Likelihood:** Medium  
**Impact:** Low

**Description:**
Migration may take longer than estimated 3 weeks.

**Estimates:**
- Week 1 (Prep): 40 hours
- Week 2 (Implementation): 40 hours
- Week 3 (Verification): 40 hours
- **Total:** 120 hours

**Risk Factors:**
- HTTP server migration more complex than expected (Day 2.5)
- Integration issues discovered late (Week 3)
- Unexpected circular dependencies

**Mitigation:**
1. **Start with low-risk crates:**
   - Build confidence
   - Learn patterns
   
2. **Buffer time:**
   - Add 1 extra day per week
   - Estimated 4 weeks (safe)

3. **Parallel work possible:**
   - Multiple crates can be migrated simultaneously
   - Registry → provisioner → monitor (independent)

**Residual Risk:** Very Low

---

### 5. Integration Issues

**Risk Level:** 🟡 Medium  
**Likelihood:** Low  
**Impact:** High

**Description:**
Crates may fail to integrate properly at runtime.

**Potential Issues:**
- Type mismatches across crate boundaries
- Missing feature flags
- Version conflicts

**Mitigation:**
1. **Continuous integration:**
   - Build full binary after each crate
   - Run BDD tests daily
   
2. **Workspace dependencies:**
   - Use workspace-level versions
   - Prevent version conflicts

3. **Integration test harness:**
   ```bash
   # test-harness/integration/test_rbee_hive_crates.sh
   cargo build --workspace
   cargo test --workspace
   cargo run -- status
   cargo run -- detect
   ```

**Residual Risk:** Low

---

### 6. Performance Regression

**Risk Level:** 🟢 Low  
**Likelihood:** Low  
**Impact:** Medium

**Description:**
Decomposition could introduce performance overhead (unlikely but possible).

**Theoretical Concerns:**
- More dynamic dispatch
- Larger binary size
- Slower compilation (opposite of goal!)

**Reality Check:**
- Static linking eliminates runtime overhead
- Binary size increase minimal (~5-10%)
- Compilation time DECREASES (goal is 93% faster)

**Mitigation:**
1. **Benchmark before/after:**
   ```bash
   # Before
   hyperfine 'cargo build --release'
   # After
   hyperfine 'cargo build --release -p rbee-hive-registry'
   ```

2. **Profile binary size:**
   ```bash
   cargo bloat --release
   ```

3. **Runtime profiling:**
   - Worker spawn time
   - Health check latency
   - HTTP response time

**Residual Risk:** Very Low

---

## DEPENDENCIES ON OTHER TEAMS

### No Blockers Identified ✅

**rbee-hive is standalone:**
- Does not depend on other binaries
- Shares crates are already mature
- No coordination needed

**Peer Dependencies (Informational Only):**
- TEAM-132 (queen-rbee): No shared code
- TEAM-133 (llm-worker-rbee): No shared code  
- TEAM-134 (rbee-keeper): No shared code

**Future Opportunity:**
- After decomposition, `rbee-hive-http-middleware` could be reused by queen-rbee
- This is optional, not required

---

## ROLLBACK PLAN

### If Migration Fails

**Scenario 1: Single crate fails**
- Keep old code in binary
- Delete failed crate
- Revert imports
- Time: 1-2 hours

**Scenario 2: Multiple crates fail**
- Revert to git checkpoint
- Analyze root cause
- Restart with smaller scope
- Time: 1 day

**Scenario 3: Fundamental design flaw**
- Abort decomposition
- Document findings
- Propose alternative approach
- Time: 1 week

**Git Strategy:**
```bash
# Create checkpoint before each phase
git checkout -b feat/rbee-hive-decomposition
git commit -m "Phase 1: Preparation complete"

# Tag each milestone
git tag decomp-phase1-complete
git tag decomp-phase2-complete
git tag decomp-phase3-complete
```

---

## GO/NO-GO CRITERIA

### GO Decision ✅

**Conditions Met:**
- ✅ All code analyzed (6,021 LOC)
- ✅ 10 crates proposed with justification
- ✅ Shared crate audit complete
- ✅ Migration plan documented
- ✅ Risks assessed and mitigated
- ✅ No blocking dependencies
- ✅ Team has capacity (3 weeks available)

### Would Be NO-GO If:
- ❌ Circular dependencies found (not found)
- ❌ Shared crates missing features (all adequate)
- ❌ Timeline insufficient (3 weeks is adequate)
- ❌ High coupling unsolvable (coupling is manageable)
- ❌ Other teams blocked (no blockers)

---

## ACCEPTANCE CRITERIA

### Phase 1 Complete When:
- [ ] All 10 crate directories created
- [ ] All Cargo.toml files written
- [ ] Workspace builds successfully
- [ ] Migration scripts tested

### Phase 2 Complete When:
- [ ] All code moved to crates
- [ ] All imports updated
- [ ] Binary builds successfully
- [ ] All tests passing

### Phase 3 Complete When:
- [ ] BDD test suite passing
- [ ] Integration tests passing
- [ ] Performance verified (93% faster)
- [ ] Documentation updated
- [ ] Cleanup complete

---

## RISK SUMMARY

**Overall Risk:** 🟢 **LOW**

**Key Insights:**
1. Code is already well-structured
2. Modules are mostly independent
3. Good test coverage exists
4. No external dependencies
5. Phased approach minimizes risk

**Recommendation:** ✅ **PROCEED WITH DECOMPOSITION**

**Estimated Success Probability:** 95%

---

**Risk Assessment Complete**  
**Analyst:** TEAM-131  
**Date:** 2025-10-19
