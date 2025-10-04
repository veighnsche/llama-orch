# Testing Remediation Plan — Narration Core

**Issued by**: Testing Team 🔍  
**Date**: 2025-10-04  
**Crate**: `bin/shared-crates/narration-core`  
**Status**: 🚨 **MANDATORY REMEDIATION REQUIRED**

---

## Executive Summary

This document provides a **step-by-step remediation plan** for the 4 violations identified in `TESTING_AUDIT.md`.

**Violations**:
1. ⚠️ **MEDIUM**: Flaky tests (2 tests) — 48h deadline
2. 🚨 **CRITICAL**: Insufficient test coverage (15 failing integration tests) — 72h deadline
3. 🔴 **HIGH**: Missing proof bundle integration — 1 week deadline
4. 🔴 **HIGH**: Missing specification — 1 week deadline

**Total Estimated Effort**: 24-32 hours across 2 weeks

---

## Phase 1: CRITICAL — Fix Test Failures (72 hours)

### Task 1.1: Fix Flaky Unit Tests (4 hours)

**Violation**: VIOLATION #1 (MEDIUM)

**Problem**: `CaptureAdapter` uses global `OnceLock` causing race conditions

**Solution**: Use `serial_test` crate to force serialization

**Steps**:

1. Add dependency to `Cargo.toml`:
```toml
[dev-dependencies]
serial_test = "3.0"
```

2. Update `src/auto.rs` tests:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial] // Force serial execution
    fn test_narrate_auto_injects_fields() {
        let adapter = CaptureAdapter::install();
        adapter.clear();
        // ... rest of test
    }

    #[test]
    #[serial] // Force serial execution
    fn test_narrate_auto_respects_existing_fields() {
        let adapter = CaptureAdapter::install();
        adapter.clear();
        // ... rest of test
    }
}
```

3. Verify fix:
```bash
cargo test -p observability-narration-core --lib -- --test-threads=8
```

**Expected Result**: All 41 unit tests pass

**Acceptance Criteria**:
- [ ] `serial_test` dependency added
- [ ] Both flaky tests annotated with `#[serial]`
- [ ] All 41 unit tests pass with parallel execution
- [ ] Tests pass 10 times in a row (no flakiness)

---

### Task 1.2: Fix Integration Tests (8 hours)

**Violation**: VIOLATION #2 (CRITICAL)

**Problem**: Same root cause — global `CaptureAdapter` state

**Solution**: Apply same fix to all integration tests

**Steps**:

1. Update `tests/integration.rs`:
```rust
use serial_test::serial;

#[test]
#[serial]
fn test_narration_basic() {
    let adapter = CaptureAdapter::install();
    adapter.clear();
    // ... rest of test
}

// Apply #[serial] to all 16 tests
```

2. Verify each test individually:
```bash
cargo test -p observability-narration-core --test integration test_narration_basic
cargo test -p observability-narration-core --test integration test_correlation_id_propagation
# ... etc for all 16 tests
```

3. Verify all tests together:
```bash
cargo test -p observability-narration-core --test integration
```

**Expected Result**: All 16 integration tests pass

**Acceptance Criteria**:
- [ ] All 16 integration tests annotated with `#[serial]`
- [ ] All 16 integration tests pass individually
- [ ] All 16 integration tests pass together
- [ ] Tests pass 10 times in a row (no flakiness)

---

### Task 1.3: Verify BDD Tests Execute (4 hours)

**Problem**: BDD tests compile but execution status unknown

**Solution**: Run BDD suite and verify scenarios pass

**Steps**:

1. Run BDD suite:
```bash
cargo run -p observability-narration-core-bdd --bin bdd-runner
```

2. If failures occur, fix step definitions:
   - Remove deprecated `human()` function usage
   - Fix unused `world` parameters
   - Ensure all steps are implemented

3. Run individual features:
```bash
LLORCH_BDD_FEATURE_PATH=bdd/features/cute_mode.feature \
  cargo run -p observability-narration-core-bdd --bin bdd-runner

LLORCH_BDD_FEATURE_PATH=bdd/features/story_mode.feature \
  cargo run -p observability-narration-core-bdd --bin bdd-runner

LLORCH_BDD_FEATURE_PATH=bdd/features/levels.feature \
  cargo run -p observability-narration-core-bdd --bin bdd-runner
```

4. Document results in `BDD_EXECUTION_RESULTS.md`

**Expected Result**: All BDD scenarios pass or have documented failures

**Acceptance Criteria**:
- [ ] BDD runner executes without panics
- [ ] All scenarios either pass or have documented failures
- [ ] Deprecated function warnings resolved
- [ ] Unused variable warnings resolved
- [ ] Execution results documented

---

### Task 1.4: Achieve 100% Test Pass Rate (2 hours)

**Goal**: All tests passing before moving to next phase

**Steps**:

1. Run full test suite:
```bash
cargo test -p observability-narration-core
cargo test -p observability-narration-core --test integration
cargo run -p observability-narration-core-bdd --bin bdd-runner
```

2. Document results:
```
Unit Tests: 41/41 passing (100%)
Integration Tests: 16/16 passing (100%)
BDD Tests: X/Y passing (Z%)
Total: X/Y passing (Z%)
```

3. Create proof of remediation document

**Acceptance Criteria**:
- [ ] 100% unit test pass rate
- [ ] 100% integration test pass rate
- [ ] BDD test pass rate documented
- [ ] No flaky tests (10 consecutive runs pass)

---

## Phase 2: HIGH — Proof Bundle Integration (1 week)

### Task 2.1: Add Proof Bundle Dependency (1 hour)

**Violation**: VIOLATION #3 (HIGH)

**Steps**:

1. Add to `Cargo.toml`:
```toml
[dev-dependencies]
proof-bundle = { path = "../../../libs/proof-bundle" }
```

2. Verify compilation:
```bash
cargo build -p observability-narration-core --tests
```

**Acceptance Criteria**:
- [ ] Dependency added
- [ ] Tests compile successfully

---

### Task 2.2: Integrate Proof Bundle in Unit Tests (4 hours)

**Steps**:

1. Create test helper in `src/lib.rs`:
```rust
#[cfg(test)]
pub(crate) mod test_helpers {
    use proof_bundle::{ProofBundle, TestType};
    
    pub fn unit_test_bundle(test_name: &str) -> ProofBundle {
        ProofBundle::for_type(TestType::Unit)
            .with_test_name(test_name)
    }
}
```

2. Update unit tests to emit proof bundles:
```rust
#[test]
#[serial]
fn test_narrate_auto_injects_fields() {
    let bundle = test_helpers::unit_test_bundle("test_narrate_auto_injects_fields");
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    // ... test logic ...
    
    // Emit proof bundle
    bundle.write_json("captured_events.json", &adapter.captured())?;
    bundle.write_markdown("test_summary.md", &format!(
        "# Test: test_narrate_auto_injects_fields\n\n\
         Captured {} events\n", 
        adapter.captured().len()
    ))?;
}
```

3. Verify proof bundles are created:
```bash
cargo test -p observability-narration-core --lib
ls -la .proof_bundle/unit/*/
```

**Acceptance Criteria**:
- [ ] All unit tests emit proof bundles
- [ ] Proof bundles created in `.proof_bundle/unit/<run_id>/`
- [ ] Bundles include JSON and Markdown outputs
- [ ] Bundles respect `LLORCH_RUN_ID` and `LLORCH_PROOF_DIR`

---

### Task 2.3: Integrate Proof Bundle in Integration Tests (4 hours)

**Steps**:

1. Update `tests/integration.rs`:
```rust
use proof_bundle::{ProofBundle, TestType};

fn integration_test_bundle(test_name: &str) -> ProofBundle {
    ProofBundle::for_type(TestType::Integration)
        .with_test_name(test_name)
}

#[test]
#[serial]
fn test_narration_basic() {
    let bundle = integration_test_bundle("test_narration_basic");
    
    let adapter = CaptureAdapter::install();
    adapter.clear();
    
    // ... test logic ...
    
    // Emit proof bundle
    bundle.write_json("captured_events.json", &adapter.captured())?;
    bundle.write_json("test_metadata.json", &json!({
        "test_name": "test_narration_basic",
        "events_captured": adapter.captured().len(),
        "assertions_passed": true,
    }))?;
}
```

2. Apply to all 16 integration tests

3. Verify proof bundles:
```bash
cargo test -p observability-narration-core --test integration
ls -la .proof_bundle/integration/*/
```

**Acceptance Criteria**:
- [ ] All 16 integration tests emit proof bundles
- [ ] Proof bundles created in `.proof_bundle/integration/<run_id>/`
- [ ] Bundles include captured events and metadata
- [ ] Bundles have autogenerated headers per PB-1012

---

### Task 2.4: Integrate Proof Bundle in BDD Tests (6 hours)

**Steps**:

1. Update `bdd/src/steps/world.rs`:
```rust
use proof_bundle::{ProofBundle, TestType};

pub struct World {
    pub adapter: CaptureAdapter,
    pub bundle: ProofBundle,
}

impl World {
    pub fn new() -> Self {
        Self {
            adapter: CaptureAdapter::install(),
            bundle: ProofBundle::for_type(TestType::Bdd),
        }
    }
}
```

2. Update step definitions to use `world.bundle`:
```rust
#[then("the captured event should include the text {string}")]
pub async fn then_captured_includes(world: &mut World, text: String) {
    world.adapter.assert_includes(&text);
    
    // Emit proof bundle
    world.bundle.write_json("captured_events.json", &world.adapter.captured())?;
}
```

3. Add after-scenario hook to emit bundles:
```rust
impl World {
    pub async fn after_scenario(&mut self, scenario: &str) {
        self.bundle.write_markdown(
            &format!("{}.md", scenario),
            &format!("# Scenario: {}\n\nCaptured {} events", 
                scenario, 
                self.adapter.captured().len()
            )
        ).ok();
    }
}
```

4. Verify proof bundles:
```bash
cargo run -p observability-narration-core-bdd --bin bdd-runner
ls -la .proof_bundle/bdd/*/
```

**Acceptance Criteria**:
- [ ] BDD World struct includes ProofBundle
- [ ] All scenarios emit proof bundles
- [ ] Proof bundles created in `.proof_bundle/bdd/<run_id>/`
- [ ] Bundles include scenario results and captured events

---

## Phase 3: HIGH — Create Specification (1 week)

### Task 3.1: Create Specification Document (8 hours)

**Violation**: VIOLATION #4 (HIGH)

**Steps**:

1. Create `.specs/00_narration-core.md`:
```markdown
# Narration Core Specification

**Version**: 0.1.0  
**Status**: Draft  
**License**: GPL-3.0-or-later

---

## Normative Requirements

### Core Narration (NARR-1000 series)

**NARR-1001**: The system MUST emit structured narration events with actor, action, target, and human fields.

**NARR-1002**: The system MUST support 7 logging levels: MUTE, TRACE, DEBUG, INFO, WARN, ERROR, FATAL.

**NARR-1003**: The system MUST automatically redact secrets in narration events.

### Correlation IDs (NARR-2000 series)

**NARR-2001**: The system MUST generate UUID v4 correlation IDs.

**NARR-2002**: The system MUST validate correlation IDs in <100ns.

**NARR-2003**: The system MUST propagate correlation IDs across service boundaries.

### Redaction (NARR-3000 series)

**NARR-3001**: The system MUST redact bearer tokens.

**NARR-3002**: The system MUST redact API keys.

**NARR-3003**: The system MUST redact JWT tokens.

**NARR-3004**: The system MUST redact private keys.

**NARR-3005**: The system MUST redact URL passwords.

**NARR-3006**: The system MAY redact UUIDs (configurable).

### Performance (NARR-4000 series)

**NARR-4001**: Correlation ID validation MUST complete in <100ns.

**NARR-4002**: Production builds MUST have zero overhead (conditional compilation).

**NARR-4003**: ASCII fast path MUST complete in <1μs.

**NARR-4004**: CRLF sanitization MUST complete in <50ns for clean strings.

### Testing (NARR-5000 series)

**NARR-5001**: The system MUST provide a test capture adapter.

**NARR-5002**: The capture adapter MUST support assertions on captured events.

**NARR-5003**: Tests MUST emit proof bundles per monorepo standard.

---

## Verification Plan

| Requirement | Test Type | Test ID | Location |
|-------------|-----------|---------|----------|
| NARR-1001 | Unit | test_narration_basic | tests/integration.rs:10 |
| NARR-1002 | BDD | levels.feature | bdd/features/levels.feature |
| NARR-1003 | Unit | test_redact_bearer_token | src/redaction.rs:150 |
| NARR-2001 | Unit | test_generate_correlation_id | src/correlation.rs:80 |
| NARR-2002 | Unit | test_validate_correlation_id | src/correlation.rs:90 |
| NARR-2003 | Integration | test_correlation_id_propagation | tests/integration.rs:34 |
| NARR-3001 | Unit | test_redact_bearer_token | src/redaction.rs:150 |
| NARR-3002 | Unit | test_redact_api_key | src/redaction.rs:160 |
| NARR-3003 | Unit | test_redact_jwt_token | src/redaction.rs:170 |
| NARR-3004 | Unit | test_redact_private_key | src/redaction.rs:180 |
| NARR-3005 | Unit | test_redact_url_password | src/redaction.rs:190 |
| NARR-3006 | Unit | test_uuid_redaction_when_enabled | src/redaction.rs:200 |
| NARR-4001 | Benchmark | correlation_validation | benches/narration_benchmarks.rs |
| NARR-4002 | Compile | production_build | CI verification |
| NARR-4003 | Benchmark | ascii_fast_path | benches/narration_benchmarks.rs |
| NARR-4004 | Benchmark | crlf_sanitization | benches/narration_benchmarks.rs |
| NARR-5001 | Unit | test_capture_adapter_basic | src/capture.rs:250 |
| NARR-5002 | Unit | test_assert_includes | src/capture.rs:260 |
| NARR-5003 | Integration | All tests | tests/integration.rs |

---

## Acceptance Criteria

- [ ] All MUST requirements have passing tests
- [ ] All SHOULD requirements have tests or documented exceptions
- [ ] All MAY requirements are documented
- [ ] Verification plan maps all requirements to tests
- [ ] All tests reference requirement IDs in comments
```

2. Update test files to reference requirement IDs:
```rust
/// Tests NARR-1001: System MUST emit structured narration events
#[test]
#[serial]
fn test_narration_basic() {
    // ...
}

/// Tests NARR-2002: System MUST validate correlation IDs in <100ns
#[test]
fn test_validate_correlation_id() {
    // ...
}
```

**Acceptance Criteria**:
- [ ] `.specs/00_narration-core.md` created
- [ ] All normative requirements documented with RFC-2119 language
- [ ] All requirements have stable IDs (NARR-XXXX)
- [ ] Verification plan maps requirements to tests
- [ ] Test files reference requirement IDs

---

### Task 3.2: Update Documentation (2 hours)

**Steps**:

1. Update `README.md` to reference specification:
```markdown
## Specifications

Implements requirements from `.specs/00_narration-core.md`:
- NARR-1001..NARR-1003: Core narration
- NARR-2001..NARR-2003: Correlation IDs
- NARR-3001..NARR-3006: Secret redaction
- NARR-4001..NARR-4004: Performance
- NARR-5001..NARR-5003: Testing
```

2. Update `TESTING_NOTES.md` to reference specification

3. Update `IMPLEMENTATION_STATUS.md` to reference specification

**Acceptance Criteria**:
- [ ] README references specification
- [ ] All documentation updated
- [ ] Links to specification are correct

---

## Phase 4: Verification & Sign-Off (1 week)

### Task 4.1: Run Full Test Suite (2 hours)

**Steps**:

1. Run all tests:
```bash
cargo test -p observability-narration-core
cargo test -p observability-narration-core --test integration
cargo run -p observability-narration-core-bdd --bin bdd-runner
```

2. Verify proof bundles:
```bash
ls -la .proof_bundle/unit/*/
ls -la .proof_bundle/integration/*/
ls -la .proof_bundle/bdd/*/
```

3. Document results in `TESTING_VERIFICATION.md`

**Acceptance Criteria**:
- [ ] 100% test pass rate
- [ ] All proof bundles created
- [ ] No flaky tests
- [ ] Results documented

---

### Task 4.2: Run Benchmarks (2 hours)

**Steps**:

1. Run benchmarks:
```bash
cargo bench -p observability-narration-core
```

2. Verify performance targets:
   - Correlation ID validation: <100ns
   - ASCII fast path: <1μs
   - CRLF sanitization: <50ns (clean)

3. Document results in `PERFORMANCE_VERIFICATION.md`

**Acceptance Criteria**:
- [ ] All benchmarks run successfully
- [ ] All performance targets met
- [ ] Results documented

---

### Task 4.3: Submit Remediation Proof (1 hour)

**Steps**:

1. Create `REMEDIATION_COMPLETE.md`:
```markdown
# Remediation Complete — Narration Core

**Date**: YYYY-MM-DD  
**Violations Addressed**: 4 (2 CRITICAL, 2 HIGH)

## Phase 1: Test Failures (COMPLETE)
- [x] Fixed flaky unit tests (VIOLATION #1)
- [x] Fixed integration tests (VIOLATION #2)
- [x] Verified BDD tests execute
- [x] Achieved 100% test pass rate

## Phase 2: Proof Bundle Integration (COMPLETE)
- [x] Added proof bundle dependency (VIOLATION #3)
- [x] Integrated in unit tests
- [x] Integrated in integration tests
- [x] Integrated in BDD tests
- [x] Verified proof bundles created

## Phase 3: Specification (COMPLETE)
- [x] Created `.specs/00_narration-core.md` (VIOLATION #4)
- [x] Documented normative requirements
- [x] Created verification plan
- [x] Updated test files with requirement IDs
- [x] Updated documentation

## Phase 4: Verification (COMPLETE)
- [x] Ran full test suite (100% pass rate)
- [x] Ran benchmarks (all targets met)
- [x] Verified proof bundles
- [x] Documented results

## Test Results
- Unit Tests: 41/41 passing (100%)
- Integration Tests: 16/16 passing (100%)
- BDD Tests: X/Y passing (Z%)
- Total: X/Y passing (Z%)

## Proof Bundle Outputs
- Unit: .proof_bundle/unit/<run_id>/
- Integration: .proof_bundle/integration/<run_id>/
- BDD: .proof_bundle/bdd/<run_id>/

## Performance Verification
- Correlation ID validation: <100ns ✅
- ASCII fast path: <1μs ✅
- CRLF sanitization: <50ns ✅

## Request Re-Audit
Ready for Testing Team re-audit.
```

2. Submit to Testing Team for re-audit

**Acceptance Criteria**:
- [ ] Remediation proof document created
- [ ] All violations addressed
- [ ] All acceptance criteria met
- [ ] Ready for re-audit

---

## Timeline

### Week 1 (CRITICAL)
- **Day 1-3**: Phase 1 (Fix test failures) — 18 hours
  - Day 1: Fix flaky tests (4h) + Fix integration tests (4h)
  - Day 2: Fix integration tests cont. (4h) + Verify BDD tests (4h)
  - Day 3: Achieve 100% pass rate (2h)

### Week 2 (HIGH)
- **Day 4-7**: Phase 2 (Proof bundle integration) — 15 hours
  - Day 4: Add dependency (1h) + Unit tests (4h)
  - Day 5: Integration tests (4h)
  - Day 6-7: BDD tests (6h)

- **Day 8-10**: Phase 3 (Specification) — 10 hours
  - Day 8-9: Create specification (8h)
  - Day 10: Update documentation (2h)

- **Day 11-12**: Phase 4 (Verification) — 5 hours
  - Day 11: Run tests + benchmarks (4h)
  - Day 12: Submit proof (1h)

**Total**: 48 hours across 12 days (2 weeks)

---

## Success Criteria

### Phase 1 (CRITICAL)
- [x] All 41 unit tests pass
- [x] All 16 integration tests pass
- [x] BDD tests execute successfully
- [x] 100% test pass rate
- [x] No flaky tests

### Phase 2 (HIGH)
- [x] Proof bundle dependency added
- [x] All unit tests emit proof bundles
- [x] All integration tests emit proof bundles
- [x] All BDD tests emit proof bundles
- [x] Proof bundles respect monorepo standard

### Phase 3 (HIGH)
- [x] Specification document created
- [x] All requirements documented with RFC-2119
- [x] Verification plan created
- [x] Tests reference requirement IDs
- [x] Documentation updated

### Phase 4 (Verification)
- [x] Full test suite passes
- [x] Benchmarks meet targets
- [x] Proof bundles verified
- [x] Remediation proof submitted

---

## Contact

**Questions?** Contact Testing Team 🔍

**Blocked?** Escalate to Testing Team lead

**Need Help?** Review `TESTING_AUDIT.md` for detailed violation descriptions

---

Issued by Testing Team — mandatory remediation required 🔍
