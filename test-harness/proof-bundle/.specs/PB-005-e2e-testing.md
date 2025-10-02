# PB-005: End-to-End Testing & Validation

**Status**: Active  
**Created**: 2025-10-02  
**Owner**: proof-bundle team  
**Priority**: CRITICAL

## Overview

V2 proof-bundle MUST be validated with real end-to-end tests before it can be considered production-ready. This spec defines the testing requirements to prove the system works.

## Problem Statement

Current state:
- ❌ V2 code exists but hasn't been proven to work end-to-end
- ❌ Parsers haven't been validated against real cargo test output
- ❌ No golden test files to verify report quality
- ❌ Can't prove the one-liner API actually works

This is **NOT acceptable** for production.

## Requirements

### R1: Real Test Execution
- MUST run `cargo test` and capture actual output
- MUST parse real test results (not 0 tests)
- MUST handle both passing and failing tests
- MUST extract test names, durations, and status

### R2: Golden Test Files
- MUST have golden files for expected outputs
- MUST validate parser output against golden files
- MUST validate formatter output against golden files
- MUST detect regressions in report quality

### R3: End-to-End Validation
- MUST generate real proof bundle from proof-bundle's own tests
- MUST verify all 7 files are created
- MUST verify reports contain actual test data
- MUST verify reports are human-readable and useful

### R4: Parser Correctness
- MUST correctly parse stable `cargo test` output
- MUST extract test count from summary line
- MUST extract individual test results
- MUST handle edge cases (0 tests, all pass, all fail, mixed)

## Test Cases

### TC1: Parse Real Cargo Output
```rust
#[test]
fn test_parse_real_cargo_output() {
    let output = include_str!("../golden/cargo_test_output.txt");
    let summary = parsers::parse_stable_output(output).unwrap();
    
    assert_eq!(summary.total, 38); // proof-bundle has 38 tests
    assert_eq!(summary.passed, 38);
    assert_eq!(summary.failed, 0);
}
```

### TC2: Generate Real Proof Bundle
```rust
#[test]
fn test_generate_real_proof_bundle() {
    let summary = api::generate_for_crate("proof-bundle", api::Mode::UnitFast).unwrap();
    
    assert!(summary.total > 0, "Must capture real tests");
    assert!(summary.pass_rate >= 90.0, "Must have high pass rate");
    
    // Verify files exist
    let pb = ProofBundle::for_type(LegacyTestType::Unit).unwrap();
    assert!(pb.root().join("summary.json").exists());
    assert!(pb.root().join("executive_summary.md").exists());
    assert!(pb.root().join("test_report.md").exists());
    assert!(pb.root().join("failure_report.md").exists());
    assert!(pb.root().join("metadata_report.md").exists());
}
```

### TC3: Validate Report Quality
```rust
#[test]
fn test_report_quality() {
    let summary = create_realistic_summary();
    let executive = formatters::generate_executive_summary(&summary);
    
    // Must contain key sections
    assert!(executive.contains("Test Results Summary"));
    assert!(executive.contains("Risk Assessment"));
    assert!(executive.contains("Recommendation"));
    
    // Must detect critical failures
    if has_critical_failures(&summary) {
        assert!(executive.contains("CRITICAL ALERT"));
        assert!(executive.contains("NOT APPROVED"));
    }
}
```

### TC4: Golden File Validation
```rust
#[test]
fn test_golden_files() {
    let output = include_str!("../golden/cargo_test_output.txt");
    let summary = parsers::parse_stable_output(output).unwrap();
    
    let expected = include_str!("../golden/expected_summary.json");
    let actual = serde_json::to_string_pretty(&summary).unwrap();
    
    assert_eq!(actual, expected, "Parser output must match golden file");
}
```

## Implementation Plan

### Phase 1: Fix Parser (CRITICAL)
1. Capture real `cargo test --package proof-bundle` output
2. Save as golden file: `tests/golden/cargo_test_output.txt`
3. Fix `parse_stable_output()` to correctly parse it
4. Validate against golden file

### Phase 2: Create Golden Files
1. `tests/golden/cargo_test_output.txt` - Real cargo output
2. `tests/golden/expected_summary.json` - Expected parsed summary
3. `tests/golden/expected_executive.md` - Expected executive summary
4. `tests/golden/expected_test_report.md` - Expected test report

### Phase 3: E2E Tests
1. Create `tests/e2e_tests.rs` with all test cases
2. Run against real proof-bundle tests
3. Validate all outputs
4. Ensure 100% pass rate

### Phase 4: Validation
1. Run `cargo test -p proof-bundle`
2. Generate proof bundle
3. Manually review all 7 generated files
4. Confirm reports are useful and accurate

## Success Criteria

✅ All E2E tests pass  
✅ Parser correctly extracts 38 tests from proof-bundle  
✅ All 7 report files generated  
✅ Reports contain real, useful data  
✅ Golden files validate correctness  
✅ Manual review confirms quality  

## Failure Modes

### FM1: Parser Returns 0 Tests
**Cause**: Regex/parsing logic doesn't match cargo output  
**Fix**: Debug with real output, fix parser logic  
**Test**: Golden file validation

### FM2: Reports Are Empty/Useless
**Cause**: No test data captured  
**Fix**: Fix parser first, then regenerate  
**Test**: Manual review + assertions

### FM3: API Fails to Run
**Cause**: Command construction wrong  
**Fix**: Debug command, check stderr  
**Test**: E2E test with real execution

## Refinement Opportunities

1. **JSON Parser**: Add JSON format support for nightly Rust
2. **More Golden Files**: Add edge cases (all fail, timeouts, etc.)
3. **Performance**: Benchmark parser on large test suites
4. **Error Messages**: Better errors when parsing fails
5. **Diff Tool**: Tool to compare against golden files

## Dependencies

- Cargo test output format (stable)
- File system access for golden files
- Real proof-bundle tests (38 tests)

## Notes

This spec is CRITICAL. V2 cannot be considered done until all tests pass and golden files validate.

**No shortcuts. No mock data. Real tests only.**
