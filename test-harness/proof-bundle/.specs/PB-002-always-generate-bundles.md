# PB-002: Always Generate Proof Bundles (Pass AND Fail)

**Status**: Normative  
**Created**: 2025-10-02  
**Updated**: 2025-10-02

---

## Requirement

**CRITICAL**: Proof bundles MUST be generated for ALL test runs, regardless of pass/fail status.

---

## Policy

### ✅ REQUIRED: Generate on ALL Outcomes

Proof bundles MUST be generated when:
- ✅ All tests pass
- ✅ Some tests fail
- ✅ All tests fail
- ✅ Tests panic or crash
- ✅ Tests timeout
- ✅ Build fails before tests run

### ❌ FORBIDDEN: Skip on Failure

**NEVER** skip proof bundle generation due to test failures.

---

## Rationale

### 1. Failure Evidence is CRITICAL

**Failed tests are MORE important to document than passing tests:**
- ✅ Shows what broke
- ✅ Provides debugging information
- ✅ Enables root cause analysis
- ✅ Tracks regression history
- ✅ Proves tests actually ran (not skipped)

### 2. Human Auditor Needs

Human auditors need to see:
- ✅ **What was tested** (even if it failed)
- ✅ **How it failed** (error messages, stack traces)
- ✅ **When it failed** (timestamp, environment)
- ✅ **Why it failed** (root cause if known)

### 3. Honesty and Transparency

Proof bundles must be **honest and objective**:
- ✅ Show failures openly
- ✅ Don't hide problems
- ✅ Provide complete picture
- ✅ Enable accountability

---

## Implementation

### Pattern: Always Generate

```rust
use proof_bundle::{ProofBundle, TestType};

#[test]
fn my_test() -> anyhow::Result<()> {
    // ALWAYS create proof bundle FIRST
    let pb = ProofBundle::for_type(TestType::Unit)?;
    
    // Write metadata
    pb.write_json("metadata", &serde_json::json!({
        "test": "my_test",
        "started_at": Utc::now().to_rfc3339(),
    }))?;
    
    // Run test (may fail)
    let result = run_actual_test();
    
    // ALWAYS capture result (pass or fail)
    pb.append_ndjson("test_results", &serde_json::json!({
        "test": "my_test",
        "status": if result.is_ok() { "pass" } else { "fail" },
        "error": result.as_ref().err().map(|e| format!("{:?}", e)),
        "completed_at": Utc::now().to_rfc3339(),
    }))?;
    
    // ALWAYS generate report
    pb.write_markdown("test_report.md", &generate_report(&result))?;
    
    // Return actual result (may fail test)
    result
}
```

### Pattern: Catch Panics

```rust
use proof_bundle::{ProofBundle, TestType};
use std::panic;

#[test]
fn test_with_panic_capture() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Catch panics to ensure proof bundle is generated
    let result = panic::catch_unwind(|| {
        // Test code that might panic
        run_risky_test()
    });
    
    // ALWAYS capture result
    pb.append_ndjson("test_results", &serde_json::json!({
        "test": "test_with_panic_capture",
        "status": if result.is_ok() { "pass" } else { "panic" },
        "panic_info": if result.is_err() { "Test panicked" } else { "" },
    })).unwrap();
    
    // Re-panic if needed (to fail test)
    if let Err(e) = result {
        panic::resume_unwind(e);
    }
}
```

### Pattern: BDD Tests (Always Generate)

```rust
// bdd/src/main.rs

use proof_bundle::{ProofBundle, TestType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ALWAYS create proof bundle FIRST
    let pb = ProofBundle::for_type(TestType::Bdd)?;
    
    pb.write_json("metadata", &serde_json::json!({
        "started_at": Utc::now().to_rfc3339(),
    }))?;
    
    // Run BDD tests (may fail)
    let result = BddWorld::cucumber()
        .run("tests/features")
        .await;
    
    // ALWAYS capture results (even if scenarios failed)
    pb.append_ndjson("scenarios", &serde_json::json!({
        "total": result.total_scenarios,
        "passed": result.passed_scenarios,
        "failed": result.failed_scenarios,
        "status": if result.failed_scenarios == 0 { "pass" } else { "fail" },
    }))?;
    
    // ALWAYS generate report
    pb.write_markdown("test_report.md", &format!(r#"
# BDD Test Report

**Status**: {}

## Results
- Total: {}
- Passed: {}
- Failed: {}

## Failed Scenarios
{}
"#,
        if result.failed_scenarios == 0 { "✅ PASS" } else { "❌ FAIL" },
        result.total_scenarios,
        result.passed_scenarios,
        result.failed_scenarios,
        result.failed_scenarios_details.join("\n"),
    ))?;
    
    // Return actual result (may fail)
    Ok(())
}
```

---

## What to Capture on Failure

### Minimum Required

1. **Test identification**
   - Test name
   - Test type (unit, BDD, integration, etc.)
   - Crate name

2. **Failure details**
   - Error message
   - Error type
   - Stack trace (if available)

3. **Context**
   - Timestamp
   - Environment (OS, Rust version, etc.)
   - Test mode (mock vs real GPU)

4. **Status**
   - Clear pass/fail indicator
   - Failure count
   - Success count

### Recommended

5. **Debugging information**
   - Input data that caused failure
   - Expected vs actual output
   - Intermediate state
   - Logs leading up to failure

6. **Root cause analysis**
   - Known issues
   - Related failures
   - Possible causes

---

## Examples

### Example 1: Unit Test Failure

```rust
#[test]
fn test_seal_model_with_invalid_size() -> anyhow::Result<()> {
    let pb = ProofBundle::for_type(TestType::Unit)?;
    
    let mut manager = VramManager::new();
    let result = manager.seal_model(&[], 0); // Zero size (should fail)
    
    // Capture failure details
    pb.append_ndjson("test_results", &serde_json::json!({
        "test": "test_seal_model_with_invalid_size",
        "status": if result.is_err() { "pass" } else { "fail" },
        "expected": "InvalidInput error",
        "actual": format!("{:?}", result),
        "input_size": 0,
    }))?;
    
    // Test expects error
    assert!(result.is_err());
    Ok(())
}
```

**Proof Bundle Output**:
```json
{
  "test": "test_seal_model_with_invalid_size",
  "status": "pass",
  "expected": "InvalidInput error",
  "actual": "Err(InvalidInput(\"Model size must be > 0\"))",
  "input_size": 0
}
```

### Example 2: BDD Scenario Failure

```gherkin
Scenario: Reject invalid shard ID
  Given a model with 1MB of data
  When I seal the model with shard_id "../etc/passwd" on GPU 0
  Then the seal should fail with "InvalidInput"
```

**Proof Bundle Output** (if scenario fails):
```json
{
  "feature": "seal_model",
  "scenario": "Reject invalid shard ID",
  "status": "fail",
  "expected": "InvalidInput error",
  "actual": "Seal succeeded (should have failed)",
  "failure_reason": "Path traversal attack not detected",
  "timestamp": "2025-10-02T13:42:30Z"
}
```

### Example 3: Panic Capture

```rust
#[test]
fn test_with_panic() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    let result = panic::catch_unwind(|| {
        panic!("Intentional panic for testing");
    });
    
    pb.append_ndjson("test_results", &serde_json::json!({
        "test": "test_with_panic",
        "status": "panic",
        "panic_message": "Intentional panic for testing",
    })).unwrap();
    
    // Re-panic to fail test
    if let Err(e) = result {
        panic::resume_unwind(e);
    }
}
```

**Proof Bundle Output**:
```json
{
  "test": "test_with_panic",
  "status": "panic",
  "panic_message": "Intentional panic for testing"
}
```

---

## CI/CD Integration

### Always Upload Proof Bundles

```yaml
# .github/workflows/test.yml

- name: Run Tests
  run: cargo test -p vram-residency
  continue-on-error: true  # Don't stop on test failure

- name: Upload Proof Bundle (ALWAYS)
  if: always()  # Upload even if tests failed
  uses: actions/upload-artifact@v3
  with:
    name: vram-residency-proof-bundle
    path: bin/worker-orcd-crates/vram-residency/.proof_bundle/
```

### Fail Job After Upload

```yaml
- name: Run Tests
  id: tests
  run: cargo test -p vram-residency
  continue-on-error: true

- name: Upload Proof Bundle (ALWAYS)
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: proof-bundle
    path: .proof_bundle/

- name: Fail if Tests Failed
  if: steps.tests.outcome == 'failure'
  run: exit 1
```

---

## Verification

### Check Proof Bundle Exists

```bash
# After test run (pass or fail)
ls -la <crate>/.proof_bundle/unit/

# Should ALWAYS have latest bundle
```

### Check Failure Captured

```bash
# After failed test
cat <crate>/.proof_bundle/unit/<run-id>/test_results.txt

# Should show failure details
```

---

## Anti-Patterns

### ❌ WRONG: Skip on Failure

```rust
// ❌ BAD: Don't do this
#[test]
fn bad_test() -> anyhow::Result<()> {
    let result = run_test()?;  // Fails early, no proof bundle
    
    // This code never runs if test fails
    let pb = ProofBundle::for_type(TestType::Unit)?;
    pb.write_json("results", &result)?;
    
    Ok(())
}
```

### ❌ WRONG: Only Capture Success

```rust
// ❌ BAD: Don't do this
#[test]
fn bad_test() -> anyhow::Result<()> {
    let pb = ProofBundle::for_type(TestType::Unit)?;
    let result = run_test();
    
    // Only captures success
    if result.is_ok() {
        pb.append_ndjson("results", &result)?;
    }
    // Failure not captured!
    
    result
}
```

### ✅ CORRECT: Always Capture

```rust
// ✅ GOOD: Always capture
#[test]
fn good_test() -> anyhow::Result<()> {
    let pb = ProofBundle::for_type(TestType::Unit)?;
    let result = run_test();
    
    // ALWAYS capture (pass or fail)
    pb.append_ndjson("results", &serde_json::json!({
        "status": if result.is_ok() { "pass" } else { "fail" },
        "error": result.as_ref().err().map(|e| format!("{:?}", e)),
    }))?;
    
    result
}
```

---

## Enforcement

### Code Review Checklist

- [ ] Proof bundle created at start of test
- [ ] Results captured for ALL outcomes (pass/fail/panic)
- [ ] Error details captured on failure
- [ ] Report generated regardless of outcome
- [ ] CI uploads proof bundle with `if: always()`

---

## Refinement Opportunities

1. **Automated verification**: Check that proof bundles exist after failed test runs
2. **Failure analysis**: Aggregate failure patterns across proof bundles
3. **Regression tracking**: Compare failures across runs
4. **Root cause database**: Link failures to known issues

---

## References

- PB-001: Proof Bundle Location Policy
- `.docs/testing/TEST_TYPES_GUIDE.md` — Test type definitions
- `test-harness/proof-bundle/README.md` — Library documentation

---

**Status**: ✅ NORMATIVE  
**Enforcement**: REQUIRED for all crates  
**Violations**: Block PR merge
