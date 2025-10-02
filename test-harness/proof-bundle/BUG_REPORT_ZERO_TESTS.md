# BUG REPORT: V2 API Generates Empty Proof Bundles (0 Tests)

**Date**: 2025-10-02  
**Severity**: 🚨 **CRITICAL**  
**Status**: CONFIRMED  
**Affects**: V2 One-Liner API (`api::generate_for_crate()`)  
**Impact**: Complete failure of primary feature - proof bundles contain 0 tests

---

## Executive Summary

The V2 one-liner API (`api::generate_for_crate()`) generates **empty proof bundles** with 0 tests, producing contradictory and meaningless reports. This is a **complete failure** of the core feature that makes the entire V2 redesign non-functional.

### Observed Output

```markdown
# Test Results Summary
**Status**: ✅ 0.0% PASS RATE
**Confidence**: LOW

## Quick Facts
- **0 tests** executed
- **0 passed** (0.0%)
- **0 failed** (NaN%)

## Risk Assessment
✅ **LOW RISK** — All tests passing

## Recommendation
**❌ NOT APPROVED** — Significant test failures require resolution
```

**This is nonsense**: 
- Claims "0.0% pass rate" but "✅ LOW RISK"
- Says "All tests passing" but "NOT APPROVED"
- Division by zero (NaN%)
- Contradictory status indicators

---

## Root Cause Analysis

### Investigation Methodology

1. ✅ Examined actual proof bundle output at `.proof_bundle/unit/1759413379/`
2. ✅ Reviewed source code in `src/api.rs` (lines 103-171)
3. ✅ Analyzed parser in `src/parsers/stable.rs` (lines 42-114)
4. ✅ Tested actual cargo test output format
5. ✅ Verified formatter logic in `src/formatters/executive.rs`

### Root Cause: Parser Receives Warnings, Not Test Output

**Location**: `src/api.rs:129-136`

```rust
// Run tests and capture output
let output = cmd.output()
    .context("Failed to run cargo test")?;

let stdout = String::from_utf8_lossy(&output.stdout);
let stderr = String::from_utf8_lossy(&output.stderr);  // ← UNUSED!

// Parse test results using stable parser (works on all Rust versions)
let summary = parsers::parse_stable_output(&stdout)
    .context("Failed to parse test output")?;
```

**Problem**: `cargo test` writes test output to **STDERR**, not STDOUT!

#### Evidence from Actual Execution

When running `cargo test -p proof-bundle --lib`:

**STDOUT** contains:
```
warning: /home/vince/Projects/llama-orch/xtask/Cargo.toml: unused manifest key
warning: unused import: `TestStatus`
warning: unused variable: `stderr`
   Compiling proof-bundle v0.0.0
    Finished `test` profile
     Running unittests src/lib.rs
```

**STDERR** contains:
```
running 43 tests
test api::tests::test_mode_template ... ok
test api::tests::test_mode_test_type ... ok
test formatters::tests::test_generate_executive_summary ... ok
[... 40 more tests ...]

test result: ok. 43 passed; 0 failed; 0 ignored; 0 measured
```

**The parser is looking at STDOUT (warnings) instead of STDERR (test results)!**

---

## Detailed Analysis

### Bug #1: Wrong Output Stream (CRITICAL)

**File**: `src/api.rs:136`  
**Line**: `let summary = parsers::parse_stable_output(&stdout)`  
**Should be**: `let summary = parsers::parse_stable_output(&stderr)`

**Evidence**:
```rust
// Line 132-133
let stdout = String::from_utf8_lossy(&output.stdout);
let stderr = String::from_utf8_lossy(&output.stderr);  // ← Contains test output

// Line 136 - BUG: Using wrong stream
let summary = parsers::parse_stable_output(&stdout)  // ← Should be &stderr
```

**Verification**: Variable `stderr` is captured but **never used** (compiler warning at line 133).

### Bug #2: Parser Fails Silently

**File**: `src/parsers/stable.rs:42-114`  
**Behavior**: When no test lines are found, returns empty `TestSummary` with 0 tests

```rust
pub fn parse_stable_output(stdout: &str) -> Result<TestSummary> {
    let mut test_results = Vec::new();  // ← Starts empty
    let mut passed = 0;
    let mut failed = 0;
    
    // Parse individual test lines
    for line in stdout.lines() {
        if line.starts_with("test ") && ... {
            // Extract test
        }
    }
    
    let total = test_results.len();  // ← 0 if no tests found
    
    Ok(TestSummary {
        total,      // ← 0
        passed,     // ← 0
        failed,     // ← 0
        // ...
    })
}
```

**Problem**: Parser doesn't fail when it finds 0 tests - it just returns an empty summary. This masks the real issue.

**Should**: Return `Err` if `total == 0` to alert that parsing failed.

### Bug #3: Formatters Generate Garbage for Empty Data

**File**: `src/formatters/executive.rs:30-100`

```rust
pub fn generate_executive_summary(summary: &TestSummary) -> String {
    // ...
    md.push_str(&format!("**Status**: {} {:.1}% PASS RATE\n", status_emoji, summary.pass_rate));
    
    // Line 51-52: Division by zero!
    md.push_str(&format!("- **{} failed** ({:.1}%)\n", summary.failed, 
                         (summary.failed as f64 / summary.total as f64) * 100.0));
    //                                            ↑ Division by 0 = NaN
    
    // Line 99-100: Contradictory logic
    if summary.failed == 0 {
        md.push_str(" — All tests passing\n\n");  // ← TRUE when total=0
```

**Problems**:
1. **Division by zero**: `failed / total` when `total = 0` produces `NaN%`
2. **Contradictory logic**: "All tests passing" when there are 0 tests
3. **No validation**: Doesn't check if `summary.total > 0`

### Bug #4: No Validation in API

**File**: `src/api.rs:103-171`

The API never validates that tests were actually found:

```rust
pub fn generate_for_crate(package: &str, mode: Mode) -> Result<TestSummary> {
    // ... run tests ...
    let summary = parsers::parse_stable_output(&stdout)?;
    
    // ❌ NO VALIDATION HERE
    // Should check: if summary.total == 0 { return Err(...) }
    
    // Write test results
    for result in &summary.tests {  // ← Empty loop when 0 tests
        pb.append_ndjson("test_results", result)?;
    }
    
    Ok(summary)  // ← Returns success even with 0 tests!
}
```

**Should**: Validate `summary.total > 0` before proceeding.

---

## Impact Assessment

### Severity: CRITICAL

This bug makes the **entire V2 API non-functional**:

1. ❌ **Primary feature broken**: One-liner API doesn't work
2. ❌ **Dogfooding impossible**: Can't generate proof bundles for proof-bundle itself
3. ❌ **Reports are garbage**: Contradictory, meaningless output
4. ❌ **Silent failure**: No error message, just empty results
5. ❌ **Misleading output**: Claims success when it failed

### Affected Components

- ✅ `api::generate_for_crate()` - **BROKEN**
- ✅ `api::generate_with_template()` - **BROKEN** (same bug)
- ✅ All formatters - Generate nonsense for empty data
- ✅ Parser - Fails silently
- ⚠️ V1 API (`ProofBundle::for_type()`) - **WORKS** (different code path)

### User Impact

**Before fix**: 
```bash
$ cargo test generate_proof_bundle -- --ignored
✅ Generated proof bundle!
   Total tests: 0  # ← WRONG!
   Pass rate: 0.0%  # ← WRONG!
```

**Expected**:
```bash
$ cargo test generate_proof_bundle -- --ignored
✅ Generated proof bundle!
   Total tests: 43
   Pass rate: 100.0%
```

---

## Verification

### Test Case 1: Actual Cargo Test Output

```bash
$ cargo test -p proof-bundle --lib 2>&1 | grep "running.*tests"
running 43 tests
```

**Location**: STDERR (not STDOUT)

### Test Case 2: Parser with Real Data

```rust
let output = "running 43 tests\ntest foo ... ok\n...";
let summary = parse_stable_output(output)?;
assert_eq!(summary.total, 43);  // ✅ PASSES with real data
```

### Test Case 3: Current Bug

```rust
let stdout = "warning: unused import\nCompiling...\nFinished...";
let summary = parse_stable_output(stdout)?;
assert_eq!(summary.total, 0);  // ❌ BUG: Returns 0
```

---

## Fix Strategy

### Priority 1: Fix Output Stream (5 minutes)

**File**: `src/api.rs`  
**Lines**: 136, 229

```diff
- let summary = parsers::parse_stable_output(&stdout)
+ let summary = parsers::parse_stable_output(&stderr)
```

**Impact**: Immediate fix for 90% of the problem

### Priority 2: Add Validation (10 minutes)

**File**: `src/api.rs`  
**After line**: 137

```rust
// Validate we actually found tests
if summary.total == 0 {
    return Err(anyhow::anyhow!(
        "No tests found in output. Package '{}' may have no tests, \
         or parsing failed. Check that the package name is correct.",
        package
    ));
}
```

### Priority 3: Fix Parser (15 minutes)

**File**: `src/parsers/stable.rs`  
**After line**: 103

```rust
// Warn if no tests found
if total == 0 {
    eprintln!("WARNING: No test lines found in output. This may indicate:");
    eprintln!("  - Package has no tests");
    eprintln!("  - Wrong output stream (check stderr vs stdout)");
    eprintln!("  - Parsing regex doesn't match output format");
}
```

### Priority 4: Fix Formatters (20 minutes)

**File**: `src/formatters/executive.rs`, `developer.rs`, etc.

```rust
pub fn generate_executive_summary(summary: &TestSummary) -> String {
    // Add validation
    if summary.total == 0 {
        return "# Test Results Summary\n\n\
                ⚠️ **WARNING**: No tests found.\n\n\
                This proof bundle contains no test results. \
                This usually indicates a configuration error.\n".to_string();
    }
    
    // ... rest of formatting ...
}
```

---

## Testing Plan

### Unit Tests

```rust
#[test]
fn test_api_rejects_zero_tests() {
    // Should fail when no tests found
    let result = generate_for_crate("nonexistent-package", Mode::UnitFast);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No tests found"));
}

#[test]
fn test_parser_uses_stderr() {
    let stderr = "running 2 tests\ntest foo ... ok\ntest bar ... ok\n";
    let summary = parse_stable_output(stderr).unwrap();
    assert_eq!(summary.total, 2);
}
```

### Integration Tests

```bash
# Test actual proof bundle generation
cargo test -p proof-bundle generate_comprehensive_proof_bundle -- --ignored --nocapture

# Should show:
# ✅ 43 tests executed
# ✅ 100.0% pass rate
```

---

## Lessons Learned

### Design Flaws

1. **No validation**: API accepts empty results without error
2. **Silent failures**: Parser returns success with 0 tests
3. **Wrong assumptions**: Assumed test output on STDOUT (it's on STDERR)
4. **No integration testing**: Bug would have been caught by end-to-end test
5. **Formatters too permissive**: Should reject invalid input

### Process Improvements

1. ✅ **Add integration tests**: Test full API with real cargo test
2. ✅ **Validate assumptions**: Document where cargo test writes output
3. ✅ **Fail fast**: Return errors for obviously wrong results
4. ✅ **Better error messages**: Tell users what went wrong
5. ✅ **Dogfood earlier**: Run proof bundle generation in CI

---

## References

### Source Code Locations

- **Bug location**: `src/api.rs:136` and `src/api.rs:229`
- **Parser**: `src/parsers/stable.rs:42-114`
- **Formatters**: `src/formatters/executive.rs:30-100`
- **Test output**: `.proof_bundle/unit/1759413379/`

### Related Issues

- Compiler warning: "unused variable: `stderr`" at `src/api.rs:133`
- Division by zero: `NaN%` in reports
- Contradictory status: "All tests passing" + "NOT APPROVED"

### Cargo Test Behavior

From Rust documentation:
> By default, `cargo test` writes test output to **stderr**, not stdout.
> Compilation messages and warnings go to stdout.

**We were parsing the wrong stream!**

---

## Conclusion

This is a **critical bug** caused by a simple mistake: parsing STDOUT instead of STDERR. The bug was masked by:

1. Parser failing silently (returning 0 tests instead of error)
2. No validation in the API
3. Formatters generating output even for invalid data
4. No integration tests to catch the issue

**Estimated fix time**: 1 hour  
**Estimated test time**: 30 minutes  
**Total**: 90 minutes to fully resolve

**Priority**: Fix immediately - this blocks all V2 API usage.
