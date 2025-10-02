# Feature Request: Cargo Test Result Capture

**Date**: 2025-10-02  
**Requestor**: vram-residency proof bundle implementation  
**Priority**: HIGH (needed for merge window)

---

## Problem

Currently, crates that want to generate comprehensive proof bundles for unit tests must:
1. Manually run `cargo test --format json`
2. Parse the JSON output themselves
3. Extract test results
4. Write to proof bundle using `append_ndjson()`

This is error-prone and duplicates logic across crates.

**Example** (current approach in `vram-residency/tests/comprehensive_proof_bundle.rs`):
```rust
// Manual cargo test invocation
let output = Command::new("cargo")
    .args(&["test", "-p", "vram-residency", "--", "--format", "json"])
    .output()?;

// Manual JSON parsing
for line in stdout.lines() {
    if let Ok(json) = serde_json::from_str::<Value>(line) {
        // Extract test results manually...
        pb.append_ndjson("test_results", &result)?;
    }
}
```

---

## Proposed Solution

Add a helper function to `ProofBundle` that captures all test results automatically:

```rust
impl ProofBundle {
    /// Capture all test results from cargo test and write to proof bundle
    ///
    /// This runs `cargo test --format json` for the current crate and captures:
    /// - Individual test results (pass/fail/ignored)
    /// - Test timing data
    /// - Test output (stdout/stderr)
    /// - Summary statistics
    ///
    /// # Arguments
    /// * `package_name` - The package to test (e.g., "vram-residency")
    /// * `test_args` - Additional args to pass to cargo test (e.g., ["--lib", "--tests"])
    ///
    /// # Returns
    /// Summary with total/passed/failed/ignored counts
    ///
    /// # Example
    /// ```rust
    /// let pb = ProofBundle::for_type(TestType::Unit)?;
    /// let summary = pb.capture_cargo_test_results("vram-residency", &["--lib", "--tests"])?;
    /// println!("Captured {} tests", summary.total);
    /// ```
    pub fn capture_cargo_test_results(
        &self,
        package_name: &str,
        test_args: &[&str],
    ) -> anyhow::Result<TestSummary> {
        // Implementation:
        // 1. Run cargo test --format json
        // 2. Parse JSON output
        // 3. Write to test_results.ndjson
        // 4. Generate summary.json
        // 5. Return TestSummary struct
    }
}

pub struct TestSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub ignored: usize,
    pub duration_secs: f64,
}
```

---

## Benefits

1. **DRY**: No duplicate test capture logic across crates
2. **Consistent**: All crates use same format for test results
3. **Maintainable**: Changes to test capture logic happen in one place
4. **Easy to use**: Single function call instead of 50+ lines of code

---

## Alternative: Keep Current Approach

If this feature is out of scope for proof-bundle, that's fine! The current manual approach works, it's just more verbose.

**Pros of manual approach**:
- More control over what gets captured
- No new dependencies in proof-bundle
- Crates can customize parsing logic

**Cons of manual approach**:
- Duplicated code across crates
- Inconsistent formats
- More maintenance burden

---

## Decision

**Proof Bundle Team**: Please decide:

- [ ] **Option A**: Implement `capture_cargo_test_results()` helper
- [ ] **Option B**: Keep manual approach (current implementation is fine)

If Option B, we'll document the manual approach as a pattern for other crates to follow.

---

**Status**: ⏳ PENDING DECISION  
**Blocked**: vram-residency comprehensive proof bundle implementation
