# Feature Decision: Cargo Test Result Capture

**Date**: 2025-10-02  
**Decision**: ✅ **APPROVED - Option A (Implement Helper)**  
**Reviewer**: proof-bundle team

---

## Analysis

### Request is Fair: ✅ YES

**Rationale**:
1. **Aligns with proof-bundle mission**: Making it easy to generate comprehensive, honest proof bundles
2. **Reduces duplication**: Every crate would need this logic
3. **Ensures consistency**: All crates use same format
4. **Follows DRY principle**: Common functionality belongs in shared library

### Implementation Assessment

**Complexity**: MEDIUM (manageable)

**Required Changes**:
1. Add `std::process::Command` usage (already using `std::fs`)
2. Parse `cargo test --format json` output
3. Add new public API: `capture_cargo_test_results()`
4. Add new type: `TestSummary`

**Breaking Changes**: NONE
- This is a new feature, not a change to existing API
- Existing code continues to work

---

## Recommended Implementation

### Design Decisions

#### 1. ✅ Make it Synchronous (Consistent with Library)

The proof-bundle library is intentionally synchronous (test-time only). Keep this pattern:

```rust
pub fn capture_cargo_test_results(...) -> Result<TestSummary>
// NOT: pub async fn capture_cargo_test_results(...)
```

#### 2. ✅ Use Builder Pattern for Flexibility

Instead of simple args, use a builder for better ergonomics:

```rust
pub struct TestCaptureBuilder {
    package: String,
    lib: bool,
    tests: bool,
    benches: bool,
    doc: bool,
    features: Vec<String>,
    no_fail_fast: bool,
}

impl ProofBundle {
    pub fn capture_tests(&self, package: &str) -> TestCaptureBuilder {
        TestCaptureBuilder::new(self, package)
    }
}

impl TestCaptureBuilder {
    pub fn lib(mut self) -> Self { self.lib = true; self }
    pub fn tests(mut self) -> Self { self.tests = true; self }
    pub fn all(mut self) -> Self { 
        self.lib = true;
        self.tests = true;
        self
    }
    pub fn run(self) -> Result<TestSummary> { /* ... */ }
}
```

**Usage**:
```rust
let summary = pb.capture_tests("vram-residency")
    .lib()
    .tests()
    .run()?;
```

#### 3. ✅ Always Capture (Even on Failure)

Per PB-002 policy, capture results even if tests fail:

```rust
let output = Command::new("cargo")
    .args(&args)
    .output()?;  // Don't check exit code

// Parse results regardless of pass/fail
// ...

// Return summary with failure count
Ok(TestSummary {
    total,
    passed,
    failed,  // May be > 0
    ...
})
```

#### 4. ✅ Capture Detailed Failure Information

When tests fail, capture:
- Error message
- Stack trace (if available)
- stdout/stderr
- Which assertion failed

```rust
pub struct TestResult {
    pub name: String,
    pub status: TestStatus,
    pub duration_secs: f64,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub error_message: Option<String>,
}

pub enum TestStatus {
    Passed,
    Failed,
    Ignored,
    Timeout,
}
```

#### 5. ✅ Handle Unstable Features Gracefully

`--format json` requires `-Z unstable-options` on stable Rust. Handle this:

```rust
// Try with --format json first
let mut cmd = Command::new("cargo");
cmd.args(&["test", "-p", package, "--", "--format", "json", "-Z", "unstable-options"]);

let output = cmd.output()?;

// If it fails due to unstable options, fall back to regular output
if !output.status.success() && String::from_utf8_lossy(&output.stderr).contains("unstable") {
    // Fall back to parsing regular test output
    // Less detailed, but still works
}
```

---

## Proposed API

### Core API

```rust
impl ProofBundle {
    /// Capture test results from cargo test
    ///
    /// Returns a builder for configuring which tests to run.
    ///
    /// # Example
    /// ```rust
    /// let pb = ProofBundle::for_type(TestType::Unit)?;
    /// 
    /// let summary = pb.capture_tests("vram-residency")
    ///     .lib()
    ///     .tests()
    ///     .run()?;
    /// 
    /// println!("Captured {} tests ({} passed, {} failed)", 
    ///     summary.total, summary.passed, summary.failed);
    /// ```
    pub fn capture_tests(&self, package: &str) -> TestCaptureBuilder {
        TestCaptureBuilder::new(self, package)
    }
}
```

### Builder API

```rust
pub struct TestCaptureBuilder<'a> {
    pb: &'a ProofBundle,
    package: String,
    lib: bool,
    tests: bool,
    benches: bool,
    doc: bool,
    features: Vec<String>,
    no_fail_fast: bool,
    test_threads: Option<usize>,
}

impl<'a> TestCaptureBuilder<'a> {
    /// Include unit tests (--lib)
    pub fn lib(mut self) -> Self;
    
    /// Include integration tests (--tests)
    pub fn tests(mut self) -> Self;
    
    /// Include benchmarks (--benches)
    pub fn benches(mut self) -> Self;
    
    /// Include doc tests (--doc)
    pub fn doc(mut self) -> Self;
    
    /// Include all test types
    pub fn all(mut self) -> Self;
    
    /// Enable specific features
    pub fn features(mut self, features: &[&str]) -> Self;
    
    /// Don't stop on first failure
    pub fn no_fail_fast(mut self) -> Self;
    
    /// Set test thread count
    pub fn test_threads(mut self, n: usize) -> Self;
    
    /// Run tests and capture results
    ///
    /// This will:
    /// 1. Run cargo test with --format json
    /// 2. Parse test results
    /// 3. Write to test_results.ndjson
    /// 4. Write summary.json
    /// 5. Generate test_report.md
    /// 6. Return TestSummary
    ///
    /// **IMPORTANT**: Results are captured even if tests fail (per PB-002 policy)
    pub fn run(self) -> Result<TestSummary>;
}
```

### Result Types

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    /// Total number of tests run
    pub total: usize,
    
    /// Number of tests that passed
    pub passed: usize,
    
    /// Number of tests that failed
    pub failed: usize,
    
    /// Number of tests that were ignored
    pub ignored: usize,
    
    /// Total duration in seconds
    pub duration_secs: f64,
    
    /// Pass rate (0.0 to 100.0)
    pub pass_rate: f64,
    
    /// Individual test results (for detailed analysis)
    pub tests: Vec<TestResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name (e.g., "vram_residency::tests::test_seal_model")
    pub name: String,
    
    /// Test status
    pub status: TestStatus,
    
    /// Duration in seconds
    pub duration_secs: f64,
    
    /// Standard output (if captured)
    pub stdout: Option<String>,
    
    /// Standard error (if captured)
    pub stderr: Option<String>,
    
    /// Error message (if failed)
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Ignored,
    Timeout,
}
```

---

## Implementation Plan

### Phase 1: Core Functionality (Week 1)

1. ✅ Add `TestCaptureBuilder` struct
2. ✅ Implement `capture_tests()` method
3. ✅ Parse `cargo test --format json` output
4. ✅ Write to `test_results.ndjson`
5. ✅ Write to `summary.json`
6. ✅ Generate `test_report.md`
7. ✅ Return `TestSummary`

**Deliverable**: Working `capture_tests()` API

### Phase 2: Robustness (Week 2)

1. ✅ Handle unstable features gracefully
2. ✅ Fall back to regular output parsing if needed
3. ✅ Capture failure details (error messages, stack traces)
4. ✅ Handle timeouts
5. ✅ Handle panics

**Deliverable**: Production-ready implementation

### Phase 3: Documentation (Week 3)

1. ✅ Add examples to README
2. ✅ Update IMPLEMENTATION_GUIDE for vram-residency
3. ✅ Add to proof-bundle spec (PB-003: Helper Functions)
4. ✅ Document best practices

**Deliverable**: Complete documentation

---

## Migration Path for vram-residency

### Before (Manual - 50+ lines)

```rust
let output = Command::new("cargo")
    .args(&["test", "-p", "vram-residency", "--lib", "--tests", "--", "--format", "json", "-Z", "unstable-options"])
    .output()?;

let stdout = String::from_utf8_lossy(&output.stdout);
let mut test_results = Vec::new();
let mut passed = 0;
let mut failed = 0;

for line in stdout.lines() {
    if let Ok(json) = serde_json::from_str::<Value>(line) {
        if json["type"] == "test" {
            // ... 30+ lines of parsing logic
        }
    }
}

for result in &test_results {
    pb.append_ndjson("test_results", result)?;
}

pb.write_json("summary", &serde_json::json!({ /* ... */ }))?;
pb.write_markdown("test_report.md", &report)?;
```

### After (Helper - 5 lines)

```rust
let summary = pb.capture_tests("vram-residency")
    .lib()
    .tests()
    .run()?;

println!("Captured {} tests", summary.total);
```

**Reduction**: 50+ lines → 5 lines (90% reduction)

---

## Dependencies

### New Dependencies Required

```toml
[dependencies]
# Already have these:
anyhow = { workspace = true }
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }

# No new dependencies needed!
# std::process::Command is in std
```

**Impact**: ZERO new dependencies

---

## Risks and Mitigations

### Risk 1: Cargo Test Format Changes

**Risk**: `cargo test --format json` output format might change

**Mitigation**:
- Parse defensively (handle missing fields)
- Fall back to regular output if JSON fails
- Version-check cargo if needed

### Risk 2: Unstable Features

**Risk**: `-Z unstable-options` not available on stable Rust

**Mitigation**:
- Try JSON format first
- Fall back to parsing regular output
- Document both approaches

### Risk 3: Performance

**Risk**: Running cargo test might be slow

**Mitigation**:
- This is test-time only (acceptable)
- Users can control which tests to run (builder pattern)
- Parallel test execution (cargo handles this)

---

## Alternatives Considered

### Alternative 1: Keep Manual Approach

**Pros**:
- No new code in proof-bundle
- Maximum flexibility for crates

**Cons**:
- Duplicated code across crates
- Inconsistent formats
- Higher maintenance burden
- Violates DRY principle

**Decision**: REJECTED (doesn't align with proof-bundle mission)

### Alternative 2: External Tool

**Pros**:
- Separate concern
- Could be used outside proof-bundle

**Cons**:
- Extra dependency
- More complex setup
- Doesn't integrate with ProofBundle API

**Decision**: REJECTED (adds complexity)

### Alternative 3: Macro-Based Approach

**Pros**:
- Compile-time generation
- No runtime overhead

**Cons**:
- Complex implementation
- Less flexible
- Harder to debug

**Decision**: REJECTED (over-engineered)

---

## Decision

✅ **APPROVED: Implement `capture_tests()` helper (Option A)**

**Rationale**:
1. ✅ Aligns with proof-bundle mission (make comprehensive proof bundles easy)
2. ✅ Reduces duplication (DRY principle)
3. ✅ Ensures consistency across crates
4. ✅ Zero new dependencies
5. ✅ Non-breaking change (new feature)
6. ✅ Manageable complexity
7. ✅ High value for users (50+ lines → 5 lines)

**Breaking Changes Allowed**: YES (but none needed)

---

## Next Steps

1. ✅ Implement `TestCaptureBuilder` and `capture_tests()`
2. ✅ Add tests for the new functionality
3. ✅ Update README with examples
4. ✅ Update vram-residency to use new API
5. ✅ Document in PB-003 spec (Helper Functions)

---

## Refinement Opportunities

1. **Parallel test capture**: Capture multiple crates in parallel
2. **Test filtering**: Capture only specific tests by name/pattern
3. **Incremental capture**: Only capture changed tests
4. **Coverage integration**: Combine with code coverage data

---

**Status**: ✅ APPROVED  
**Priority**: HIGH  
**Estimated Effort**: 2-3 days  
**Blocked**: vram-residency comprehensive proof bundle (unblocked after implementation)
