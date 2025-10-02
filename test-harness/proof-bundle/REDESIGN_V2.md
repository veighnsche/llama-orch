# Proof Bundle Crate Redesign (V2)

**Date**: 2025-10-02  
**Status**: PROPOSAL  
**Trigger**: Lessons learned from vram-residency (first real use case)

---

## Executive Summary

The current proof-bundle crate has a **critical design flaw** discovered during vram-residency integration:

**The `capture_tests()` API does not work when called from within the same package.**

This renders the primary feature (automatic test capture) unusable for its main use case: crates capturing their own tests.

**Management Decision**: Complete redesign from ground up.

---

## What Went Wrong

### Critical Issue: Same-Package Capture Fails

**Problem**: 
```rust
// In vram-residency/tests/comprehensive_proof_bundle.rs
#[test]
fn generate_proof_bundle() {
    let pb = ProofBundle::for_type(TestType::Unit)?;
    let summary = pb.capture_tests("vram-residency")
        .lib()
        .tests()
        .run()?;
    
    // Result: 0 tests captured ❌
}
```

**Root Cause**:
1. Test runs in package `vram-residency`
2. Calls `cargo test -p vram-residency --format json`
3. Cargo filters out currently running test file to avoid recursion
4. All tests in that file are excluded
5. Result: 0 tests captured

**Impact**: **100% of use cases broken** for same-package capture.

### Workaround Required

Had to create standalone shell script instead:
```bash
#!/usr/bin/env bash
# NOT a test, runs externally
cargo +nightly test -p vram-residency --features skip-long-tests \
    -- --format json -Z unstable-options \
    | parse_output.sh
```

**Result**: 360 tests captured successfully.

**Conclusion**: The core API is fundamentally flawed for its primary use case.

---

## What Went Right

### 1. ✅ Directory Structure

**Crate-local proof bundles** work well:
```
crate/.proof_bundle/
├── unit-fast/
├── unit-full/
└── bdd/
```

**Benefits**:
- Easy to find
- Version controlled alongside code
- Self-contained per crate

**Keep this**: ✅

### 2. ✅ File Formats

**NDJSON + JSON + Markdown** is the right combination:
- `test_results.ndjson` — Machine-parseable, append-friendly
- `summary.json` — Aggregate statistics
- `test_report.md` — Human-readable

**Keep this**: ✅

### 3. ✅ Builder Pattern

The builder API is ergonomic:
```rust
pb.capture_tests("crate-name")
    .lib()
    .tests()
    .features(&["skip-long-tests"])
    .run()?
```

**Keep this pattern**: ✅ (but fix the implementation)

### 4. ✅ Comprehensive Testing

The proof-bundle crate itself has 25+ unit tests covering:
- Builder pattern
- Serialization
- Edge cases
- Type safety

**Keep this discipline**: ✅

---

## Requirements for V2

### Must Have

1. **✅ Works from same package**
   - Primary use case: crate captures its own tests
   - Must not require external scripts

2. **✅ No nightly required**
   - Stable Rust compatibility
   - Fallback to parsing regular test output if needed

3. **✅ No recursion issues**
   - Safely handle test-in-test scenarios
   - Clear error messages if misconfigured

4. **✅ Feature flags supported**
   - `--features skip-long-tests` and similar
   - Conditional compilation awareness

5. **✅ Cross-package capture works**
   - proof-bundle captures proof-bundle tests (dogfooding)
   - Bonus: external audit tools can capture any crate

### Should Have

6. **⚠️ Stable Rust output parsing**
   - Parse non-JSON test output if `--format json` unavailable
   - Graceful degradation

7. **⚠️ Incremental capture**
   - Append to existing proof bundles
   - Don't overwrite on each run

8. **⚠️ Filtering support**
   - Capture specific tests by name/pattern
   - Exclude certain tests

### Nice to Have

9. **💡 Performance tracking**
   - Track test duration over time
   - Detect performance regressions

10. **💡 Coverage integration**
    - Combine with code coverage data
    - Show untested code paths

---

## Root Cause Analysis

### Why Same-Package Capture Fails

**Current Implementation**:
```rust
impl ProofBundle {
    pub fn capture_tests(&self, package: &str) -> TestCaptureBuilder {
        TestCaptureBuilder::new(self, package)
    }
}

impl TestCaptureBuilder {
    pub fn run(&self) -> Result<TestSummary> {
        // Builds: cargo test -p <package> -- --format json
        let mut cmd = Command::new("cargo");
        cmd.arg("test");
        cmd.arg("-p");
        cmd.arg(&self.package);  // ❌ Problem: same as calling package
        // ...
        cmd.output()?
    }
}
```

**When called from vram-residency test**:
- Runs: `cargo test -p vram-residency -- --format json`
- Cargo sees: "test is already running in vram-residency"
- Filters: Excludes currently running test file
- Result: 0 tests

**The Fundamental Problem**: Cannot run `cargo test` recursively in same package.

---

## Design Solutions

### Option 1: Binary Wrapper (RECOMMENDED)

**Approach**: Extract test capture to separate binary.

**Structure**:
```
test-harness/
├── proof-bundle/        # Library (types, writers, etc.)
└── proof-bundle-cli/    # Binary (test capture)
```

**Usage**:
```bash
# From any crate
cargo run --bin proof-bundle-cli -- \
    --package vram-residency \
    --features skip-long-tests \
    --output .proof_bundle/unit-fast
```

**Benefits**:
- ✅ No same-package issues (runs externally)
- ✅ Works from any crate
- ✅ Can be CI/CD integrated
- ✅ Stable Rust compatible

**Drawbacks**:
- ⚠️ Requires separate binary
- ⚠️ Not a library-only solution

**Recommendation**: **This is the way.** ✅

### Option 2: Macro-Based Capture

**Approach**: Use macros to capture at compile time.

```rust
#[proof_bundle::capture]
#[test]
fn test_something() {
    // Test automatically captured
}
```

**Benefits**:
- ✅ No runtime issues
- ✅ Works in same package

**Drawbacks**:
- ❌ Requires proc macros (complex)
- ❌ Can't capture existing tests without annotation
- ❌ Doesn't capture test output/timing

**Recommendation**: **Not viable.** ❌

### Option 3: Build Script Integration

**Approach**: Generate proof bundles during `cargo test`.

```toml
[build-dependencies]
proof-bundle = "0.2"
```

```rust
// build.rs
fn main() {
    proof_bundle::generate_build_script();
}
```

**Benefits**:
- ✅ Automatic integration
- ✅ No manual scripts

**Drawbacks**:
- ❌ Build scripts don't run during `cargo test`
- ❌ Wrong lifecycle phase
- ❌ Can't access test results

**Recommendation**: **Not viable.** ❌

### Option 4: Test Harness Replacement

**Approach**: Replace default test harness with proof-bundle aware one.

```toml
[[test]]
name = "my_tests"
harness = false
```

**Benefits**:
- ✅ Full control over test execution
- ✅ Can capture everything

**Drawbacks**:
- ❌ User must opt-in per test file
- ❌ Lose default test runner features
- ❌ Complex implementation

**Recommendation**: **Too invasive.** ❌

---

## Recommended Design: V2 Architecture

### Two-Component System

#### Component 1: Library (`proof-bundle`)

**Purpose**: Types, writers, utilities

**API**:
```rust
// Core types (keep these)
pub struct ProofBundle { /* ... */ }
pub struct TestSummary { /* ... */ }
pub struct TestResult { /* ... */ }

// Writers (keep these)
impl ProofBundle {
    pub fn for_type(test_type: TestType) -> Result<Self>;
    pub fn write_json<T: Serialize>(&self, name: &str, value: &T) -> Result<()>;
    pub fn append_ndjson<T: Serialize>(&self, name: &str, value: &T) -> Result<()>;
    pub fn write_markdown(&self, name: &str, content: &str) -> Result<()>;
}

// NEW: Parser utilities (not executors)
pub mod parsers {
    pub fn parse_json_output(output: &str) -> Result<Vec<TestResult>>;
    pub fn parse_stable_output(output: &str) -> Result<Vec<TestResult>>;
}

// REMOVED: capture_tests() (doesn't work)
```

**Changes**:
- ❌ Remove `capture_tests()` API
- ✅ Add parser utilities for external use
- ✅ Keep all file writing functionality

#### Component 2: CLI (`proof-bundle-cli`)

**Purpose**: Executable test capture tool

**Usage**:
```bash
proof-bundle-cli \
    --package vram-residency \
    --features skip-long-tests \
    --output .proof_bundle/unit-fast \
    --mode unit
```

**Implementation**:
```rust
// proof-bundle-cli/src/main.rs
use proof_bundle::{ProofBundle, TestType, parsers};
use std::process::Command;

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Run cargo test externally (not recursive!)
    let output = Command::new("cargo")
        .arg("test")
        .arg("-p").arg(&args.package)
        // ... features, etc.
        .output()?;
    
    // Parse results
    let results = if json_available() {
        parsers::parse_json_output(&output.stdout)?
    } else {
        parsers::parse_stable_output(&output.stdout)?
    };
    
    // Write proof bundle
    let pb = ProofBundle::for_type(TestType::Unit)?;
    for result in results {
        pb.append_ndjson("test_results", &result)?;
    }
    
    Ok(())
}
```

**Benefits**:
- ✅ Runs externally (no recursion)
- ✅ Works with any crate
- ✅ Can be scripted/CI integrated
- ✅ Fallback to stable Rust parsing

---

## Migration Path

### Phase 1: Create CLI Tool

1. Create `test-harness/proof-bundle-cli/` crate
2. Implement argument parsing
3. Implement test execution
4. Implement output parsing (JSON + stable)
5. Write to proof bundle

**Deliverable**: Working `proof-bundle-cli` binary

### Phase 2: Extract Parsers

1. Extract JSON parsing from `test_capture.rs` to `parsers/json.rs`
2. Add stable output parsing to `parsers/stable.rs`
3. Export parsers from library
4. Make CLI use library parsers

**Deliverable**: Reusable parser utilities

### Phase 3: Update Documentation

1. Update README with CLI usage
2. Update examples to use CLI
3. Document limitations of library-only approach
4. Provide migration guide

**Deliverable**: Clear documentation

### Phase 4: Deprecate Old API

1. Add `#[deprecated]` to `capture_tests()`
2. Document why it doesn't work
3. Point to CLI tool instead
4. Keep for proof-bundle's own tests (cross-package works)

**Deliverable**: Clean deprecation path

### Phase 5: Clean Up vram-residency

1. Remove `tests/comprehensive_proof_bundle.rs` (doesn't work)
2. Update to use `proof-bundle-cli`
3. Update documentation
4. Verify proof bundles generate correctly

**Deliverable**: Working vram-residency proof bundles

---

## API Comparison

### V1 (Current - Broken)

```rust
// In vram-residency/tests/
#[test]
#[ignore]
fn generate_proof_bundle() -> Result<()> {
    let pb = ProofBundle::for_type(TestType::Unit)?;
    let summary = pb.capture_tests("vram-residency")  // ❌ Doesn't work
        .lib()
        .tests()
        .run()?;
    Ok(())
}
```

**Result**: 0 tests captured

### V2 (Proposed - Working)

```bash
# From command line or script
proof-bundle-cli \
    --package vram-residency \
    --lib \
    --tests \
    --output .proof_bundle/unit-full
```

**Result**: All tests captured

**Alternative** (for CI/CD):
```yaml
# .github/workflows/test.yml
- name: Generate proof bundle
  run: |
    cargo install proof-bundle-cli
    proof-bundle-cli --package ${{ matrix.crate }} \
        --output .proof_bundle/${{ github.sha }}
```

---

## Success Criteria

### V2 Must Achieve

1. ✅ **Same-package capture works**
   - vram-residency can generate its own proof bundles
   - No workarounds or shell scripts required

2. ✅ **Stable Rust compatible**
   - Works without nightly
   - Graceful degradation if `--format json` unavailable

3. ✅ **Zero recursion issues**
   - Clear error messages
   - Safe to run from any context

4. ✅ **Feature flag support**
   - `--features skip-long-tests` works
   - Multiple features supported

5. ✅ **CI/CD friendly**
   - Single command to generate proof bundles
   - Exit codes indicate success/failure
   - JSON output for machine parsing

### V2 Should Achieve

6. ⚠️ **Incremental capture**
   - Don't regenerate entire proof bundle each time
   - Append new results

7. ⚠️ **Performance tracking**
   - Compare test duration against previous runs
   - Flag regressions

### V2 Nice to Have

8. 💡 **Coverage integration**
   - Show code coverage in proof bundles
   - Identify untested code paths

9. 💡 **Parallel capture**
   - Capture multiple crates simultaneously
   - Workspace-level proof bundles

---

## Lessons Learned

### From vram-residency Experience

1. **❌ Don't assume library-only solution**
   - Test capture needs external execution
   - Binary tools are OK for this use case

2. **❌ Don't test API without real use case**
   - proof-bundle's own tests worked (cross-package)
   - Missed that same-package fails

3. **✅ Shell scripts reveal the real solution**
   - The working script shows what the CLI should do
   - Don't fight against cargo's design

4. **✅ Separate fast/full modes is valuable**
   - skip-long-tests saves significant time
   - Different proof bundle directories makes sense

5. **✅ Comprehensive testing is essential**
   - 25+ tests for proof-bundle caught many bugs
   - Just not the big architectural one

### From Management Perspective

1. **⚠️ First use case is critical**
   - Should have validated with same-package scenario
   - Cross-package testing alone was insufficient

2. **⚠️ Core feature must work**
   - If `capture_tests()` doesn't work, what's the point?
   - Better to have working CLI than broken library API

3. **✅ Rapid iteration is valuable**
   - Discovered issue quickly
   - Can pivot to correct solution

---

## Implementation Estimate

### Phase 1: CLI Tool (2-3 days)

- Binary crate setup
- Argument parsing (clap)
- Test execution
- Output parsing
- Proof bundle writing

### Phase 2: Parsers (1 day)

- Extract JSON parser
- Add stable parser
- Tests for both

### Phase 3: Documentation (1 day)

- README updates
- Examples
- Migration guide

### Phase 4: Deprecation (1 day)

- Deprecate old API
- Update proof-bundle's own tests
- Version bump to 0.2.0

### Phase 5: vram-residency Cleanup (1 day)

- Remove broken tests
- Update to CLI
- Verify works

**Total**: 6-7 days

---

## Open Questions

1. **Should we keep `capture_tests()` for cross-package use?**
   - Works for proof-bundle capturing proof-bundle
   - Could be useful for external audit tools
   - But confusing if it only works sometimes

2. **Should CLI be in same repo or separate?**
   - Same repo: Easier to version together
   - Separate: Cleaner dependency graph

3. **How to handle nightly requirement?**
   - `--format json` only on nightly
   - Stable fallback: parse human-readable output
   - Which to prefer?

4. **Should we support filtering tests?**
   - `--test-name pattern`
   - `--exclude pattern`
   - Complexity vs. value?

---

## Recommendation

**Proceed with V2 redesign using binary CLI approach.**

**Rationale**:
1. Only viable solution for same-package capture
2. Aligns with how cargo ecosystem works
3. Can be implemented quickly (1 week)
4. Solves all current limitations
5. Sets up for future enhancements

**Next Step**: Get management approval, then start Phase 1.

---

**Status**: ✅ PROPOSAL COMPLETE  
**Confidence**: HIGH  
**Risk**: LOW (clear path forward)  
**Timeline**: 1 week for full implementation
