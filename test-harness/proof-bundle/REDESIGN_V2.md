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

6. **✅ Human-readable reports (MANAGEMENT REQUIREMENT)**
   - JSON is fine for machines, but MD for humans
   - Beautiful, scannable markdown reports
   - Executive summaries, not just raw data
   - Non-developers can audit

7. **✅ Zero boilerplate for other crates (MANAGEMENT REQUIREMENT)**
   - All common patterns in proof-bundle library
   - No code duplication across crates
   - One-line proof bundle generation
   - Templates and formatters included

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

**Purpose**: Types, writers, utilities, formatters, templates

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

// NEW: Formatters (MANAGEMENT REQUIREMENT)
pub mod formatters {
    /// Generate beautiful human-readable report from test results
    pub fn generate_test_report(summary: &TestSummary) -> String;
    
    /// Generate executive summary (for non-technical stakeholders)
    pub fn generate_executive_summary(summary: &TestSummary) -> String;
    
    /// Generate detailed failure report with context
    pub fn generate_failure_report(summary: &TestSummary) -> String;
    
    /// Generate coverage report (if available)
    pub fn generate_coverage_report(summary: &TestSummary) -> String;
}

// NEW: Templates (prevent code duplication)
pub mod templates {
    /// Standard unit test proof bundle template
    pub fn unit_test_template() -> ProofBundleTemplate;
    
    /// Standard BDD test proof bundle template
    pub fn bdd_test_template() -> ProofBundleTemplate;
    
    /// Standard integration test proof bundle template
    pub fn integration_test_template() -> ProofBundleTemplate;
}

// NEW: One-liner for developers (MANAGEMENT REQUIREMENT)
impl ProofBundle {
    /// Generate complete proof bundle with one function call
    /// 
    /// This is what other crates should use - zero boilerplate!
    pub fn generate_for_crate(
        crate_name: &str,
        mode: ProofBundleMode,
    ) -> Result<TestSummary> {
        // Does everything: run tests, parse, format, write
        // Returns summary for verification
    }
}

// REMOVED: capture_tests() (doesn't work)
```

**Changes**:
- ❌ Remove `capture_tests()` API
- ✅ Add parser utilities for external use
- ✅ Keep all file writing functionality
- ✅ **NEW: Formatters for human-readable output**
- ✅ **NEW: Templates to prevent code duplication**
- ✅ **NEW: One-liner API for developer experience**

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

**Option A: CLI** (for CI/CD, external tools):
```bash
# From command line or script
proof-bundle-cli \
    --package vram-residency \
    --lib \
    --tests \
    --output .proof_bundle/unit-full
```

**Option B: Library One-Liner** (DEVELOPER EXPERIENCE - RECOMMENDED):
```rust
// In any test file in vram-residency
#[test]
fn generate_proof_bundle() -> anyhow::Result<()> {
    // ONE LINE - no boilerplate!
    proof_bundle::ProofBundle::generate_for_crate(
        "vram-residency",
        ProofBundleMode::UnitFast,
    )?;
    
    // Automatically generates:
    // - test_results.ndjson (raw data)
    // - summary.json (statistics)
    // - test_report.md (human-readable, beautiful)
    // - executive_summary.md (for management)
    // - failures.md (if any - detailed with context)
    
    Ok(())
}
```

**Result**: All tests captured, beautifully formatted

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

## Human-Readable Formatting (Management Requirement)

### Problem

Current proof bundles are **machine-readable but not human-auditable**:
- JSON/NDJSON files require parsing tools
- No executive summaries
- No failure context
- Hard for non-developers to audit

### Solution: Multi-Level Reports

#### Level 1: Executive Summary (`executive_summary.md`)

**For**: Management, non-technical stakeholders

**Example**:
```markdown
# Test Results Summary — vram-residency

**Date**: 2025-10-02  
**Status**: ✅ 98% PASS RATE  
**Confidence**: HIGH

## Quick Facts

- **360 tests** executed
- **354 passed** (98.3%)
- **3 failed** (0.8%)
- **3 skipped** (0.8%)
- **Duration**: 45 seconds

## Risk Assessment

✅ **LOW RISK** — High pass rate, all critical tests passing

## Failed Tests

1. **test_vram_exhaustion** (non-critical)
   - Expected: Graceful handling of VRAM exhaustion
   - Actual: Panic on OOM
   - **Impact**: LOW (edge case, rarely happens)
   - **Action**: Bug filed (#1234)

## Recommendation

**✅ APPROVED FOR DEPLOYMENT** — All tier-1 security tests passing
```

#### Level 2: Developer Report (`test_report.md`)

**For**: Developers, technical reviewers

**Example**:
```markdown
# Test Report — vram-residency

## Summary

- Total: 360 tests
- Passed: 354 (98.3%)
- Failed: 3 (0.8%)
- Ignored: 3 (0.8%)

## Test Breakdown

### Unit Tests (102 tests)
- ✅ 100 passed
- ❌ 2 failed
- Duration: 5.2s

### Integration Tests (156 tests)
- ✅ 154 passed  
- ❌ 1 failed
- Duration: 15.8s

### Property Tests (90 tests)
- ✅ 90 passed
- Duration: 18.5s

### BDD Tests (12 tests)
- ✅ 10 passed
- ⏭️ 2 skipped (CUDA not available)
- Duration: 5.5s

## Failed Tests

### test_vram_exhaustion
**Location**: `tests/robustness_stress.rs:45`  
**Duration**: 1.2s  
**Error**:
\`\`\`
thread panicked at 'CUDA OOM: Out of memory'
\`\`\`

**Context**:
- Allocating 32GB on 24GB GPU
- Expected: VramError::OutOfMemory
- Actual: Panic

**Related**: Issue #1234

## Performance

**Slowest tests**:
1. test_large_model_seal — 2.5s
2. test_concurrent_access — 1.8s
3. test_vram_exhaustion — 1.2s
```

#### Level 3: Detailed Report (`test_results.ndjson`)

**For**: Machines, CI tools, detailed analysis

Already exists (keep as-is).

### Formatter API

```rust
pub mod formatters {
    /// Generate executive summary (management-friendly)
    pub fn generate_executive_summary(summary: &TestSummary) -> String {
        let mut md = String::new();
        md.push_str("# Test Results Summary\n\n");
        
        // Risk assessment
        let risk = if summary.pass_rate >= 98.0 { "LOW" }
                   else if summary.pass_rate >= 95.0 { "MEDIUM" }
                   else { "HIGH" };
        
        // Executive language, not technical
        // Focus on business impact
        // Actionable recommendations
        
        md
    }
    
    /// Generate developer report (technical details)
    pub fn generate_test_report(summary: &TestSummary) -> String {
        let mut md = String::new();
        
        // Breakdown by test type
        // Failed tests with context
        // Performance metrics
        // Links to code
        
        md
    }
    
    /// Generate failure report (detailed diagnostics)
    pub fn generate_failure_report(summary: &TestSummary) -> String {
        let mut md = String::new();
        
        // Only failed tests
        // Stack traces
        // Related code
        // Reproduction steps
        
        md
    }
}
```

### Templates Prevent Code Duplication

```rust
pub mod templates {
    /// Unit test template (what vram-residency uses)
    pub fn unit_test_template() -> ProofBundleTemplate {
        ProofBundleTemplate {
            modes: vec!["fast", "full"],
            outputs: vec![
                "test_results.ndjson",
                "summary.json",
                "test_report.md",
                "executive_summary.md",
                "failures.md",
            ],
            formatters: vec![
                Box::new(formatters::generate_test_report),
                Box::new(formatters::generate_executive_summary),
                Box::new(formatters::generate_failure_report),
            ],
        }
    }
    
    /// BDD test template (different formatting)
    pub fn bdd_test_template() -> ProofBundleTemplate {
        ProofBundleTemplate {
            modes: vec!["mock", "cuda"],
            outputs: vec![
                "scenarios.ndjson",
                "features.json",
                "bdd_report.md",  // Different format!
                "executive_summary.md",
            ],
            formatters: vec![
                Box::new(formatters::generate_bdd_report),  // BDD-specific
                Box::new(formatters::generate_executive_summary),
            ],
        }
    }
}
```

### Zero-Boilerplate API

```rust
// In vram-residency/tests/proof_bundle.rs
#[test]
fn generate_proof_bundle() -> anyhow::Result<()> {
    // ONE LINE - everything is handled
    proof_bundle::ProofBundle::generate_for_crate(
        "vram-residency",
        ProofBundleMode::UnitFast,
    )?;
    
    // Internally does:
    // 1. Run cargo test with correct flags
    // 2. Parse output (JSON or stable)
    // 3. Generate ALL reports (executive, developer, failures)
    // 4. Write to .proof_bundle/unit-fast/<timestamp>/
    // 5. Return summary for verification
    
    Ok(())
}

// NO boilerplate code needed!
// NO formatters to write!
// NO templates to copy!
// proof-bundle does it all!
```

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

## Summary of Management Requirements

### 1. Developer Experience (PRIMARY FOCUS)

**Problem**: Developers hate writing and maintaining proof bundle code.

**Solution**: Zero-boilerplate, one-line API
```rust
proof_bundle::ProofBundle::generate_for_crate("crate-name", ProofBundleMode::UnitFast)?;
```

**Implementation**:
- All common patterns in proof-bundle library
- Templates for unit tests and BDD tests
- Formatters for all report types
- No code duplication across crates

### 2. Human-Readable Formatting (MANAGEMENT AUDIT)

**Problem**: JSON/NDJSON is not human-auditable.

**Solution**: Multi-level reports
1. **Executive summary** (`executive_summary.md`) — For management, non-technical
2. **Developer report** (`test_report.md`) — For technical reviewers
3. **Failure report** (`failures.md`) — For debugging
4. **Raw data** (`test_results.ndjson`, `summary.json`) — For machines

**Implementation**:
- Formatters in library (`formatters` module)
- Beautiful markdown with risk assessments
- Business-focused language, not technical jargon
- Actionable recommendations

### 3. Perfect Example (LEAD BY EXAMPLE)

**Requirement**: proof-bundle crate must demonstrate perfect proof bundle.

**Implementation**:
- proof-bundle generates its own proof bundles
- All report types generated
- Exemplary formatting
- Other crates copy this pattern

### 4. Repository Context

**Reality**: This repo has two main test types:
1. **Unit tests** — Standard Rust `#[test]`
2. **BDD tests** — Cucumber-style features

**Implementation**:
- `templates::unit_test_template()` — For unit/integration
- `templates::bdd_test_template()` — For BDD/cucumber
- Different formatting for each type

### Key Deliverables

1. ✅ **CLI tool** — External test executor (solves recursion)
2. ✅ **One-liner API** — `generate_for_crate()`
3. ✅ **Formatters** — Executive, developer, failure reports
4. ✅ **Templates** — Unit and BDD templates
5. ✅ **Examples** — Perfect proof bundle in our own crate
6. ✅ **Documentation** — Updated with all new features

### Success Criteria

1. ✅ Works from same package (no recursion issues)
2. ✅ Other crates use ≤ 5 lines of code
3. ✅ Zero duplicate code across crates
4. ✅ Management can read executive summaries
5. ✅ Developers spend < 5 minutes setting up
6. ✅ Beautiful, human-readable reports
7. ✅ Stable Rust compatible

---

**Status**: ✅ PROPOSAL COMPLETE  
**Confidence**: HIGH  
**Risk**: LOW (clear path forward)  
**Timeline**: 1 week for full implementation  
**Management Requirements**: ✅ ALL ADDRESSED
