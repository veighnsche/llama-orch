# Spec: proof-bundle — Test Evidence Collection & Human-Readable Reporting

**Status**: Normative  
**Owner**: @llama-orch-proof-bundle-team  
**Version**: 0.2.0 (V2 Redesign)  
**Date**: 2025-10-02

---

## 0) Motivation

Provide comprehensive test evidence collection with **zero boilerplate** for developers and **human-readable reports** for management audit. The V2 redesign addresses critical design flaws discovered in V1 (same-package test capture failure) and introduces developer-experience-first philosophy per management directive.

**Key Problems Solved**:
1. V1 `capture_tests()` API failed when called from same package (100% of use cases)
2. JSON/NDJSON output not auditable by non-technical stakeholders
3. Code duplication across crates for proof bundle generation
4. Developers hate writing test infrastructure

**Alignment**: This spec supports ORCH-3200 series (testing ownership), home profile quality gates, and establishes proof bundle as the standard test evidence mechanism across all crates.

---

## 1) Scope

### In Scope

- Test execution orchestration and output parsing
- Multi-level report generation (executive, developer, failure, raw data)
- Standard templates for unit tests and BDD tests
- Zero-boilerplate one-liner API for crates
- Human-readable markdown formatting
- CLI tool for external test capture
- Developer experience optimization

### Out of Scope

- Test implementation (each crate owns per ORCH-3200)
- CI/CD pipeline integration details (covered in CI workflows)
- Code coverage computation (may be added in refinements)
- Performance benchmarking (separate concern)

---

## 2) Architecture Overview

proof-bundle V2 uses a **two-component architecture**:

1. **Library (`proof-bundle`)** — Types, parsers, formatters, templates, one-liner API
2. **CLI (`proof-bundle-cli`)** — External test executor (solves same-package recursion)

**Key Innovation**: The library provides `generate_for_crate()` which handles everything internally, avoiding the broken V1 pattern.

---

## 3) Normative Requirements (RFC-2119)

IDs use the **ORCH-38xx range** (proof-bundle system).

### Core Functionality

- **[ORCH-3800]** The proof-bundle crate MUST provide a library for test evidence collection, parsing, formatting, and file generation.

- **[ORCH-3801]** The proof-bundle-cli binary MUST execute `cargo test` externally to avoid same-package recursion issues.

- **[ORCH-3802]** All test output MUST be parseable from both `--format json` (nightly) and stable output formats.

- **[ORCH-3803]** Proof bundles MUST be written to crate-local `.proof_bundle/` directories, NOT repository root.

### Output Format Requirements

- **[ORCH-3804]** Every proof bundle MUST generate at minimum:
  - `test_results.ndjson` — Newline-delimited JSON of all test results
  - `summary.json` — Aggregate statistics (total, passed, failed, ignored, duration, pass_rate)
  - `test_report.md` — Human-readable technical report
  - `executive_summary.md` — Non-technical management summary

- **[ORCH-3805]** When tests fail, proof bundles MUST additionally generate:
  - `failures.md` — Detailed failure report with stack traces and context

- **[ORCH-3806]** All markdown reports MUST use proper formatting: headers, lists, code blocks, emoji indicators (✅❌⏭️⚠️).

### Report Content Requirements

- **[ORCH-3807]** Executive summaries MUST include:
  - Pass rate and confidence level
  - Risk assessment (LOW/MEDIUM/HIGH)
  - Failed test impact analysis
  - Deployment recommendation
  - Non-technical language suitable for management

- **[ORCH-3808]** Developer reports MUST include:
  - Test breakdown by type (unit, integration, property, BDD)
  - Failed tests with locations, durations, and error messages
  - Performance metrics (slowest tests)
  - Links to source code

- **[ORCH-3809]** Failure reports MUST include:
  - Stack traces (if available)
  - Test context and reproduction steps
  - Related code references
  - Issue tracker links (if present)

### Developer Experience Requirements

- **[ORCH-3810]** Crates MUST be able to generate proof bundles with ≤ 5 lines of code.

- **[ORCH-3811]** The primary API MUST be a one-liner:
  ```rust
  proof_bundle::ProofBundle::generate_for_crate(crate_name, mode)?
  ```

- **[ORCH-3812]** All formatting logic MUST be contained in the proof-bundle library; crates MUST NOT implement custom formatters.

- **[ORCH-3813]** Standard templates MUST be provided for:
  - Unit/integration tests (`templates::unit_test_template()`)
  - BDD/cucumber tests (`templates::bdd_test_template()`)

### Template Requirements

- **[ORCH-3814]** The unit test template MUST support modes: `fast` (skip long tests) and `full` (all tests).

- **[ORCH-3815]** The BDD test template MUST support modes: `mock` and `real` (GPU/CUDA required).

- **[ORCH-3816]** Templates MUST define output files, formatters, and feature flags automatically.

### Parser Requirements

- **[ORCH-3817]** The JSON parser MUST handle nightly `cargo test --format json -Z unstable-options` output.

- **[ORCH-3818]** The stable parser MUST handle standard `cargo test` output without requiring nightly Rust.

- **[ORCH-3819]** Parsers MUST extract: test name, status (passed/failed/ignored), duration, error messages, stdout/stderr.

### Formatter Requirements

- **[ORCH-3820]** The formatters module MUST provide:
  - `generate_executive_summary(summary: &TestSummary) -> String`
  - `generate_test_report(summary: &TestSummary) -> String`
  - `generate_failure_report(summary: &TestSummary) -> String`

- **[ORCH-3821]** Formatters MUST be pure functions (no I/O, no side effects).

- **[ORCH-3822]** Executive summaries MUST use business language, not technical jargon.

### CLI Requirements

- **[ORCH-3823]** proof-bundle-cli MUST accept arguments:
  - `--package` (crate name)
  - `--features` (cargo features)
  - `--output` (proof bundle directory)
  - `--mode` (unit-fast, unit-full, bdd-mock, bdd-real)

- **[ORCH-3824]** proof-bundle-cli MUST run `cargo test` as a subprocess, NOT in-process.

- **[ORCH-3825]** proof-bundle-cli MUST return exit code 0 on success, non-zero on failure.

### Feature Flag Requirements

- **[ORCH-3826]** The `skip-long-tests` feature MUST be recognized and passed to cargo test.

- **[ORCH-3827]** Multiple features MUST be supported via comma-separated or repeated arguments.

### Error Handling Requirements

- **[ORCH-3828]** All errors MUST use `anyhow::Result` for library functions.

- **[ORCH-3829]** Error messages MUST be actionable and include:
  - What went wrong
  - Why it went wrong
  - How to fix it

- **[ORCH-3830]** When test execution fails, proof bundles MUST still be generated with partial results.

### Directory Structure Requirements

- **[ORCH-3831]** Proof bundles MUST use timestamp-based subdirectories: `.proof_bundle/<type>/<timestamp>/`

- **[ORCH-3832]** Multiple proof bundle types MUST coexist:
  - `.proof_bundle/unit-fast/`
  - `.proof_bundle/unit-full/`
  - `.proof_bundle/bdd-mock/`
  - `.proof_bundle/bdd-real/`

- **[ORCH-3833]** Old proof bundles MAY be automatically cleaned up after N runs (configurable).

### Testing Requirements (Dogfooding)

- **[ORCH-3834]** The proof-bundle crate MUST generate its own proof bundles demonstrating perfect formatting.

- **[ORCH-3835]** The proof-bundle crate MUST have ≥ 30 unit tests per major feature.

- **[ORCH-3836]** The proof-bundle crate's own proof bundle MUST serve as the exemplar for other crates.

### Documentation Requirements

- **[ORCH-3837]** README MUST include copy-paste examples for common use cases.

- **[ORCH-3838]** API documentation MUST include examples that compile and run.

- **[ORCH-3839]** TEAM_RESPONSIBILITIES.md MUST document developer experience philosophy.

---

## 4) API Specification

### Primary API (One-Liner)

```rust
pub fn generate_for_crate(
    crate_name: &str,
    mode: ProofBundleMode,
) -> Result<TestSummary>
```

**Modes**:
- `ProofBundleMode::UnitFast` — Unit tests with `--features skip-long-tests`
- `ProofBundleMode::UnitFull` — All unit tests
- `ProofBundleMode::BddMock` — BDD tests with mocked dependencies
- `ProofBundleMode::BddReal` — BDD tests with real GPU/CUDA

**Returns**: `TestSummary` with statistics for verification.

**Example**:
```rust
#[test]
fn generate_proof_bundle() -> anyhow::Result<()> {
    proof_bundle::ProofBundle::generate_for_crate(
        "my-crate",
        ProofBundleMode::UnitFast,
    )
}
```

### Formatters Module

```rust
pub mod formatters {
    /// Generate executive summary (management-friendly)
    pub fn generate_executive_summary(summary: &TestSummary) -> String;
    
    /// Generate developer report (technical details)
    pub fn generate_test_report(summary: &TestSummary) -> String;
    
    /// Generate failure report (debugging info)
    pub fn generate_failure_report(summary: &TestSummary) -> String;
}
```

### Templates Module

```rust
pub mod templates {
    /// Standard unit test template
    pub fn unit_test_template() -> ProofBundleTemplate;
    
    /// Standard BDD test template
    pub fn bdd_test_template() -> ProofBundleTemplate;
}
```

### Parsers Module

```rust
pub mod parsers {
    /// Parse JSON output from cargo test --format json
    pub fn parse_json_output(output: &str) -> Result<Vec<TestResult>>;
    
    /// Parse stable output from cargo test
    pub fn parse_stable_output(output: &str) -> Result<Vec<TestResult>>;
}
```

---

## 5) Output Structure

### Directory Layout

```
crate/.proof_bundle/
├── unit-fast/
│   └── <timestamp>/
│       ├── test_results.ndjson
│       ├── summary.json
│       ├── test_report.md
│       ├── executive_summary.md
│       └── failures.md (if any)
├── unit-full/
│   └── <timestamp>/
│       └── ...
├── bdd-mock/
│   └── <timestamp>/
│       └── ...
└── bdd-real/
    └── <timestamp>/
        └── ...
```

### File Formats

**test_results.ndjson**:
```json
{"name":"test_foo","status":"passed","duration_secs":0.001}
{"name":"test_bar","status":"failed","duration_secs":0.05,"error":"assertion failed"}
```

**summary.json**:
```json
{
  "total": 100,
  "passed": 98,
  "failed": 2,
  "ignored": 0,
  "duration_secs": 5.2,
  "pass_rate": 98.0,
  "timestamp": "1696262400",
  "mode": "unit-fast",
  "features": ["skip-long-tests"]
}
```

**executive_summary.md**: See example in section 6.

---

## 6) Report Format Examples

### Executive Summary (Management-Facing)

```markdown
# Test Results Summary — my-crate

**Date**: 2025-10-02  
**Status**: ✅ 98% PASS RATE  
**Confidence**: HIGH

## Quick Facts

- **100 tests** executed
- **98 passed** (98.0%)
- **2 failed** (2.0%)
- **0 skipped**
- **Duration**: 5.2 seconds

## Risk Assessment

⚠️ **MEDIUM RISK** — 2 non-critical failures

## Failed Tests

1. **test_edge_case_timeout** (non-critical)
   - Expected: Graceful timeout handling
   - Actual: Panic after 30s
   - **Impact**: LOW (rare edge case)
   - **Action**: Bug filed (#1234)

2. **test_large_input_oom** (non-critical)
   - Expected: OOM error returned
   - Actual: Process crash
   - **Impact**: MEDIUM (affects large deployments)
   - **Action**: Fix in progress (#1235)

## Recommendation

**✅ APPROVED FOR STAGING** — Critical path tests passing, non-critical issues tracked
```

### Developer Report (Technical)

```markdown
# Test Report — my-crate

## Summary

- Total: 100 tests
- Passed: 98 (98.0%)
- Failed: 2 (2.0%)
- Ignored: 0
- Duration: 5.2s

## Test Breakdown

### Unit Tests (60 tests)
- ✅ 59 passed
- ❌ 1 failed
- Duration: 2.1s

### Integration Tests (30 tests)
- ✅ 29 passed
- ❌ 1 failed
- Duration: 2.5s

### Property Tests (10 tests)
- ✅ 10 passed
- Duration: 0.6s

## Failed Tests

### test_edge_case_timeout
**Location**: `tests/timeout_tests.rs:45`  
**Duration**: 30.1s  
**Error**:
\`\`\`
thread 'test_edge_case_timeout' panicked at 'timeout exceeded'
\`\`\`

**Context**: Testing 30s timeout boundary

### test_large_input_oom
**Location**: `tests/memory_tests.rs:78`  
**Duration**: 0.8s  
**Error**:
\`\`\`
memory allocation of 10GB failed
\`\`\`

**Context**: Allocating 10GB input

## Performance

**Slowest tests**:
1. test_edge_case_timeout — 30.1s
2. test_large_data_processing — 1.2s
3. test_concurrent_access — 0.8s
```

---

## 7) Migration from V1

### V1 API (Broken - Being Removed)

```rust
// ❌ Don't use - doesn't work
let pb = ProofBundle::for_type(TestType::Unit)?;
let summary = pb.capture_tests("my-crate")
    .lib()
    .tests()
    .run()?;
```

**Problem**: Fails with 0 tests when called from same package.

### V2 API (Working)

```rust
// ✅ Use this instead
proof_bundle::ProofBundle::generate_for_crate(
    "my-crate",
    ProofBundleMode::UnitFast,
)?;
```

**Fix**: Handles execution internally, no same-package issues.

### Migration Steps for Existing Crates

1. Update `Cargo.toml`: `proof-bundle = "0.2"`
2. Replace `capture_tests()` calls with `generate_for_crate()`
3. Remove custom formatting code
4. Verify proof bundles generate correctly

---

## 8) Integration with Other Specs

### ORCH-3200 Series (Testing Ownership)

- Proof bundles implement evidence collection per ORCH-3200
- Each crate generates its own proof bundles per ORCH-3207
- Cross-crate BDD tests use BDD template per ORCH-3202

### ORCH-325x Series (Hardening Requirements)

- Per-crate tests generate proof bundles per ORCH-3250+
- Proof bundle reports show hardening compliance

### Home Profile Quality Gates

- Proof bundles provide evidence for PR merges
- Executive summaries enable non-technical review
- Failure reports guide remediation

---

## 9) Implementation Phases

### Phase 1: Core Library (Week 1, Days 1-3)

- Create formatters module
- Create templates module
- Extract parsers from V1
- Implement `generate_for_crate()`
- Write 30+ unit tests

**Deliverable**: Working library with one-liner API

### Phase 2: CLI Tool (Week 1, Days 4-5)

- Create proof-bundle-cli crate
- Implement argument parsing (clap)
- Implement subprocess execution
- Wire up library parsers and formatters
- Test with vram-residency

**Deliverable**: Working CLI tool

### Phase 3: Dogfooding (Week 1, Day 6)

- Generate proof-bundle's own proof bundle
- Polish report formatting
- Create exemplary executive summary
- Verify all 4 report types

**Deliverable**: Perfect proof bundle example

### Phase 4: Documentation (Week 1, Day 7)

- Update README with examples
- Document API with doctests
- Create migration guide from V1
- Update TEAM_RESPONSIBILITIES.md

**Deliverable**: Complete documentation

### Phase 5: Rollout (Week 2)

- Update vram-residency to V2
- Update other crates (if any)
- Remove V1 API with deprecation warnings
- Version bump to 0.2.0

**Deliverable**: V2 deployed across repository

---

## 10) Acceptance Criteria

- ✅ proof-bundle library provides one-liner API
- ✅ Formatters generate all 4 report types
- ✅ Templates for unit and BDD tests
- ✅ CLI tool works from any crate
- ✅ proof-bundle generates its own perfect proof bundle
- ✅ vram-residency successfully uses V2 API
- ✅ Executive summaries readable by non-developers
- ✅ Zero code duplication across crates
- ✅ 30+ unit tests per major feature
- ✅ Documentation includes working examples

---

## 11) Refinement Opportunities

### Short-term (Next Quarter)

1. **Code coverage integration**
   - Parse coverage data from `cargo-tarpaulin` or `cargo-llvm-cov`
   - Include coverage % in reports
   - Highlight untested code paths

2. **Performance tracking**
   - Store historical test durations
   - Detect performance regressions (tests slower than previous runs)
   - Graph test duration trends

3. **Filtering support**
   - Allow capturing specific tests by name pattern
   - Exclude certain tests from proof bundles
   - Tag-based filtering for BDD scenarios

### Medium-term (2-3 Quarters)

4. **Workspace-level proof bundles**
   - Aggregate proof bundles from all crates
   - Generate workspace-wide executive summary
   - Cross-crate dependency analysis

5. **CI/CD integration helpers**
   - GitHub Actions workflow templates
   - GitLab CI templates
   - Automatic PR commenting with test results

6. **Interactive HTML reports**
   - Generate static HTML pages
   - Interactive test result filtering
   - Visual performance graphs

### Long-term (Future)

7. **Test impact analysis**
   - Correlate code changes with test results
   - Identify affected tests from diffs
   - Suggest relevant tests for changes

8. **Flaky test detection**
   - Track test stability over time
   - Flag intermittent failures
   - Recommend quarantine or fix

9. **Compliance reporting**
   - Map tests to requirements (ORCH-IDs)
   - Generate compliance matrices
   - Audit trail for regulatory review

---

## 12) Security & Privacy Considerations

- **[ORCH-3840]** Proof bundles MUST NOT include secrets, API keys, or credentials in any output.
- **[ORCH-3841]** Error messages and stack traces MUST redact sensitive data.
- **[ORCH-3842]** Proof bundles MAY be committed to version control (contain no secrets).
- **[ORCH-3843]** Personal information MUST NOT be included in reports (developer names OK, but no emails/phone numbers).

---

## 13) Performance Requirements

- **[ORCH-3844]** Proof bundle generation MUST NOT add > 5% overhead to test execution time.
- **[ORCH-3845]** Formatting MUST complete in < 1 second for 1000 tests.
- **[ORCH-3846]** File writes MUST be buffered to minimize I/O overhead.

---

## 14) Compatibility

- **[ORCH-3847]** Library MUST work on Rust stable (current - 2 versions).
- **[ORCH-3848]** CLI tool MUST support Rust nightly for `--format json`.
- **[ORCH-3849]** Fallback to stable parsing MUST work when nightly unavailable.
- **[ORCH-3850]** Platform support: Linux (required), macOS (best effort), Windows (unsupported for CLI).

---

**End of Specification**

**Status**: ✅ READY FOR IMPLEMENTATION  
**Version**: 0.2.0  
**Next Review**: After Phase 5 completion
