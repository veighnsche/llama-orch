# PB-003: V2 Formatters and Templates (Zero-Boilerplate API)

**Status**: ✅ NORMATIVE (REQUIRED)  
**Owner**: @llama-orch-proof-bundle-team  
**Date**: 2025-10-02  
**Version**: 0.2.0

---

## Purpose & Scope

Define the V2 API for proof-bundle that provides **zero-boilerplate proof bundle generation** with **human-readable reports** for management audit.

This spec is normative for:
- Formatter functions (executive, developer, failure reports)
- Template system (unit, BDD)
- One-liner `generate_for_crate()` API
- Report format requirements

Out of scope:
- V1 API (being deprecated, see PB-004)
- Test execution details (handled internally)
- CI/CD integration (separate concern)

---

## Motivation

**V1 Problem**: The `capture_tests()` API failed when called from the same package (recursion issue), making it unusable for 100% of use cases.

**V2 Solution**: 
1. **Formatters** — Human-readable reports (executive, developer, failure)
2. **Templates** — Standard patterns for unit and BDD tests
3. **One-liner API** — `generate_for_crate()` handles everything internally

**Management Requirements**:
- JSON/NDJSON not auditable by non-technical stakeholders
- Code duplication across crates unacceptable
- Developers need zero-boilerplate solution

---

## Normative Requirements

### PB-3001: Formatter Functions

The proof-bundle crate MUST provide three formatter functions in the `formatters` module:

```rust
pub fn generate_executive_summary(summary: &TestSummary) -> String;
pub fn generate_test_report(summary: &TestSummary) -> String;
pub fn generate_failure_report(summary: &TestSummary) -> String;
```

**Requirements**:
- ✅ Pure functions (no I/O, no side effects)
- ✅ Return markdown strings
- ✅ Handle empty/zero test cases gracefully

### PB-3002: Executive Summary Format

Executive summaries MUST include:

1. **Pass rate and confidence level**
   - HIGH: ≥ 98% pass rate
   - MEDIUM: 95-98% pass rate  
   - LOW: < 95% pass rate

2. **Risk assessment**
   - LOW: 0 failures
   - MEDIUM: 1-2 failures
   - HIGH: > 2 failures

3. **Quick facts**
   - Total tests executed
   - Passed count and percentage
   - Failed count and percentage
   - Skipped count
   - Total duration

4. **Failed test summaries** (if any)
   - Simplified error messages (non-technical)
   - Impact assessment
   - Action items

5. **Deployment recommendation**
   - ✅ APPROVED FOR DEPLOYMENT (0 failures, ≥98% pass)
   - ⚠️ APPROVED FOR STAGING (≤2 failures, ≥95% pass)
   - ❌ NOT APPROVED (>2 failures or <95% pass)

**Language Requirements**:
- ✅ Use business language, not technical jargon
- ✅ Focus on impact and risk
- ✅ Provide actionable recommendations
- ❌ Do NOT include stack traces or technical details

### PB-3003: Developer Report Format

Developer reports MUST include:

1. **Summary statistics**
   - Total, passed, failed, ignored counts
   - Pass rate percentage
   - Total duration

2. **Test breakdown by category**
   - Unit Tests
   - Integration Tests
   - Property Tests
   - BDD Tests
   - Stress Tests
   - Benchmarks
   - (Auto-detected from test names)

3. **Failed test details** (if any)
   - Test name and location
   - Duration
   - Full error message with code blocks
   - Context (assertion, panic, timeout, etc.)

4. **Performance metrics**
   - Top 10 slowest tests
   - Duration in seconds for each

**Language Requirements**:
- ✅ Technical details appropriate
- ✅ Include code locations
- ✅ Show full error messages
- ✅ Performance data included

### PB-3004: Failure Report Format

Failure reports MUST include (when failures exist):

1. **Per-failure sections**
   - Module and test name
   - Duration
   - Full error message (code block)
   - Context hints (assertion/panic/timeout analysis)
   - Reproduction command

2. **Recommendations**
   - Run individually with --nocapture
   - Check consistency vs intermittent
   - Review recent changes
   - Add logging if unclear

**When no failures**:
- MUST indicate "NO FAILURES"
- MUST state "All tests passed successfully"

### PB-3005: Template System

The proof-bundle crate MUST provide a `templates` module with:

```rust
pub fn unit_test_template() -> ProofBundleTemplate;
pub fn bdd_test_template() -> ProofBundleTemplate;
```

**ProofBundleTemplate MUST include**:
- Supported modes (e.g., ["fast", "full"] or ["mock", "real"])
- Output file list (test_results.ndjson, summary.json, etc.)
- Formatter functions to use
- Feature flags to enable

### PB-3006: Unit Test Template

The unit test template MUST define:

**Modes**:
- `fast` — With `--features skip-long-tests`
- `full` — All tests

**Output Files**:
- `test_results.ndjson` — Raw test data
- `summary.json` — Statistics
- `test_report.md` — Developer report
- `executive_summary.md` — Management report
- `failures.md` — Failure details (if failures exist)

**Directory Structure**:
- `.proof_bundle/unit-fast/<timestamp>/` — Fast mode
- `.proof_bundle/unit-full/<timestamp>/` — Full mode

### PB-3007: BDD Test Template

The BDD test template MUST define:

**Modes**:
- `mock` — Mocked dependencies
- `real` — Real GPU/CUDA

**Output Files**:
- `scenarios.ndjson` — Scenario results
- `features.json` — Feature statistics
- `bdd_report.md` — BDD-specific report
- `executive_summary.md` — Management report

**Directory Structure**:
- `.proof_bundle/bdd-mock/<timestamp>/` — Mock mode
- `.proof_bundle/bdd-real/<timestamp>/` — Real mode

### PB-3008: One-Liner API (DEVELOPER EXPERIENCE)

The proof-bundle crate MUST provide:

```rust
impl ProofBundle {
    pub fn generate_for_crate(
        crate_name: &str,
        mode: ProofBundleMode,
    ) -> Result<TestSummary>;
}

pub enum ProofBundleMode {
    UnitFast,
    UnitFull,
    BddMock,
    BddReal,
}
```

**This function MUST**:
1. ✅ Run cargo test with appropriate flags
2. ✅ Parse output (JSON or stable)
3. ✅ Apply template based on mode
4. ✅ Generate all report files using formatters
5. ✅ Write to correct directory
6. ✅ Return summary for verification

**Usage Example**:
```rust
#[test]
fn generate_proof_bundle() -> anyhow::Result<()> {
    proof_bundle::ProofBundle::generate_for_crate(
        "my-crate",
        ProofBundleMode::UnitFast,
    )
}
```

**MUST be ≤ 5 lines** for crate usage.

### PB-3009: Helper Functions

The formatters module MUST provide internal helpers:

```rust
fn categorize_tests(tests: &[TestResult]) -> HashMap<String, Vec<TestResult>>;
fn simplify_error(error: &str) -> String;
```

**categorize_tests** MUST detect:
- Unit tests (default)
- Property tests (name contains "property" or "proptest")
- Integration tests (name contains "integration" or "e2e")
- BDD tests (name contains "bdd" or "scenario")
- Stress tests (name contains "stress" or "load")
- Benchmarks (name contains "bench")

**simplify_error** MUST convert:
- "assertion" → "Test expectation not met"
- "panicked" → "Test encountered unexpected condition"
- "timeout" → "Test exceeded time limit"
- "memory"/"OOM" → "Memory allocation issue"
- Other → First line, max 80 chars

### PB-3010: Error Handling

All V2 APIs MUST:
- ✅ Return `anyhow::Result`
- ✅ Provide actionable error messages
- ✅ Include: what went wrong, why, how to fix
- ✅ Generate partial proof bundles on failure (best effort)
- ❌ NEVER panic on expected conditions

### PB-3011: Performance Requirements

V2 APIs MUST:
- ✅ Add ≤ 5% overhead to test execution time
- ✅ Format 1000 tests in < 1 second
- ✅ Use buffered I/O for file writes

### PB-3012: Security & Privacy

Proof bundle reports MUST:
- ✅ Redact secrets/API keys from error messages
- ✅ Redact sensitive data from stack traces
- ❌ NEVER include personal information (except developer names)
- ✅ Be safe to commit to version control

### PB-3013: Compatibility

V2 MUST:
- ✅ Work on Rust stable (current - 2 versions)
- ✅ Support nightly for `--format json`
- ✅ Fall back to stable output parsing when nightly unavailable
- ✅ Work on Linux (required)
- ⚠️ Best effort on macOS
- ❌ Windows unsupported for CLI

---

## Verification Tests

The following tests MUST exist:

### Formatter Tests

- **PBV-3001**: `generate_executive_summary` produces valid markdown with all required sections
- **PBV-3002**: Executive summary includes risk assessment (LOW/MEDIUM/HIGH)
- **PBV-3003**: Executive summary uses non-technical language
- **PBV-3004**: `generate_test_report` includes test breakdown by category
- **PBV-3005**: `generate_test_report` includes top 10 slowest tests
- **PBV-3006**: `generate_failure_report` includes reproduction commands
- **PBV-3007**: `generate_failure_report` shows "NO FAILURES" when all pass

### Template Tests

- **PBV-3010**: `unit_test_template` defines both fast and full modes
- **PBV-3011**: `bdd_test_template` defines both mock and real modes
- **PBV-3012**: Templates specify correct output files
- **PBV-3013**: Templates specify correct formatters

### Helper Tests

- **PBV-3020**: `categorize_tests` correctly identifies test types
- **PBV-3021**: `simplify_error` converts technical errors to business language

### Integration Tests

- **PBV-3030**: `generate_for_crate` generates all 4 report types
- **PBV-3031**: `generate_for_crate` writes to correct directory based on mode
- **PBV-3032**: `generate_for_crate` returns accurate TestSummary
- **PBV-3033**: Proof bundle crate generates its own perfect proof bundle (dogfooding)

---

## Examples

### Example: Crate Using V2 API

```rust
// In any crate's tests/proof_bundle.rs

#[test]
fn generate_proof_bundle_fast() -> anyhow::Result<()> {
    proof_bundle::ProofBundle::generate_for_crate(
        "my-crate",
        proof_bundle::ProofBundleMode::UnitFast,
    )
}

#[test]
fn generate_proof_bundle_full() -> anyhow::Result<()> {
    proof_bundle::ProofBundle::generate_for_crate(
        "my-crate",
        proof_bundle::ProofBundleMode::UnitFull,
    )
}
```

**That's it!** No formatters, no templates, no boilerplate.

### Example: Executive Summary Output

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
   - Issue: Test exceeded time limit
   - **Impact**: LOW (rare edge case)
   - **Action**: Engineering review in progress

2. **test_large_input_oom** (non-critical)
   - Issue: Memory allocation issue
   - **Impact**: MEDIUM (affects large deployments)
   - **Action**: Engineering review in progress

## Recommendation

**⚠️ APPROVED FOR STAGING** — Minor issues tracked, review recommended
```

---

## Migration from V1

### V1 API (Deprecated)

```rust
// ❌ DON'T USE - Being deprecated
let pb = ProofBundle::for_type(TestType::Unit)?;
let summary = pb.capture_tests("my-crate")
    .lib()
    .tests()
    .run()?;
```

**Problem**: Fails with 0 tests when called from same package.

### V2 API (Use This)

```rust
// ✅ USE THIS
proof_bundle::ProofBundle::generate_for_crate(
    "my-crate",
    ProofBundleMode::UnitFast,
)?;
```

**Fix**: Handles execution internally, no recursion issues.

---

## Refinement Opportunities

1. **Code coverage integration** — Parse coverage data, include in reports
2. **Performance tracking** — Store historical durations, detect regressions
3. **Filtering support** — Capture specific tests by name pattern
4. **Workspace-level bundles** — Aggregate across all crates
5. **CI/CD helpers** — GitHub Actions/GitLab CI templates
6. **Interactive HTML** — Generate static HTML reports with graphs
7. **Test impact analysis** — Correlate code changes with test results
8. **Flaky test detection** — Track stability over time
9. **Compliance reporting** — Map tests to requirements (ORCH-IDs)

---

## Cross-References

- **PB-001**: Proof bundle location policy (crate-local)
- **PB-002**: Always generate bundles (pass AND fail)
- **PB-004**: V1 deprecation plan (TBD)
- **00_proof-bundle.md**: V1 API specification (PB-1xxx)
- **REDESIGN_V2.md**: Complete V2 redesign rationale
- **MANAGEMENT_REQUIREMENTS_ADDRESSED.md**: Management feedback addressed
- **.specs/75-proof-bundle.md**: Root-level specification (ORCH-38xx)

---

**Status**: ✅ IN PROGRESS (Phase 1.1 complete - formatters)  
**Next**: Phase 1.2 (templates module)  
**Target**: 2025-10-09 (1 week)
