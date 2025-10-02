# Proof Bundle Team Responsibilities

**Team**: proof-bundle  
**Scope**: Repository-wide test quality and audit evidence  
**Status**: Active

---

## Mission

**Ensure all crates generate honest, objective, information-rich, human-auditable proof bundles.**

---

## Core Responsibilities

### 1. Test Coverage Auditing

**Responsibility**: Audit test coverage across all crates and identify gaps.

**Actions**:
- ✅ Review test code (not just claims)
- ✅ Identify missing tests
- ✅ Identify insufficient proof bundle coverage
- ✅ Document gaps in audit reports
- ✅ Recommend improvements

**Deliverables**:
- `<crate>/.proof_bundle/TESTING_CODE_AUDIT.md` — Real code audit
- `<crate>/.proof_bundle/AUDIT_FINDINGS_PROOF_BUNDLE_GAPS.md` — Gap analysis
- `<crate>/.proof_bundle/IMPLEMENTATION_GUIDE.md` — How to fix gaps

**Example**: vram-residency audit found:
- ❌ Only 6/173 tests captured (3.5% coverage)
- ❌ NO BDD proof bundles (10 features, 40 scenarios exist but no evidence)
- ⚠️ Missing integration tests
- ⚠️ No performance regression tests

---

### 2. Proof Bundle Quality Enforcement

**Responsibility**: Ensure proof bundles are information-rich and human-auditable.

**Quality Criteria**:
- ✅ **Complete**: ALL tests captured (not just passing)
- ✅ **Honest**: Failures documented openly
- ✅ **Detailed**: Rich explanations, not just "✅ PASS"
- ✅ **Provenance**: Git SHA, environment, timestamps
- ✅ **Traceable**: Maps to requirements/specs
- ✅ **Auditable**: Human can verify claims

**Actions**:
- ✅ Review proof bundle contents
- ✅ Identify skimpy or missing evidence
- ✅ Request improvements
- ✅ Block PRs with insufficient evidence

**Example Issues**:
- ❌ "✅ PASS" without explanation
- ❌ No error details on failure
- ❌ No performance measurements
- ❌ No compliance mapping
- ❌ No threat model coverage

---

### 3. Specification Maintenance

**Responsibility**: Maintain normative specifications for proof bundle system.

**Specifications**:
- ✅ PB-001: Proof Bundle Location Policy (crate-local required)
- ✅ PB-002: Always Generate Bundles (pass AND fail)
- ⚠️ PB-003: Proof Bundle Content Requirements (TODO)
- ⚠️ PB-004: Proof Bundle Naming Conventions (TODO)
- ⚠️ PB-005: Proof Bundle Security (TODO)

**Actions**:
- ✅ Write and maintain specs
- ✅ Ensure specs have "Refinement Opportunities" sections
- ✅ Update specs based on learnings
- ✅ Enforce specs across crates

---

### 4. Cross-Crate Consistency

**Responsibility**: Ensure consistent proof bundle practices across all crates.

**Actions**:
- ✅ Standardize directory structure
- ✅ Standardize file formats (NDJSON, JSON, Markdown)
- ✅ Standardize naming conventions
- ✅ Standardize metadata fields
- ✅ Provide templates and examples

**Deliverables**:
- `/.proof_bundle/templates/` — Templates for each test type
- `test-harness/proof-bundle/README.md` — Library documentation
- Implementation guides per crate

---

### 5. Human Auditor Advocacy

**Responsibility**: Ensure proof bundles serve human auditors, not just machines.

**Principles**:
- ✅ **Explain WHY**, not just WHAT
- ✅ **Provide context** for non-experts
- ✅ **Map to threat models** and compliance requirements
- ✅ **Show provenance** (reproducibility)
- ✅ **Be transparent** about failures

**Bad Example**:
```
✅ PASS: Model sealed successfully
```

**Good Example**:
```
## Test: Model Sealing (Cryptographic Integrity)

**Purpose**: Verify that model data loaded into VRAM is cryptographically
sealed with HMAC-SHA256 to prevent tampering.

**Security Property**: Integrity protection (TIER 1 requirement)

**Threat Model Coverage**:
- ✅ Prevents model corruption attacks
- ✅ Detects unauthorized model modifications
- ✅ Ensures inference uses verified model weights

**Test Execution**:
1. Created 1MB test model (1,000,000 bytes)
2. Loaded into VRAM on GPU device 0
3. Computed SHA-256 digest: a1b2c3d4...
4. Generated HMAC-SHA256 seal: e5f6g7h8...
5. Verified seal signature matches

**Result**: ✅ PASS

**Evidence**:
- Shard ID: abc123-def456-ghi789
- VRAM allocation: 1,000,000 bytes
- GPU device: 0 (RTX 3090)
- Digest: a1b2c3d4e5f6g7h8... (64 hex chars)
- Signature: e5f6g7h8i9j0k1l2... (64 hex chars)
- Sealed at: 2025-10-02T10:18:33Z
- Verification: SUCCESS (timing-safe comparison)

**Performance**:
- Seal operation: 1.8ms
- Verification: 0.3ms

**Compliance**:
- ✅ GDPR Art. 32 (Integrity protection)
- ✅ SOC2 CC6.1 (Cryptographic controls)
- ✅ ISO 27001 A.10.1.1 (Cryptographic policy)
```

---

### 6. Continuous Improvement

**Responsibility**: Identify and implement improvements to proof bundle system.

**Actions**:
- ✅ Learn from audit findings
- ✅ Update specs and templates
- ✅ Improve tooling and automation
- ✅ Share best practices
- ✅ Mentor crate teams

**Refinement Opportunities** (tracked in specs):
- Automated verification
- Performance regression detection
- Failure pattern analysis
- Root cause databases

---

### 7. Lead by Example: Extensive Testing

**Responsibility**: Set the standard for comprehensive testing across the repository.

**Principle**: **The proof-bundle team must practice what it preaches.**

**Requirements**:
- ✅ **100% test coverage** for all proof-bundle features
- ✅ **Extensive unit tests** (not just happy path)
- ✅ **Edge case testing** (error conditions, boundary values)
- ✅ **Integration tests** (real-world usage scenarios)
- ✅ **Documentation tests** (examples must compile and run)

**Example**: `capture_tests()` feature testing:
- ✅ 30+ unit tests covering:
  - Builder pattern (chaining, all methods)
  - Type serialization/deserialization
  - Pass rate calculations
  - Error handling
  - Edge cases (empty features, zero threads, large values)
  - Trait implementations (Debug, Clone, Copy, Eq)
  - File generation verification
  - Multiple builders independence

**Why This Matters**:
1. **Credibility**: We can't audit others if our own code is untested
2. **Trust**: Teams trust our tools when they see comprehensive tests
3. **Quality**: Extensive tests catch bugs before they reach users
4. **Documentation**: Tests serve as executable examples
5. **Leadership**: We model the behavior we want to see

**Anti-Pattern** (What NOT to do):
```rust
// ❌ BAD: Minimal testing
#[test]
fn test_capture_tests() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    pb.capture_tests("test").run().unwrap();
}
```

**Best Practice** (What TO do):
```rust
// ✅ GOOD: Comprehensive testing
#[test]
fn test_capture_builder_creation() { /* ... */ }

#[test]
fn test_capture_builder_chaining() { /* ... */ }

#[test]
fn test_capture_builder_all() { /* ... */ }

#[test]
fn test_test_status_serialization() { /* ... */ }

#[test]
fn test_test_status_equality() { /* ... */ }

#[test]
fn test_test_summary_pass_rate_calculation() { /* ... */ }

// ... 25+ more tests covering all aspects
```

**Metrics**:
- ✅ Test count: > 30 tests per major feature
- ✅ Line coverage: > 90%
- ✅ Branch coverage: > 85%
- ✅ Edge case coverage: All error paths tested

**Deliverables**:
- `tests/test_capture_tests.rs` — Comprehensive unit tests for `capture_tests()`
- `tests/test_proof_bundle.rs` — Core functionality tests
- `tests/integration_tests.rs` — Real-world usage tests

---

### 8. Developer Experience First (MANAGEMENT DIRECTIVE)

**Responsibility**: Make proof bundle generation effortless for developers.

**Principle**: **Developers don't want to test. We make it so easy they don't have to think about it.**

**Reality**:
- Developers HATE writing tests
- Developers HATE writing boilerplate
- Developers HATE maintaining proof bundle code
- Developers just want tests to "work"

**Our Solution**: **Zero-boilerplate, one-line API**

```rust
// This is ALL developers should write:
#[test]
fn generate_proof_bundle() -> anyhow::Result<()> {
    proof_bundle::ProofBundle::generate_for_crate(
        "my-crate",
        ProofBundleMode::UnitFast,
    )
}
```

**What We Handle (So Developers Don't Have To)**:

1. **✅ Test execution** — We run cargo test
2. **✅ Output parsing** — We parse JSON or stable output
3. **✅ Formatting** — We generate beautiful reports
4. **✅ File writing** — We write all proof bundle files
5. **✅ Error handling** — We provide clear error messages
6. **✅ Templates** — We provide standard templates
7. **✅ Human-readable reports** — We format for management

**Code Patterns We Prevent**:

```rust
// ❌ Developers should NEVER write this:
let output = Command::new("cargo")
    .arg("test")
    .arg("-p").arg("my-crate")
    .output()?;

let stdout = String::from_utf8_lossy(&output.stdout);
for line in stdout.lines() {
    if let Ok(json) = serde_json::from_str::<Value>(line) {
        // ... 50 lines of parsing
    }
}

let report = format!("# Test Report\n\n...");  // 100 lines of formatting
pb.write_markdown("test_report.md", &report)?;
```

```rust
// ✅ Developers SHOULD write this:
proof_bundle::ProofBundle::generate_for_crate("my-crate", ProofBundleMode::UnitFast)?;
```

**Repository Context**:

In this repository, developers mainly write:
1. **Unit tests** — Standard Rust `#[test]` functions
2. **BDD tests** — Cucumber-style feature files

**We provide templates for both**:
- `templates::unit_test_template()` — For unit/integration tests
- `templates::bdd_test_template()` — For BDD/cucumber tests

**Example: Perfect Proof Bundle**

The `proof-bundle` crate itself must demonstrate the **perfect proof bundle**:

```
test-harness/proof-bundle/.proof_bundle/unit/<timestamp>/
├── test_results.ndjson           # Raw data (machines)
├── summary.json                  # Statistics (CI/CD)
├── test_report.md                # Technical (developers)
├── executive_summary.md          # Business (management)
└── failures.md                   # Diagnostics (if any)
```

**Each file must be exemplary**:
- Executive summary: Non-technical, actionable, risk-focused
- Test report: Detailed, organized, performance metrics
- Failures: Context, reproduction steps, related code

**Deliverables**:
1. ✅ **One-liner API** — `generate_for_crate()`
2. ✅ **Templates** — Unit and BDD templates
3. ✅ **Formatters** — Executive, developer, failure reports
4. ✅ **Examples** — Perfect proof bundle in our own crate
5. ✅ **Documentation** — Copy-paste examples for other teams

**Success Metrics**:
- ✅ Other crates use ≤ 5 lines of code for proof bundles
- ✅ Zero duplicate formatting code across crates
- ✅ Management can read executive summaries without technical knowledge
- ✅ Developers spend < 5 minutes adding proof bundles to their crate

**Why This Matters**:
1. **Adoption**: Easy tools get used, complex tools get ignored
2. **Consistency**: Templates ensure uniform quality
3. **Maintenance**: Less code = less to maintain
4. **Compliance**: Easier to enforce standards
5. **Morale**: Developers hate busywork; we eliminate it

---

## Authority

### What We Can Do

1. ✅ **Audit any crate's tests** and proof bundles
2. ✅ **Request improvements** to test coverage or proof bundle quality
3. ✅ **Block PRs** with insufficient test evidence
4. ✅ **Create specifications** for proof bundle system
5. ✅ **Enforce specifications** across all crates

### What We Cannot Do

1. ❌ **Write tests for other crates** (we audit, they implement)
2. ❌ **Override crate-specific test decisions** (we advise, they decide)
3. ❌ **Change test frameworks** without crate team agreement

---

## Workflow

### 1. Audit Request

**Trigger**: User requests audit, or periodic review

**Process**:
1. Read actual test code (not claims)
2. Count tests (unit, property, BDD, etc.)
3. Check proof bundle coverage
4. Identify gaps

**Deliverable**: `TESTING_CODE_AUDIT.md`

---

### 2. Gap Analysis

**Process**:
1. Document what's missing
2. Explain why it matters
3. Estimate effort to fix
4. Prioritize gaps (P1/P2/P3)

**Deliverable**: `AUDIT_FINDINGS_PROOF_BUNDLE_GAPS.md`

---

### 3. Implementation Guide

**Process**:
1. Provide code examples
2. Show before/after
3. Explain best practices
4. Link to specs

**Deliverable**: `IMPLEMENTATION_GUIDE.md`

---

### 4. Follow-Up

**Process**:
1. Review improvements
2. Verify gaps addressed
3. Update audit status
4. Share learnings

---

## Communication

### With Crate Teams

**Tone**: Helpful, not punitive

**Approach**:
- ✅ "We found these gaps..." (objective)
- ✅ "Here's why it matters..." (educational)
- ✅ "Here's how to fix it..." (actionable)
- ❌ "Your tests are bad" (judgmental)

**Example**:
> "We audited vram-residency and found 173 tests exist but only 6 are captured in proof bundles (3.5% coverage). This means human auditors can't verify the other 167 tests actually ran. We've created an implementation guide showing how to capture all tests using the proof-bundle library. Estimated effort: 8-12 hours."

---

### With Human Auditors

**Tone**: Transparent and honest

**Approach**:
- ✅ Show what's proven
- ✅ Show what's missing
- ✅ Explain limitations
- ✅ Provide evidence trails

**Example**:
> "vram-residency claims 173 tests passing, but only 6 are captured in proof bundles. We cannot verify the other 167 tests. BDD tests exist (10 features, 40 scenarios) but no BDD proof bundles are generated. Recommendation: Address gaps before production deployment."

---

## Metrics

### Coverage Metrics

- **Test capture rate**: Tests captured / Total tests
- **Proof bundle completeness**: Required files present / Required files
- **Evidence richness**: Lines of evidence / Test

**Targets**:
- ✅ Test capture rate: > 95%
- ✅ Proof bundle completeness: 100%
- ✅ Evidence richness: > 10 lines/test

### Quality Metrics

- **Failure documentation rate**: Failures documented / Total failures
- **Explanation quality**: Human-readable explanations present
- **Provenance completeness**: Git SHA, environment, timestamps present

**Targets**:
- ✅ Failure documentation: 100%
- ✅ Explanation quality: All tests have context
- ✅ Provenance: 100%

---

## Escalation

### When to Escalate

1. ⚠️ Crate team refuses to address critical gaps
2. ⚠️ Proof bundles consistently missing or low-quality
3. ⚠️ Security-critical crate has insufficient test coverage
4. ⚠️ Repeated violations of PB-001 or PB-002 policies

### Escalation Path

1. **Level 1**: Document in audit report, request improvements
2. **Level 2**: Block PR merge, require fixes
3. **Level 3**: Escalate to project lead
4. **Level 4**: Mark crate as "not production-ready"

---

## Tools and Resources

### Specifications

- `test-harness/proof-bundle/.specs/` — Normative specs
- `test-harness/proof-bundle/README.md` — Library docs

### Templates

- `/.proof_bundle/templates/` — Templates for each test type

### Audit Tools

- `grep_search` — Find tests in code
- `cargo test --format json` — Capture test results
- Manual code review — Real gap analysis

---

## Success Criteria

### For Proof Bundle Team

- ✅ All crates have > 95% test capture rate
- ✅ All crates generate proof bundles on pass AND fail
- ✅ All proof bundles are human-auditable
- ✅ Zero security-critical crates with insufficient coverage
- ✅ Specifications maintained and enforced

### For Repository

- ✅ High auditor confidence across all crates
- ✅ Transparent test evidence
- ✅ Rapid root cause analysis on failures
- ✅ Compliance-ready proof bundles

---

## Current Status

### Completed

- ✅ PB-001: Proof Bundle Location Policy (normative)
- ✅ PB-002: Always Generate Bundles (normative)
- ✅ vram-residency audit (TESTING_CODE_AUDIT.md)
- ✅ vram-residency gap analysis (AUDIT_FINDINGS_PROOF_BUNDLE_GAPS.md)
- ✅ vram-residency implementation guide (IMPLEMENTATION_GUIDE.md)
- ✅ Proof bundle library (async-safe, crate-local)

### In Progress

- ⚠️ vram-residency improvements (BDD proof bundles, full unit test capture)

### Pending

- ⚠️ Audit other crates (pool-managerd, orchestratord, etc.)
- ⚠️ PB-003: Content Requirements spec
- ⚠️ PB-004: Naming Conventions spec
- ⚠️ PB-005: Security spec
- ⚠️ Automated verification tooling

---

## Contact

**Team**: proof-bundle  
**Location**: `test-harness/proof-bundle/`  
**Specs**: `test-harness/proof-bundle/.specs/`  
**Audits**: `<crate>/.proof_bundle/TESTING_CODE_AUDIT.md`

---

**Last Updated**: 2025-10-02  
**Next Review**: Monthly
