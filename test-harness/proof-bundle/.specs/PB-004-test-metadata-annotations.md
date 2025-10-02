# PB-004: Test Metadata Annotations (Custom Proof Bundle Content)

**Status**: 🔵 PROPOSED  
**Owner**: @llama-orch-proof-bundle-team  
**Date**: 2025-10-02  
**Version**: 0.2.0

---

## Purpose & Scope

Allow teams to add **custom metadata** to their tests that gets captured and displayed in proof bundles. This enables:

- Tagging tests with priority/criticality
- Linking tests to requirements/specs
- Documenting known issues
- Identifying test owners
- Adding business context
- Custom categorization

**In scope**:
- Metadata annotation syntax
- Metadata extraction from tests
- Metadata display in reports
- API for programmatic metadata

**Out of scope**:
- Proc macros (future enhancement)
- Test execution filtering based on metadata
- Metadata validation (future enhancement)

---

## Motivation

**Management Request**: Teams need to annotate tests with business context that appears in proof bundles.

**Use Cases**:

1. **Priority/Criticality**
   ```rust
   /// @priority: critical
   /// @spec: ORCH-3250
   #[test]
   fn test_queue_invariants() { }
   ```
   → Proof bundle shows: "Critical test - ORCH-3250"

2. **Requirements Tracing**
   ```rust
   /// @requirement: REQ-AUTH-001
   /// @compliance: SOC2
   #[test]
   fn test_api_key_validation() { }
   ```
   → Proof bundle shows: "Compliance: SOC2, Requirement: REQ-AUTH-001"

3. **Known Issues**
   ```rust
   /// @known-issue: #1234
   /// @flaky: 5% failure rate
   #[test]
   fn test_network_retry() { }
   ```
   → Proof bundle shows: "Known issue #1234, intermittent failures"

4. **Ownership**
   ```rust
   /// @team: core-infra
   /// @owner: alice@example.com
   #[test]
   fn test_deployment() { }
   ```
   → Proof bundle shows: "Owner: core-infra team"

---

## Proposed Solutions

### Option A: Doc Comment Annotations (RECOMMENDED)

**Syntax**: Use special markers in doc comments

```rust
/// Test description here.
///
/// @priority: critical
/// @spec: ORCH-3250
/// @team: orchestrator
#[test]
fn test_something() {
    // test body
}
```

**Pros**:
- ✅ No proc macros needed
- ✅ Works today with stable Rust
- ✅ Readable in source code
- ✅ Standard Rust syntax

**Cons**:
- ⚠️ Requires parsing test source files
- ⚠️ Not enforced by compiler

### Option B: Companion Metadata Files

**Syntax**: Create `.meta.toml` files alongside tests

```toml
# tests/queue_tests.meta.toml

[test.test_queue_invariants]
priority = "critical"
spec = "ORCH-3250"
team = "orchestrator"

[test.test_queue_ordering]
priority = "high"
spec = "ORCH-3251"
```

**Pros**:
- ✅ Structured data (TOML)
- ✅ Easy to parse
- ✅ No source parsing needed

**Cons**:
- ⚠️ Separate files to maintain
- ⚠️ Can get out of sync

### Option C: Test Name Conventions

**Syntax**: Embed metadata in test names

```rust
#[test]
fn critical_orch3250_test_queue_invariants() {
    // priority: critical, spec: ORCH-3250
}
```

**Pros**:
- ✅ No extra files
- ✅ Easy to parse

**Cons**:
- ❌ Ugly test names
- ❌ Limited metadata

### Option D: Programmatic API

**Syntax**: Use proof-bundle API in tests

```rust
#[test]
fn test_queue_invariants() -> anyhow::Result<()> {
    proof_bundle::test_metadata()
        .priority("critical")
        .spec("ORCH-3250")
        .team("orchestrator")
        .record()?;
    
    // actual test
    assert!(true);
    Ok(())
}
```

**Pros**:
- ✅ Type-safe
- ✅ Compiler-enforced
- ✅ Easy to implement

**Cons**:
- ⚠️ Requires code in every test
- ⚠️ More boilerplate

---

## Recommended Approach: Hybrid (A + D)

**Use doc comments for simple metadata** (most tests):
```rust
/// @priority: critical
/// @spec: ORCH-3250
#[test]
fn test_queue_invariants() { }
```

**Use programmatic API for complex metadata** (when needed):
```rust
#[test]
fn test_queue_invariants() -> anyhow::Result<()> {
    proof_bundle::test_metadata()
        .priority("critical")
        .spec("ORCH-3250")
        .related_tests(&["test_queue_ordering", "test_queue_limits"])
        .record()?;
    
    // test logic
}
```

---

## Normative Requirements

### PB-4001: Standard Metadata Fields

The proof-bundle crate MUST recognize these standard fields:

| Field | Type | Example | Purpose |
|-------|------|---------|---------|
| `@priority` | enum | `critical`, `high`, `medium`, `low` | Test importance |
| `@spec` | string | `ORCH-3250` | Spec/requirement ID |
| `@team` | string | `orchestrator` | Owning team |
| `@owner` | email | `alice@example.com` | Responsible person |
| `@issue` | string | `#1234` | Related issue tracker |
| `@flaky` | string | `5% failure rate` | Known flakiness |
| `@timeout` | duration | `30s` | Expected max duration |
| `@requires` | list | `gpu, cuda` | Required resources |
| `@tags` | list | `integration, slow` | Custom tags |

### PB-4002: Doc Comment Parsing

When doc comments are used:

1. Parser MUST look for lines starting with `@` in doc comments
2. Format: `/// @key: value`
3. Multi-line values: Use `\` continuation or YAML-style
4. Location: Anywhere in test doc comment

**Example**:
```rust
/// Test description.
///
/// @priority: critical
/// @spec: ORCH-3250
/// @tags: integration, slow
/// @requires: gpu, cuda
#[test]
fn test_something() { }
```

### PB-4003: Programmatic API

The proof-bundle crate MUST provide:

```rust
pub fn test_metadata() -> TestMetadataBuilder;

impl TestMetadataBuilder {
    pub fn priority(self, level: &str) -> Self;
    pub fn spec(self, id: &str) -> Self;
    pub fn team(self, name: &str) -> Self;
    pub fn owner(self, email: &str) -> Self;
    pub fn issue(self, id: &str) -> Self;
    pub fn flaky(self, description: &str) -> Self;
    pub fn timeout(self, duration: &str) -> Self;
    pub fn requires(self, resources: &[&str]) -> Self;
    pub fn tags(self, tags: &[&str]) -> Self;
    pub fn custom(self, key: &str, value: &str) -> Self;
    pub fn record(self) -> Result<()>;
}
```

### PB-4004: Metadata in Reports

**Executive Summary** MUST include:
- Count of critical tests
- Count of failed critical tests (highlight)

**Developer Report** MUST include:
- Metadata for each test in failure section
- Filterable by metadata (future)

**New: Metadata Report** (`metadata.md`):
```markdown
# Test Metadata Report

## By Priority

### Critical (15 tests)
- test_queue_invariants (ORCH-3250) ✅
- test_seal_verification (ORCH-3275) ✅
- test_auth_validation (ORCH-3254) ❌ **FAILED**

### High (42 tests)
...

## By Spec

### ORCH-3250 (Queue Invariants)
- test_queue_invariants ✅
- test_queue_ordering ✅
- test_queue_limits ✅

## By Team

### orchestrator (35 tests)
- 34 passed, 1 failed

### pool-managerd (28 tests)
- 28 passed
```

### PB-4005: Metadata Storage

Metadata MUST be stored in:
- `test_metadata.json` — Structured JSON for machines
- `metadata.md` — Human-readable markdown

**Format** (`test_metadata.json`):
```json
{
  "tests": [
    {
      "name": "test_queue_invariants",
      "metadata": {
        "priority": "critical",
        "spec": "ORCH-3250",
        "team": "orchestrator",
        "status": "passed"
      }
    }
  ]
}
```

### PB-4006: Custom Fields

Teams MAY add custom fields:
```rust
/// @custom:deployment-stage: staging
/// @custom:security-level: 3
#[test]
fn test_something() { }
```

**Namespace**: Prefix custom fields with `custom:` to avoid conflicts.

### PB-4007: Validation (Optional)

Proof-bundle MAY validate:
- Priority is one of: critical, high, medium, low
- Spec IDs match pattern: ORCH-\d+
- Email format for owners
- (Validation is optional, non-blocking)

---

## Implementation Plan

### Phase 1: Programmatic API (Immediate)

1. Add `TestMetadataBuilder` to proof-bundle
2. Store metadata in thread-local or global
3. Write to `test_metadata.json` on bundle creation
4. Add to test reports

**Timeline**: 2 days

### Phase 2: Doc Comment Parsing (Week 2)

1. Add source file parsing
2. Extract `@` annotations from doc comments
3. Merge with programmatic metadata
4. Add to reports

**Timeline**: 3 days

### Phase 3: Metadata Report (Week 2)

1. Generate `metadata.md`
2. Add to executive summary
3. Highlight critical test failures

**Timeline**: 2 days

### Phase 4: Proc Macro (Future)

```rust
#[test]
#[metadata(priority = "critical", spec = "ORCH-3250")]
fn test_something() { }
```

**Timeline**: Future (requires proc macro crate)

---

## Examples

### Example 1: Simple Annotation

```rust
/// Validates queue ordering under load.
///
/// @priority: critical
/// @spec: ORCH-3250
/// @team: orchestrator
#[test]
fn test_queue_ordering() {
    // test body
}
```

**In proof bundle**:
```markdown
### test_queue_ordering ✅

**Priority**: CRITICAL  
**Spec**: ORCH-3250  
**Team**: orchestrator

Test validates queue ordering under load.
```

### Example 2: Known Flaky Test

```rust
/// @priority: high
/// @flaky: 5% timeout rate
/// @issue: #1234
/// @timeout: 30s
#[test]
fn test_network_retry() {
    // test body
}
```

**In proof bundle**:
```markdown
### test_network_retry ✅ (Flaky)

**Priority**: HIGH  
**Known Issue**: #1234  
**Flakiness**: 5% timeout rate  
**Expected Timeout**: 30s

⚠️ This test is known to be intermittently flaky.
```

### Example 3: Compliance Test

```rust
/// @priority: critical
/// @compliance: SOC2, GDPR
/// @requirement: REQ-AUTH-001
/// @team: security
#[test]
fn test_api_key_validation() -> anyhow::Result<()> {
    proof_bundle::test_metadata()
        .custom("audit-log", "required")
        .custom("retention", "7-years")
        .record()?;
    
    // test logic
}
```

**In proof bundle**:
```markdown
### test_api_key_validation ✅

**Priority**: CRITICAL  
**Compliance**: SOC2, GDPR  
**Requirement**: REQ-AUTH-001  
**Team**: security  
**Custom**:
- audit-log: required
- retention: 7-years

🔒 Compliance-critical test
```

---

## Benefits

1. **Requirements Tracing** — Link tests to specs/requirements
2. **Risk Assessment** — Identify critical test failures
3. **Ownership** — Clear team/owner for each test
4. **Compliance** — Track compliance-related tests
5. **Debugging** — Known issues documented inline
6. **Reporting** — Better management summaries

---

## Refinement Opportunities

1. **Proc macro attributes** — `#[metadata(priority = "critical")]`
2. **Metadata validation** — Enforce required fields, check formats
3. **Filtering** — Run only tests with specific metadata
4. **IDE integration** — Show metadata in test runners
5. **CI integration** — Block PRs if critical tests fail
6. **Compliance reporting** — Generate compliance matrices
7. **Test impact analysis** — Link code changes to affected specs
8. **Historical tracking** — Track metadata changes over time
9. **Schema validation** — JSON schema for custom fields

---

## Cross-References

- **PB-001**: Proof bundle location policy
- **PB-002**: Always generate bundles
- **PB-003**: V2 formatters and templates
- **ORCH-3200 series**: Testing ownership requirements
- **ORCH-38xx series**: Proof-bundle system specs

---

**Status**: ✅ APPROVED - IMPLEMENT IMMEDIATELY  
**Implementation**: Starting Phase 1 (in V2)  
**Priority**: 🔴 CRITICAL (Management priority - wanted yesterday)  
**Estimated Effort**: 1 week (3 phases)  
**Integration**: Part of V2 release (not separate)
