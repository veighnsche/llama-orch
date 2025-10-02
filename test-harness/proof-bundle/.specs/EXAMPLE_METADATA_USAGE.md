# Example: Test Metadata Usage

**Purpose**: Show how teams would use PB-004 test metadata annotations

---

## Simple Example (Doc Comments)

```rust
// tests/queue_tests.rs

/// Validates queue maintains FIFO ordering under load.
///
/// @priority: critical
/// @spec: ORCH-3250
/// @team: orchestrator
#[test]
fn test_queue_ordering() {
    let queue = Queue::new();
    // test logic...
    assert!(queue.is_ordered());
}

/// Validates queue capacity limits.
///
/// @priority: high
/// @spec: ORCH-3251
/// @team: orchestrator
/// @timeout: 30s
#[test]
fn test_queue_limits() {
    // test logic...
}
```

**Result in Proof Bundle**:
```markdown
## Failed Tests

### test_queue_ordering ❌ **CRITICAL**

**Spec**: ORCH-3250  
**Team**: orchestrator  
**Priority**: CRITICAL

⚠️ CRITICAL TEST FAILURE - Requires immediate attention

**Error**: assertion failed: queue.is_ordered()
```

---

## Advanced Example (Programmatic API)

```rust
// tests/security_tests.rs

/// @priority: critical
/// @compliance: SOC2, GDPR
/// @requirement: REQ-AUTH-001
#[test]
fn test_api_key_validation() -> anyhow::Result<()> {
    // Record additional metadata programmatically
    proof_bundle::test_metadata()
        .team("security")
        .owner("security-team@example.com")
        .requires(&["redis", "postgres"])
        .custom("audit-log", "required")
        .custom("retention", "7-years")
        .record()?;
    
    // Actual test logic
    let key = generate_api_key();
    assert!(validate_key(&key));
    
    Ok(())
}
```

**Result in Proof Bundle**:

**metadata.md**:
```markdown
# Test Metadata Report

## Critical Tests (15)

### ✅ Passing (14)
- test_queue_ordering (ORCH-3250)
- test_seal_verification (ORCH-3275)
- test_api_key_validation (REQ-AUTH-001) 🔒
- ...

### ❌ **FAILED (1)**
- **test_auth_token_expiry** (REQ-AUTH-002) ⚠️ **CRITICAL FAILURE**

## Compliance Tests

### SOC2 (8 tests)
- ✅ test_api_key_validation
- ✅ test_audit_logging
- ❌ **test_auth_token_expiry** ⚠️
- ...

### GDPR (5 tests)
- ✅ test_data_deletion
- ✅ test_data_export
- ...
```

**executive_summary.md**:
```markdown
# Test Results Summary

**Status**: ⚠️ 97% PASS RATE  
**Confidence**: MEDIUM

## ⚠️ CRITICAL ALERT

**1 CRITICAL TEST FAILURE**:
- test_auth_token_expiry (REQ-AUTH-002)
- Compliance: SOC2
- Impact: HIGH - Security requirement

## Compliance Status

- ✅ GDPR: 5/5 passing (100%)
- ⚠️ SOC2: 7/8 passing (87.5%) - **1 FAILURE**

## Recommendation

**❌ NOT APPROVED** — Critical security test failure requires resolution
```

---

## Known Flaky Test

```rust
/// Network retry logic with exponential backoff.
///
/// @priority: medium
/// @spec: ORCH-3280
/// @flaky: 5% timeout rate on slow CI
/// @issue: #1234
/// @timeout: 60s
#[test]
#[ignore] // Run explicitly
fn test_network_retry() {
    // test logic...
}
```

**Result in Proof Bundle**:
```markdown
### test_network_retry ✅ (Known Flaky)

**Priority**: MEDIUM  
**Spec**: ORCH-3280  
**Known Issue**: #1234  
**Flakiness**: 5% timeout rate on slow CI  
**Expected Timeout**: 60s  

⚠️ This test is known to be intermittently flaky. See issue #1234.
```

---

## Multiple Specs

```rust
/// Validates placement feasibility predicate.
///
/// @priority: critical
/// @spec: ORCH-3251, ORCH-3252, ORCH-3253
/// @team: orchestrator
/// @tags: placement, algorithm, critical-path
#[test]
fn test_placement_feasibility() {
    // test logic...
}
```

**Result in Proof Bundle**:

**metadata.md**:
```markdown
## By Spec

### ORCH-3251 (Placement Feasibility)
- ✅ test_placement_feasibility

### ORCH-3252 (Deterministic Tie-break)
- ✅ test_placement_feasibility
- ✅ test_tie_break_ordering

### ORCH-3253 (Metrics Labeling)
- ✅ test_placement_feasibility
- ✅ test_metrics_labels
```

---

## Custom Fields for Team-Specific Needs

```rust
/// @priority: high
/// @spec: ORCH-3269
/// @custom:deployment-stage: canary
/// @custom:rollout-percentage: 5%
/// @custom:monitoring-dashboard: grafana/llama-cpp-provisioner
/// @custom:alert-channel: #alerts-provisioner
#[test]
fn test_llama_cpp_provisioning() -> anyhow::Result<()> {
    proof_bundle::test_metadata()
        .team("provisioners")
        .custom("sla", "99.9%")
        .custom("oncall-rotation", "https://oncall.example.com/provisioners")
        .record()?;
    
    // test logic...
}
```

**Result in Proof Bundle**:
```markdown
### test_llama_cpp_provisioning ✅

**Priority**: HIGH  
**Spec**: ORCH-3269  
**Team**: provisioners

**Custom Metadata**:
- deployment-stage: canary
- rollout-percentage: 5%
- monitoring-dashboard: grafana/llama-cpp-provisioner
- alert-channel: #alerts-provisioner
- sla: 99.9%
- oncall-rotation: https://oncall.example.com/provisioners

📊 Monitoring: [Dashboard](grafana/llama-cpp-provisioner)  
🚨 Alerts: #alerts-provisioner
```

---

## Benefits for Teams

### 1. Requirements Tracing

Link every test to requirements:
```rust
/// @spec: ORCH-3250
/// @requirement: REQ-QUEUE-001
#[test]
fn test_queue_invariants() { }
```

→ Generate traceability matrix for compliance

### 2. Priority-Based Reporting

Highlight critical failures in executive summaries:
```
⚠️ **2 CRITICAL TEST FAILURES** (out of 5 critical tests)
```

→ Management sees critical issues immediately

### 3. Team Ownership

Clear responsibility:
```rust
/// @team: pool-managerd
/// @owner: bob@example.com
#[test]
fn test_pool_health() { }
```

→ Know who to contact for failures

### 4. Resource Requirements

Document dependencies:
```rust
/// @requires: gpu, cuda, 16gb-vram
#[test]
fn test_large_model() { }
```

→ CI can skip tests when resources unavailable

### 5. Known Issues

Document flakiness:
```rust
/// @flaky: 3% rate
/// @issue: #1234
#[test]
fn test_network_timeout() { }
```

→ Don't panic when flaky tests fail

---

## Migration Path

### Week 1: Add metadata to critical tests

```rust
/// @priority: critical
/// @spec: ORCH-XXXX
#[test]
fn critical_test() { }
```

### Week 2: Add team/owner metadata

```rust
/// @priority: critical
/// @spec: ORCH-XXXX
/// @team: my-team
#[test]
fn critical_test() { }
```

### Week 3: Document flaky tests

```rust
/// @flaky: known issue
/// @issue: #1234
#[test]
fn flaky_test() { }
```

### Week 4: Full compliance metadata

```rust
/// @compliance: SOC2
/// @requirement: REQ-XXX
/// @priority: critical
#[test]
fn compliance_test() { }
```

---

**This is how teams would use the metadata feature!**
