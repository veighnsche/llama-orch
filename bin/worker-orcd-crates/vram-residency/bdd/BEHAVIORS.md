# VRAM Residency — Behavior Catalog

**Purpose**: Document all behaviors tested via BDD  
**Status**: Initial implementation  
**Last Updated**: 2025-10-02

---

## 1. Seal Model Behaviors

### 1.1 Valid Seal Operations

**Behavior**: Successfully seal model in VRAM  
**Feature**: `seal_model.feature:7`  
**Given**: VramManager with 10MB capacity, model with 1MB data  
**When**: Seal model with valid shard_id  
**Then**: Seal succeeds, shard has correct metadata, audit event emitted

**Behavior**: Accept model at exact capacity  
**Feature**: `seal_model.feature:33`  
**Given**: VramManager with 10MB capacity, model with 10MB data  
**When**: Seal model  
**Then**: Seal succeeds (no off-by-one errors)

---

### 1.2 Input Validation Behaviors

**Behavior**: Reject path traversal in shard_id  
**Feature**: `seal_model.feature:21`  
**Given**: Shard ID contains "../"  
**When**: Attempt to seal model  
**Then**: Seal fails with InvalidInput, no audit event

**Behavior**: Reject null byte in shard_id  
**Feature**: `seal_model.feature:27`  
**Given**: Shard ID contains null byte  
**When**: Attempt to seal model  
**Then**: Seal fails with InvalidInput, no audit event

---

### 1.3 Capacity Management Behaviors

**Behavior**: Fail on insufficient VRAM  
**Feature**: `seal_model.feature:33`  
**Given**: Model size exceeds available VRAM  
**When**: Attempt to seal model  
**Then**: Seal fails with InsufficientVram, error shows needed/available

**Behavior**: Reject oversized model  
**Feature**: `seal_model.feature:39`  
**Given**: Model exceeds max size (100GB)  
**When**: Attempt to seal model  
**Then**: Seal fails with InvalidInput

---

## 2. Seal Verification Behaviors

### 2.1 Valid Verification

**Behavior**: Verify valid seal  
**Feature**: `verify_seal.feature:7`  
**Given**: Sealed shard with valid digest  
**When**: Verify seal  
**Then**: Verification succeeds, audit event emitted

---

### 2.2 Tampering Detection

**Behavior**: Reject tampered digest  
**Feature**: `verify_seal.feature:12`  
**Given**: Shard digest modified after sealing  
**When**: Verify seal  
**Then**: Verification fails, critical audit event emitted

**Behavior**: Reject forged signature  
**Feature**: `verify_seal.feature:19`  
**Given**: Shard signature replaced with zeros  
**When**: Verify seal  
**Then**: Verification fails, critical audit event emitted

---

## 3. VRAM-Only Policy Behaviors

### 3.1 Policy Enforcement

**Behavior**: Enforce VRAM-only policy at initialization  
**Feature**: `vram_policy.feature:6`  
**Given**: VramManager initialized  
**When**: Enforce VRAM-only policy  
**Then**: UMA/zero-copy/pinned memory disabled

---

### 3.2 Policy Violation Detection

**Behavior**: Detect policy violation  
**Feature**: `vram_policy.feature:15`  
**Given**: Unified memory enabled  
**When**: Enforce VRAM-only policy  
**Then**: Policy enforcement fails, worker stops, audit event emitted

---

## 4. Behavior Coverage Matrix

| Behavior Category | Scenarios | Implemented | Status |
|-------------------|-----------|-------------|--------|
| **Seal Operations** | 6 | 6 | ✅ COMPLETE |
| **Seal Verification** | 3 | 3 | ✅ COMPLETE |
| **Policy Enforcement** | 2 | 2 | 🚧 PARTIAL (pending CUDA) |
| **Input Validation** | 2 | 2 | ✅ COMPLETE |
| **Capacity Management** | 2 | 2 | ✅ COMPLETE |

**Total**: 15 scenarios defined

---

## 5. Behavior Traceability

### 5.1 Requirements to Behaviors

| Requirement | Behavior | Feature |
|-------------|----------|---------|
| WORKER-4110 | Successfully seal model | `seal_model.feature:7` |
| WORKER-4111 | VRAM pointer not exposed | (unit test) |
| WORKER-4112 | Sealed flag set | `seal_model.feature:7` |
| WORKER-4113 | SHA-256 digest computed | `seal_model.feature:7` |
| WORKER-4120 | HMAC-SHA256 signature | `verify_seal.feature:19` |
| WORKER-4121 | Verify before Execute | `verify_seal.feature:7` |
| WORKER-4122 | Fail on verification failure | `verify_seal.feature:12` |
| WORKER-4100 | VRAM-only inference | `vram_policy.feature:6` |
| WORKER-4101 | Disable UMA/zero-copy | `vram_policy.feature:6` |
| WORKER-4103 | Fail fast on OOM | `seal_model.feature:33` |

---

## 6. Behavior Patterns

### 6.1 Happy Path Pattern

```gherkin
Given valid preconditions
When I perform valid operation
Then operation succeeds
And audit event emitted
```

**Examples**:
- Successfully seal model
- Verify valid seal

---

### 6.2 Validation Failure Pattern

```gherkin
Given invalid input
When I attempt operation
Then operation fails with InvalidInput
And no audit event emitted
```

**Examples**:
- Reject path traversal
- Reject null byte

---

### 6.3 Resource Exhaustion Pattern

```gherkin
Given insufficient resources
When I attempt operation
Then operation fails with InsufficientVram
And error shows needed/available
```

**Examples**:
- Fail on insufficient VRAM

---

### 6.4 Security Violation Pattern

```gherkin
Given tampered data
When I verify integrity
Then verification fails
And critical audit event emitted
```

**Examples**:
- Reject tampered digest
- Reject forged signature

---

## 7. Behavior Evolution

### 7.1 Phase 1: Core Behaviors (Current)
- ✅ Seal model operations
- ✅ Basic verification
- ✅ Input validation
- ✅ Capacity management

### 7.2 Phase 2: Security Behaviors (Next)
- ⬜ HMAC signature verification
- ⬜ Timing-safe comparison
- ⬜ Seal key management
- ⬜ Audit logging integration

### 7.3 Phase 3: CUDA Behaviors (Post-M0)
- ⬜ Real VRAM allocation
- ⬜ UMA detection
- ⬜ Zero-copy detection
- ⬜ Multi-GPU coordination

---

## 8. Refinement Opportunities

### 8.1 Additional Behaviors to Test

- Concurrent seal operations
- Seal timestamp freshness
- Digest re-verification timing
- Multi-shard coordination
- CUDA error handling
- Drop behavior (resource cleanup)

### 8.2 Behavior Documentation Improvements

- Add behavior diagrams
- Document failure modes
- Add performance expectations
- Document recovery procedures

---

**Status**: 15 behaviors defined, 13 fully testable, 2 pending CUDA integration
