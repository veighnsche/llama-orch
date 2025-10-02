# VRAM Residency — Security Verification Matrix

**Purpose**: Proof that all attack surfaces are closed with tests and mitigations  
**Status**: 🚧 IN PROGRESS - Implementation Phase  
**Last Updated**: 2025-10-02

---

## Executive Summary

🚧 **IMPLEMENTATION IN PROGRESS**  
📋 **33 SECURITY REQUIREMENTS IDENTIFIED**  
⬜ **0/33 SECURITY REQUIREMENTS IMPLEMENTED**  
📋 **7 ATTACK SURFACES MAPPED**  
⬜ **0/7 ATTACK SURFACES CLOSED**  
⬜ **0% TEST COVERAGE**

**Production Status**: 🚧 Not ready - requires implementation of all security requirements

---

## 1. Attack Surface Coverage Matrix

### 1.1 VRAM Pointer Leakage Attack Surface ⬜ OPEN

| Attack Type | Requirement | Mitigation | Unit Test | BDD Test | Status |
|-------------|-------------|------------|-----------|----------|--------|
| Pointer in API Response | MS-001, WORKER-4111 | Private `vram_ptr` field | `test_vram_ptr_not_exposed()` | N/A | ⬜ PLANNED |
| Pointer in JSON Serialization | MS-001 | `#[serde(skip)]` on vram_ptr | `test_vram_ptr_not_in_json()` | N/A | ⬜ PLANNED |
| Pointer in Debug Output | MS-001 | Custom Debug impl | `test_vram_ptr_not_in_debug()` | N/A | ⬜ PLANNED |
| Pointer in Error Messages | WORKER-4152 | Generic error messages | `test_errors_dont_expose_pointers()` | N/A | ⬜ PLANNED |
| Pointer in Logs | MS-001 | Only shard_id logged | Code review | N/A | ⬜ PLANNED |

**Verification**: ⬜ 0/5 pointer leakage vectors closed

**Implementation Plan**:
- `src/types.rs`: Make `vram_ptr` private, add `#[serde(skip)]`
- `src/types.rs`: Custom `Debug` impl that omits pointer
- `src/error.rs`: Ensure no pointer values in error messages
- `tests/security_tests.rs`: Add all 5 test cases

---

### 1.2 Seal Forgery Attack Surface ⬜ OPEN

| Attack Type | Requirement | Mitigation | Unit Test | BDD Test | Status |
|-------------|-------------|------------|-----------|----------|--------|
| Forged Signature | CI-001, WORKER-4120 | HMAC-SHA256 seal | `test_seal_forgery_rejected()` | `verify_seal.feature:18` | ⬜ PLANNED |
| Modified Digest | CI-004, WORKER-4113 | SHA-256 verification | `test_tampered_digest_rejected()` | `verify_seal.feature:13` | ⬜ PLANNED |
| Replayed Seal | CI-002 | Timestamp in signature | `test_seal_replay_rejected()` | N/A | ⬜ PLANNED |
| Seal Key Exposure | CI-006 | `secrets-management` integration | `test_seal_key_not_logged()` | N/A | ⬜ PLANNED |
| Timing Attack on Verification | CI-003 | `subtle::ConstantTimeEq` | `test_timing_safe_verification()` | N/A | ⬜ PLANNED |

**Verification**: ⬜ 0/5 seal forgery vectors closed

**Implementation Plan**:
- Add `hmac = "0.12"` dependency
- Add `secrets-management` dependency
- `src/seal.rs`: Implement HMAC-SHA256 signature computation
- `src/seal.rs`: Implement timing-safe signature verification
- `tests/security_tests.rs`: Add all 5 test cases
- `bdd/features/verify_seal.feature`: Add BDD scenarios

---

### 1.3 Digest TOCTOU Attack Surface ⬜ OPEN

| Attack Type | Requirement | Mitigation | Unit Test | BDD Test | Status |
|-------------|-------------|------------|-----------|----------|--------|
| VRAM Modification After Seal | CI-007, WORKER-4121 | Re-verify before Execute | `test_digest_reverification()` | N/A | ⬜ PLANNED |
| Stale Digest | CI-007 | Fresh digest computation | `test_stale_digest_rejected()` | N/A | ⬜ PLANNED |
| Concurrent Modification | CI-007 | Digest mismatch detection | `test_concurrent_modification_detected()` | N/A | ⬜ PLANNED |

**Verification**: ⬜ 0/3 TOCTOU vectors closed

**Implementation Plan**:
- `src/manager.rs`: Add `verify_sealed()` that re-computes digest
- `src/manager.rs`: Call `verify_sealed()` before every Execute
- `tests/security_tests.rs`: Add TOCTOU test cases

---

### 1.4 CUDA FFI Buffer Overflow Attack Surface ⬜ OPEN

| Attack Type | Requirement | Mitigation | Unit Test | Status |
|-------------|-------------|------------|-----------|--------|
| Out-of-Bounds Write | MS-002, MS-007 | Bounds checking wrapper | `test_bounds_checking()` | ⬜ PLANNED |
| Unchecked cudaMemcpy | MS-002 | Safe wrapper | `test_safe_memcpy()` | ⬜ PLANNED |
| Invalid Pointer Arithmetic | MS-006 | Checked arithmetic | `test_pointer_arithmetic_safe()` | ⬜ PLANNED |
| Allocation Size Overflow | MS-003 | Saturating add | `test_allocation_overflow_prevented()` | ⬜ PLANNED |

**Verification**: ⬜ 0/4 buffer overflow vectors closed

**Implementation Plan**:
- `src/cuda_ffi.rs`: Implement `SafeVramPtr` wrapper
- `src/cuda_ffi.rs`: Add bounds checking to all operations
- `src/cuda_ffi.rs`: Use checked/saturating arithmetic
- `tests/security_tests.rs`: Add buffer overflow test cases

---

### 1.5 Integer Overflow Attack Surface ✅ PARTIALLY CLOSED

| Attack Type | Requirement | Mitigation | Unit Test | Status |
|-------------|-------------|------------|-----------|--------|
| VRAM Allocation Overflow | MS-003, RP-002 | Saturating arithmetic | `test_integer_overflow_prevented()` | ✅ IMPLEMENTED |
| Capacity Calculation Overflow | RP-002 | `saturating_add()` | Code review | ✅ IMPLEMENTED |
| Size Parameter Overflow | IV-001 | Max size validation | `test_max_size_enforced()` | ⬜ PLANNED |
| Offset Overflow | MS-007 | `checked_add()` | `test_offset_overflow_prevented()` | ⬜ PLANNED |

**Verification**: ✅ 2/4 integer overflow vectors closed

**Current Implementation**:
- `src/lib.rs:180`: `self.used_vram.saturating_add(model_bytes.len())`
- `src/lib.rs:183`: `self.total_vram.saturating_sub(self.used_vram)`

**Remaining Work**:
- Add max size validation (100GB default per WORKER-4170)
- Add offset overflow checks in CUDA operations
- Add test cases for size and offset validation

---

### 1.6 VRAM-Only Policy Bypass Attack Surface ⬜ OPEN

| Attack Type | Requirement | Mitigation | Unit Test | Status |
|-------------|-------------|------------|-----------|--------|
| Unified Memory Enabled | VP-002, WORKER-4101 | Disable UMA at init | `test_uma_detection()` | ⬜ PLANNED |
| Zero-Copy Enabled | VP-003, WORKER-4101 | Disable zero-copy | `test_zero_copy_disabled()` | ⬜ PLANNED |
| Pinned Host Memory | VP-003, WORKER-4101 | Disable pinned memory | `test_pinned_memory_disabled()` | ⬜ PLANNED |
| RAM Fallback | VP-005, WORKER-4103 | Fail fast on OOM | `test_ram_fallback_rejected()` | ⬜ PLANNED |
| Policy Verification Bypass | VP-006 | Cryptographic attestation | `test_policy_attestation()` | ⬜ PLANNED |

**Verification**: ⬜ 0/5 policy bypass vectors closed

**Implementation Plan**:
- `src/policy.rs`: Implement `enforce_vram_only_policy()`
- `src/policy.rs`: Add UMA/zero-copy/pinned memory detection
- `src/policy.rs`: Add CUDA device property queries
- `tests/security_tests.rs`: Add policy enforcement test cases
- Emit `AuditEvent::PolicyViolation` on detection

---

### 1.7 Panic/DoS Attack Surface ⬜ OPEN

| Attack Type | Requirement | Mitigation | Unit Test | Clippy Lint | Status |
|-------------|-------------|------------|-----------|-------------|--------|
| Panic via `.unwrap()` | MS-005 | No unwrap in code | Code review | `deny(clippy::unwrap_used)` | ✅ ENFORCED |
| Panic via `.expect()` | MS-005 | No expect in code | Code review | `deny(clippy::expect_used)` | ✅ ENFORCED |
| Panic via `panic!()` | MS-005 | No panic in code | Code review | `deny(clippy::panic)` | ✅ ENFORCED |
| Panic via indexing `[]` | MS-005 | Use `.get()` | Code review | `deny(clippy::indexing_slicing)` | ✅ ENFORCED |
| Panic via integer overflow | MS-003 | Saturating arithmetic | Code review | `deny(clippy::integer_arithmetic)` | ✅ ENFORCED |
| Panic in Drop | MS-004, MS-005 | Graceful error handling | `test_drop_never_panics()` | Manual review | ⬜ PLANNED |
| DoS via huge allocation | RP-003 | Max model size limit | `test_max_model_size_enforced()` | N/A | ⬜ PLANNED |

**Verification**: ✅ 5/7 panic vectors enforced via Clippy, 2 require implementation

**Current Implementation**:
- `src/lib.rs:94-114`: TIER 1 Clippy lints enforced

**Remaining Work**:
- Implement `Drop` for `SealedShard` with error handling
- Add max model size validation (100GB default)
- Add test cases for Drop and size limits

---

## 2. Security Requirements Compliance Matrix

### 2.1 Memory Safety (MS-001 to MS-007) ⬜ 1/7 COMPLIANT

| Requirement | Implementation | Verification | Status |
|-------------|----------------|--------------|--------|
| MS-001: Private VRAM pointers | `vram_ptr` field is private | `test_vram_ptr_not_exposed()` | ✅ IMPLEMENTED |
| MS-002: Safe CUDA FFI wrappers | `SafeVramPtr` wrapper | `test_safe_cuda_operations()` | ⬜ PLANNED |
| MS-003: Integer overflow prevention | Saturating arithmetic | Code review | ✅ IMPLEMENTED |
| MS-004: Graceful deallocation | Error handling in Drop | `test_drop_never_panics()` | ⬜ PLANNED |
| MS-005: No-panic Drop | Result-based error handling | Clippy + manual review | ⬜ PLANNED |
| MS-006: Checked pointer arithmetic | `checked_add()` usage | `test_pointer_arithmetic()` | ⬜ PLANNED |
| MS-007: Bounds validation | Offset + length checks | `test_bounds_checking()` | ⬜ PLANNED |

---

### 2.2 Cryptographic Integrity (CI-001 to CI-007) ⬜ 0/7 COMPLIANT

| Requirement | Implementation | Verification | Status |
|-------------|----------------|--------------|--------|
| CI-001: HMAC-SHA256 signatures | `hmac` crate integration | `test_seal_signature()` | ⬜ PLANNED |
| CI-002: Signature coverage | `(shard_id, digest, sealed_at, gpu_device)` | `test_signature_coverage()` | ⬜ PLANNED |
| CI-003: Timing-safe comparison | `subtle::ConstantTimeEq` | `test_timing_safe_verification()` | ⬜ PLANNED |
| CI-004: SHA-256 digests | `sha2` crate (FIPS 140-2) | Code review | ✅ IMPLEMENTED |
| CI-005: Seal key derivation | `secrets-management` integration | `test_key_derivation()` | ⬜ PLANNED |
| CI-006: Key zeroization | `secrets-management` handles | Code review | ⬜ PLANNED |
| CI-007: Digest re-verification | Before each Execute | `test_digest_reverification()` | ⬜ PLANNED |

---

### 2.3 VRAM-Only Policy (VP-001 to VP-006) ⬜ 0/6 COMPLIANT

| Requirement | Implementation | Verification | Status |
|-------------|----------------|--------------|--------|
| VP-001: VRAM-only inference | Policy enforcement at init | `test_vram_only_enforced()` | ⬜ PLANNED |
| VP-002: Disable UMA | `cudaDeviceSetLimit()` call | `test_uma_disabled()` | ⬜ PLANNED |
| VP-003: Disable zero-copy | Device property validation | `test_zero_copy_disabled()` | ⬜ PLANNED |
| VP-004: Fail fast on OOM | `InsufficientVram` error | `test_insufficient_vram()` | ✅ IMPLEMENTED |
| VP-005: Detect RAM inference | Memory type validation | `test_ram_inference_rejected()` | ⬜ PLANNED |
| VP-006: Cryptographic attestation | Seal signature proves residency | `test_attestation()` | ⬜ PLANNED |

---

### 2.4 Input Validation (IV-001 to IV-005) ⬜ 0/5 COMPLIANT

| Requirement | Implementation | Verification | Status |
|-------------|----------------|--------------|--------|
| IV-001: Model size validation | Max 100GB check | `test_max_model_size()` | ⬜ PLANNED |
| IV-002: GPU device validation | `input-validation::validate_range()` | `test_gpu_device_validation()` | ⬜ PLANNED |
| IV-003: Shard ID validation | `input-validation::validate_identifier()` | `test_shard_id_validation()` | ⬜ PLANNED |
| IV-004: Digest validation | `input-validation::validate_hex_string()` | `test_digest_validation()` | ⬜ PLANNED |
| IV-005: Null byte checking | Built into validators | `test_null_byte_rejection()` | ⬜ PLANNED |

---

### 2.5 Resource Protection (RP-001 to RP-005) ⬜ 2/5 COMPLIANT

| Requirement | Implementation | Verification | Status |
|-------------|----------------|--------------|--------|
| RP-001: Capacity enforcement | Capacity check before allocation | `test_capacity_enforcement()` | ✅ IMPLEMENTED |
| RP-002: Overflow prevention | Saturating arithmetic | Code review | ✅ IMPLEMENTED |
| RP-003: Configurable limits | `VramConfig::max_model_size` | `test_configurable_limits()` | ⬜ PLANNED |
| RP-004: Actionable errors | `InsufficientVram(needed, available)` | Code review | ✅ IMPLEMENTED |
| RP-005: Audit trail | `audit-logging` integration | `test_audit_trail()` | ⬜ PLANNED |

---

### 2.6 Error Handling (WORKER-4150 to WORKER-4153) ✅ 4/4 COMPLIANT

| Requirement | Implementation | Verification | Status |
|-------------|----------------|--------------|--------|
| WORKER-4150: Result types | All ops return `Result<T, VramError>` | Code review | ✅ IMPLEMENTED |
| WORKER-4151: Error classification | Retriable vs fatal distinction | `error.rs` | ✅ IMPLEMENTED |
| WORKER-4152: No sensitive data in errors | Generic error messages | Code review | ✅ IMPLEMENTED |
| WORKER-4153: Actionable diagnostics | `InsufficientVram(needed, available)` | Code review | ✅ IMPLEMENTED |

---

### 2.7 Audit Requirements (WORKER-4160 to WORKER-4163) ⬜ 0/4 COMPLIANT

| Requirement | Implementation | Verification | Status |
|-------------|----------------|--------------|--------|
| WORKER-4160: Seal operations | `AuditEvent::VramSealed` | `test_seal_audited()` | ⬜ PLANNED |
| WORKER-4161: Verification failures | `AuditEvent::SealVerificationFailed` | `test_verification_failure_audited()` | ⬜ PLANNED |
| WORKER-4162: Deallocation | `AuditEvent::VramDeallocated` | `test_deallocation_audited()` | ⬜ PLANNED |
| WORKER-4163: Policy violations | `AuditEvent::PolicyViolation` | `test_policy_violation_audited()` | ⬜ PLANNED |

---

## 3. Test Coverage Matrix

### 3.1 Unit Tests ⬜ 0/35 IMPLEMENTED

| Test Category | Test Count | Implemented | Status |
|---------------|------------|-------------|--------|
| **Pointer Leakage** | 5 | 0 | ⬜ PLANNED |
| **Seal Forgery** | 5 | 0 | ⬜ PLANNED |
| **TOCTOU** | 3 | 0 | ⬜ PLANNED |
| **Buffer Overflow** | 4 | 0 | ⬜ PLANNED |
| **Integer Overflow** | 4 | 1 | 🚧 PARTIAL |
| **Policy Bypass** | 5 | 0 | ⬜ PLANNED |
| **Panic/DoS** | 7 | 0 | ⬜ PLANNED |
| **Input Validation** | 5 | 0 | ⬜ PLANNED |
| **Audit Logging** | 4 | 0 | ⬜ PLANNED |

**Total**: 1/42 tests implemented (2.4%)

---

### 3.2 BDD Tests ⬜ 0/3 IMPLEMENTED

| Feature File | Scenarios | Implemented | Status |
|--------------|-----------|-------------|--------|
| `seal_model.feature` | 3 | 0 | ⬜ PLANNED |
| `verify_seal.feature` | 3 | 0 | ⬜ PLANNED |
| `vram_policy.feature` | 2 | 0 | ⬜ PLANNED |

**Total**: 0/8 BDD scenarios implemented

---

### 3.3 Property Tests ⬜ 0/3 IMPLEMENTED

| Property | Test | Status |
|----------|------|--------|
| Seal determinism | `seal_verification_deterministic()` | ⬜ PLANNED |
| Capacity never exceeded | `vram_allocation_never_exceeds_capacity()` | ⬜ PLANNED |
| Modified data rejected | `seal_verification_rejects_modified_data()` | ⬜ PLANNED |

**Total**: 0/3 property tests implemented

---

## 4. Dependency Security Verification

### 4.1 Cryptographic Dependencies ⬜ NOT ADDED

| Crate | Version | Purpose | Audit Status | Status |
|-------|---------|---------|--------------|--------|
| `hmac` | 0.12 | HMAC-SHA256 signatures | ✅ RustCrypto audited | ⬜ NOT ADDED |
| `sha2` | 0.10 | SHA-256 digests | ✅ RustCrypto audited | ✅ ADDED |
| `subtle` | 2.5 | Timing-safe comparison | ✅ RustCrypto audited | ⬜ NOT ADDED (via secrets-management) |

**Action Required**: Add `hmac` to `Cargo.toml`

---

### 4.2 Shared Security Crates ⬜ NOT INTEGRATED

| Crate | Purpose | Security Tier | Status |
|-------|---------|---------------|--------|
| `input-validation` | Input sanitization | TIER 2 | ⬜ NOT ADDED |
| `secrets-management` | Key management | TIER 1 | ⬜ NOT ADDED |
| `audit-logging` | Security audit trail | TIER 1 | ⬜ NOT ADDED |

**Action Required**: Add all three shared crates to `Cargo.toml`

---

## 5. Implementation Checklist

### 5.1 Phase 1: Core Security (Priority P0)

- [ ] Add `hmac`, `input-validation`, `secrets-management`, `audit-logging` dependencies
- [ ] Implement HMAC-SHA256 seal signature computation
- [ ] Implement timing-safe seal verification
- [ ] Integrate `secrets-management` for seal key derivation
- [ ] Add `#[serde(skip)]` to `vram_ptr` field
- [ ] Implement custom `Debug` that omits pointer
- [ ] Add input validation for shard_id, gpu_device, digest
- [ ] Implement max model size validation (100GB)
- [ ] Add audit logging for all operations

**Estimated effort**: 3-4 days

---

### 5.2 Phase 2: CUDA Integration (Priority P1)

- [ ] Implement `SafeVramPtr` wrapper with bounds checking
- [ ] Add checked arithmetic to all pointer operations
- [ ] Implement `enforce_vram_only_policy()`
- [ ] Add UMA/zero-copy/pinned memory detection
- [ ] Implement digest re-verification from VRAM
- [ ] Add CUDA device property queries
- [ ] Implement graceful Drop with error handling

**Estimated effort**: 2-3 days

---

### 5.3 Phase 3: Testing (Priority P1)

- [ ] Write all 42 unit tests
- [ ] Write 8 BDD scenarios
- [ ] Write 3 property tests
- [ ] Add fuzzing harness
- [ ] Achieve >90% code coverage
- [ ] Run security audit

**Estimated effort**: 3-4 days

---

## 6. Security Gaps Summary

### 6.1 Critical Gaps (Block Production)

1. **No seal signature implementation** → Seal forgery possible
2. **No seal key management** → Cannot derive/store keys securely
3. **No digest re-verification** → TOCTOU attacks possible
4. **No audit logging** → No security trail
5. **No input validation** → Injection attacks possible

---

### 6.2 High Priority Gaps (Security Risk)

6. **No VRAM-only policy enforcement** → RAM fallback possible
7. **No CUDA bounds checking** → Buffer overflow possible
8. **No Drop implementation** → Resource leaks possible
9. **No max size validation** → DoS via huge allocations

---

### 6.3 Medium Priority Gaps (Hardening)

10. **No BDD tests** → Behavior not verified
11. **No property tests** → Invariants not tested
12. **No fuzzing** → Edge cases not covered

---

## 7. Refinement Opportunities

### 7.1 Enhanced Security

- Add hardware-based seal keys (TPM integration)
- Implement seal timestamp freshness validation
- Add periodic digest re-verification (not just on Execute)
- Implement secure VRAM wipe on deallocation

---

### 7.2 Testing Improvements

- Add mutation testing for security-critical code
- Implement chaos testing for CUDA error injection
- Add performance regression tests for seal operations
- Create security test suite with known attack vectors

---

### 7.3 Monitoring & Observability

- Add metrics for seal verification failures
- Implement alerting for policy violations
- Add tracing for all security-critical operations
- Create security dashboard for VRAM operations

---

## 8. References

**Security specifications**:
- `.specs/20_security.md` — Security requirements and threat model
- `.specs/00_vram-residency.md` — Functional specification
- `.specs/30_dependencies.md` — Dependency security analysis

**Related crates**:
- `bin/shared-crates/audit-logging/.specs/21_security_verification.md` — Audit logging verification
- `bin/shared-crates/secrets-management/.specs/21_security_verification.md` — Secrets management verification
- `bin/shared-crates/input-validation/.specs/21_security_verification.md` — Input validation verification

**Security audits**:
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Overall security posture
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Operational security

---

**Status**: 🚧 IN PROGRESS - Requires implementation of all security requirements  
**Next Steps**: Begin Phase 1 implementation (core security features)  
**Blocking Issues**: None - ready to proceed with implementation  
**Target Completion**: 8-11 days for full security implementation
