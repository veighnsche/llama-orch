# Security Alert: GGUF Parsing Vulnerabilities

**Date**: 2025-10-04  
**Severity**: HIGH  
**Component**: worker-orcd GGUF parser (M0)  
**Threat**: Heap overflow via unchecked offsets  
**Status**: 🔴 REQUIRES IMMEDIATE ATTENTION

---

## Executive Summary

The Llama team's research (LT-000) has uncovered **critical security vulnerabilities** in GGUF file parsing that can lead to heap overflows. These vulnerabilities are well-documented in the wild and affect multiple GGUF parser implementations, including llama.cpp.

**Impact**: An attacker with the ability to provide a malicious GGUF file to worker-orcd could:
1. Trigger heap overflow via crafted tensor offsets
2. Cause out-of-bounds memory access
3. Potentially achieve arbitrary code execution
4. Crash the worker process (denial of service)

**Affected Component**: worker-orcd GGUF parser (M0-W-1211)

**Required Action**: Add explicit bounds validation to prevent heap overflows before M0 implementation begins.

---

## Vulnerability Details

### Source of Discovery

**Research Assignment**: LT-000 (GGUF & BPE Spec Study)  
**Research Document**: `REASERCH_pt1.md` line 45  
**External Reference**: [GGUF File Format Vulnerabilities: A Guide for Hackers](https://blog.huntr.com/gguf-file-format-vulnerabilities-a-guide-for-hackers)

### Vulnerability Description

**CWE-119**: Improper Restriction of Operations within the Bounds of a Memory Buffer  
**CWE-787**: Out-of-bounds Write

**Attack Vector**:
```
Attacker provides malicious GGUF file with:
1. Valid magic bytes (0x47475546 "GGUF")
2. Valid version (3)
3. Crafted tensor metadata with:
   - offset = 0xFFFFFFFF (near SIZE_MAX)
   - size = 0x1000 (4KB)
   - offset + size wraps around (integer overflow)
   - OR offset points beyond file bounds
```

**Exploitation**:
```cpp
// VULNERABLE CODE (current M0 spec does not prevent this)
void load_tensor(const GGUFTensor& tensor, const void* mapped_file) {
    // No bounds checking!
    const void* tensor_data = (char*)mapped_file + tensor.offset;  // ❌ Unchecked offset
    cudaMemcpy(device_ptr, tensor_data, tensor.size, cudaMemcpyHostToDevice);  // ❌ Heap overflow
}
```

**Result**: Out-of-bounds read from memory-mapped file, potential heap corruption, worker crash, or arbitrary code execution.

---

## Current M0 Spec Status

### Existing Validation (M0-W-1210)

The M0 spec **already includes basic validation**:
- ✅ File exists and is readable
- ✅ GGUF magic bytes correct (0x47475546)
- ✅ GGUF version supported (version 3)
- ✅ Tensor count reasonable (<10,000)
- ✅ Total size fits in VRAM

### Critical Gap: Missing Bounds Validation

**What's Missing**:
- ❌ Tensor offset bounds checking (offset < file_size)
- ❌ Tensor size bounds checking (offset + size <= file_size)
- ❌ Integer overflow protection (offset + size doesn't wrap)
- ❌ Metadata string length validation
- ❌ Array length validation

**Why This Is Critical**:
- Memory-mapped I/O (M0-W-1221) exposes raw file bytes to CUDA
- Unchecked offsets can read arbitrary memory
- Integer overflows can bypass size checks
- Malicious GGUF files can crash worker or worse

---

## Threat Model

### Attack Scenarios

**Scenario 1: Malicious Model Upload (Platform Mode)**
```
Attacker → Uploads crafted GGUF to platform
         → Orchestrator schedules to worker
         → Worker loads malicious GGUF
         → Heap overflow → Worker crash or RCE
```

**Scenario 2: Supply Chain Attack**
```
Attacker → Compromises Hugging Face model repo
         → Replaces legitimate GGUF with malicious version
         → User downloads and loads model
         → Worker loads malicious GGUF
         → Heap overflow → Worker crash or RCE
```

**Scenario 3: Local Development (Home Mode)**
```
Developer → Downloads untrusted GGUF from internet
          → Tests with worker-orcd
          → Worker loads malicious GGUF
          → Heap overflow → Developer machine compromised
```

### Attacker Capabilities

**Required**:
- Ability to provide GGUF file to worker-orcd
- Knowledge of GGUF file format
- Ability to craft valid magic bytes and version

**Not Required**:
- Network access to worker
- Authentication credentials
- Privileged access

**Difficulty**: LOW (GGUF format is well-documented, tools exist)

### Impact Assessment

| Impact | Severity | Likelihood | Risk |
|--------|----------|------------|------|
| Worker crash (DoS) | Medium | High | HIGH |
| Heap corruption | High | Medium | HIGH |
| Arbitrary code execution | Critical | Low | MEDIUM |
| Data exfiltration | High | Low | MEDIUM |

**Overall Risk**: 🔴 **HIGH** (requires immediate mitigation)

---

## Recommended Mitigation

### New Requirement: M0-W-1211a

Add explicit bounds validation requirement to M0 spec:

```markdown
#### [M0-W-1211a] GGUF Bounds Validation (Security)

Worker-orcd MUST validate all GGUF offsets and sizes to prevent heap overflows.

**Required Checks**:
1. Tensor offset MUST be >= header_size + metadata_size
2. Tensor offset MUST be < file_size
3. Tensor offset + tensor_size MUST be <= file_size
4. Tensor offset + tensor_size MUST NOT overflow (check for integer overflow)
5. Metadata string lengths MUST be < 1MB (sanity check)
6. Array lengths MUST be < 1M elements (sanity check)

**Implementation**:
```cpp
bool validate_tensor_bounds(const GGUFTensor& tensor, size_t file_size, size_t data_start) {
    // Check offset is after metadata
    if (tensor.offset < data_start) {
        tracing::error!("Tensor offset before data section: {}", tensor.offset);
        return false;
    }
    
    // Check offset is within file
    if (tensor.offset >= file_size) {
        tracing::error!("Tensor offset beyond file: {} >= {}", tensor.offset, file_size);
        return false;
    }
    
    // Calculate tensor size
    size_t tensor_size = calculate_tensor_size(tensor);
    
    // Check for integer overflow (offset + size wraps around)
    if (tensor_size > SIZE_MAX - tensor.offset) {
        tracing::error!("Integer overflow: offset={} size={}", tensor.offset, tensor_size);
        return false;
    }
    
    // Check end is within file
    if (tensor.offset + tensor_size > file_size) {
        tracing::error!("Tensor extends beyond file: {}+{} > {}", 
                       tensor.offset, tensor_size, file_size);
        return false;
    }
    
    return true;
}
```

**Error Handling**:
- Log security error with details (offset, size, file_size)
- Reject GGUF file with clear error message
- Exit worker with non-zero code
- DO NOT attempt to load malformed GGUF

**Rationale**: Research (LT-000) identified heap overflow vulnerabilities in GGUF parsers due to unchecked offsets. This validation prevents exploitation.

**Spec Reference**: Security audit from LT-000 research, CWE-119, CWE-787
```

### Story Card Updates

**LT-001: GGUF Header Parser**

Add to acceptance criteria:
```markdown
- [ ] Security: Tensor offset validation (offset >= data_start && offset < file_size)
- [ ] Security: Tensor bounds validation (offset + size <= file_size)
- [ ] Security: Integer overflow protection (offset + size doesn't wrap)
- [ ] Security: Metadata string length validation (< 1MB)
- [ ] Security: Array length validation (< 1M elements)
- [ ] Security: Fuzzing test with malformed GGUF files (invalid offsets, sizes, overflows)
- [ ] Security: Unit test for offset at file boundary (edge case)
- [ ] Security: Unit test for integer overflow (SIZE_MAX - 1 + 2)
```

**LT-005: Pre-Load Validation**

Add to acceptance criteria:
```markdown
- [ ] Security: All tensor offsets validated before mmap
- [ ] Security: All tensor sizes validated before VRAM allocation
- [ ] Security: Clear error messages for malformed GGUF files
- [ ] Security: Worker exits cleanly on validation failure (no partial load)
- [ ] Security: Audit log entry for rejected GGUF files (fingerprint file hash)
```

### Testing Requirements

**1. Fuzzing Tests** (REQUIRED for M0):
```rust
#[test]
fn test_gguf_fuzzing_invalid_offsets() {
    // Test 1: Offset beyond file
    let malicious_gguf = create_gguf_with_offset(file_size + 1);
    assert!(validate_gguf(&malicious_gguf).is_err());
    
    // Test 2: Integer overflow (offset + size wraps)
    let malicious_gguf = create_gguf_with_offset_size(SIZE_MAX - 100, 200);
    assert!(validate_gguf(&malicious_gguf).is_err());
    
    // Test 3: Offset before data section
    let malicious_gguf = create_gguf_with_offset(10);  // Before metadata end
    assert!(validate_gguf(&malicious_gguf).is_err());
    
    // Test 4: Size extends beyond file
    let malicious_gguf = create_gguf_with_offset_size(file_size - 100, 200);
    assert!(validate_gguf(&malicious_gguf).is_err());
}
```

**2. Property Tests** (REQUIRED for M0):
```rust
#[quickcheck]
fn prop_gguf_bounds_always_validated(offset: u64, size: u64, file_size: u64) -> bool {
    // Property: validate_tensor_bounds never allows out-of-bounds access
    let tensor = GGUFTensor { offset, size, .. };
    let valid = validate_tensor_bounds(&tensor, file_size, 0);
    
    // If valid, then offset + size MUST be <= file_size
    if valid {
        offset < file_size && offset + size <= file_size
    } else {
        true  // Rejection is always safe
    }
}
```

**3. Edge Case Tests** (REQUIRED for M0):
```rust
#[test]
fn test_gguf_edge_cases() {
    // Edge case 1: Offset at exact file boundary
    let gguf = create_gguf_with_offset(file_size);
    assert!(validate_gguf(&gguf).is_err());
    
    // Edge case 2: Size is zero (valid but unusual)
    let gguf = create_gguf_with_offset_size(100, 0);
    assert!(validate_gguf(&gguf).is_ok());
    
    // Edge case 3: Maximum valid offset
    let gguf = create_gguf_with_offset_size(file_size - 1, 1);
    assert!(validate_gguf(&gguf).is_ok());
}
```

---

## Implementation Timeline

### Immediate Actions (Before LT-001 Implementation)

**Week 1 (Before Sprint 1)**:
1. ✅ Add M0-W-1211a to M0 spec (security requirement)
2. ✅ Update LT-001 story card (add security acceptance criteria)
3. ✅ Update LT-005 story card (add validation requirements)
4. ✅ Create security alert document (this document)
5. ✅ Notify Llama team of security requirements

**Week 2 (Sprint 1 - LT-001 Implementation)**:
1. Implement bounds validation in GGUF parser
2. Add fuzzing tests for malformed GGUF files
3. Add property tests for bounds validation
4. Add edge case tests
5. Security review by auth-min team

**Week 3 (Sprint 1 - LT-005 Implementation)**:
1. Integrate validation into pre-load checks
2. Add audit logging for rejected files
3. Test with real malicious GGUF files (if available)
4. Final security review

### Estimated Impact

**Timeline Impact**: +1 day to LT-001 (security implementation + tests)  
**Complexity Impact**: Low (validation is straightforward)  
**Risk Reduction**: HIGH → LOW (eliminates heap overflow vector)

---

## Defense in Depth

### Additional Security Measures

**1. File Hash Validation** (M1+):
```rust
// Verify GGUF file hash against known-good models
let file_hash = sha256_file(&gguf_path)?;
if !is_trusted_model_hash(&file_hash) {
    tracing::warn!(hash = %file_hash, "Untrusted GGUF file");
    // Proceed with extra caution or reject
}
```

**2. Sandboxing** (M2+):
```rust
// Load GGUF in sandboxed process with restricted permissions
let sandbox = Sandbox::new()
    .restrict_network()
    .restrict_filesystem()
    .limit_memory(1GB);
sandbox.run(|| load_gguf(&path))?;
```

**3. Audit Logging** (M0):
```rust
// Log all GGUF validation failures for security monitoring
audit_logger.emit(AuditEvent::GGUFValidationFailed {
    file_path: gguf_path,
    file_hash: sha256_file(&gguf_path)?,
    reason: "Tensor offset beyond file bounds",
    offset: tensor.offset,
    file_size: file_size,
})?;
```

**4. Rate Limiting** (M1+):
```rust
// Limit GGUF load attempts to prevent DoS via malformed files
if failed_loads_last_hour > 10 {
    return Err("Too many failed GGUF loads, possible attack");
}
```

---

## Security Review Checklist

### Pre-Implementation Review (auth-min Team)

- [ ] M0-W-1211a requirement added to spec
- [ ] LT-001 story card updated with security criteria
- [ ] LT-005 story card updated with validation requirements
- [ ] Fuzzing test plan approved
- [ ] Property test plan approved
- [ ] Edge case test plan approved

### Implementation Review (auth-min Team)

- [ ] Bounds validation implemented correctly
- [ ] Integer overflow protection verified
- [ ] Fuzzing tests passing (100+ malformed files)
- [ ] Property tests passing (1000+ random inputs)
- [ ] Edge case tests passing
- [ ] No timing vulnerabilities in validation
- [ ] Error messages do not leak sensitive information

### Post-Implementation Review (auth-min Team)

- [ ] Security tests integrated into CI
- [ ] Audit logging for rejected files
- [ ] Documentation updated (security considerations)
- [ ] Threat model updated (heap overflow mitigated)
- [ ] Sign-off by auth-min team

---

## References

### External References

1. **GGUF Parsing Vulnerabilities**: https://blog.huntr.com/gguf-file-format-vulnerabilities-a-guide-for-hackers
2. **CWE-119**: Improper Restriction of Operations within the Bounds of a Memory Buffer
3. **CWE-787**: Out-of-bounds Write
4. **GGUF Specification**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

### Internal References

1. **Research Document**: `bin/worker-orcd/.plan/llama-team/stories/LT-000-prep/REASERCH_pt1.md` line 45
2. **M0 Spec**: `bin/.specs/01_M0_worker_orcd.md` §6.2 (Model Validation)
3. **Story Card**: `bin/worker-orcd/.plan/llama-team/stories/LT-001-to-LT-010/LT-001-gguf-header-parser.md`
4. **Story Card**: `bin/worker-orcd/.plan/llama-team/stories/LT-001-to-LT-010/LT-005-pre-load-validation.md`

---

## Approval and Sign-Off

### Security Review

**Reviewed by**: auth-min Team  
**Date**: 2025-10-04  
**Severity**: HIGH  
**Mitigation**: REQUIRED before M0 implementation  
**Status**: 🔴 AWAITING IMPLEMENTATION

### Recommendations

1. ✅ **APPROVED**: Add M0-W-1211a to M0 spec
2. ✅ **APPROVED**: Update LT-001 and LT-005 story cards
3. ✅ **APPROVED**: Implement bounds validation with fuzzing tests
4. ✅ **APPROVED**: Security review before merge

### Sign-Off

This security alert has been reviewed and approved by the auth-min team. The recommended mitigations are **mandatory** for M0 and MUST be implemented before worker-orcd GGUF parser is deployed.

**Timing-Safe**: Not applicable (file parsing, not token comparison)  
**Leak-Safe**: Ensure error messages do not leak file contents  
**Injection-Safe**: Validate all metadata strings for control characters

---

Security verified by auth-min Team 🎭
