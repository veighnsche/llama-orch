# VRAM Residency — Security Specification

**Status**: Draft  
**Security Tier**: TIER 1 (Critical)  
**Last Updated**: 2025-10-01

---

## 0. Security Classification

### 0.1 Criticality Assessment

**Tier**: TIER 1 — Security-Critical

**Rationale**:
- Controls VRAM allocation and deallocation (memory safety boundary)
- Implements sealed shard contract (integrity guarantees)
- Enforces VRAM-only policy (prevents RAM inference attacks)
- Handles cryptographic seal verification (trust anchor)
- Manages CUDA FFI boundary (unsafe code interface)

**Impact of compromise**:
- Remote code execution via VRAM corruption
- Model exfiltration via RAM offload
- Seal forgery enabling poisoned models
- GPU memory exhaustion (DoS)
- Cross-tenant data leakage

---

## 1. Threat Model

### 1.1 Adversary Capabilities

**External Attacker** (network access):
- Can send malicious requests to worker-orcd endpoints
- Can attempt to trigger VRAM OOM conditions
- Can probe for memory layout information
- Cannot directly access GPU memory

**Compromised pool-managerd**:
- Can send forged Commit requests
- Can attempt to load malicious model bytes
- Can try to forge sealed shard handles
- Can attempt VRAM pointer leakage

**Compromised worker process**:
- Can attempt to read other workers' VRAM
- Can try to modify sealed shards
- Can attempt to bypass VRAM-only policy
- Can try to exfiltrate model weights

**Malicious model**:
- Can contain exploits in GGUF format
- Can trigger buffer overflows during parsing
- Can cause VRAM exhaustion
- Can contain backdoors in weights

### 1.2 Assets to Protect

**Primary Assets**:
1. **Model weights** — Intellectual property (millions in training cost)
2. **Sealed shard integrity** — Trust anchor for audited staging
3. **VRAM isolation** — Prevents cross-tenant leakage
4. **CUDA memory safety** — Prevents arbitrary code execution

**Secondary Assets**:
5. **Seal secret keys** — Enables seal verification
6. **VRAM allocation state** — Prevents resource exhaustion
7. **Digest computation** — Integrity verification

---

## 2. Security Requirements (RFC-2119)

### 2.1 Memory Safety

**MS-001**: VRAM pointers MUST be private and MUST NOT be exposed in API responses, logs, or error messages.

**MS-002**: All CUDA FFI calls MUST be wrapped in safe Rust abstractions with bounds checking.

**MS-003**: VRAM allocation MUST validate size parameters to prevent integer overflow.

**MS-004**: VRAM deallocation MUST handle CUDA errors gracefully without panicking.

**MS-005**: `Drop` implementation MUST NOT panic under any circumstances (per WORKER-4083).

**MS-006**: All pointer arithmetic MUST use checked operations or saturating arithmetic.

**MS-007**: VRAM reads/writes MUST validate offset + length <= allocated size.

### 2.2 Cryptographic Integrity

**CI-001**: Seal signatures MUST use HMAC-SHA256 with per-worker secret keys.

**CI-002**: Seal signature MUST cover `(shard_id, digest, sealed_at, gpu_device)`.

**CI-003**: Seal verification MUST use timing-safe comparison for signatures.

**CI-004**: Digest computation MUST use SHA-256 (FIPS 140-2 approved).

**CI-005**: Seal secret keys MUST be derived from worker API token or hardware ID.

**CI-006**: Seal secret keys MUST NOT be logged, exposed in API, or written to disk.

**CI-007**: Digest re-verification MUST occur before each Execute operation (per WORKER-4245).

### 2.3 VRAM-Only Policy Enforcement

**VP-001**: Model weights MUST reside entirely in GPU VRAM during inference (per ORCH-2.13).

**VP-002**: Unified memory (UMA) MUST be disabled at initialization.

**VP-003**: Zero-copy and pinned host memory MUST be disabled for model weights.

**VP-004**: Workers MUST fail fast if VRAM capacity is insufficient (per WORKER-4103).

**VP-005**: Workers MUST detect and reject RAM inference attempts.

**VP-006**: VRAM residency attestation MUST be cryptographically verifiable.

### 2.4 Input Validation

**IV-001**: Model bytes size MUST be validated before allocation (max 100GB default).

**IV-002**: GPU device index MUST be validated against available devices.

**IV-003**: Shard IDs MUST be validated (max length 256, alphanumeric + dash).

**IV-004**: Digest strings MUST be validated (64 hex chars for SHA-256).

**IV-005**: All string inputs MUST be checked for null bytes.

### 2.5 Resource Protection

**RP-001**: Total VRAM allocation MUST NOT exceed physical VRAM capacity.

**RP-002**: VRAM allocation MUST use saturating arithmetic to prevent overflow.

**RP-003**: Maximum model size MUST be configurable (default 100GB).

**RP-004**: VRAM allocation failures MUST return actionable error messages.

**RP-005**: VRAM deallocation MUST be tracked for audit trail.

---

## 3. Vulnerability Analysis

### 3.1 VRAM Pointer Leakage (HIGH)

**Vulnerability**: Exposing VRAM pointers in API responses or logs.

**Attack Vector**:
```rust
// VULNERABLE CODE
pub struct SealedShard {
    pub vram_ptr: *mut c_void,  // ← EXPOSED
}

// Response contains: vram_ptr: 0x7f8a4c000000
```

**Impact**:
- Defeats ASLR (Address Space Layout Randomization)
- Enables targeted buffer overflow attacks
- Facilitates ROP chain construction
- Enables cross-worker memory probing

**Mitigation** (MS-001):
```rust
pub struct SealedShard {
    pub shard_id: String,
    vram_ptr: *mut c_void,  // ← PRIVATE
    // Never expose in Debug, Display, or Serialize
}
```

**Status**: ✅ Implemented in current code

---

### 3.2 Seal Forgery (HIGH)

**Vulnerability**: Sealed shard attestation without cryptographic proof.

**Attack Vector**:
```rust
// VULNERABLE CODE
pub struct SealedShard {
    pub sealed: bool,  // ← Anyone can set to true
    pub digest: String,  // ← Can be forged
}

// Attacker creates fake sealed shard
let fake = SealedShard {
    sealed: true,
    digest: "fake_digest".to_string(),
    ...
};
```

**Impact**:
- Compromised pool-managerd can forge seals
- Malicious models bypass validation
- Integrity guarantees broken
- Audited staging compromised

**Mitigation** (CI-001, CI-002):
```rust
pub struct SealedShard {
    pub digest: String,
    pub sealed_at: SystemTime,
    signature: Vec<u8>,  // ← HMAC-SHA256 seal proof (private)
}

impl SealedShard {
    pub fn verify_seal(&self, secret_key: &[u8]) -> Result<()> {
        let message = format!("{}|{}|{}|{}", 
            self.shard_id, 
            self.digest, 
            self.sealed_at.duration_since(UNIX_EPOCH)?.as_secs(),
            self.gpu_device
        );
        
        let mut mac = HmacSha256::new_from_slice(secret_key)
            .map_err(|_| VramError::IntegrityViolation)?;
        mac.update(message.as_bytes());
        
        let expected = mac.finalize().into_bytes();
        
        // Timing-safe comparison
        if !constant_time_eq(&expected[..], &self.signature[..]) {
            return Err(VramError::SealVerificationFailed);
        }
        
        Ok(())
    }
}
```

**Status**: ⬜ Not yet implemented (Phase 1 priority)

---

### 3.3 Digest TOCTOU (MEDIUM)

**Vulnerability**: Time-of-check to time-of-use race in digest verification.

**Attack Vector**:
```
Time 0: Compute digest → "abc123" (valid model)
Time 1: Seal shard with digest "abc123"
Time 2: <GPU driver exploit modifies VRAM>
Time 3: Execute with modified weights (digest not re-checked)
```

**Impact**:
- Model poisoning after seal
- Backdoor injection post-validation
- Integrity guarantees bypassed
- Silent model corruption

**Mitigation** (CI-007):
```rust
pub fn execute(&self, shard: &SealedShard) -> Result<()> {
    // Re-verify seal before EVERY execution
    self.verify_sealed(shard)?;
    
    // Re-compute digest from VRAM
    let current_digest = self.compute_vram_digest(shard)?;
    
    // Verify digest matches
    shard.verify(&current_digest)?;
    
    // Now safe to execute
    self.run_inference(shard)
}
```

**Status**: ⬜ Not yet implemented (Phase 3 priority)

---

### 3.4 CUDA FFI Buffer Overflow (CRITICAL)

**Vulnerability**: Unchecked CUDA memory operations.

**Attack Vector**:
```rust
// VULNERABLE CODE
unsafe {
    cuda_memcpy(vram_ptr, model_bytes.as_ptr(), model_bytes.len());
    // No bounds checking!
}

// Attacker sends model_bytes.len() > allocated VRAM
// → Out-of-bounds write → GPU memory corruption
```

**Impact**:
- Arbitrary GPU memory write
- Potential code execution
- GPU driver crash
- Cross-worker memory corruption

**Mitigation** (MS-002, MS-007):
```rust
pub struct SafeVramPtr {
    ptr: *mut c_void,
    size: usize,
}

impl SafeVramPtr {
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        // Bounds check
        let end = offset.checked_add(data.len())
            .ok_or(VramError::IntegrityViolation)?;
        
        if end > self.size {
            return Err(VramError::IntegrityViolation);
        }
        
        // Safe to write
        unsafe {
            let dst = self.ptr.add(offset);
            cuda_memcpy_checked(dst, data.as_ptr(), data.len())?;
        }
        
        Ok(())
    }
}
```

**Status**: ⬜ Not yet implemented (Phase 3 priority)

---

### 3.5 Integer Overflow in VRAM Allocation (HIGH)

**Vulnerability**: Unchecked arithmetic in VRAM size calculations.

**Attack Vector**:
```rust
// VULNERABLE CODE
let total_needed = self.used_vram + model_bytes.len();  // ← Can overflow
if total_needed > self.total_vram {
    return Err(...);
}

// Attacker sends model_bytes.len() = usize::MAX
// → Overflow wraps to small value → check passes → OOM
```

**Impact**:
- VRAM exhaustion
- GPU OOM crash
- Denial of service
- Worker process crash

**Mitigation** (MS-003, RP-002):
```rust
// Use saturating arithmetic
let total_needed = self.used_vram.saturating_add(model_bytes.len());

if total_needed > self.total_vram {
    return Err(VramError::InsufficientVram(
        model_bytes.len(),
        self.total_vram.saturating_sub(self.used_vram),
    ));
}
```

**Status**: ✅ Implemented in current code

---

### 3.6 VRAM-Only Policy Bypass (HIGH)

**Vulnerability**: Unified memory or RAM fallback enabled.

**Attack Vector**:
```
1. Worker starts with UMA enabled (default on some systems)
2. Model "loaded to VRAM" but actually uses host RAM
3. Attacker reads /proc/pid/maps → sees RAM mapping
4. Attacker dumps model weights from RAM
```

**Impact**:
- Model exfiltration
- VRAM-only policy violated
- Audited staging guarantees broken
- Intellectual property theft

**Mitigation** (VP-002, VP-003):
```rust
pub fn enforce_vram_only_policy() -> Result<()> {
    // Disable unified memory
    unsafe {
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0)?;
    }
    
    // Verify no host memory fallback
    let props = get_device_properties()?;
    if props.unified_addressing {
        return Err(VramError::PolicyViolation(
            "Unified memory detected, VRAM-only policy cannot be enforced"
        ));
    }
    
    Ok(())
}
```

**Status**: ⬜ Not yet implemented (Phase 3 priority)

---

### 3.7 Seal Key Exposure (CRITICAL)

**Vulnerability**: Seal secret key logged or exposed.

**Attack Vector**:
```rust
// VULNERABLE CODE
tracing::info!("Seal key: {:?}", secret_key);  // ← LOGGED
eprintln!("Debug: key={:?}", secret_key);  // ← STDERR

// Or in error message
Err(format!("Seal failed with key {:?}", secret_key))
```

**Impact**:
- Attacker can forge seals
- All sealed shards compromised
- Integrity guarantees broken
- Complete trust anchor failure

**Mitigation** (CI-006):
```rust
// Never log secret keys
// Use opaque types that don't impl Debug/Display

pub struct SealKey([u8; 32]);

// No Debug/Display impl
// No serialization
// Only used internally

impl SealKey {
    pub fn from_token(token: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"llorch-seal-key-v1");
        hasher.update(token.as_bytes());
        let hash = hasher.finalize();
        
        let mut key = [0u8; 32];
        key.copy_from_slice(&hash[..]);
        SealKey(key)
    }
    
    pub(crate) fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

// Zeroize on drop
impl Drop for SealKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
```

**Status**: ⬜ Not yet implemented (Phase 1 priority)

---

## 4. Clippy Security Configuration

### 4.1 TIER 1 Configuration

**Applied to**: `bin/worker-orcd-crates/vram-residency/src/lib.rs`

```rust
// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::must_use_candidate)]
```

**Rationale**: VRAM management is security-critical; any panic or UB could compromise sealed shard guarantees.

**Status**: ✅ Partially applied (current code has TIER 1 lints)

---

### 4.2 Forbidden Patterns

**MUST NOT use**:
- `.unwrap()` — Use `?` or explicit error handling
- `.expect()` — Use `?` or explicit error handling
- `panic!()` — Return `Result` instead
- Array indexing `arr[i]` — Use `.get(i)` or bounds checking
- Unchecked arithmetic `a + b` — Use `a.saturating_add(b)` or `a.checked_add(b)`
- `mem::forget()` — Prevents Drop, leaks resources
- `todo!()` / `unimplemented!()` — Complete all code paths

**Example violations**:
```rust
// ❌ FORBIDDEN
let shard = shards[0];  // Indexing
let total = used + needed;  // Unchecked arithmetic
let ptr = ptr_from_int(addr).unwrap();  // Unwrap

// ✅ CORRECT
let shard = shards.get(0).ok_or(VramError::NotFound)?;
let total = used.saturating_add(needed);
let ptr = ptr_from_int(addr)?;
```

---

## 5. Secure Coding Guidelines

### 5.1 Error Handling

**Principle**: Never panic, always return `Result`.

```rust
// ❌ BAD: Panics on error
pub fn seal_model(&mut self, bytes: &[u8]) -> SealedShard {
    let ptr = cuda_malloc(bytes.len()).unwrap();  // PANIC
    ...
}

// ✅ GOOD: Returns Result
pub fn seal_model(&mut self, bytes: &[u8]) -> Result<SealedShard> {
    let ptr = cuda_malloc(bytes.len())
        .map_err(|e| VramError::CudaAllocationFailed(e.to_string()))?;
    ...
}
```

### 5.2 Bounds Checking

**Principle**: Validate all array/slice access.

```rust
// ❌ BAD: Unchecked indexing
let byte = bytes[offset];

// ✅ GOOD: Checked access
let byte = bytes.get(offset)
    .ok_or(VramError::IntegrityViolation)?;
```

### 5.3 Integer Arithmetic

**Principle**: Use saturating or checked arithmetic.

```rust
// ❌ BAD: Can overflow
let total = a + b;

// ✅ GOOD: Saturating arithmetic
let total = a.saturating_add(b);

// ✅ ALSO GOOD: Checked arithmetic
let total = a.checked_add(b)
    .ok_or(VramError::IntegrityViolation)?;
```

### 5.4 Pointer Safety

**Principle**: Minimize unsafe, validate all pointers.

```rust
// ❌ BAD: Unchecked pointer arithmetic
unsafe {
    let dst = ptr.add(offset);  // No bounds check
    *dst = value;
}

// ✅ GOOD: Validated pointer operations
pub fn write_at(&mut self, offset: usize, value: u8) -> Result<()> {
    if offset >= self.size {
        return Err(VramError::IntegrityViolation);
    }
    
    unsafe {
        let dst = self.ptr.add(offset);
        *dst = value;
    }
    
    Ok(())
}
```

### 5.5 Secret Handling

**Principle**: Never log, display, or serialize secrets.

```rust
// ❌ BAD: Secret in log
tracing::info!("Key: {:?}", secret_key);

// ❌ BAD: Secret in error
return Err(format!("Failed with key {:?}", secret_key));

// ✅ GOOD: Opaque error
return Err(VramError::SealVerificationFailed);

// ✅ GOOD: Zeroize on drop
impl Drop for SealKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
```

---

## 6. Security Testing Requirements

### 6.1 Unit Tests

**Required test coverage**:

```rust
#[cfg(test)]
mod security_tests {
    #[test]
    fn test_vram_ptr_not_exposed() {
        let shard = create_test_shard();
        let json = serde_json::to_string(&shard).unwrap();
        assert!(!json.contains("vram_ptr"));
        assert!(!json.contains("0x"));
    }
    
    #[test]
    fn test_seal_forgery_rejected() {
        let mut shard = create_test_shard();
        shard.signature = vec![0u8; 32];  // Forged signature
        
        let result = verify_seal(&shard, &secret_key);
        assert!(matches!(result, Err(VramError::SealVerificationFailed)));
    }
    
    #[test]
    fn test_integer_overflow_prevented() {
        let mut manager = VramManager::new();
        let huge_size = usize::MAX;
        
        let result = manager.seal_model(&vec![0u8; huge_size], 0);
        assert!(matches!(result, Err(VramError::InsufficientVram(_, _))));
    }
    
    #[test]
    fn test_bounds_checking() {
        let shard = create_test_shard();
        let result = shard.read_at(shard.vram_bytes + 1);
        assert!(matches!(result, Err(VramError::IntegrityViolation)));
    }
    
    #[test]
    fn test_seal_key_not_logged() {
        // Capture logs
        let logs = capture_logs(|| {
            let key = SealKey::from_token("secret");
            let _ = compute_seal(&key, "data");
        });
        
        // Verify no key material in logs
        assert!(!logs.contains("secret"));
        assert!(!logs.contains("SealKey"));
    }
}
```

### 6.2 Fuzzing

**Required fuzz targets**:

```rust
// fuzz/fuzz_targets/seal_verification.rs
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(shard) = SealedShard::deserialize(data) {
        let _ = verify_seal(&shard, &test_key());
        // Should never panic, always return Result
    }
});

// fuzz/fuzz_targets/vram_allocation.rs
fuzz_target!(|size: usize| {
    let mut manager = VramManager::new();
    let _ = manager.seal_model(&vec![0u8; size], 0);
    // Should handle any size without panic
});
```

### 6.3 Property Tests

**Required properties**:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn seal_verification_deterministic(
        data in prop::collection::vec(any::<u8>(), 0..1000)
    ) {
        let shard1 = seal_model(&data);
        let shard2 = seal_model(&data);
        
        // Same data → same digest
        prop_assert_eq!(shard1.digest, shard2.digest);
    }
    
    #[test]
    fn vram_allocation_never_exceeds_capacity(
        sizes in prop::collection::vec(0usize..1_000_000, 0..100)
    ) {
        let mut manager = VramManager::new();
        let mut total_allocated = 0usize;
        
        for size in sizes {
            if let Ok(shard) = manager.seal_model(&vec![0u8; size], 0) {
                total_allocated = total_allocated.saturating_add(shard.vram_bytes);
            }
        }
        
        // Never exceed capacity
        prop_assert!(total_allocated <= manager.total_vram());
    }
}
```

---

## 7. Audit Trail Requirements

### 7.1 Security Events to Log

**MUST log** (with structured fields):

```rust
// Seal operations
tracing::info!(
    shard_id = %shard.shard_id,
    gpu_device = %shard.gpu_device,
    vram_bytes = %shard.vram_bytes,
    digest = %shard.digest,
    "Model sealed in VRAM"
);

// Seal verification
tracing::info!(
    shard_id = %shard.shard_id,
    result = "success",
    "Seal verification passed"
);

tracing::error!(
    shard_id = %shard.shard_id,
    result = "failure",
    "Seal verification FAILED - integrity violation"
);

// VRAM allocation
tracing::info!(
    requested = %size,
    available = %available,
    used = %used,
    "VRAM allocation request"
);

// Policy violations
tracing::error!(
    violation = "vram_only_policy",
    details = "Unified memory detected",
    "VRAM-only policy violation"
);
```

**MUST NOT log**:
- VRAM pointers
- Seal secret keys
- Raw model bytes
- Internal memory addresses

---

## 8. Deployment Security

### 8.1 Process Privileges

**Requirements**:
- Workers MUST run as non-root user (dedicated `worker-orcd` user)
- Workers MUST drop unnecessary capabilities after GPU initialization
- Workers MUST NOT require root privileges for normal operation

**Implementation**:
```bash
# Create dedicated user
sudo useradd -r -s /bin/false worker-orcd

# Grant GPU access
sudo usermod -aG video worker-orcd

# Run with reduced privileges
sudo -u worker-orcd /usr/bin/worker-orcd --gpu=0
```

### 8.2 File System Permissions

**Requirements**:
- Model files MUST be readable only by worker user
- Seal keys MUST be stored in protected directory (0600 permissions)
- VRAM state MUST NOT be written to disk

**Implementation**:
```bash
# Seal key file
sudo mkdir -p /etc/llorch/secrets
sudo chown worker-orcd:worker-orcd /etc/llorch/secrets
sudo chmod 0700 /etc/llorch/secrets

# Model directory
sudo mkdir -p /var/lib/llorch/models
sudo chown worker-orcd:worker-orcd /var/lib/llorch/models
sudo chmod 0750 /var/lib/llorch/models
```

### 8.3 Container Isolation

**Requirements** (post-M0):
- Workers SHOULD run in containers (Docker/Podman)
- Workers SHOULD use Linux namespaces (CLONE_NEWPID, CLONE_NEWNET)
- Workers SHOULD use SELinux or AppArmor policies

---

## 9. Incident Response

### 9.1 Seal Verification Failure

**Detection**:
```rust
if let Err(VramError::SealVerificationFailed) = verify_seal(&shard) {
    // SECURITY INCIDENT
}
```

**Response**:
1. Immediately stop worker process
2. Log security alert with full context
3. Notify pool-managerd of failure
4. Quarantine affected GPU
5. Dump VRAM for forensic analysis
6. Restart worker with clean state

### 9.2 VRAM Corruption Detected

**Detection**:
```rust
let current_digest = compute_vram_digest(&shard)?;
if current_digest != shard.digest {
    // VRAM CORRUPTION
}
```

**Response**:
1. Halt all inference on affected GPU
2. Log security alert
3. Transition worker to Stopped state
4. Trigger restart via pool-managerd
5. Investigate root cause (driver bug, hardware fault, attack)

### 9.3 Policy Violation

**Detection**:
```rust
if unified_memory_detected() {
    // POLICY VIOLATION
}
```

**Response**:
1. Refuse to start worker
2. Log security alert
3. Notify operator
4. Provide remediation steps
5. Do not proceed with inference

---

## 10. Security Review Checklist

### 10.1 Code Review

**Before merging**:
- [ ] All TIER 1 Clippy lints pass
- [ ] No `.unwrap()` or `.expect()` in code
- [ ] No `panic!()` in code
- [ ] All array access is bounds-checked
- [ ] All arithmetic uses saturating/checked operations
- [ ] No VRAM pointers in public API
- [ ] Seal keys never logged or displayed
- [ ] All unsafe blocks have safety comments
- [ ] Error messages don't leak sensitive info

### 10.2 Testing

**Before release**:
- [ ] All unit tests pass
- [ ] Fuzz tests run for 1 hour without crashes
- [ ] Property tests pass with 10,000 cases
- [ ] Security tests cover all vulnerabilities
- [ ] Proof bundles generated for all tests

### 10.3 Documentation

**Before deployment**:
- [ ] Security spec reviewed and approved
- [ ] Threat model documented
- [ ] Incident response procedures defined
- [ ] Deployment security guidelines followed
- [ ] Audit trail requirements implemented

---

## 11. References

**Security Audits**:
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Issues #9-#20
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Operational security
- `.docs/security/SECURITY_OVERSEER_SUMMARY.md` — Overall posture

**Specifications**:
- `bin/worker-orcd-crates/vram-residency/.specs/00_vram-residency.md` — Functional spec
- `bin/worker-orcd-crates/vram-residency/.specs/10_expectations.md` — Consumer contracts
- `bin/worker-orcd/.specs/00_worker-orcd.md` — Parent spec

**Standards**:
- FIPS 140-2 — Cryptographic module validation
- RFC 2119 — Requirement levels (MUST/SHOULD/MAY)
- CWE-119 — Buffer overflow
- CWE-190 — Integer overflow
- CWE-200 — Information exposure

---

**End of Security Specification**
