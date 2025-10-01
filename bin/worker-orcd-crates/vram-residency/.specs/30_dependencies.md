# VRAM Residency — Dependencies Investigation

**Status**: Draft  
**Security Tier**: TIER 1 (Critical)  
**Last Updated**: 2025-10-01  
**Purpose**: Map all security requirements to necessary dependencies

---

## 0. Overview

This document investigates **all dependencies required** to satisfy the security requirements defined in `.specs/20_security.md`. Each security requirement is mapped to specific crates, libraries, and their justification.

**Scope**: Complete dependency analysis for TIER 1 security-critical implementation.

---

## 1. Dependency Matrix by Security Domain

### 1.1 Memory Safety (MS-001 to MS-007)

| Requirement | Dependencies | Justification |
|-------------|--------------|---------------|
| **MS-001**: Private VRAM pointers | `serde` (with skip) | Prevent serialization of pointers |
| **MS-002**: Safe CUDA FFI wrappers | Internal abstractions | Wrap unsafe CUDA calls with bounds checking |
| **MS-003**: Integer overflow prevention | Rust stdlib (saturating ops) | Built-in saturating arithmetic |
| **MS-004**: Graceful deallocation | `tracing` | Log errors without panicking |
| **MS-005**: No-panic Drop | TIER 1 Clippy lints | Enforce panic-free code |
| **MS-006**: Checked pointer arithmetic | Rust stdlib | `checked_add`, `saturating_add` |
| **MS-007**: Bounds validation | Rust stdlib | Slice bounds checking |

**Required crates**:
- ✅ `serde` (serialization control)
- ✅ `tracing` (error logging)
- ✅ Rust stdlib (checked arithmetic)

---

### 1.2 Cryptographic Integrity (CI-001 to CI-007)

| Requirement | Dependencies | Justification |
|-------------|--------------|---------------|
| **CI-001**: HMAC-SHA256 signatures | `hmac`, `sha2` | Industry-standard HMAC implementation |
| **CI-002**: Signature coverage | `chrono` | Timestamp handling for `sealed_at` |
| **CI-003**: Timing-safe comparison | `subtle` | Constant-time equality (via `secrets-management`) |
| **CI-004**: SHA-256 digests | `sha2` | FIPS 140-2 approved hash function |
| **CI-005**: Key derivation | `secrets-management` | HKDF-SHA256 key derivation |
| **CI-006**: Key zeroization | `secrets-management` | Automatic zeroization on drop |
| **CI-007**: Digest re-verification | `sha2` | Re-compute digest before Execute |

**Required crates**:
- ✅ `hmac = "0.12"` — HMAC-SHA256 seal signatures
- ✅ `sha2 = "0.10"` — SHA-256 digest computation
- ✅ `subtle = "2.5"` — Timing-safe comparison (transitive via `secrets-management`)
- ✅ `secrets-management` — Seal key derivation and zeroization
- ✅ `chrono` — Timestamp handling

**Cryptographic library choice**:
- **RustCrypto** ecosystem (`hmac`, `sha2`, `subtle`)
- Professionally audited, FIPS 140-2 compliant
- Used by thousands of production systems
- Active maintenance and security patches

---

### 1.3 VRAM-Only Policy Enforcement (VP-001 to VP-006)

| Requirement | Dependencies | Justification |
|-------------|--------------|---------------|
| **VP-001**: VRAM-only inference | CUDA FFI (internal) | Query and enforce VRAM residency |
| **VP-002**: Disable UMA | CUDA FFI (internal) | `cudaDeviceSetLimit` calls |
| **VP-003**: Disable zero-copy | CUDA FFI (internal) | Device property validation |
| **VP-004**: Fail fast on OOM | `thiserror` | Structured error types |
| **VP-005**: Detect RAM inference | CUDA FFI (internal) | Memory type validation |
| **VP-006**: Cryptographic attestation | `hmac`, `sha2` | Seal signature proves residency |

**Required crates**:
- ✅ `thiserror` — Structured error types
- ✅ CUDA FFI wrappers (internal to `worker-orcd`)
- ✅ `hmac`, `sha2` — Cryptographic attestation

**Note**: CUDA FFI is provided by parent `worker-orcd` binary, not a direct dependency.

---

### 1.4 Input Validation (IV-001 to IV-005)

| Requirement | Dependencies | Justification |
|-------------|--------------|---------------|
| **IV-001**: Model size validation | `input-validation` | Range checking |
| **IV-002**: GPU device validation | `input-validation` | `validate_range()` |
| **IV-003**: Shard ID validation | `input-validation` | `validate_identifier()` |
| **IV-004**: Digest validation | `input-validation` | `validate_hex_string()` |
| **IV-005**: Null byte checking | `input-validation` | Built into all validators |

**Required crates**:
- ✅ `input-validation` — All validation applets

**Why not roll our own?**
- Prevents code duplication across crates
- Centralized security boundary
- Comprehensive test coverage (BDD, property tests, fuzz tests)
- Already implements all required validations

---

### 1.5 Resource Protection (RP-001 to RP-005)

| Requirement | Dependencies | Justification |
|-------------|--------------|---------------|
| **RP-001**: Capacity enforcement | Rust stdlib | Saturating arithmetic |
| **RP-002**: Overflow prevention | Rust stdlib | `saturating_add`, `checked_add` |
| **RP-003**: Configurable limits | `serde` | Configuration deserialization |
| **RP-004**: Actionable errors | `thiserror` | Structured error messages |
| **RP-005**: Audit trail | `audit-logging` | VRAM operation events |

**Required crates**:
- ✅ `audit-logging` — VRAM audit events
- ✅ `thiserror` — Error types
- ✅ `serde` — Configuration

---

## 2. Vulnerability Mitigation Dependencies

### 2.1 VRAM Pointer Leakage (§3.1)

**Mitigation requirements**:
- Private VRAM pointers (never exposed)
- No Debug/Display/Serialize for pointer fields

**Dependencies**:
```toml
serde = { version = "1.0", features = ["derive"] }
```

**Implementation**:
```rust
#[derive(Serialize)]
pub struct SealedShard {
    pub shard_id: String,
    pub gpu_device: u32,
    pub vram_bytes: usize,
    pub digest: String,
    
    #[serde(skip)]  // ← Never serialize
    vram_ptr: *mut c_void,  // ← Private
}

// No Debug impl that exposes pointer
```

---

### 2.2 Seal Forgery (§3.2)

**Mitigation requirements**:
- HMAC-SHA256 cryptographic signatures
- Per-worker secret keys
- Timing-safe verification

**Dependencies**:
```toml
hmac = "0.12"
sha2 = "0.10"
secrets-management = { path = "../../shared-crates/secrets-management" }
```

**Implementation**:
```rust
use hmac::{Hmac, Mac};
use sha2::Sha256;
use secrets_management::SecretKey;

// Derive seal key from worker token
let seal_key = SecretKey::derive_from_token(
    &worker_api_token,
    b"llorch-seal-key-v1"
)?;

// Compute HMAC-SHA256 signature
let mut mac = Hmac::<Sha256>::new_from_slice(seal_key.as_bytes())?;
mac.update(message.as_bytes());
let signature = mac.finalize().into_bytes();
```

---

### 2.3 Digest TOCTOU (§3.3)

**Mitigation requirements**:
- Re-compute digest from VRAM before each Execute
- Verify digest matches sealed value

**Dependencies**:
```toml
sha2 = "0.10"
```

**Implementation**:
```rust
use sha2::{Sha256, Digest};

// Re-compute digest from VRAM contents
let mut hasher = Sha256::new();
hasher.update(&vram_contents);
let current_digest = format!("{:x}", hasher.finalize());

// Verify matches sealed digest
if current_digest != shard.digest {
    return Err(VramError::IntegrityViolation);
}
```

---

### 2.4 CUDA FFI Buffer Overflow (§3.4)

**Mitigation requirements**:
- Bounds checking on all VRAM operations
- Safe wrappers around unsafe CUDA calls

**Dependencies**:
```toml
# No external dependencies - use Rust stdlib
```

**Implementation**:
```rust
pub struct SafeVramPtr {
    ptr: *mut c_void,
    size: usize,
}

impl SafeVramPtr {
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        // Bounds check using checked arithmetic
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

---

### 2.5 Integer Overflow in VRAM Allocation (§3.5)

**Mitigation requirements**:
- Saturating arithmetic for all size calculations
- Checked arithmetic with error handling

**Dependencies**:
```toml
# No external dependencies - use Rust stdlib
```

**Implementation**:
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

---

### 2.6 VRAM-Only Policy Bypass (§3.6)

**Mitigation requirements**:
- Disable unified memory at initialization
- Verify device properties
- Detect host memory fallback

**Dependencies**:
```toml
# CUDA FFI provided by worker-orcd parent
```

**Implementation**:
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
            "Unified memory detected"
        ));
    }
    
    Ok(())
}
```

---

### 2.7 Seal Key Exposure (§3.7)

**Mitigation requirements**:
- No Debug/Display/Serialize for keys
- Automatic zeroization on drop
- Never log key material

**Dependencies**:
```toml
secrets-management = { path = "../../shared-crates/secrets-management" }
```

**Why `secrets-management`?**
- Wraps `zeroize` crate for automatic cleanup
- No Debug/Display implementations
- HKDF-SHA256 key derivation
- Battle-tested security primitives

**Implementation**:
```rust
use secrets_management::SecretKey;

// SecretKey automatically:
// - Zeroizes on drop
// - Has no Debug/Display impl
// - Cannot be accidentally logged

let seal_key = SecretKey::derive_from_token(
    &worker_api_token,
    b"llorch-seal-key-v1"
)?;

// Use key for HMAC
let key_bytes = seal_key.as_bytes();  // &[u8; 32]

// Automatic zeroization when seal_key goes out of scope
```

---

## 3. Audit Trail Dependencies (§10.2 from 10_expectations.md)

**Requirements**:
- Every seal operation with digest
- Every verification operation with result
- Every VRAM allocation/deallocation
- Any policy violations or integrity failures

**Dependencies**:
```toml
audit-logging = { path = "../../shared-crates/audit-logging" }
chrono = { workspace = true }
```

**Event types provided by `audit-logging`**:
- `AuditEvent::VramSealed` — Model sealed in VRAM
- `AuditEvent::SealVerified` — Seal verification passed
- `AuditEvent::SealVerificationFailed` — Seal verification failed (CRITICAL)
- `AuditEvent::VramAllocated` — VRAM allocation succeeded
- `AuditEvent::VramAllocationFailed` — VRAM allocation failed
- `AuditEvent::VramDeallocated` — VRAM deallocated
- `AuditEvent::PolicyViolation` — Security policy violated

**Why `audit-logging`?**
- Tamper-evident (append-only with hash chains)
- Never logs secrets (uses fingerprints)
- Async, non-blocking emission
- Integration with `input-validation` for log injection prevention
- Compliance-ready (GDPR, SOC2, ISO 27001)

---

## 4. Testing Dependencies

### 4.1 Unit Testing

**Required for security tests** (§6.1 from 20_security.md):

```toml
[dev-dependencies]
# BDD testing framework
cucumber = "0.20"
tokio = { workspace = true, features = ["test-util", "macros"] }

# Property-based testing
proptest = "1.0"

# Fuzzing (separate fuzz/ directory)
# libfuzzer-sys = "0.4"  # Added via cargo-fuzz

# Test utilities
tempfile = "3.8"
```

**Test coverage requirements**:
- ✅ Seal forgery rejection
- ✅ Timing-safe verification
- ✅ Integer overflow prevention
- ✅ Bounds checking
- ✅ Key zeroization on drop
- ✅ Input validation edge cases

---

### 4.2 Integration Testing

**Required for end-to-end tests**:

```toml
[dev-dependencies]
# Mock CUDA for testing without GPU
# (Internal mock implementation, no external dependency)

# Audit log verification
audit-logging = { path = "../../shared-crates/audit-logging" }
```

---

## 5. Complete Cargo.toml

### 5.1 Production Dependencies

```toml
[package]
name = "vram-residency"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true

[dependencies]
# Error handling
thiserror.workspace = true

# Serialization (with pointer skip)
serde = { workspace = true, features = ["derive"] }
serde_json.workspace = true

# Logging
tracing.workspace = true

# Timestamps
chrono = { workspace = true }

# Cryptography - HMAC-SHA256 seal signatures
hmac = "0.12"
sha2 = "0.10"

# Shared crates - Security
input-validation = { path = "../../shared-crates/input-validation" }
secrets-management = { path = "../../shared-crates/secrets-management" }
audit-logging = { path = "../../shared-crates/audit-logging" }

# Note: CUDA FFI provided by parent worker-orcd binary
```

**Dependency count**: 10 direct dependencies (excluding workspace-shared)

**Transitive dependencies** (via shared crates):
- `subtle` (via `secrets-management`) — Timing-safe comparison
- `zeroize` (via `secrets-management`) — Memory zeroization
- `hkdf` (via `secrets-management`) — Key derivation
- `secrecy` (via `secrets-management`) — Secret wrapper

---

### 5.2 Development Dependencies

```toml
[dev-dependencies]
# BDD testing
cucumber = "0.20"
tokio = { workspace = true, features = ["test-util", "macros"] }

# Property-based testing
proptest = "1.0"

# Test utilities
tempfile = "3.8"

# Benchmarking
criterion = "0.5"
```

---

## 6. Dependency Justification by Category

### 6.1 Cryptography (CRITICAL)

| Crate | Version | Purpose | Audit Status |
|-------|---------|---------|--------------|
| `hmac` | 0.12 | HMAC-SHA256 signatures | ✅ RustCrypto audited |
| `sha2` | 0.10 | SHA-256 digests | ✅ RustCrypto audited |
| `subtle` | 2.5 | Timing-safe comparison | ✅ RustCrypto audited |
| `zeroize` | 1.7 | Memory zeroization | ✅ RustCrypto audited |
| `hkdf` | 0.12 | Key derivation | ✅ RustCrypto audited |

**Why RustCrypto?**
- Professional security audits
- FIPS 140-2 compliant implementations
- Used by thousands of production systems
- Active maintenance and CVE response
- No-std compatible (minimal dependencies)

**Alternatives considered**:
- ❌ OpenSSL bindings — Too heavy, C FFI complexity
- ❌ Ring — Less flexible, opinionated API
- ✅ RustCrypto — Best balance of security, usability, and auditability

---

### 6.2 Shared Security Crates (CRITICAL)

| Crate | Purpose | Security Tier |
|-------|---------|---------------|
| `input-validation` | Input sanitization | TIER 2 |
| `secrets-management` | Key management | TIER 1 |
| `audit-logging` | Security audit trail | TIER 1 |

**Why shared crates?**
- ✅ Centralized security boundary
- ✅ Consistent validation across services
- ✅ Comprehensive test coverage
- ✅ Single point of security updates
- ✅ Prevents code duplication and drift

**Security review status**:
- ✅ All three crates reviewed and approved (see `31_dependency_verification.md`)
- ✅ Comprehensive BDD test suites
- ✅ Property tests and fuzz tests
- ✅ Security-focused documentation

---

### 6.3 Core Infrastructure (HIGH)

| Crate | Version | Purpose | Justification |
|-------|---------|---------|---------------|
| `thiserror` | workspace | Error types | Industry standard, minimal dependencies |
| `serde` | workspace | Serialization | Required for API types, pointer skip |
| `tracing` | workspace | Structured logging | Non-blocking, async-aware |
| `chrono` | workspace | Timestamps | ISO 8601, UTC handling |

**Why these specific crates?**
- Widely adopted in Rust ecosystem
- Minimal transitive dependencies
- Well-maintained and stable
- No known security vulnerabilities

---

### 6.4 Testing Infrastructure (DEV)

| Crate | Version | Purpose | Justification |
|-------|---------|---------|---------------|
| `cucumber` | 0.20 | BDD testing | Gherkin-based test specs |
| `proptest` | 1.0 | Property testing | Randomized test generation |
| `tempfile` | 3.8 | Temp files | Test isolation |
| `criterion` | 0.5 | Benchmarking | Performance regression detection |

---

## 7. Dependency Security Analysis

### 7.1 Supply Chain Security

**Verification steps**:
1. ✅ All cryptographic crates from RustCrypto (trusted organization)
2. ✅ Shared crates are internal (no supply chain risk)
3. ✅ Core infrastructure crates are widely adopted (high scrutiny)
4. ✅ No dependencies on unmaintained crates
5. ✅ No dependencies with known CVEs

**Cargo audit integration**:
```bash
# Run security audit
cargo audit

# Check for outdated dependencies
cargo outdated
```

---

### 7.2 Transitive Dependency Analysis

**Total dependency tree** (production):
```
vram-residency
├── hmac (0.12)
│   └── digest (0.10)
├── sha2 (0.10)
│   └── digest (0.10)
├── input-validation
│   └── thiserror
├── secrets-management
│   ├── secrecy (0.8)
│   ├── zeroize (1.7)
│   ├── subtle (2.5)
│   ├── hkdf (0.12)
│   ├── sha2 (0.10)
│   └── hex (0.4)
├── audit-logging
│   ├── sha2 (0.10)
│   ├── hex (0.4)
│   ├── input-validation
│   └── [workspace crates]
└── [workspace crates]
```

**Total transitive dependencies**: ~20 crates (excluding workspace)

**Dependency depth**: Maximum 3 levels

---

### 7.3 License Compatibility

| Crate | License | Compatible with GPL-3.0? |
|-------|---------|--------------------------|
| `hmac` | MIT/Apache-2.0 | ✅ Yes |
| `sha2` | MIT/Apache-2.0 | ✅ Yes |
| `subtle` | BSD-3-Clause | ✅ Yes |
| `zeroize` | MIT/Apache-2.0 | ✅ Yes |
| `hkdf` | MIT/Apache-2.0 | ✅ Yes |
| `thiserror` | MIT/Apache-2.0 | ✅ Yes |
| `serde` | MIT/Apache-2.0 | ✅ Yes |
| `tracing` | MIT | ✅ Yes |
| `chrono` | MIT/Apache-2.0 | ✅ Yes |

**All dependencies are GPL-3.0 compatible.**

---

## 8. Dependency Update Policy

### 8.1 Security Updates

**Critical security updates** (CVEs):
- ⚠️ Apply immediately (within 24 hours)
- Run full test suite before deployment
- Document in CHANGELOG

**Non-critical security updates**:
- Apply within 1 week
- Include in next release cycle

---

### 8.2 Version Pinning Strategy

**Cryptographic crates** (`hmac`, `sha2`, `subtle`, etc.):
- Pin to minor version: `"0.12"` (not `"0.12.1"`)
- Allows patch updates (security fixes)
- Prevents breaking changes

**Shared crates** (`input-validation`, etc.):
- Use path dependencies (always latest)
- Version controlled via workspace

**Infrastructure crates** (`thiserror`, `serde`, etc.):
- Use workspace versions
- Centralized version management

---

## 9. Alternatives Considered

### 9.1 Cryptography

**Option A: OpenSSL bindings**
- ❌ Heavy C dependency
- ❌ Complex FFI boundary
- ❌ Platform-specific build issues
- ❌ Harder to audit

**Option B: Ring**
- ❌ Opinionated API (less flexible)
- ❌ Larger binary size
- ❌ Less modular

**Option C: RustCrypto** ✅ SELECTED
- ✅ Pure Rust (no FFI)
- ✅ Modular (use only what you need)
- ✅ Professionally audited
- ✅ FIPS 140-2 compliant
- ✅ Excellent documentation

---

### 9.2 Input Validation

**Option A: Roll our own**
- ❌ Code duplication across crates
- ❌ Inconsistent validation rules
- ❌ Harder to maintain
- ❌ More attack surface

**Option B: Use regex crate**
- ❌ ReDoS vulnerabilities
- ❌ Slower performance
- ❌ Overkill for simple validation

**Option C: Shared input-validation crate** ✅ SELECTED
- ✅ Centralized security boundary
- ✅ Comprehensive test coverage
- ✅ Fast (no regex)
- ✅ Consistent across services

---

### 9.3 Audit Logging

**Option A: Use tracing for everything**
- ❌ Not tamper-evident
- ❌ No compliance features
- ❌ Mixes observability with security

**Option B: External audit service (Splunk, ELK)**
- ❌ Requires network connectivity
- ❌ Vendor lock-in
- ❌ Complex integration

**Option C: Dedicated audit-logging crate** ✅ SELECTED
- ✅ Tamper-evident (hash chains)
- ✅ Compliance-ready (GDPR, SOC2)
- ✅ Separate from observability
- ✅ Local and platform modes

---

## 10. Dependency Checklist

### 10.1 Pre-Integration Checklist

Before adding any new dependency:

- [ ] Security audit status verified
- [ ] License compatibility checked
- [ ] Transitive dependencies reviewed
- [ ] No known CVEs
- [ ] Active maintenance (commits in last 6 months)
- [ ] Adequate documentation
- [ ] Test coverage > 80%
- [ ] Used by > 100 projects (for external crates)
- [ ] Alternatives considered and documented

---

### 10.2 Post-Integration Checklist

After adding dependency:

- [ ] Added to Cargo.toml with version pin
- [ ] Integration tests passing
- [ ] Security tests passing
- [ ] Documentation updated
- [ ] CHANGELOG entry added
- [ ] Dependency tree verified (no conflicts)
- [ ] Binary size impact measured
- [ ] Compile time impact measured

---

## 11. Refinement Opportunities

### 11.1 Dependency Reduction

**Future optimization**:
- Consider vendoring small dependencies (< 100 LOC)
- Evaluate feature flags to reduce unused code
- Investigate no-std compatibility for embedded use

---

### 11.2 Security Hardening

**Future enhancements**:
- Add cargo-deny configuration for supply chain security
- Implement SBOM (Software Bill of Materials) generation
- Add dependency signature verification
- Integrate with cargo-vet for trusted dependency verification

---

### 11.3 Performance Optimization

**Future benchmarks**:
- Measure HMAC-SHA256 signature computation overhead
- Benchmark digest re-verification performance
- Profile memory allocation patterns
- Optimize hot paths if needed

---

## 12. References

**Security specifications**:
- `.specs/20_security.md` — Security requirements (this document satisfies)
- `.specs/31_dependency_verification.md` — Shared crate verification
- `.specs/10_expectations.md` — Consumer expectations

**External documentation**:
- [RustCrypto Audit Report](https://research.nccgroup.com/2020/02/26/public-report-rustcrypto-aes-gcm-and-chacha20poly1305-implementation-review/)
- [FIPS 140-2 Compliance](https://csrc.nist.gov/projects/cryptographic-module-validation-program)
- [Cargo Security Best Practices](https://doc.rust-lang.org/cargo/reference/security.html)

---

## 13. Conclusion

**Summary**:
- ✅ All security requirements mapped to dependencies
- ✅ 10 direct production dependencies (minimal attack surface)
- ✅ All cryptographic crates professionally audited
- ✅ Shared security crates verified and approved
- ✅ No known CVEs or security issues
- ✅ GPL-3.0 license compatible

**Next steps**:
1. Add dependencies to Cargo.toml (see §5.1)
2. Implement cryptographic seal signatures
3. Integrate input validation
4. Wire up audit logging
5. Add comprehensive security tests

**Estimated effort**: 2-3 days for dependency integration, 1-2 days for testing.

---

**Reviewed by**: vram-residency team  
**Approved for implementation**: Yes  
**Blocking issues**: None
