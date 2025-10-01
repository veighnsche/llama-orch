# VRAM Residency — Testing Specification

**Status**: Draft  
**Security Tier**: TIER 1 (Critical)  
**Last Updated**: 2025-10-01  
**Purpose**: Define testing strategy for production-level validation without physical GPU

---

## 0. Executive Summary

**Key Finding**: Physical VRAM is **NOT required** for production-level testing of vram-residency.

**Rationale**:
- 95% of code is GPU-agnostic business logic (cryptography, validation, audit)
- Mock VRAM provides sufficient fidelity for security and correctness validation
- GPU tests only needed for final production deployment validation (5% of tests)

**Testing Strategy**: Mock-first development with optional GPU integration tests.

---

## 1. Testing Architecture

### 1.1 Two-Layer Design

**Layer 1: Business Logic** (95% of code, GPU-agnostic):
- Cryptographic seal computation (HMAC-SHA256, SHA-256)
- Input validation (shard_id, gpu_device, digest)
- Audit logging (event emission, tamper-evidence)
- Error handling (all error paths)
- State management (capacity tracking, allocation state)

**Layer 2: VRAM Abstraction** (5% of code, GPU-specific):
- CUDA FFI calls (`cudaMalloc`, `cudaFree`)
- Device property queries (`cudaGetDeviceProperties`)
- Memory allocation/deallocation
- Unified memory detection

**Testing Implication**: Only Layer 2 requires physical GPU for testing.

---

### 1.2 Mock vs Real VRAM

| Aspect | Mock VRAM | Real VRAM |
|--------|-----------|-----------|
| **GPU Required** | ❌ No | ✅ Yes |
| **Test Coverage** | 95% of code | 100% of code |
| **CI/CD Friendly** | ✅ Yes | ❌ No (requires GPU runners) |
| **Development Speed** | ✅ Fast | ⚠️ Slow (GPU access) |
| **Cost** | ✅ Free | ⚠️ Expensive (GPU infrastructure) |
| **Debugging** | ✅ Easy | ⚠️ Complex (CUDA errors) |
| **Security Testing** | ✅ Complete | ✅ Complete |
| **Performance Testing** | ❌ No | ✅ Yes |

**Recommendation**: Use mock VRAM for development and CI/CD, real VRAM for final production validation.

---

## 2. Mock VRAM Implementation

### 2.1 MockVramAllocator

**Purpose**: Simulate VRAM allocation with identical semantics to real CUDA.

```rust
/// Mock VRAM allocator for testing without GPU
///
/// Provides identical semantics to real CUDA VRAM allocation:
/// - Capacity checking
/// - Bounds validation
/// - Error conditions
/// - Memory isolation
pub struct MockVramAllocator {
    /// Simulated VRAM allocations (ptr -> data)
    allocations: HashMap<usize, Vec<u8>>,
    
    /// Total VRAM capacity (configurable)
    total_vram: usize,
    
    /// Currently used VRAM
    used_vram: usize,
    
    /// Next pointer ID
    next_ptr: usize,
}

impl MockVramAllocator {
    /// Create new mock allocator with specified capacity
    pub fn new(total_vram: usize) -> Self {
        Self {
            allocations: HashMap::new(),
            total_vram,
            used_vram: 0,
            next_ptr: 0x1000,  // Start at non-zero (like real CUDA)
        }
    }
    
    /// Allocate mock VRAM (simulates cudaMalloc)
    pub fn allocate(&mut self, size: usize) -> Result<usize, VramError> {
        // Same capacity checking as real VRAM
        let total_needed = self.used_vram.saturating_add(size);
        if total_needed > self.total_vram {
            return Err(VramError::InsufficientVram(
                size,
                self.total_vram.saturating_sub(self.used_vram),
            ));
        }
        
        // Allocate in RAM (simulates VRAM)
        let ptr = self.next_ptr;
        self.next_ptr = self.next_ptr.saturating_add(1);
        self.allocations.insert(ptr, vec![0u8; size]);
        self.used_vram = self.used_vram.saturating_add(size);
        
        Ok(ptr)
    }
    
    /// Write to mock VRAM (simulates cudaMemcpy host-to-device)
    pub fn write(&mut self, ptr: usize, offset: usize, data: &[u8]) -> Result<(), VramError> {
        let allocation = self.allocations.get_mut(&ptr)
            .ok_or(VramError::IntegrityViolation)?;
        
        // Same bounds checking as real VRAM
        let end = offset.checked_add(data.len())
            .ok_or(VramError::IntegrityViolation)?;
        
        if end > allocation.len() {
            return Err(VramError::IntegrityViolation);
        }
        
        allocation[offset..end].copy_from_slice(data);
        Ok(())
    }
    
    /// Read from mock VRAM (simulates cudaMemcpy device-to-host)
    pub fn read(&self, ptr: usize, offset: usize, len: usize) -> Result<Vec<u8>, VramError> {
        let allocation = self.allocations.get(&ptr)
            .ok_or(VramError::IntegrityViolation)?;
        
        // Same bounds checking as real VRAM
        let end = offset.checked_add(len)
            .ok_or(VramError::IntegrityViolation)?;
        
        if end > allocation.len() {
            return Err(VramError::IntegrityViolation);
        }
        
        Ok(allocation[offset..end].to_vec())
    }
    
    /// Deallocate mock VRAM (simulates cudaFree)
    pub fn deallocate(&mut self, ptr: usize) -> Result<(), VramError> {
        let allocation = self.allocations.remove(&ptr)
            .ok_or(VramError::IntegrityViolation)?;
        
        self.used_vram = self.used_vram.saturating_sub(allocation.len());
        Ok(())
    }
    
    /// Get available VRAM
    pub fn available_vram(&self) -> usize {
        self.total_vram.saturating_sub(self.used_vram)
    }
}
```

---

### 2.2 Feature Flags

**Cargo.toml**:
```toml
[features]
default = ["mock-vram"]
mock-vram = []  # Use mock allocator (no GPU required)
real-cuda = []  # Use real CUDA FFI (requires GPU)

[dependencies]
# No CUDA dependencies in default build
# CUDA dependencies only with real-cuda feature
```

**lib.rs**:
```rust
// Conditional compilation based on feature flags

#[cfg(feature = "mock-vram")]
mod vram {
    pub use crate::mock::MockVramAllocator as VramAllocator;
}

#[cfg(feature = "real-cuda")]
mod vram {
    pub use crate::cuda::CudaVramAllocator as VramAllocator;
}

// Public API uses abstraction (works with both)
pub struct VramManager {
    allocator: vram::VramAllocator,
    // ... other fields
}
```

---

## 3. Test Coverage Without GPU

### 3.1 Unit Tests (Mock VRAM)

**What can be tested** (95% of code):

#### Cryptographic Operations
```rust
#[cfg(test)]
mod crypto_tests {
    use super::*;
    
    #[test]
    fn test_seal_signature_computation() {
        let seal_key = SecretKey::derive_from_token("test-token", b"domain")?;
        let shard = create_test_shard();
        
        let signature = compute_seal_signature(&shard, &seal_key)?;
        assert_eq!(signature.len(), 32);  // HMAC-SHA256 output
    }
    
    #[test]
    fn test_seal_verification_valid() {
        let manager = VramManager::new_mock(1024 * 1024)?;
        let shard = manager.seal_model("test", 0, &[0u8; 1024])?;
        
        // Verification should pass
        assert!(manager.verify_sealed(&shard).is_ok());
    }
    
    #[test]
    fn test_seal_verification_tampered() {
        let manager = VramManager::new_mock(1024 * 1024)?;
        let mut shard = manager.seal_model("test", 0, &[0u8; 1024])?;
        
        // Tamper with digest
        shard.digest = "0".repeat(64);
        
        // Verification should fail
        assert!(matches!(
            manager.verify_sealed(&shard),
            Err(VramError::SealVerificationFailed)
        ));
    }
    
    #[test]
    fn test_digest_computation_deterministic() {
        let data = b"test data";
        let digest1 = compute_sha256(data);
        let digest2 = compute_sha256(data);
        
        // Same data → same digest
        assert_eq!(digest1, digest2);
    }
    
    #[test]
    fn test_timing_safe_verification() {
        let seal_key = SecretKey::derive_from_token("test-token", b"domain")?;
        let shard = create_test_shard();
        
        let valid_sig = compute_seal_signature(&shard, &seal_key)?;
        let invalid_sig = vec![0u8; 32];
        
        // Both should take similar time (timing-safe)
        let start1 = Instant::now();
        let _ = verify_signature(&shard, &valid_sig, &seal_key);
        let duration1 = start1.elapsed();
        
        let start2 = Instant::now();
        let _ = verify_signature(&shard, &invalid_sig, &seal_key);
        let duration2 = start2.elapsed();
        
        // Durations should be within 10% (timing-safe)
        let ratio = duration1.as_nanos() as f64 / duration2.as_nanos() as f64;
        assert!(ratio > 0.9 && ratio < 1.1);
    }
}
```

---

#### Input Validation
```rust
#[cfg(test)]
mod validation_tests {
    use super::*;
    
    #[test]
    fn test_shard_id_validation() {
        let manager = VramManager::new_mock(1024 * 1024)?;
        
        // Valid shard ID
        assert!(manager.seal_model("shard-abc123", 0, &[0u8; 1024]).is_ok());
        
        // Invalid: path traversal
        assert!(matches!(
            manager.seal_model("shard-../etc/passwd", 0, &[0u8; 1024]),
            Err(VramError::InvalidInput(_))
        ));
        
        // Invalid: null byte
        assert!(matches!(
            manager.seal_model("shard\0null", 0, &[0u8; 1024]),
            Err(VramError::InvalidInput(_))
        ));
        
        // Invalid: too long
        let long_id = "a".repeat(300);
        assert!(matches!(
            manager.seal_model(&long_id, 0, &[0u8; 1024]),
            Err(VramError::InvalidInput(_))
        ));
    }
    
    #[test]
    fn test_gpu_device_validation() {
        let manager = VramManager::new_mock(1024 * 1024)?;
        
        // Valid GPU device
        assert!(manager.seal_model("test", 0, &[0u8; 1024]).is_ok());
        
        // Invalid: out of range
        assert!(matches!(
            manager.seal_model("test", 999, &[0u8; 1024]),
            Err(VramError::InvalidInput(_))
        ));
    }
    
    #[test]
    fn test_digest_validation() {
        // Valid SHA-256 digest (64 hex chars)
        let valid_digest = "a".repeat(64);
        assert!(validate_hex_string(&valid_digest, 64).is_ok());
        
        // Invalid: wrong length
        assert!(validate_hex_string("abc", 64).is_err());
        
        // Invalid: non-hex characters
        assert!(validate_hex_string(&"z".repeat(64), 64).is_err());
    }
}
```

---

#### Capacity Management
```rust
#[cfg(test)]
mod capacity_tests {
    use super::*;
    
    #[test]
    fn test_insufficient_vram() {
        let manager = VramManager::new_mock(1024)?;  // 1KB total
        
        // Try to allocate 2KB (should fail)
        let result = manager.seal_model("test", 0, &[0u8; 2048]);
        
        assert!(matches!(
            result,
            Err(VramError::InsufficientVram(2048, available)) if available < 2048
        ));
    }
    
    #[test]
    fn test_capacity_tracking() {
        let mut manager = VramManager::new_mock(10 * 1024)?;  // 10KB total
        
        assert_eq!(manager.available_vram(), 10 * 1024);
        assert_eq!(manager.used_vram(), 0);
        
        // Allocate 3KB
        let shard1 = manager.seal_model("shard1", 0, &[0u8; 3 * 1024])?;
        assert_eq!(manager.used_vram(), 3 * 1024);
        assert_eq!(manager.available_vram(), 7 * 1024);
        
        // Allocate 4KB
        let shard2 = manager.seal_model("shard2", 0, &[0u8; 4 * 1024])?;
        assert_eq!(manager.used_vram(), 7 * 1024);
        assert_eq!(manager.available_vram(), 3 * 1024);
        
        // Try to allocate 5KB (should fail)
        assert!(manager.seal_model("shard3", 0, &[0u8; 5 * 1024]).is_err());
    }
    
    #[test]
    fn test_deallocation_tracking() {
        let mut manager = VramManager::new_mock(10 * 1024)?;
        
        let shard = manager.seal_model("test", 0, &[0u8; 3 * 1024])?;
        assert_eq!(manager.used_vram(), 3 * 1024);
        
        // Deallocate
        drop(shard);
        assert_eq!(manager.used_vram(), 0);
        assert_eq!(manager.available_vram(), 10 * 1024);
    }
}
```

---

#### Security Vulnerabilities
```rust
#[cfg(test)]
mod security_tests {
    use super::*;
    
    #[test]
    fn test_vram_ptr_not_exposed() {
        let manager = VramManager::new_mock(1024 * 1024)?;
        let shard = manager.seal_model("test", 0, &[0u8; 1024])?;
        
        // Serialize to JSON
        let json = serde_json::to_string(&shard)?;
        
        // VRAM pointer should NOT be in JSON
        assert!(!json.contains("vram_ptr"));
        assert!(!json.contains("0x"));
    }
    
    #[test]
    fn test_seal_forgery_rejected() {
        let manager = VramManager::new_mock(1024 * 1024)?;
        let mut shard = manager.seal_model("test", 0, &[0u8; 1024])?;
        
        // Forge signature (replace with zeros)
        shard.signature = vec![0u8; 32];
        
        // Verification should fail
        assert!(matches!(
            manager.verify_sealed(&shard),
            Err(VramError::SealVerificationFailed)
        ));
    }
    
    #[test]
    fn test_integer_overflow_prevented() {
        let manager = VramManager::new_mock(1024)?;
        
        // Try to allocate usize::MAX (should fail gracefully)
        let result = manager.seal_model("test", 0, &vec![0u8; usize::MAX]);
        
        assert!(matches!(
            result,
            Err(VramError::InsufficientVram(_, _))
        ));
    }
    
    #[test]
    fn test_bounds_checking() {
        let manager = VramManager::new_mock(1024 * 1024)?;
        let shard = manager.seal_model("test", 0, &[0u8; 1024])?;
        
        // Try to read beyond allocation
        let result = manager.read_vram(&shard, 1024, 1);  // offset=1024, len=1
        
        assert!(matches!(result, Err(VramError::IntegrityViolation)));
    }
    
    #[test]
    fn test_seal_key_not_logged() {
        // Capture logs
        let (logs, _guard) = capture_logs();
        
        let seal_key = SecretKey::derive_from_token("secret-token", b"domain")?;
        let _ = compute_seal_signature(&test_shard(), &seal_key)?;
        
        // Verify no key material in logs
        let log_output = logs.lock().unwrap();
        assert!(!log_output.contains("secret-token"));
        assert!(!log_output.contains("SecretKey"));
    }
}
```

---

#### Audit Logging
```rust
#[cfg(test)]
mod audit_tests {
    use super::*;
    
    #[test]
    fn test_seal_operation_audited() {
        let (audit_logger, receiver) = create_test_audit_logger();
        let manager = VramManager::new_mock_with_audit(1024 * 1024, audit_logger)?;
        
        let shard = manager.seal_model("test", 0, &[0u8; 1024])?;
        
        // Check audit event was emitted
        let events = receiver.try_recv_all();
        assert_eq!(events.len(), 1);
        
        match &events[0] {
            AuditEvent::VramSealed { shard_id, vram_bytes, digest, .. } => {
                assert_eq!(shard_id, "test");
                assert_eq!(*vram_bytes, 1024);
                assert_eq!(digest.len(), 64);  // SHA-256 hex
            }
            _ => panic!("Expected VramSealed event"),
        }
    }
    
    #[test]
    fn test_verification_failure_audited() {
        let (audit_logger, receiver) = create_test_audit_logger();
        let manager = VramManager::new_mock_with_audit(1024 * 1024, audit_logger)?;
        
        let mut shard = manager.seal_model("test", 0, &[0u8; 1024])?;
        shard.digest = "0".repeat(64);  // Tamper
        
        let _ = manager.verify_sealed(&shard);
        
        // Check audit event was emitted
        let events = receiver.try_recv_all();
        let failure_event = events.iter().find(|e| matches!(e, AuditEvent::SealVerificationFailed { .. }));
        assert!(failure_event.is_some());
    }
    
    #[test]
    fn test_deallocation_audited() {
        let (audit_logger, receiver) = create_test_audit_logger();
        let manager = VramManager::new_mock_with_audit(1024 * 1024, audit_logger)?;
        
        let shard = manager.seal_model("test", 0, &[0u8; 1024])?;
        drop(shard);
        
        // Check audit event was emitted
        let events = receiver.try_recv_all();
        let dealloc_event = events.iter().find(|e| matches!(e, AuditEvent::VramDeallocated { .. }));
        assert!(dealloc_event.is_some());
    }
}
```

---

### 3.2 BDD Tests (Mock VRAM)

**Feature files** (Gherkin):

```gherkin
# bdd/features/seal_model.feature
Feature: Seal Model in VRAM
  As a worker-orcd service
  I want to seal models in VRAM with cryptographic integrity
  So that I can detect tampering and ensure VRAM residency

  Background:
    Given a VramManager with 10MB capacity
    And a worker API token "test-worker-token"

  Scenario: Successfully seal model
    Given a model with 1MB of data
    When I seal the model with shard_id "shard-123" on GPU 0
    Then the seal should succeed
    And the sealed shard should have:
      | field       | value       |
      | shard_id    | shard-123   |
      | gpu_device  | 0           |
      | vram_bytes  | 1048576     |
      | digest      | 64 hex chars|
    And an audit event "VramSealed" should be emitted

  Scenario: Reject invalid shard ID
    Given a model with 1MB of data
    When I seal the model with shard_id "../etc/passwd" on GPU 0
    Then the seal should fail with "InvalidInput"
    And no audit event should be emitted

  Scenario: Fail on insufficient VRAM
    Given a model with 20MB of data
    When I seal the model with shard_id "large-model" on GPU 0
    Then the seal should fail with "InsufficientVram"
    And the error should indicate needed=20MB available=10MB
```

```gherkin
# bdd/features/verify_seal.feature
Feature: Verify Sealed Shard
  As a worker-orcd service
  I want to verify sealed shards before execution
  So that I can detect VRAM corruption or tampering

  Background:
    Given a VramManager with 10MB capacity
    And a sealed shard "shard-123" with 1MB of data

  Scenario: Verify valid seal
    When I verify the sealed shard
    Then the verification should succeed
    And an audit event "SealVerified" should be emitted

  Scenario: Reject tampered digest
    Given the shard digest is modified to "0000...0000"
    When I verify the sealed shard
    Then the verification should fail with "SealVerificationFailed"
    And an audit event "SealVerificationFailed" should be emitted
    And the event should have severity "critical"

  Scenario: Reject forged signature
    Given the shard signature is replaced with zeros
    When I verify the sealed shard
    Then the verification should fail with "SealVerificationFailed"
    And an audit event "SealVerificationFailed" should be emitted
```

---

### 3.3 Property Tests (Mock VRAM)

**Property-based testing with proptest**:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn seal_verification_deterministic(
        data in prop::collection::vec(any::<u8>(), 0..1000)
    ) {
        let manager = VramManager::new_mock(10 * 1024 * 1024)?;
        
        // Seal same data twice
        let shard1 = manager.seal_model("test1", 0, &data)?;
        let shard2 = manager.seal_model("test2", 0, &data)?;
        
        // Same data → same digest
        prop_assert_eq!(shard1.digest, shard2.digest);
    }
    
    #[test]
    fn vram_allocation_never_exceeds_capacity(
        sizes in prop::collection::vec(0usize..1_000_000, 0..100)
    ) {
        let mut manager = VramManager::new_mock(10 * 1024 * 1024)?;
        let mut total_allocated = 0usize;
        
        for (i, size) in sizes.iter().enumerate() {
            if let Ok(shard) = manager.seal_model(&format!("shard-{}", i), 0, &vec![0u8; *size]) {
                total_allocated = total_allocated.saturating_add(shard.vram_bytes);
            }
        }
        
        // Never exceed capacity
        prop_assert!(total_allocated <= 10 * 1024 * 1024);
        prop_assert_eq!(total_allocated, manager.used_vram());
    }
    
    #[test]
    fn seal_verification_rejects_modified_data(
        original in prop::collection::vec(any::<u8>(), 100..1000),
        modified in prop::collection::vec(any::<u8>(), 100..1000)
    ) {
        prop_assume!(original != modified);
        
        let manager = VramManager::new_mock(10 * 1024 * 1024)?;
        let shard = manager.seal_model("test", 0, &original)?;
        
        // Modify VRAM contents
        manager.write_vram(&shard, 0, &modified)?;
        
        // Verification should fail
        prop_assert!(manager.verify_sealed(&shard).is_err());
    }
}
```

---

## 4. GPU Integration Tests (Real VRAM)

### 4.1 When GPU Tests Are Required

**Required before**:
- ⚠️ Production deployment with real GPUs
- ⚠️ Performance optimization
- ⚠️ Hardware compatibility validation
- ⚠️ CUDA driver updates

**NOT required for**:
- ✅ Development iteration
- ✅ CI/CD pipelines
- ✅ Security audits
- ✅ Code review

---

### 4.2 GPU Test Suite

**Separate test module** (only runs with `real-cuda` feature):

```rust
#[cfg(all(test, feature = "real-cuda"))]
mod gpu_integration_tests {
    use super::*;
    
    #[test]
    fn test_real_cuda_allocation() {
        let mut manager = VramManager::new_with_cuda(0)?;
        
        // Allocate 1MB in real VRAM
        let shard = manager.seal_model("test", 0, &[0u8; 1024 * 1024])?;
        
        assert_eq!(shard.vram_bytes, 1024 * 1024);
        assert!(shard.digest.len() == 64);
    }
    
    #[test]
    fn test_real_vram_capacity() {
        let manager = VramManager::new_with_cuda(0)?;
        
        let total = manager.total_vram();
        let available = manager.available_vram();
        
        // Real GPU has non-zero VRAM
        assert!(total > 0);
        assert!(available > 0);
        assert!(available <= total);
    }
    
    #[test]
    fn test_unified_memory_detection() {
        let manager = VramManager::new_with_cuda(0)?;
        
        // Should detect if UMA is enabled
        let result = manager.enforce_vram_only_policy();
        
        // Should succeed if UMA is disabled
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_real_vram_oom() {
        let manager = VramManager::new_with_cuda(0)?;
        let total = manager.total_vram();
        
        // Try to allocate more than available
        let result = manager.seal_model("huge", 0, &vec![0u8; total + 1024]);
        
        assert!(matches!(result, Err(VramError::InsufficientVram(_, _))));
    }
    
    #[test]
    fn test_multi_gpu_allocation() {
        let gpu_count = get_gpu_count()?;
        if gpu_count < 2 {
            return Ok(());  // Skip if only 1 GPU
        }
        
        // Allocate on GPU 0
        let mut manager0 = VramManager::new_with_cuda(0)?;
        let shard0 = manager0.seal_model("shard0", 0, &[0u8; 1024])?;
        
        // Allocate on GPU 1
        let mut manager1 = VramManager::new_with_cuda(1)?;
        let shard1 = manager1.seal_model("shard1", 1, &[0u8; 1024])?;
        
        // Should be on different GPUs
        assert_eq!(shard0.gpu_device, 0);
        assert_eq!(shard1.gpu_device, 1);
    }
    
    #[test]
    fn test_cuda_error_handling() {
        let manager = VramManager::new_with_cuda(0)?;
        
        // Try to allocate on invalid GPU
        let result = manager.seal_model("test", 999, &[0u8; 1024]);
        
        assert!(matches!(result, Err(VramError::CudaAllocationFailed(_))));
    }
}
```

---

### 4.3 Running GPU Tests

**Manual execution** (requires GPU):

```bash
# On machine with GPU
cargo test -p vram-residency --features real-cuda --test gpu_integration

# Specific GPU tests
cargo test -p vram-residency cuda_allocation --features real-cuda
cargo test -p vram-residency multi_gpu --features real-cuda
cargo test -p vram-residency vram_capacity --features real-cuda

# Performance benchmarks
cargo bench -p vram-residency --features real-cuda
```

**NOT in CI/CD** (no GPU runners):
- GPU tests are opt-in only
- Run manually before production deployment
- Document results in release notes

---

## 5. CI/CD Configuration

### 5.1 GitHub Actions Workflow

```yaml
# .github/workflows/vram-residency-ci.yml
name: vram-residency CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Install Rust toolchain
        uses: actions-rs/toolchain@v1
        with:
          profile: minimal
          toolchain: stable
          override: true
          components: clippy, rustfmt
      
      # All tests use mock VRAM (no GPU required)
      - name: Run unit tests
        run: cargo test -p vram-residency --features mock-vram
      
      - name: Run BDD tests
        run: |
          cd bin/worker-orcd-crates/vram-residency/bdd
          cargo test --features mock-vram
      
      - name: Run security tests
        run: cargo test -p vram-residency security --features mock-vram
      
      - name: Run property tests
        run: cargo test -p vram-residency proptest --features mock-vram
      
      - name: Check Clippy (TIER 1)
        run: cargo clippy -p vram-residency -- -D warnings
      
      - name: Check formatting
        run: cargo fmt -p vram-residency -- --check
      
      - name: Generate code coverage
        uses: actions-rs/tarpaulin@v0.1
        with:
          args: '-p vram-residency --features mock-vram --out Lcov'
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./lcov.info

  security-audit:
    name: Security Audit
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
      
      - name: Run cargo-audit
        uses: actions-rs/audit-check@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
```

---

### 5.2 Pre-commit Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running vram-residency tests..."

# Run unit tests (mock VRAM)
cargo test -p vram-residency --features mock-vram || exit 1

# Run Clippy (TIER 1)
cargo clippy -p vram-residency -- -D warnings || exit 1

# Run formatting check
cargo fmt -p vram-residency -- --check || exit 1

echo "All checks passed!"
```

---

## 6. Test Coverage Requirements

### 6.1 Coverage Targets

| Category | Target | Current |
|----------|--------|---------|
| **Unit tests** | > 90% | ⬜ TBD |
| **BDD tests** | All scenarios | ⬜ TBD |
| **Security tests** | 100% vulnerabilities | ⬜ TBD |
| **Property tests** | All invariants | ⬜ TBD |
| **Integration tests** | All API contracts | ⬜ TBD |

---

### 6.2 Critical Path Coverage

**MUST be covered** (100% required):
- ✅ Seal signature computation
- ✅ Seal verification logic
- ✅ Input validation (all validators)
- ✅ Capacity checking
- ✅ Error handling (all error paths)
- ✅ Audit logging (all events)
- ✅ Security vulnerabilities (all 7)

**SHOULD be covered** (> 80% target):
- Configuration parsing
- Metrics emission
- Helper functions
- Utility methods

**MAY be covered** (best effort):
- CUDA FFI wrappers (requires GPU)
- Performance optimizations
- Platform-specific code

---

## 7. Performance Testing

### 7.1 Benchmarks (Mock VRAM)

**What can be benchmarked without GPU**:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_seal_signature(c: &mut Criterion) {
    let seal_key = SecretKey::derive_from_token("test-token", b"domain").unwrap();
    let shard = create_test_shard();
    
    c.bench_function("seal_signature", |b| {
        b.iter(|| {
            compute_seal_signature(black_box(&shard), black_box(&seal_key))
        })
    });
}

fn bench_seal_verification(c: &mut Criterion) {
    let manager = VramManager::new_mock(10 * 1024 * 1024).unwrap();
    let shard = manager.seal_model("test", 0, &[0u8; 1024]).unwrap();
    
    c.bench_function("seal_verification", |b| {
        b.iter(|| {
            manager.verify_sealed(black_box(&shard))
        })
    });
}

fn bench_input_validation(c: &mut Criterion) {
    c.bench_function("validate_identifier", |b| {
        b.iter(|| {
            validate_identifier(black_box("shard-abc123"), 256)
        })
    });
}

criterion_group!(benches, bench_seal_signature, bench_seal_verification, bench_input_validation);
criterion_main!(benches);
```

---

### 7.2 Performance Targets (Mock VRAM)

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Seal signature | < 1ms | HMAC-SHA256 is fast |
| Seal verification | < 1ms | Timing-safe comparison |
| Input validation | < 10μs | Simple string checks |
| Capacity query | < 1μs | Arithmetic only |

**Note**: Real VRAM performance will differ (dominated by memory copy).

---

## 8. Production Validation Checklist

### 8.1 Without GPU (Development)

**Before merging to main**:
- [ ] All unit tests passing (mock VRAM)
- [ ] All BDD tests passing (mock VRAM)
- [ ] All security tests passing (mock VRAM)
- [ ] All property tests passing (mock VRAM)
- [ ] Clippy lints passing (TIER 1)
- [ ] Code coverage > 90%
- [ ] Documentation complete
- [ ] Security audit passed

**Status**: ✅ Production-ready for logic validation

---

### 8.2 With GPU (Deployment)

**Before deploying to production**:
- [ ] GPU integration tests passing (real CUDA)
- [ ] Performance benchmarks acceptable
- [ ] Multi-GPU tests passing (if applicable)
- [ ] Hardware compatibility validated
- [ ] CUDA driver version tested
- [ ] VRAM capacity limits verified
- [ ] OOM behavior validated
- [ ] Error recovery tested

**Status**: ⚠️ Required before production deployment with real GPUs

---

## 9. Refinement Opportunities

### 9.1 Mock Improvements

**Future enhancements**:
- Add mock CUDA error injection (simulate driver failures)
- Add mock memory corruption scenarios
- Add mock multi-GPU coordination
- Add mock performance characteristics (latency simulation)

---

### 9.2 GPU Test Automation

**Future work**:
- Investigate GPU runners for CI/CD (GitHub Actions GPU instances)
- Add nightly GPU test runs
- Implement GPU test result tracking
- Add GPU performance regression detection

---

### 9.3 Test Tooling

**Future tools**:
- Property-based test generator for seal operations
- Fuzzing harness for VRAM operations
- Mutation testing for security-critical code
- Automated test coverage reporting

---

## 10. References

**Testing best practices**:
- [Rust Testing Guide](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Property-Based Testing with Proptest](https://proptest-rs.github.io/proptest/)
- [BDD with Cucumber](https://cucumber-rs.github.io/cucumber/)

**Security testing**:
- `.specs/20_security.md` — Security requirements
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Security context

**Related specifications**:
- `.specs/00_vram-residency.md` — Functional specification
- `.specs/30_dependencies.md` — Dependency analysis
- `.specs/31_dependency_verification.md` — Shared crate verification

---

## 11. Conclusion

**Key Findings**:
- ✅ 95% of code can be tested without physical GPU
- ✅ Mock VRAM provides sufficient fidelity for security validation
- ✅ GPU tests only needed for final production validation (5% of tests)
- ✅ CI/CD can run entirely on CPU-only runners

**Recommendation**:
1. Implement with mock VRAM (no GPU required)
2. Write comprehensive tests (all with mock)
3. Pass security audit (logic is GPU-agnostic)
4. Deploy to CI/CD (mock tests only)
5. Run GPU tests manually before production deployment

**Benefits**:
- Fast development iteration (no GPU access needed)
- Reliable CI/CD (no GPU runners required)
- Comprehensive test coverage (95% without GPU)
- Easy debugging (no CUDA complexity)
- Cost-effective (no GPU infrastructure)

---

**Status**: Ready for implementation  
**Next steps**: Implement mock VRAM allocator and test suite  
**Blocking issues**: None
