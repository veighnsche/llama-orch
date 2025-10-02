# Test Coverage Analysis — vram-residency

**Date**: 2025-10-02  
**Target**: 80% unit test coverage  
**Current Status**: ~15% coverage (estimated)

---

## Executive Summary

**Current Coverage**: ~15% (3 tests covering basic functionality only)  
**Target Coverage**: 80%  
**Gap**: ~65% additional coverage needed  
**Estimated Tests Needed**: ~60-80 additional unit tests

### Critical Gaps

- ❌ **0%** - Validation modules (shard_id, gpu_device, model_size)
- ❌ **0%** - Seal cryptography (signature, digest, key_derivation)
- ❌ **0%** - Policy enforcement and validation
- ❌ **0%** - Audit event emission
- ❌ **0%** - Error handling paths
- ✅ **~30%** - CUDA FFI (cuda_kernel_tests.rs covers basic operations)
- ✅ **~20%** - VramManager (lib.rs has 1 basic test)

---

## Current Test Inventory

### Existing Tests (3 total)

#### 1. `tests/cuda_kernel_tests.rs` (396 lines, 37 tests)
**Coverage**: CUDA FFI layer only

**Tests**:
- Context creation (2 tests)
- Allocation (6 tests)
- Memory copy (11 tests)
- VRAM info queries (3 tests)
- Drop/cleanup (1 test)
- Alignment (1 test)
- Stress tests (2 tests)
- Error recovery (2 tests)

**What's Covered**:
- ✅ Basic CUDA operations
- ✅ Bounds checking
- ✅ Error conditions
- ✅ Memory management

**What's Missing**:
- ❌ Integration with VramManager
- ❌ Seal operations
- ❌ Multi-GPU scenarios

#### 2. `src/lib.rs::tests::test_seal_model` (1 test)
**Coverage**: Basic VramManager seal operation

**What's Covered**:
- ✅ Basic seal_model() call
- ✅ Basic verify_sealed() call

**What's Missing**:
- ❌ Error paths
- ❌ Edge cases
- ❌ Audit events
- ❌ Signature verification

#### 3. BDD Tests (bdd/ directory)
**Status**: Skeleton only, no actual tests implemented

---

## Module-by-Module Coverage Analysis

### 1. **src/validation/** (0% coverage)

#### `shard_id.rs` (77 lines, 1 public function)
**Function**: `validate_shard_id()`

**Untested Scenarios** (12 needed):
- [ ] Valid shard ID (alphanumeric + `-`, `_`, `:`)
- [ ] Empty shard ID (should fail)
- [ ] Too long shard ID (>256 chars, should fail)
- [ ] Path traversal with `..` (should fail)
- [ ] Path traversal with `/` (should fail)
- [ ] Path traversal with `\` (should fail)
- [ ] Null byte injection (should fail)
- [ ] Control characters (should fail)
- [ ] Invalid characters (special chars, should fail)
- [ ] Exactly 256 chars (boundary, should pass)
- [ ] 257 chars (boundary, should fail)
- [ ] Unicode characters (should fail)

**Estimated Coverage**: 0/12 = **0%**

---

#### `gpu_device.rs` (58 lines, 1 public function)
**Function**: `validate_gpu_device()`

**Untested Scenarios** (8 needed):
- [ ] Valid device (0 <= device < max_devices)
- [ ] Device 0 with max_devices=1 (should pass)
- [ ] Device out of range (should fail)
- [ ] max_devices = 0 (should fail)
- [ ] max_devices > 16 (should fail)
- [ ] Device = max_devices (boundary, should fail)
- [ ] Device = max_devices - 1 (boundary, should pass)
- [ ] Overflow scenarios

**Estimated Coverage**: 0/8 = **0%**

---

#### `model_size.rs` (35 lines, 1 public function)
**Function**: `validate_model_size()`

**Untested Scenarios** (6 needed):
- [ ] Valid size (0 < size <= max_size)
- [ ] Zero size (should fail)
- [ ] Size > max_size (should fail)
- [ ] Size = max_size (boundary, should pass)
- [ ] Size = max_size + 1 (boundary, should fail)
- [ ] Overflow scenarios

**Estimated Coverage**: 0/6 = **0%**

---

### 2. **src/seal/** (0% coverage)

#### `digest.rs` (59 lines, 2 public functions)
**Functions**: `compute_digest()`, `recompute_digest_from_vram()`

**Untested Scenarios** (8 needed):
- [ ] compute_digest() with empty data
- [ ] compute_digest() with small data (1 byte)
- [ ] compute_digest() with large data (1MB)
- [ ] compute_digest() deterministic (same input → same output)
- [ ] compute_digest() different data → different output
- [ ] recompute_digest_from_vram() matches original
- [ ] recompute_digest_from_vram() detects corruption
- [ ] Digest format validation (64 hex chars)

**Estimated Coverage**: 0/8 = **0%**

---

#### `signature.rs` (120 lines, 2 public functions)
**Functions**: `compute_signature()`, `verify_signature()`

**Untested Scenarios** (12 needed):
- [ ] compute_signature() with valid inputs
- [ ] compute_signature() with short seal_key (<32 bytes, should fail)
- [ ] compute_signature() deterministic
- [ ] verify_signature() with valid signature (should pass)
- [ ] verify_signature() with invalid signature (should fail)
- [ ] verify_signature() with wrong seal_key (should fail)
- [ ] verify_signature() with tampered shard_id (should fail)
- [ ] verify_signature() with tampered digest (should fail)
- [ ] verify_signature() with tampered timestamp (should fail)
- [ ] verify_signature() with tampered gpu_device (should fail)
- [ ] verify_signature() timing-safe (constant time)
- [ ] Signature length mismatch (should fail)

**Estimated Coverage**: 0/12 = **0%**

---

#### `key_derivation.rs` (68 lines, 1 public function)
**Function**: `derive_seal_key()`

**Untested Scenarios** (8 needed):
- [ ] Valid worker token and domain
- [ ] Empty worker token (should fail)
- [ ] Empty domain (should fail)
- [ ] Short worker token (<16 chars, should fail)
- [ ] Deterministic (same inputs → same key)
- [ ] Different tokens → different keys
- [ ] Different domains → different keys
- [ ] Output length validation (32 bytes)

**Estimated Coverage**: 0/8 = **0%**

---

### 3. **src/policy/** (0% coverage)

#### `validation.rs` (112 lines, 2 public functions)
**Functions**: `validate_device_properties()`, `check_unified_memory()`

**Untested Scenarios** (10 needed):
- [ ] validate_device_properties() with valid GPU
- [ ] validate_device_properties() with invalid device
- [ ] validate_device_properties() with no GPU (should fail)
- [ ] validate_device_properties() compute capability check
- [ ] validate_device_properties() VRAM capacity check
- [ ] check_unified_memory() in test mode
- [ ] check_unified_memory() with UMA disabled
- [ ] check_unified_memory() with UMA enabled (should fail)
- [ ] Test mode vs production mode behavior
- [ ] GPU info integration

**Estimated Coverage**: 0/10 = **0%**

---

#### `enforcement.rs` (87 lines, 2 public functions)
**Functions**: `enforce_vram_only_policy()`, `is_policy_enforced()`

**Untested Scenarios** (8 needed):
- [ ] enforce_vram_only_policy() success
- [ ] enforce_vram_only_policy() with invalid device (should fail)
- [ ] enforce_vram_only_policy() with UMA enabled (should fail)
- [ ] enforce_vram_only_policy() audit event emission
- [ ] is_policy_enforced() returns true when enforced
- [ ] is_policy_enforced() returns false when not enforced
- [ ] Policy violation scenarios
- [ ] Multi-GPU policy enforcement

**Estimated Coverage**: 0/8 = **0%**

---

### 4. **src/audit/** (0% coverage)

#### `events.rs` (127 lines, 5 public functions)
**Functions**: `emit_vram_sealed()`, `emit_seal_verified()`, `emit_seal_verification_failed()`, `emit_vram_deallocated()`, `emit_policy_violation()`

**Untested Scenarios** (10 needed):
- [ ] emit_vram_sealed() logs correct fields
- [ ] emit_seal_verified() logs correct fields
- [ ] emit_seal_verification_failed() logs CRITICAL severity
- [ ] emit_vram_deallocated() logs correct fields
- [ ] emit_policy_violation() logs CRITICAL severity
- [ ] Event format validation
- [ ] Structured logging fields
- [ ] No sensitive data in logs (seal_key, vram_ptr)
- [ ] Event ordering
- [ ] Integration with tracing subscriber

**Estimated Coverage**: 0/10 = **0%**

---

### 5. **src/allocator/** (~20% coverage)

#### `vram_manager.rs` (271 lines, 6 public functions)
**Functions**: `new()`, `new_with_token()`, `seal_model()`, `verify_sealed()`, `available_vram()`, `total_vram()`

**Current Coverage**: 1 test (basic seal + verify)

**Untested Scenarios** (25 needed):
- [ ] new() creates valid manager
- [ ] new_with_token() with valid token
- [ ] new_with_token() with invalid token (should fail)
- [ ] new_with_token() with no GPU (should fail)
- [ ] seal_model() with zero-size model (should fail)
- [ ] seal_model() with insufficient VRAM (should fail)
- [ ] seal_model() with invalid GPU device (should fail)
- [ ] seal_model() audit event emission
- [ ] seal_model() signature generation
- [ ] seal_model() digest computation
- [ ] seal_model() VRAM allocation tracking
- [ ] verify_sealed() with unsealed shard (should fail)
- [ ] verify_sealed() with invalid signature (should fail)
- [ ] verify_sealed() with corrupted VRAM (should fail)
- [ ] verify_sealed() with missing allocation (should fail)
- [ ] verify_sealed() audit event emission (success)
- [ ] verify_sealed() audit event emission (failure)
- [ ] available_vram() returns correct value
- [ ] total_vram() returns correct value
- [ ] Multiple allocations tracking
- [ ] Allocation cleanup on drop
- [ ] Capacity exhaustion
- [ ] Concurrent operations (if applicable)
- [ ] Error recovery
- [ ] Integration with all modules

**Estimated Coverage**: 1/25 = **4%**

---

#### `cuda_allocator.rs` (0% coverage)
**Note**: Currently unused (dead code), but should be tested if activated

---

### 6. **src/cuda_ffi/** (~30% coverage)

#### `mod.rs` (SafeCudaPtr, CudaContext)
**Current Coverage**: 37 tests in cuda_kernel_tests.rs

**Well Covered**:
- ✅ Basic allocation/deallocation
- ✅ Read/write operations
- ✅ Bounds checking
- ✅ Error handling
- ✅ VRAM queries

**Missing Coverage** (10 needed):
- [ ] Multi-threaded access
- [ ] Concurrent allocations
- [ ] Device switching
- [ ] Large allocation stress test (>10GB)
- [ ] Fragmentation scenarios
- [ ] Memory leak detection
- [ ] Integration with VramManager
- [ ] Error propagation
- [ ] Mock vs real CUDA behavior differences
- [ ] Edge cases in mock implementation

**Estimated Coverage**: 37/47 = **79%** (close to target!)

---

### 7. **src/types/** (0% coverage)

#### `sealed_shard.rs` (SealedShard struct)
**Untested Scenarios** (8 needed):
- [ ] new() creates valid shard
- [ ] Getters return correct values
- [ ] is_sealed() returns false before signature
- [ ] is_sealed() returns true after signature
- [ ] set_signature() works correctly
- [ ] Serialization (if applicable)
- [ ] Clone behavior
- [ ] Field validation

**Estimated Coverage**: 0/8 = **0%**

---

#### `vram_config.rs` (VramConfig struct)
**Untested Scenarios** (4 needed):
- [ ] Config creation
- [ ] Field access
- [ ] Validation
- [ ] Clone behavior

**Estimated Coverage**: 0/4 = **0%**

---

### 8. **src/error.rs** (0% coverage)

**Untested Scenarios** (8 needed):
- [ ] All error variants can be created
- [ ] Error messages are correct
- [ ] Error Display formatting
- [ ] Error source chain
- [ ] From conversions
- [ ] Result type usage
- [ ] Error propagation
- [ ] Error context preservation

**Estimated Coverage**: 0/8 = **0%**

---

## Coverage Summary by Module

| Module | Functions | Current Tests | Needed Tests | Coverage |
|--------|-----------|---------------|--------------|----------|
| **validation/shard_id** | 1 | 0 | 12 | 0% |
| **validation/gpu_device** | 1 | 0 | 8 | 0% |
| **validation/model_size** | 1 | 0 | 6 | 0% |
| **seal/digest** | 2 | 0 | 8 | 0% |
| **seal/signature** | 2 | 0 | 12 | 0% |
| **seal/key_derivation** | 1 | 0 | 8 | 0% |
| **policy/validation** | 2 | 0 | 10 | 0% |
| **policy/enforcement** | 2 | 0 | 8 | 0% |
| **audit/events** | 5 | 0 | 10 | 0% |
| **allocator/vram_manager** | 6 | 1 | 25 | 4% |
| **cuda_ffi** | ~10 | 37 | 10 | 79% |
| **types/sealed_shard** | ~8 | 0 | 8 | 0% |
| **types/vram_config** | ~4 | 0 | 4 | 0% |
| **error** | ~8 | 0 | 8 | 0% |
| **TOTAL** | ~53 | 38 | 137 | **~22%** |

---

## Recommended Test Implementation Plan

### Phase 1: Critical Security Functions (P0) — 40 tests
**Target**: Cover all cryptographic and validation code

1. **Validation** (26 tests)
   - shard_id.rs: 12 tests
   - gpu_device.rs: 8 tests
   - model_size.rs: 6 tests

2. **Seal Cryptography** (28 tests)
   - digest.rs: 8 tests
   - signature.rs: 12 tests
   - key_derivation.rs: 8 tests

**Estimated Coverage After Phase 1**: ~45%

---

### Phase 2: Policy & Audit (P1) — 28 tests
**Target**: Cover policy enforcement and audit logging

3. **Policy** (18 tests)
   - validation.rs: 10 tests
   - enforcement.rs: 8 tests

4. **Audit** (10 tests)
   - events.rs: 10 tests

**Estimated Coverage After Phase 2**: ~65%

---

### Phase 3: Integration & Types (P2) — 25 tests
**Target**: Cover VramManager integration and types

5. **VramManager** (24 additional tests)
   - Complete coverage of all functions
   - Error paths
   - Integration scenarios

6. **Types** (12 tests)
   - sealed_shard.rs: 8 tests
   - vram_config.rs: 4 tests

7. **Error** (8 tests)
   - error.rs: 8 tests

**Estimated Coverage After Phase 3**: ~85%

---

### Phase 4: CUDA FFI Polish (P3) — 10 tests
**Target**: Fill remaining gaps in CUDA layer

8. **CUDA FFI** (10 additional tests)
   - Multi-threading
   - Stress tests
   - Integration tests

**Final Coverage**: ~90%

---

## Test File Organization

**Per .specs/40_testing.md**:
- ✅ **Unit tests**: `#[cfg(test)] mod tests` blocks in each source file
- ✅ **Integration tests**: BDD tests in `bdd/` directory
- ✅ **CUDA-specific tests**: `tests/cuda_kernel_tests.rs` (already exists)

Recommended structure:

```
src/
├── validation/
│   ├── shard_id.rs               (#[cfg(test)] mod tests - 12 tests needed)
│   ├── gpu_device.rs             (#[cfg(test)] mod tests - 8 tests needed)
│   └── model_size.rs             (#[cfg(test)] mod tests - 6 tests needed)
├── seal/
│   ├── digest.rs                 (#[cfg(test)] mod tests - 8 tests needed)
│   ├── signature.rs              (#[cfg(test)] mod tests - 12 tests needed)
│   └── key_derivation.rs         (#[cfg(test)] mod tests - 8 tests needed)
├── policy/
│   ├── validation.rs             (#[cfg(test)] mod tests - 10 tests needed)
│   └── enforcement.rs            (#[cfg(test)] mod tests - 8 tests needed)
├── audit/
│   └── events.rs                 (#[cfg(test)] mod tests - 10 tests needed)
├── allocator/
│   └── vram_manager.rs           (#[cfg(test)] mod tests - 24 tests needed)
└── types/
    ├── sealed_shard.rs           (#[cfg(test)] mod tests - 8 tests needed)
    └── vram_config.rs            (#[cfg(test)] mod tests - 4 tests needed)

tests/
└── cuda_kernel_tests.rs          (✅ exists, 37 tests)

bdd/
├── features/
│   ├── seal_model.feature        (❌ needs implementation)
│   └── verify_seal.feature       (❌ needs implementation)
└── src/
    └── steps/                    (❌ needs implementation)
```

---

## Effort Estimate

| Phase | Tests | Estimated Hours | Priority |
|-------|-------|-----------------|----------|
| Phase 1 | 54 tests | 12-16 hours | P0 (Critical) |
| Phase 2 | 28 tests | 6-8 hours | P1 (High) |
| Phase 3 | 44 tests | 10-12 hours | P2 (Medium) |
| Phase 4 | 10 tests | 2-4 hours | P3 (Low) |
| **TOTAL** | **136 tests** | **30-40 hours** | |

---

## Immediate Next Steps

1. **Create test file structure** (15 min)
2. **Implement Phase 1: Validation tests** (4-5 hours)
3. **Implement Phase 1: Seal tests** (6-8 hours)
4. **Run coverage tool to validate** (30 min)
5. **Iterate until 80% coverage achieved**

---

## Notes

- Current coverage is estimated at **~22%** (38 tests covering ~22% of code)
- To reach **80% coverage**, we need approximately **100+ additional tests**
- CUDA FFI layer is already well-tested (79% coverage)
- Biggest gaps are in validation, seal, policy, and audit modules (0% coverage)
- All critical security functions are currently untested
- BDD tests exist but are not implemented

---

## Coverage Measurement

To measure actual coverage (requires cargo-tarpaulin or cargo-llvm-cov):

```bash
# Install coverage tool
cargo install cargo-tarpaulin
# OR
cargo install cargo-llvm-cov

# Run coverage
cargo tarpaulin -p vram-residency --out Html
# OR
cargo llvm-cov --package vram-residency --html

# View report
open tarpaulin-report.html
# OR
open target/llvm-cov/html/index.html
```

---

**Conclusion**: The crate needs significant test expansion to reach 80% coverage. Focus on Phase 1 (validation + seal) first as these are security-critical functions.
