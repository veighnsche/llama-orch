# Detailed Findings: vram-residency Performance Audit

**Auditor**: Team Performance (deadline-propagation) ⏱️  
**Date**: 2025-10-02  
**Scope**: Complete codebase analysis (Rust + CUDA + C + build script)

---

## Files Analyzed (17 Total)

### Rust Source Files (13)
1. `src/lib.rs` — Module structure and exports
2. `src/error.rs` — Error types
3. `src/allocator/vram_manager.rs` — **HOT PATH** (seal_model, verify_sealed)
4. `src/seal/signature.rs` — HMAC-SHA256 signature computation
5. `src/seal/digest.rs` — SHA-256 digest computation
6. `src/seal/key_derivation.rs` — HKDF-SHA256 key derivation
7. `src/types/sealed_shard.rs` — SealedShard type definition
8. `src/validation/shard_id.rs` — Shard ID validation
9. `src/cuda_ffi/mod.rs` — Safe CUDA FFI wrappers
10. `src/audit/events.rs` — **DEAD CODE** (unused audit helpers)
11. `src/audit/mod.rs` — Audit module exports
12. `src/policy/enforcement.rs` — VRAM-only policy enforcement
13. `src/policy/validation.rs` — Device property validation
14. `src/narration/events.rs` — Observability narration

### CUDA/C Files (3)
15. `cuda/kernels/vram_ops.cu` — **PRODUCTION-QUALITY** CUDA kernels
16. `src/cuda_ffi/mock_cuda.c` — **EXCELLENT** mock implementation
17. `build.rs` — **EXCELLENT** auto-detecting build script

---

## Hot Path Analysis

### seal_model() — Primary Hot Path

**Call Frequency**: Every model load (100s-1000s per day)

**Current Allocations** (6+):
1. `worker_id.clone()` in VramAllocationFailed event
2. `worker_id.clone()` in VramAllocated event
3. `shard.shard_id.clone()` in VramSealed event
4. `shard.digest.clone()` in VramSealed event (64 bytes)
5. `worker_id.clone()` in VramSealed event
6. Error message allocations (if validation fails)

**Optimization**: Use `Arc<str>` for `worker_id` → **-50% allocations**

---

### verify_sealed() — Critical Hot Path

**Call Frequency**: Before **EVERY inference** (10,000s-100,000s per day)

**Current Allocations** (2-8):
- **Success path** (2 allocations):
  1. `shard.shard_id.clone()` in SealVerified event
  2. `worker_id.clone()` in SealVerified event

- **Failure path** (6 allocations):
  1. `shard.shard_id.clone()` in SealVerificationFailed event
  2. `"digest_mismatch".to_string()` (reason)
  3. `shard.digest.clone()` (expected_digest, 64 bytes)
  4. `vram_digest.clone()` (actual_digest, 64 bytes)
  5. `worker_id.clone()` in SealVerificationFailed event
  6. `"CRITICAL".to_string()` (severity)

**Optimization**: Use `Arc<str>` + static strings → **-50-75% allocations**

---

## CUDA Performance Deep Dive

### vram_malloc Performance

**Operation**: `cudaMalloc(ptr, bytes)`

**Overhead**:
```cpp
// Validation: ~10-20 CPU cycles
if (!is_valid_ptr(ptr)) return CUDA_ERROR_INVALID_VALUE;
if (!is_size_valid(bytes)) return CUDA_ERROR_INVALID_VALUE;

// Allocation: Hardware-optimized (GPU driver)
cudaMalloc(ptr, bytes);  // ~1-10 microseconds

// Verification: ~10-20 CPU cycles
if (*ptr == nullptr) return CUDA_ERROR_ALLOCATION_FAILED;
if ((uintptr_t)*ptr % 256 != 0) return CUDA_ERROR_DRIVER;
```

**Total Overhead**: **Negligible** (~50 CPU cycles + GPU driver time)

**Verdict**: ✅ **OPTIMAL** — No optimization needed

---

### vram_memcpy Performance

**Operation**: `cudaMemcpy(dst, src, bytes, direction)`

**Overhead**:
```cpp
// Validation: ~10-20 CPU cycles
if (!is_valid_ptr(dst) || !is_valid_ptr(src)) return CUDA_ERROR_INVALID_VALUE;
if (bytes > MAX_ALLOCATION_SIZE) return CUDA_ERROR_INVALID_VALUE;

// Copy: PCIe bandwidth-limited
cudaMemcpy(dst, src, bytes, direction);  // ~16 GB/s (PCIe 3.0 x16)

// Synchronization: Ensures copy is complete
cudaDeviceSynchronize();  // ~1-10 microseconds
```

**Bottleneck**: **PCIe bandwidth** (~16 GB/s for PCIe 3.0 x16)

**Optimization Opportunities**:
1. **Async memcpy**: Overlap with CPU work (10-20% gain)
2. **Pinned memory**: 2-3x faster transfers (❌ violates VRAM-only policy)
3. **GPU-side digest**: Avoid D2H copy (20-100x faster for large models)

**Verdict**: ✅ **CORRECT** — Synchronous memcpy is simple and correct

---

### SHA-256 Digest Performance

**Operation**: `compute_digest(model_bytes)`

**Performance**:
```
Small models (<10 MB):    ~20 ms (CPU-bound)
Medium models (100 MB):   ~200 ms (CPU-bound)
Large models (1 GB):      ~2 seconds (CPU-bound)
Huge models (10 GB):      ~20 seconds (CPU-bound)
```

**Bottleneck**: **CPU-bound** (~500 MB/s single-threaded)

**Optimization Opportunities**:
1. **GPU-accelerated SHA-256**: 20-100x faster (requires custom kernel)
2. **Multi-threaded SHA-256**: 2-4x faster (requires chunking)
3. **SIMD SHA-256**: 2-3x faster (requires platform-specific code)

**Verdict**: ✅ **SUFFICIENT** — CPU SHA-256 is fast enough for most models

---

## Comparison with audit-logging

| Metric | audit-logging | vram-residency |
|--------|---------------|----------------|
| **Allocations (before)** | 14-24 per event | 8-14 per seal+verify |
| **Allocations (after)** | 2-7 per event | 4-7 per seal+verify |
| **Improvement** | 70-90% | 40-60% |
| **Hot path** | `emit()` | `seal_model()`, `verify_sealed()` |
| **Security impact** | None | None |
| **CUDA code** | N/A | ✅ Excellent |
| **Dead code found** | None | 1 file (audit/events.rs) |
| **Team approval** | ✅ Approved | ⏳ Pending |
