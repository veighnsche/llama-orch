# Stubs and TODOs Inventory

**Date**: 2025-10-01  
**Purpose**: Comprehensive inventory of all stubs, shims, placeholders, and unimplemented functionality in `bin/`  
**Context**: Added after studying ARCHITECTURE_CHANGE_PLAN.md and SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md

---

## Executive Summary

This document catalogs all incomplete implementations, stubs, and placeholders found in the `bin/` directory. Each item has been annotated with TODO comments referencing the architecture change plan and security audit.

**Total items identified**: 15 major stubs/placeholders across 3 binaries

---

## orchestratord Stubs

### 1. Control Service (ENTIRE FILE STUB)
**File**: `bin/orchestratord/src/services/control.rs`  
**Status**: Empty stub file  
**TODO Added**: Lines 3-8

**Required Implementation**:
- Remove adapter-host dependencies
- Implement direct worker-orcd communication
- Add pool-managerd lifecycle integration
- Implement control plane operations (drain, reload, purge)

**References**: ARCHITECTURE_CHANGE_PLAN.md Phase 2-3

---

### 2. Catalog Service (ENTIRE FILE STUB)
**File**: `bin/orchestratord/src/services/catalog.rs`  
**Status**: Empty stub file  
**TODO Added**: Lines 3-8

**Required Implementation**:
- Implement catalog operations (list, get, retire models)
- Integrate with pool-managerd model-catalog crate
- Add model lifecycle management
- Implement verification and digest checking

**References**: ARCHITECTURE_CHANGE_PLAN.md Phase 2-3

---

### 3. Filesystem Artifact Store (PLACEHOLDER)
**File**: `bin/orchestratord/src/infra/storage/fs.rs`  
**Status**: Basic implementation, missing advanced features  
**TODO Added**: Lines 3-9

**Required Implementation**:
- Proper content-addressed storage with deduplication
- Garbage collection for old artifacts
- Atomic writes with fsync
- Compression support for large transcripts
- Error recovery and partial write handling

**References**: ARCHITECTURE_CHANGE_PLAN.md Phase 4-5

---

### 4. Pool Health Checking (STUB)
**File**: `bin/orchestratord/src/services/streaming.rs`  
**Function**: `should_dispatch()`  
**Status**: Always returns `true` (stub)  
**TODO Added**: Lines 234-241

**Required Implementation**:
- Implement actual pool health checking via pool-managerd HTTP API
- Check Live+Ready status from registry
- Verify slots_free > 0 before dispatch
- Add backoff/retry logic when pool not ready
- Emit proper error SSE frames with retry hints

**Security Impact**: SECURITY_AUDIT Issue #17 (resource limits)

---

### 5. Adapter Binding Shim (MVP SHIM)
**File**: `bin/orchestratord/src/app/bootstrap.rs`  
**Lines**: 28-46  
**Status**: Feature-gated MVP shim  
**TODO Added**: Lines 29-36

**Required Implementation**:
- **REMOVE** this entire adapter binding block
- Remove adapter-host dependency
- Replace with pool-manager driven worker-orcd registration
- Implement config-schema backed worker sources
- Ensure reload/drain lifecycle integrates with new worker protocol

**Security Impact**: SECURITY_AUDIT Issue #1 (worker-orcd auth)

---

### 6. Purge Endpoint (STUB)
**File**: `bin/orchestratord/src/api/control.rs`  
**Function**: `purge_pool_v2()`  
**Status**: Returns 202 Accepted without doing anything  
**TODO Added**: Lines 119-125

**Required Implementation**:
- Implement actual purge logic via pool-managerd API
- Add authentication/authorization checks
- Implement proper async job tracking
- Add audit logging for purge operations
- Return job_id for tracking purge progress

**Security Impact**: SECURITY_AUDIT Issue #8 (pool-managerd auth)

---

### 7. Adapter Abstraction (SCHEDULED FOR REMOVAL)
**File**: `bin/orchestratord/src/ports/adapters.rs`  
**Status**: Entire file will be removed  
**TODO Added**: Lines 11-13

**Action Required**: Delete this file in Phase 2 cleanup per ARCHITECTURE_CHANGE_PLAN.md §2

---

## pool-managerd Stubs

### 8. Model Fetchers (NOT IMPLEMENTED)
**File**: `bin/pool-managerd-crates/model-catalog/src/lib.rs`  
**Functions**: `FileFetcher::ensure_present()` for HF and URL schemes  
**Status**: Returns `NotImplemented` errors  
**TODO Added**: Lines 13-18, 271-284

**Required Implementation**:
- **HuggingFace fetcher** (hf: scheme)
  - Use huggingface_hub crate or HTTP API
  - Handle authentication tokens
  - Implement caching and resume support
  - Add progress reporting
- **HTTP/HTTPS fetcher**
  - Support HTTP/HTTPS with resume
  - Add digest verification
- **S3 fetcher** (s3: scheme)
  - Support S3 with AWS SDK
- **OCI registry fetcher** (oci: scheme)
  - Support OCI with registry client

**References**: ARCHITECTURE_CHANGE_PLAN.md §2.11 Model Provisioner evaluation

---

### 9. Worker Lifecycle Manager (PLACEHOLDER)
**File**: `bin/pool-managerd-crates/lifecycle/src/lib.rs`  
**Function**: `spawn_worker()`  
**Status**: Returns placeholder PID 1234  
**TODO Added**: Lines 34-41

**Required Implementation**:
- Parse worker config (GPU device, model path, etc.)
- Spawn worker-orcd process with proper arguments
- Set up process monitoring and health checks
- Implement graceful shutdown and restart
- Add privilege dropping (run as non-root user)
- Return actual PID, not placeholder

**Security Impact**: SECURITY_AUDIT Issue #18 (unchecked privileges)

---

### 10. Engine Preload (SCHEDULED FOR REMOVAL)
**File**: `bin/pool-managerd/src/lifecycle/preload.rs`  
**Function**: `execute()` (commented out)  
**Status**: Entire function scheduled for removal  
**TODO Added**: Lines 22-29

**Action Required**: Remove this function in Phase 2. Replace with:
1. pool-managerd spawns worker-orcd process (via lifecycle crate)
2. worker-orcd loads model via Commit endpoint
3. worker-orcd attests sealed status via Ready endpoint
4. pool-managerd updates registry when worker reports ready

**References**: ARCHITECTURE_CHANGE_PLAN.md §2 Components to Remove

---

## worker-orcd Stubs

### 11. Main Entry Point (EMPTY)
**File**: `bin/worker-orcd/src/main.rs`  
**Status**: Empty main function  
**TODO Added**: Lines 8-30

**Required Implementation** (M0 Pilot - 7 Task Groups):

**Task Group 1 (Rust Control Layer)**:
- Parse CLI args (GPU device, config path, etc.)
- Initialize VramManager and ModelLoader
- Set up telemetry and structured logging
- Implement RPC server (Plan/Commit/Ready/Execute endpoints)
- Add Bearer token authentication middleware

**Task Group 2 (CUDA FFI)**:
- Initialize CUDA context and cuBLAS handle
- Set up safe FFI wrappers with bounds checking

**Task Group 3 (Kernels)**:
- Load initial kernel set (GEMM, RoPE, attention, sampling)

**Task Group 4 (Model Loading)**:
- Implement GGUF loader with validation
- Wire up inference engine with token streaming

**Task Group 5 (MCD/ECP)**:
- Implement capability matching logic

**Task Group 6 (Integration)**:
- Add health monitoring and registration with pool-managerd

**Task Group 7 (Validation)**:
- Test with TinyLlama-1.1B
- Verify determinism and VRAM-only policy

**References**: ARCHITECTURE_CHANGE_PLAN.md Phase 3 (all task groups)  
**Security Impact**: SECURITY_AUDIT M0 Must-Fix items 1-10

---

### 12. VRAM Allocation (PLACEHOLDER)
**File**: `bin/worker-orcd-crates/vram-residency/src/lib.rs`  
**Function**: `VramManager::seal_model()`  
**Status**: Hardcoded placeholder pointer `0x7f8a4c000000`  
**TODO Added**: Lines 127-134

**Required Implementation**:
- Use cudarc or cust for CUDA bindings
- Allocate VRAM via cudaMalloc
- Copy model_bytes to GPU via cudaMemcpy
- Verify allocation succeeded
- Add bounds checking and safety wrappers

**Security Impact**: SECURITY_AUDIT Issue #11 (unsafe CUDA FFI)

---

### 13. Seal Verification (STUB)
**File**: `bin/worker-orcd-crates/vram-residency/src/lib.rs`  
**Function**: `VramManager::verify_sealed()`  
**Status**: Only checks if pointer is non-zero  
**TODO Added**: Lines 159-164

**Required Implementation**:
- Re-compute SHA-256 digest from VRAM contents
- Compare with shard.digest to detect tampering
- Add periodic re-verification option
- Emit security alert on mismatch

**Security Impact**: SECURITY_AUDIT Issue #15 (digest TOCTOU)

---

### 14. GGUF Validation (INCOMPLETE)
**File**: `bin/worker-orcd-crates/model-loader/src/lib.rs`  
**Function**: `ModelLoader::validate_gguf()`  
**Status**: Only checks magic number  
**TODO Added**: Lines 131-138

**Required Implementation**:
- Validate all header fields (version, tensor_count, metadata_kv_count)
- Check tensor_count against MAX_TENSORS limit
- Validate metadata key-value pairs
- Check for buffer overflows in string lengths
- Validate tensor shapes and data types
- Add fuzz testing for parser

**Security Impact**: SECURITY_AUDIT Issue #19 (GGUF parser trusts input) - **CRITICAL**

---

## Summary by Priority

### P0 - Critical Security Issues
1. **GGUF parser validation** (Issue #19) - Buffer overflow risk
2. **Worker authentication** (Issue #1) - Unauthorized access
3. **Pool-managerd authentication** (Issue #8) - Unauthorized control
4. **CUDA FFI safety** (Issue #11) - Memory corruption risk

### P1 - M0 Pilot Blockers
5. **worker-orcd main implementation** - All 7 task groups
6. **VRAM allocation** - Core functionality
7. **Worker spawning** - Lifecycle management
8. **Pool health checking** - Dispatch gating

### P2 - Phase 2 Cleanup
9. **Remove adapter-host shim** - Architecture cleanup
10. **Remove adapter abstraction** - Architecture cleanup
11. **Remove engine preload** - Architecture cleanup
12. **Remove control service stub** - Replace with real implementation
13. **Remove catalog service stub** - Replace with real implementation

### P3 - Post-M0 Enhancements
14. **Model fetchers** (HF, HTTP, S3, OCI)
15. **Artifact store improvements** (compression, GC, etc.)

---

## Cross-References

### Architecture Change Plan
- **Phase 2 (Cleanup)**: Items 9-13
- **Phase 3 (M0 Pilot)**: Items 5, 11-14
- **Phase 4-5 (Production)**: Item 15

### Security Audit
- **M0 Must-Fix**: Items 1-4, 12-14
- **Post-M0 Hardening**: Items 6-8

---

## Verification Commands

```bash
# Find all TODO(ARCH-CHANGE) comments
rg "TODO\(ARCH-CHANGE\)" bin/

# Find all placeholder values
rg "Placeholder|placeholder|PLACEHOLDER" bin/ --type rust

# Find all NotImplemented errors
rg "NotImplemented" bin/ --type rust

# Find all stub comments
rg "stub|STUB|Stub" bin/ --type rust
```

---

## Next Steps

1. **Immediate**: Review this inventory with team
2. **Phase 2**: Begin cleanup of adapter-related code (items 9-13)
3. **Phase 3**: Implement worker-orcd M0 pilot (items 5, 11-14)
4. **Security**: Address P0 critical issues before M0 release (items 1-4)

---

**Document Status**: Complete  
**Last Updated**: 2025-10-01  
**Maintainer**: Architecture team
