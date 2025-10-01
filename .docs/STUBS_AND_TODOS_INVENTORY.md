# Stubs and TODOs Inventory

**Date**: 2025-10-01  
**Purpose**: Comprehensive inventory of all stubs, shims, placeholders, and unimplemented functionality in `bin/`  
**Context**: Added after studying ARCHITECTURE_CHANGE_PLAN.md and SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md

---

## Executive Summary

This document catalogs all incomplete implementations, stubs, and placeholders found in the `bin/` directory (binaries and their crates). Each item has been annotated with TODO comments referencing the architecture change plan and security audit.

**Total items identified**: 38 items cataloged
- **14 in binaries** (orchestratord, pool-managerd, worker-orcd)
- **18 in crates** (minimal/stub implementations)
- **6 production-ready crates** (circuit-breaker, rate-limiting, secrets-management, input-validation, narration-core, pool-registry-types)

**Files modified**: 27 files with TODO annotations

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

## Crate-Level Stubs (Additional 8 items)

### 15. Resource Limits (MINIMAL)
**File**: `bin/shared-crates/resource-limits/src/lib.rs`  
**Status**: Struct definition only, no enforcement  
**TODO Added**: Lines 5-12, 37-42

**Required Implementation**:
- Add enforcement logic (not just struct definition)
- Implement VRAM tracking per job
- Add execution timeout enforcement with cancellation
- Implement concurrent job limiting
- Add resource exhaustion detection

**Security Impact**: SECURITY_AUDIT Issue #17 (no resource limits) - **HIGH**

---

### 16. Retry Policy (MINIMAL)
**File**: `bin/shared-crates/retry-policy/src/lib.rs`  
**Status**: Basic exponential backoff, no jitter  
**TODO Added**: Lines 5-11, 39-42

**Required Implementation**:
- Add jitter to prevent thundering herd
- Implement circuit breaker integration
- Add retry budget tracking
- Implement per-error-type retry policies
- Add metrics for retry attempts

---

### 17. Job Timeout (STUB)
**File**: `bin/orchestratord-crates/job-timeout/src/lib.rs`  
**Status**: Empty struct, no timeout enforcement  
**TODO Added**: Lines 3-10, 31-35

**Required Implementation**:
- Add timeout tracking per job_id
- Implement async timeout with tokio::time::timeout
- Add cancellation token propagation
- Emit timeout events for observability
- Integrate with task-cancellation crate

---

### 18. Agentic API (STUB)
**File**: `bin/orchestratord-crates/agentic-api/src/lib.rs`  
**Status**: Empty struct, no implementation  
**TODO Added**: Lines 3-10, 25-29

**Required Implementation**:
- Define agentic workflow types (tool calls, function calling)
- Implement multi-turn conversation state management
- Add tool/function registry
- Implement streaming responses for agentic workflows
- Add context window management

---

### 19. Platform API (MINIMAL)
**File**: `bin/orchestratord-crates/platform-api/src/lib.rs`  
**Status**: Only /health endpoint  
**TODO Added**: Lines 3-10, 21-25

**Required Implementation**:
- Add /metrics endpoint (Prometheus format)
- Add /version endpoint (build info, git commit)
- Add /status endpoint (service health, dependencies)
- Add /config endpoint (safe config inspection)
- Add authentication middleware integration

---

### 20. Inference Engine (STUB)
**File**: `bin/worker-orcd-crates/inference/src/lib.rs`  
**Status**: Returns hardcoded tokens  
**TODO Added**: Lines 5-17, 48-54

**Required Implementation**:
- Implement cuBLAS GEMM integration
- Implement RoPE kernel (rope_llama, rope_neox variants)
- Implement attention kernel (prefill + decode, GQA support)
- Implement RMSNorm kernel
- Implement sampling kernel (greedy, top-k, temperature)
- Implement forward pass and KV cache management

**Security Impact**: SECURITY_AUDIT Issue #11 (unsafe CUDA FFI) - **CRITICAL**

---

### 21. Worker API (STUB)
**File**: `bin/worker-orcd-crates/api/src/lib.rs`  
**Status**: Only /ready stub endpoint  
**TODO Added**: Lines 5-14, 30-35, 41-44

**Required Implementation**:
- Implement POST /worker/plan endpoint
- Implement POST /worker/commit endpoint
- Implement GET /worker/ready endpoint (full response)
- Implement POST /worker/execute endpoint
- Add Bearer token authentication middleware
- Add input validation for all endpoints

**Security Impact**: SECURITY_AUDIT Issue #1 (worker-orcd endpoint auth) - **CRITICAL**

---

### 22. Circuit Breaker (COMPLETE)
**File**: `bin/shared-crates/circuit-breaker/src/lib.rs`  
**Status**: ✅ Fully implemented with tests  
**No TODOs needed** - This crate is production-ready

---

### 23. Rate Limiting (COMPLETE)
**File**: `bin/shared-crates/rate-limiting/src/lib.rs`  
**Status**: ✅ Fully implemented with token bucket algorithm  
**No TODOs needed** - This crate is production-ready

---

### 24. Secrets Management (COMPLETE)
**File**: `bin/shared-crates/secrets-management/src/lib.rs`  
**Status**: ✅ Fully implemented with secure loading and zeroization  
**No TODOs needed** - This crate is production-ready

---

### 25. Audit Logging (MINIMAL)
**File**: `bin/shared-crates/audit-logging/src/lib.rs`  
**Status**: Logs to tracing only, no file persistence  
**TODO Added**: Lines 81-88

**Required Implementation**:
- Write audit events to append-only file
- Add tamper-evident logging (checksums, signatures)
- Implement log rotation with retention policy
- Add external system integration (syslog, SIEM)
- Implement structured query interface for forensics
- Add compliance reporting (GDPR, SOC2)

**Security Impact**: Audit trail requirements for compliance

---

### 26. Deadline Propagation (MINIMAL)
**File**: `bin/shared-crates/deadline-propagation/src/lib.rs`  
**Status**: Basic deadline tracking, missing HTTP integration  
**TODO Added**: Lines 103-107

**Required Implementation**:
- Add `from_header()` to parse X-Deadline header
- Add `to_tokio_timeout()` for async timeout integration
- Add `propagate_to_request()` to add header to outbound requests
- Add `with_buffer()` to add safety margin for multi-hop calls
- Integrate with orchestratord → pool-managerd → worker-orcd chain

---

### 27. Input Validation (COMPLETE)
**File**: `bin/shared-crates/input-validation/src/lib.rs`  
**Status**: ✅ Fully implemented with path traversal, null byte, and range validation  
**No TODOs needed** - This crate is production-ready

---

### 28. Narration Core (COMPLETE)
**File**: `bin/shared-crates/narration-core/src/lib.rs`  
**Status**: ✅ Fully implemented with structured logging, secret redaction, and OpenTelemetry integration  
**No TODOs needed** - This crate is production-ready

---

### 29. Pool Registry Types (COMPLETE)
**File**: `bin/shared-crates/pool-registry-types/src/lib.rs`  
**Status**: ✅ Fully implemented with health, node, and pool types  
**No TODOs needed** - This crate is production-ready

---

## pool-managerd-crates Stubs (Additional 7 items)

### 30. Model Cache (STUB)
**File**: `bin/pool-managerd-crates/model-cache/src/lib.rs`  
**Status**: Empty struct, no caching logic  
**TODO Added**: Lines 3-11, 26-31

**Required Implementation**:
- Implement LRU cache for frequently used models
- Add cache hit/miss tracking and metrics
- Implement cache warming (preload popular models)
- Add cache eviction policy (coordinate with model-eviction crate)
- Implement cache persistence across restarts
- Add cache size limits (disk space, VRAM)

---

### 31. Model Eviction (MINIMAL)
**File**: `bin/pool-managerd-crates/model-eviction/src/lib.rs`  
**Status**: Enum defined, no eviction logic  
**TODO Added**: Lines 3-11, 34-39

**Required Implementation**:
- Implement LRU (Least Recently Used) eviction
- Implement LFU (Least Frequently Used) eviction
- Implement cost-based eviction (evict cheapest to reload)
- Add VRAM pressure detection
- Implement eviction scoring algorithm
- Add metrics for eviction events

---

### 32. Pool-managerd API (MINIMAL)
**File**: `bin/pool-managerd-crates/api/src/lib.rs`  
**Status**: Only /health endpoint  
**TODO Added**: Lines 5-13, 29-35, 41-44

**Required Implementation**:
- Add POST /pools/:id/preload endpoint (model preloading)
- Add GET /pools/:id/status endpoint (pool health and capacity)
- Add POST /pools/:id/drain endpoint (graceful shutdown)
- Add POST /pools/:id/reload endpoint (config reload)
- Add POST /workers/register endpoint (worker registration)
- Add Bearer token authentication middleware

**Security Impact**: SECURITY_AUDIT Issue #8 (pool-managerd auth) - **CRITICAL**

---

### 33. Error Recovery (STUB)
**File**: `bin/pool-managerd-crates/error-recovery/src/lib.rs`  
**Status**: Empty struct, no recovery logic  
**TODO Added**: Lines 3-11, 26-31

**Required Implementation**:
- Implement worker restart on failure
- Add exponential backoff for restart attempts
- Implement circuit breaker for failing workers
- Add automatic model reload on corruption
- Implement health check recovery actions
- Add recovery metrics and alerting

---

### 34. Health Monitor (MINIMAL)
**File**: `bin/pool-managerd-crates/health-monitor/src/lib.rs`  
**Status**: Basic heartbeat tracking only  
**TODO Added**: Lines 41-47

**Required Implementation**:
- Add multi-worker health tracking
- Implement active health checks (HTTP polling)
- Add unhealthy worker detection
- Emit health metrics
- Integrate with error-recovery crate

---

### 35. Pool Registry (MINIMAL)
**File**: `bin/pool-managerd-crates/pool-registry/src/lib.rs`  
**Status**: Basic HashMap storage only  
**TODO Added**: Lines 46-53

**Required Implementation**:
- Add health update methods
- Implement pool listing and filtering
- Add slot allocation/release tracking
- Implement draining state management
- Add pool removal with cleanup

---

### 36. Pool Router (STUB)
**File**: `bin/pool-managerd-crates/router/src/lib.rs`  
**Status**: Stub router with single endpoint  
**TODO Added**: Lines 3-10, 21-25

**Required Implementation**:
- Implement request routing to appropriate pool
- Add load balancing across pools (round-robin, least-loaded)
- Implement sticky routing for sessions
- Add health-aware routing (skip unhealthy pools)
- Implement routing metrics
- Add circuit breaker integration

---

### Model Provisioner (COMPLETE)
**File**: `bin/pool-managerd-crates/model-provisioner/src/lib.rs`  
**Status**: ✅ Fully implemented with API, config, and metadata modules  
**No TODOs needed** - This crate has complete structure

---

## worker-orcd-crates Stubs (Additional 2 items)

### 37. Error Handler (MINIMAL)
**File**: `bin/worker-orcd-crates/error-handler/src/lib.rs`  
**Status**: Basic error enum, minimal handling  
**TODO Added**: Lines 3-11, 43-49

**Required Implementation**:
- Add comprehensive error classification (retryable, fatal, transient)
- Implement error recovery strategies per error type
- Add error rate tracking and circuit breaking
- Implement structured error logging with context
- Add error metrics and alerting
- Integrate with CUDA error codes (cudaGetLastError)
- Add memory error detection and recovery

**Security Impact**: SECURITY_AUDIT Issue #11 (unsafe CUDA FFI)

---

### 38. Execution Planner (MINIMAL)
**File**: `bin/worker-orcd-crates/execution-planner/src/lib.rs`  
**Status**: Hardcoded plan values  
**TODO Added**: Lines 3-11, 32-49

**Required Implementation**:
- Implement KV cache allocation planning
- Add continuous batching support
- Implement dynamic batch size optimization
- Add memory budget tracking
- Implement prefill/decode phase planning
- Add scheduling for multi-request batches
- Integrate with vram-residency for capacity checks

---

## Summary by Priority

### P0 - Critical Security Issues
1. **GGUF parser validation** (Item #14) - Buffer overflow risk
2. **Worker authentication** (Item #21) - Unauthorized access
3. **Pool-managerd authentication** (Items #6, #32) - Unauthorized control
4. **CUDA FFI safety** (Item #12) - Memory corruption risk
5. **Inference engine** (Item #20) - Core security for GPU operations

### P1 - M0 Pilot Blockers
6. **worker-orcd main implementation** (Item #11) - All 7 task groups
7. **VRAM allocation** (Item #12) - Core functionality
8. **Worker spawning** (Item #9) - Lifecycle management
9. **Pool health checking** (Item #4) - Dispatch gating
10. **Worker API endpoints** (Item #21) - RPC protocol
11. **Resource limits enforcement** (Item #15) - DoS protection
12. **Pool-managerd API** (Item #32) - Preload, status, drain endpoints
13. **Health monitoring** (Item #34) - Worker health tracking

### P2 - Phase 2 Cleanup
12. **Remove adapter-host shim** (Item #5) - Architecture cleanup
13. **Remove adapter abstraction** (Item #7) - Architecture cleanup
14. **Remove engine preload** (Item #10) - Architecture cleanup
15. **Remove control service stub** (Item #1) - Replace with real implementation
16. **Remove catalog service stub** (Item #2) - Replace with real implementation

### P3 - Post-M0 Enhancements
17. **Model fetchers** (Item #8) - HF, HTTP, S3, OCI
18. **Artifact store improvements** (Item #3) - Compression, GC, etc.
19. **Job timeout enforcement** (Item #17) - Production hardening
20. **Retry policy enhancements** (Item #16) - Jitter, circuit breaker integration
21. **Agentic API** (Item #18) - Tool calling, multi-turn conversations
22. **Platform API expansion** (Item #19) - Metrics, version, status endpoints
23. **Audit logging persistence** (Item #25) - File-based audit trail
24. **Deadline propagation integration** (Item #26) - HTTP header propagation
25. **Model cache** (Item #30) - LRU caching for models
26. **Model eviction** (Item #31) - VRAM pressure management
27. **Error recovery** (Item #33) - Automated self-healing
28. **Pool registry** (Item #35) - Advanced pool management
29. **Pool router** (Item #36) - Load balancing and routing
30. **Worker error handler** (Item #37) - CUDA error classification and recovery
31. **Execution planner** (Item #38) - KV cache planning and continuous batching

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

# Count TODOs by file
rg "TODO\(ARCH-CHANGE\)" bin/ --count-matches

# List all shared-crates
ls -1 bin/shared-crates/
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
