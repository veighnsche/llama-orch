# VRAM Residency — Consumer Expectations

**Status**: Draft  
**Purpose**: Documents what other crates expect from `vram-residency`  
**Last Updated**: 2025-10-01

---

## 0. Overview

This document catalogs the expectations and dependencies that other worker-orcd crates have on `vram-residency`. It serves as a contract specification to guide implementation priorities and API stability.

**Consuming crates**:
- `worker-api` — HTTP endpoints (Plan/Commit/Ready/Execute)
- `model-loader` — Model validation and loading
- `scheduler` — Job state tracking and VRAM capacity checks
- `worker-orcd` binary — Main integration point

---

## 1. Core Type Expectations

### 1.1 `SealedShard` Struct

**Required by**: `worker-api`, `model-loader`, `scheduler`

**Expected public fields**:
```rust
pub struct SealedShard {
    pub shard_id: String,           // Unique opaque identifier
    pub gpu_device: u32,            // CUDA device index
    pub vram_bytes: usize,          // Size in VRAM
    pub digest: String,             // SHA-256 hex string
    pub sealed_at: SystemTime,      // Seal timestamp
    // vram_ptr: private (MUST NOT be exposed)
}
```

**Expected methods**:
```rust
impl SealedShard {
    pub fn verify(&self, current_digest: &str) -> Result<()>;
}
```

**Usage contexts**:
- **API responses**: Returned in Commit and Ready endpoint responses
- **Seal verification**: Called before Execute endpoint runs inference
- **State tracking**: Scheduler needs to track which shards are loaded

**Security requirements**:
- VRAM pointer MUST remain private (WORKER-4111)
- Digest MUST be SHA-256 hex string
- Clone/Debug traits needed for API serialization

---

### 1.2 `ModelShardHandle` (Alias or Wrapper)

**Required by**: `worker-api`

**Context**: API spec (WORKER-4226, WORKER-4231) references `ModelShardHandle` as the type returned in responses. Current implementation uses `SealedShard` directly.

**Decision needed**:
- **Option A**: `pub type ModelShardHandle = SealedShard;` (type alias)
- **Option B**: Keep `SealedShard` as internal, expose `ModelShardHandle` wrapper
- **Option C**: Rename `SealedShard` → `ModelShardHandle`

**Recommendation**: Use type alias for now, can refactor later if needed.

```rust
pub type ModelShardHandle = SealedShard;
```

---

## 2. VramManager API Expectations

### 2.1 Allocation API

**Required by**: `worker-api` (Commit endpoint), `model-loader`

```rust
pub fn seal_model(
    &mut self,
    model_bytes: &[u8],
    gpu_device: u32
) -> Result<SealedShard>
```

**Expected behavior** (per WORKER-4110-4113):
1. Check available VRAM capacity
2. Allocate VRAM via CUDA FFI (or mock for M0)
3. Copy model bytes to VRAM
4. Compute SHA-256 digest
5. Generate HMAC-SHA256 seal signature
6. Return sealed shard with `sealed: true`

**Error cases**:
- `VramError::InsufficientVram(needed, available)` — Not enough VRAM
- `VramError::CudaAllocationFailed` — CUDA malloc failed
- `VramError::IntegrityViolation` — Digest computation failed

**Usage**:
- Called by Commit endpoint after model validation
- Must be idempotent (calling twice with same bytes should fail or return existing)

---

### 2.2 Verification API

**Required by**: `worker-api` (Execute endpoint)

```rust
pub fn verify_sealed(&self, shard: &SealedShard) -> Result<()>
```

**Expected behavior** (per WORKER-4120-4122):
1. Re-compute digest from VRAM contents
2. Verify HMAC-SHA256 signature
3. Optionally check seal timestamp freshness
4. Return Ok(()) if valid, Err otherwise

**Error cases**:
- `VramError::SealVerificationFailed` — Digest mismatch (tampering detected)
- `VramError::NotSealed` — Shard not properly sealed
- `VramError::IntegrityViolation` — VRAM corruption detected

**Usage**:
- MUST be called before each Execute request (WORKER-4245)
- Detects VRAM corruption or tampering
- Worker transitions to Stopped state on failure (WORKER-4122)

---

### 2.3 Capacity Query API

**Required by**: `worker-api` (Plan endpoint), `scheduler`

**Expected methods**:
```rust
pub fn available_vram(&self) -> usize;
pub fn total_vram(&self) -> usize;
pub fn used_vram(&self) -> usize;
```

**Usage**:
- Plan endpoint checks if model fits before accepting
- Scheduler validates job can be accepted
- Telemetry reports VRAM usage metrics

---

### 2.4 Deallocation

**Required by**: `worker-orcd` binary (shutdown), `model-loader` (on error)

```rust
impl Drop for SealedShard {
    fn drop(&mut self) {
        // MUST release VRAM via CUDA FFI
        // MUST log deallocation for audit trail
        // MUST NOT panic in Drop (WORKER-4083)
    }
}
```

**Expected behavior**:
- Automatic cleanup when shard goes out of scope
- Graceful handling of CUDA errors during deallocation
- Audit log entry for security trail

---

## 3. Error Type Expectations

### 3.1 `VramError` Enum

**Required by**: All consuming crates

**Expected variants** (per spec WORKER-4950-4952):
```rust
pub enum VramError {
    InsufficientVram(usize, usize),  // (needed, available)
    SealVerificationFailed,
    NotSealed,
    IntegrityViolation,
    CudaAllocationFailed(String),    // CUDA error details
    CudaDeallocationFailed(String),
}
```

**Integration with `worker-error-handler`**:
- `VramError` should be convertible to `WorkerError`
- Error handler needs to classify VRAM errors as retriable/fatal
- VRAM OOM should be distinct from host OOM (WORKER-4952)

**Expected traits**:
- `Error`, `Debug`, `Display` (via thiserror)
- Structured error messages for operators

---

## 4. VRAM-Only Policy Enforcement

### 4.1 Runtime Checks

**Required by**: `worker-orcd` binary, `model-loader`

**Expected validation** (per WORKER-4100-4103):
- Disable unified memory (UMA) at initialization
- Disable zero-copy and pinned host memory modes
- Fail fast if VRAM capacity insufficient
- Detect and reject RAM inference attempts

**API expectations**:
```rust
pub fn enforce_vram_only_policy(&self) -> Result<()>;
pub fn is_vram_only(&self) -> bool;
```

**Usage**:
- Called at worker startup
- Periodic validation during inference
- Emit security alert if policy violated

---

### 4.2 Attestation

**Required by**: `worker-api` (Ready endpoint)

**Expected method**:
```rust
pub fn attest_vram_residency(&self, shard: &SealedShard) -> Result<()>;
```

**Purpose**:
- Cryptographically attest that shard is VRAM-resident
- Used in Ready endpoint response
- Enables audited staging guarantees

---

## 5. Seal Cryptography

### 5.1 HMAC-SHA256 Seal Signature

**Required by**: `worker-api` (Commit, Execute endpoints)

**Expected implementation** (per WORKER-4120-4122):
```rust
// Internal, not exposed
fn compute_seal_signature(
    shard_id: &str,
    digest: &str,
    sealed_at: SystemTime,
    secret_key: &[u8]
) -> String;

fn verify_seal_signature(
    shard: &SealedShard,
    secret_key: &[u8]
) -> Result<()>;
```

**Key management**:
- Secret key MUST be per-worker (not global)
- Key SHOULD be derived from worker token or hardware ID
- Key MUST NOT be logged or exposed in API

**Dependencies**:
- Add `hmac` crate to Cargo.toml (currently missing)
- Use `hmac::Hmac<sha2::Sha256>`

---

## 6. Tensor-Parallel Support (Post-M0)

### 6.1 Multi-Shard Tracking

**Required by**: `worker-api` (Plan endpoint), future TP implementation

**Expected fields in `SealedShard`**:
```rust
pub shard_index: Option<usize>,      // For tensor-parallel
pub total_shards: Option<usize>,     // For tensor-parallel
```

**Expected API**:
```rust
pub fn seal_model_shard(
    &mut self,
    model_bytes: &[u8],
    gpu_device: u32,
    shard_index: usize,
    total_shards: usize
) -> Result<SealedShard>;
```

**Usage** (deferred to post-M0):
- Multi-GPU tensor-parallel models
- NCCL group coordination
- Cross-shard integrity verification

---

## 7. Testing & Proof Bundle Integration

### 7.1 Test Support

**Required by**: All crate test suites

**Expected test utilities**:
```rust
#[cfg(test)]
pub mod test_utils {
    pub fn mock_vram_manager() -> VramManager;
    pub fn mock_sealed_shard(gpu_device: u32) -> SealedShard;
}
```

**Proof bundle requirements** (per `.specs/00_proof-bundle.md`):
- Unit tests MUST emit proof bundles to `.proof_bundle/unit/<run_id>/`
- Include: seeds, test metadata, VRAM allocation timeline
- Respect `LLORCH_RUN_ID` and `LLORCH_PROOF_DIR`

---

### 7.2 Determinism Requirements

**Required by**: `determinism-suite`

**Expected behavior**:
- Seal computation MUST be deterministic (same bytes → same digest)
- VRAM allocation order MUST NOT affect seal
- Timestamp in seal SHOULD be optional for determinism tests

---

## 8. Configuration & Initialization

### 8.1 VramManager Construction

**Required by**: `worker-orcd` binary

**Expected constructors**:
```rust
impl VramManager {
    pub fn new() -> Self;  // Default config
    pub fn with_capacity(total_vram: usize) -> Self;
    pub fn from_gpu_device(gpu_device: u32) -> Result<Self>;  // Query CUDA
}
```

**Configuration**:
- Total VRAM capacity (default: query from CUDA)
- GPU device index
- Seal secret key (from worker token)
- Policy enforcement flags (UMA disabled, etc.)

---

### 8.2 CUDA FFI Integration

**Required by**: `worker-orcd` binary (CUDA FFI module)

**Expected FFI boundary**:
```rust
// Internal, called by VramManager
fn cuda_malloc(size: usize, device: u32) -> Result<*mut c_void>;
fn cuda_memcpy_host_to_device(dst: *mut c_void, src: &[u8]) -> Result<()>;
fn cuda_free(ptr: *mut c_void) -> Result<()>;
fn cuda_get_device_properties(device: u32) -> Result<CudaDeviceProps>;
```

**Safety requirements** (per WORKER-4400-4403):
- All FFI calls MUST be wrapped in safe Rust abstractions
- Bounds checking on all pointer operations
- No raw pointer exposure in public API

---

## 9. Observability & Telemetry

### 9.1 Structured Logging

**Required by**: All consuming crates

**Expected log events**:
- Model sealed: `shard_id`, `vram_bytes`, `gpu_device`, `digest`
- Seal verification: `shard_id`, `result` (pass/fail)
- VRAM allocation: `requested`, `available`, `used`
- Deallocation: `shard_id`, `freed_bytes`

**Log levels**:
- `info` — Normal operations (seal, verify)
- `warn` — Low VRAM, approaching capacity
- `error` — Seal verification failed, CUDA errors

---

### 9.2 Metrics

**Required by**: `worker-orcd` binary (Prometheus exporter)

**Expected metrics** (per WORKER-4810-4812):
- `vram_total_bytes{gpu_device}` — Total VRAM capacity
- `vram_used_bytes{gpu_device}` — Currently used VRAM
- `vram_available_bytes{gpu_device}` — Available VRAM
- `vram_seal_operations_total{result}` — Seal operations (success/failure)
- `vram_verify_operations_total{result}` — Verification operations

**Access**:
```rust
pub fn metrics(&self) -> VramMetrics;
```

---

## 10. Security & Compliance

### 10.1 Clippy Configuration

**Required**: TIER 1 (security-critical)

**Enforced lints** (per spec §5):
```rust
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
```

**Rationale**: VRAM management is security-critical; any panic or UB could compromise sealed shard guarantees.

---

### 10.2 Audit Trail

**Required by**: Security audit requirements

**Expected logging**:
- Every seal operation with digest
- Every verification operation with result
- Every VRAM allocation/deallocation
- Any policy violations or integrity failures

**Format**: Structured JSON logs with correlation IDs

---

## 11. Implementation Priority

### Phase 1: M0 Essentials (Immediate)
1. ✅ `SealedShard` struct with public fields
2. ✅ `VramManager::seal_model()` with mock VRAM allocation
3. ✅ `VramManager::verify_sealed()` stub
4. ✅ `VramError` enum with core variants
5. ⬜ Add `ModelShardHandle` type alias
6. ⬜ Add HMAC-SHA256 seal signature computation
7. ⬜ Implement capacity query methods

### Phase 2: API Integration (Next)
8. ⬜ Wire into `worker-api` Commit endpoint
9. ⬜ Wire into `worker-api` Ready endpoint
10. ⬜ Wire into `worker-api` Execute endpoint (verification)
11. ⬜ Add structured logging for all operations
12. ⬜ Implement `Drop` for `SealedShard` with audit logging

### Phase 3: CUDA Integration (Post-M0)
13. ⬜ Replace mock VRAM with real CUDA FFI
14. ⬜ Implement `cuda_malloc`/`cuda_free` wrappers
15. ⬜ Add CUDA device property queries
16. ⬜ Implement digest re-verification from VRAM contents
17. ⬜ Add VRAM-only policy enforcement checks

### Phase 4: Production Hardening (Post-M0)
18. ⬜ Add metrics emission
19. ⬜ Implement tensor-parallel multi-shard support
20. ⬜ Add seal timestamp freshness checks
21. ⬜ Comprehensive unit tests with proof bundles
22. ⬜ Integration tests with real GPU

---

## 12. Open Questions

### Q1: Seal Key Management
**Question**: Where does the HMAC secret key come from?  
**Options**:
- A) Derived from worker API token (hash of token)
- B) Generated at worker startup, stored in memory
- C) Hardware-based key (TPM, GPU device ID)

**Recommendation**: Option A for M0 (simple, deterministic), Option C for production.

---

### Q2: Digest Re-Verification Performance
**Question**: Re-computing SHA-256 from VRAM on every Execute is expensive. Optimize?  
**Options**:
- A) Cache digest, only re-verify periodically
- B) Use incremental hashing with VRAM checkpoints
- C) Accept the cost for security guarantees

**Recommendation**: Option C for M0, Option B for production optimization.

---

### Q3: Multi-Shard Coordination
**Question**: How do multiple shards coordinate for tensor-parallel?  
**Deferred**: Post-M0. Requires NCCL integration and cross-GPU seal verification.

---

## 13. Breaking Changes Policy

**Pre-1.0 status**: Breaking changes allowed (per user rules).

**Expected breaking changes**:
- Rename `SealedShard` → `ModelShardHandle` (if needed)
- Add required fields for tensor-parallel support
- Change seal signature algorithm (if security audit requires)
- Refactor VRAM allocation API for CUDA integration

**Migration support**: None required (pre-1.0).

---

## 14. References

**Specs**:
- `bin/worker-orcd/.specs/00_worker-orcd.md` — Parent spec
- `bin/worker-orcd-crates/vram-residency/.specs/00_vram-residency.md` — This crate's spec
- `bin/worker-orcd-crates/api/.specs/00_api.md` — API consumer
- `.specs/00_proof-bundle.md` — Testing requirements

**Related docs**:
- `.docs/ARCHITECTURE_CHANGE_PLAN.md` — Strategic context
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Security requirements

---

**End of Expectations Document**
