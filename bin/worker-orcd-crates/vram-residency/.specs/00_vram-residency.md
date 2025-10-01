# VRAM Residency SPEC — VRAM-Only Policy & Sealed Shards (WORKER-4xxx)

**Status**: Draft  
**Applies to**: `bin/worker-orcd-crates/vram-residency/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

This crate enforces VRAM-only inference policy and implements the sealed ModelShardHandle contract with cryptographic integrity verification.

**Parent spec**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

## 1. VRAM-Only Policy

- [WORKER-4100] During inference, model weights, KV cache, activations, and intermediate tensors MUST reside entirely in GPU VRAM (per ORCH-2.13).
- [WORKER-4101] Workers MUST disable unified memory (UMA), zero-copy, pinned host memory, and BAR modes that keep weights outside VRAM.
- [WORKER-4102] Host RAM MAY be used for staging, catalog verification, and decompression; it MUST NOT be used for live decode.
- [WORKER-4103] Workers MUST fail fast with actionable diagnostics if VRAM capacity is insufficient for the requested model.

---

## 2. ModelShardHandle Contract

- [WORKER-4110] Workers MUST implement the `ModelShardHandle` abstraction with fields:
  - `shard_id: String` — Unique opaque identifier
  - `gpu_device: u32` — CUDA device index
  - `vram_bytes: usize` — Size in VRAM
  - `sealed: bool` — Attestation flag (true = resident, immutable)
  - `digest: String` — SHA-256 of shard data
  - `model_ref: String` — Original model reference
  - `shard_index: Option<usize>` — For tensor-parallel
  - `total_shards: Option<usize>` — For tensor-parallel
- [WORKER-4111] VRAM pointers (`vram_ptr: *mut c_void`) MUST be private and MUST NOT be exposed in API responses or logs.
- [WORKER-4112] When `sealed == true`, the shard MUST be resident in VRAM, immutable, and ready for immediate inference.
- [WORKER-4113] Digest computation MUST use SHA-256 and MUST be verified before sealing.

---

## 3. Seal Integrity

- [WORKER-4120] Workers MUST compute a cryptographic seal signature covering `(shard_id, digest, sealed_at)` using HMAC-SHA256.
- [WORKER-4121] Seal verification MUST occur before each execution to detect VRAM corruption or tampering.
- [WORKER-4122] If seal verification fails, the worker MUST reject the execution request and transition to `Stopped` state.

---

## 4. VRAM Manager API

### 4.1 Allocation

```rust
pub fn seal_model(&mut self, model_bytes: &[u8], gpu_device: u32) -> Result<SealedShard>
```

- MUST allocate VRAM via CUDA FFI
- MUST compute SHA-256 digest
- MUST generate HMAC-SHA256 seal signature
- MUST return sealed ModelShardHandle

### 4.2 Verification

```rust
pub fn verify_sealed(&self, shard: &SealedShard) -> Result<()>
```

- MUST re-compute digest from VRAM contents
- MUST verify HMAC-SHA256 signature
- MUST check seal timestamp freshness (optional)

### 4.3 Deallocation

```rust
impl Drop for SealedShard
```

- MUST release VRAM via CUDA FFI
- MUST log deallocation for audit trail
- MUST NOT panic in Drop

---

## 5. Security Properties

- **TIER 1 Clippy configuration** (security-critical)
- Deny: unwrap, expect, panic, indexing_slicing, integer_arithmetic
- Private VRAM pointers (never exposed)
- Cryptographic seal verification
- Bounds checking on all operations

---

## 6. Dependencies

**Crates used**:
- `sha2` — SHA-256 digest computation
- `hmac` — HMAC-SHA256 seal signatures
- `cuda_ffi` (via worker-orcd) — VRAM allocation/deallocation

---

## 7. Error Handling

- [WORKER-4150] All operations MUST return `Result<T, VramError>` with explicit error handling.
- [WORKER-4151] `VramError` MUST distinguish retriable errors (InsufficientVram, CudaAllocationFailed) from fatal errors (SealVerificationFailed, IntegrityViolation, PolicyViolation).
- [WORKER-4152] Error messages MUST NOT expose VRAM pointers, seal keys, or other sensitive information.
- [WORKER-4153] VRAM allocation failures MUST provide actionable diagnostics (needed bytes, available bytes).

---

## 8. Audit Requirements

- [WORKER-4160] All seal operations MUST emit `AuditEvent::VramSealed` with shard_id, gpu_device, vram_bytes, and digest.
- [WORKER-4161] Seal verification failures MUST emit `AuditEvent::SealVerificationFailed` with severity "critical".
- [WORKER-4162] VRAM deallocation MUST emit `AuditEvent::VramDeallocated` for audit trail.
- [WORKER-4163] Policy violations MUST emit `AuditEvent::PolicyViolation` and prevent worker startup.

---

## 9. Configuration

- [WORKER-4170] `VramManager` MUST accept configuration for:
  - `worker_api_token: String` — For seal key derivation
  - `gpu_device: u32` — CUDA device index
  - `max_model_size: usize` — Maximum allowed model size (default 100GB)
  - `audit_logger: Arc<AuditLogger>` — Audit event sink
- [WORKER-4171] Configuration MUST be validated at initialization (gpu_device exists, max_model_size > 0).
- [WORKER-4172] Invalid configuration MUST fail fast with actionable error messages.

---

## 10. Traceability

**Code**: `bin/worker-orcd-crates/vram-residency/src/lib.rs`  
**Tests**: `bin/worker-orcd-crates/vram-residency/tests/`  
**Parent**: `bin/worker-orcd/.specs/00_worker-orcd.md` §2

---

## 11. Refinement Opportunities

### 11.1 Tensor-Parallel Support
- Define multi-shard coordination requirements for tensor-parallel models
- Specify NCCL group coordination for cross-GPU seal verification
- Add requirements for shard_index and total_shards validation

### 11.2 Seal Timestamp Freshness
- Define policy for seal timestamp freshness validation (optional vs required)
- Specify acceptable time skew for distributed workers
- Add requirements for timestamp monotonicity checks

### 11.3 VRAM-Only Policy Enforcement Timing
- Clarify when policy enforcement occurs (startup only vs periodic runtime checks)
- Define behavior when policy violations are detected post-startup
- Add requirements for policy re-verification after driver updates

### 11.4 CUDA Driver Compatibility
- Specify minimum CUDA driver version requirements
- Define compatibility matrix for different GPU architectures
- Add requirements for driver version validation at startup

### 11.5 Performance Optimization
- Consider incremental hashing for large models (reduce digest re-verification cost)
- Evaluate seal signature caching strategies (balance security vs performance)
- Define performance targets for seal operations (< 1ms for signature, < 100ms for digest)

### 11.6 Multi-GPU Coordination
- Add requirements for cross-GPU VRAM isolation verification
- Define seal key derivation strategy for multi-GPU workers
- Specify behavior when one GPU fails seal verification (fail all vs isolate)

---

**End of Specification**
