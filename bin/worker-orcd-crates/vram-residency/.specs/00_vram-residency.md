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

## 7. Traceability

**Code**: `bin/worker-orcd-crates/vram-residency/src/lib.rs`  
**Tests**: `bin/worker-orcd-crates/vram-residency/tests/`  
**Parent**: `bin/worker-orcd/.specs/00_worker-orcd.md` §2
