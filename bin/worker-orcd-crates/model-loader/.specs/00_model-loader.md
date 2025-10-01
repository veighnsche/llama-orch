# Model Loader SPEC — Model Validation & GGUF Parsing (WORKER-4xxx)

**Status**: Draft  
**Applies to**: `bin/worker-orcd-crates/model-loader/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

This crate validates model files (signature, hash, GGUF format) before loading into VRAM, providing a security boundary against malicious models.

**Parent spec**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

## 1. Model Validation

- [WORKER-4310] Workers MUST validate model bytes before loading into VRAM.
- [WORKER-4311] GGUF parser MUST enforce max tensor count (default 10,000) and max file size (default 100GB).
- [WORKER-4312] GGUF parser MUST use bounds-checked reads; out-of-bounds access MUST fail fast.
- [WORKER-4313] Workers MUST verify model signature (if provided) using cryptographic verification.
- [WORKER-4314] Workers MUST compute and verify SHA-256 digest against expected value (if provided).

---

## 2. GGUF Format Validation

### 2.1 Header Validation

- MUST validate magic number (`0x46554747` = "GGUF")
- MUST validate version field
- MUST check header size >= minimum (12 bytes)
- MUST validate tensor_count <= MAX_TENSORS
- MUST validate metadata_kv_count is reasonable

### 2.2 Bounds Checking

- MUST validate all string lengths before reading
- MUST check tensor dimensions for overflow
- MUST validate data type enums
- MUST check file size matches expected from header

### 2.3 Security Limits

```rust
const MAX_TENSORS: usize = 10_000;
const MAX_FILE_SIZE: usize = 100_000_000_000; // 100GB
const MAX_STRING_LEN: usize = 65536; // 64KB
const MAX_METADATA_PAIRS: usize = 1000;
```

---

## 3. Hash Verification

- [WORKER-4320] Workers MUST compute SHA-256 digest of entire model file.
- [WORKER-4321] If `expected_hash` provided, MUST compare and fail on mismatch.
- [WORKER-4322] Hash computation MUST occur before GGUF parsing (fail fast).
- [WORKER-4323] Workers MUST log hash for audit trail (even if not verified).

---

## 4. Signature Verification (Optional)

- [WORKER-4330] Workers MAY support cryptographic signature verification (Ed25519 or RSA).
- [WORKER-4331] If signature provided, verification MUST occur before loading.
- [WORKER-4332] Signature verification failure MUST reject model load.
- [WORKER-4333] Signature format MUST be documented (e.g., detached `.sig` file).

---

## 5. Path Validation

- [WORKER-4340] Workers MUST validate filesystem paths to prevent directory traversal.
- [WORKER-4341] Paths MUST be canonicalized (resolve `..` and symlinks) before use.
- [WORKER-4342] Paths MUST be checked against allowed root directory (e.g., `/var/lib/llorch/models`).
- [WORKER-4343] Workers MUST reject paths outside allowed directory with actionable error.

---

## 6. API

### 6.1 Load and Validate

```rust
pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>>
```

**LoadRequest fields**:
- `model_path: &Path` — Path to model file
- `expected_hash: Option<&str>` — SHA-256 hex string
- `max_size: usize` — Maximum allowed file size

**Returns**: Model bytes (validated and ready for VRAM loading)

### 6.2 Validate GGUF

```rust
fn validate_gguf(&self, bytes: &[u8]) -> Result<()>
```

- MUST check magic number
- MUST validate header fields
- MUST enforce security limits
- MUST use bounds-checked parsing

---

## 7. Security Properties

- **TIER 1 Clippy configuration** (security-critical)
- Deny: unwrap, expect, panic, indexing_slicing
- Defensive parsing (all bounds checked)
- Fail-fast on invalid input
- No buffer overflows

---

## 8. Dependencies

**Crates used**:
- `sha2` — SHA-256 digest computation
- `ed25519-dalek` (optional) — Signature verification
- `thiserror` — Error types

---

## 9. Traceability

**Code**: `bin/worker-orcd-crates/model-loader/src/lib.rs`  
**Tests**: `bin/worker-orcd-crates/model-loader/tests/`  
**Parent**: `bin/worker-orcd/.specs/00_worker-orcd.md` §4
