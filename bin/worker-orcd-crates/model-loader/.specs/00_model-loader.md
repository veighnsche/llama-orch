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

## 9. Error Handling

- [WORKER-4350] All operations MUST return `Result<T, LoadError>` with explicit error handling.
- [WORKER-4351] `LoadError` MUST distinguish retriable errors (Io) from fatal errors (HashMismatch, InvalidFormat, SignatureVerificationFailed).
- [WORKER-4352] Error messages MUST NOT expose sensitive file contents or internal paths.
- [WORKER-4353] GGUF parsing errors MUST provide actionable diagnostics (offset, expected vs actual values).

### 9.1 Error Types

```rust
pub enum LoadError {
    Io(std::io::Error),
    HashMismatch { expected: String, actual: String },
    TooLarge(usize, usize),
    InvalidFormat(String),
    SignatureVerificationFailed,
    PathValidationFailed(String),
    TensorCountExceeded { count: usize, max: usize },
    StringTooLong { length: usize, max: usize },
    InvalidDataType(u8),
    BufferOverflow { offset: usize, length: usize, available: usize },
}
```

---

## 10. Integration with input-validation

- [WORKER-4360] Model loader MUST use `input-validation` crate for all path and hash validation.
- [WORKER-4361] Path validation MUST use `validate_path()` with allowed root directory.
- [WORKER-4362] Hash validation MUST use `validate_hex_string()` for expected_hash parameter.
- [WORKER-4363] Model loader MUST NOT implement custom path traversal checks (use shared crate).

**Example integration**:
```rust
use input_validation::{validate_path, validate_hex_string};

pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>> {
    // Validate path
    let canonical_path = validate_path(request.model_path, &self.allowed_root)
        .map_err(|e| LoadError::PathValidationFailed(e.to_string()))?;
    
    // Validate expected hash format
    if let Some(hash) = request.expected_hash {
        validate_hex_string(hash, 64)
            .map_err(|e| LoadError::InvalidFormat(e.to_string()))?;
    }
    
    // Proceed with loading...
}
```

---

## 11. GGUF Parsing Details

### 11.1 Header Structure

```rust
struct GgufHeader {
    magic: u32,           // 0x46554747 ("GGUF")
    version: u32,         // GGUF version (2 or 3)
    tensor_count: u64,    // Number of tensors
    metadata_kv_count: u64, // Number of metadata key-value pairs
}
```

### 11.2 Metadata Parsing

- [WORKER-4370] Metadata keys MUST be validated as UTF-8 strings.
- [WORKER-4371] Metadata string lengths MUST be checked against MAX_STRING_LEN before allocation.
- [WORKER-4372] Metadata value types MUST be validated against known enum values.
- [WORKER-4373] Unknown metadata keys SHOULD be logged but MUST NOT fail parsing (forward compatibility).

### 11.3 Tensor Info Parsing

- [WORKER-4380] Tensor names MUST be validated as UTF-8 strings.
- [WORKER-4381] Tensor dimensions MUST be checked for integer overflow before multiplication.
- [WORKER-4382] Tensor data offsets MUST be validated against file size.
- [WORKER-4383] Tensor data types MUST be validated against known enum values.

### 11.4 Bounds-Checked Reading

```rust
fn read_u32(&self, offset: usize) -> Result<u32> {
    let end = offset.checked_add(4)
        .ok_or(LoadError::BufferOverflow { offset, length: 4, available: self.bytes.len() })?;
    
    if end > self.bytes.len() {
        return Err(LoadError::BufferOverflow { 
            offset, 
            length: 4, 
            available: self.bytes.len() 
        });
    }
    
    Ok(u32::from_le_bytes([
        self.bytes[offset],
        self.bytes[offset + 1],
        self.bytes[offset + 2],
        self.bytes[offset + 3],
    ]))
}
```

---

## 12. Audit Requirements

- [WORKER-4390] Model load attempts MUST emit `AuditEvent::ModelLoadStarted` with path and expected_hash.
- [WORKER-4391] Hash verification MUST emit `AuditEvent::ModelHashVerified` with computed digest.
- [WORKER-4392] Load failures MUST emit `AuditEvent::ModelLoadFailed` with error reason.
- [WORKER-4393] Successful loads MUST emit `AuditEvent::ModelLoadCompleted` with file size and tensor count.

---

## 13. Performance Considerations

- [WORKER-4400] Hash computation SHOULD use streaming API to avoid loading entire file into memory twice.
- [WORKER-4401] GGUF validation SHOULD be zero-copy where possible (validate in-place).
- [WORKER-4402] Large model loads (>10GB) SHOULD report progress via logging.
- [WORKER-4403] Model loader SHOULD support async I/O for non-blocking file reads.

**Performance targets**:
- Hash verification: < 1s per GB
- GGUF validation: < 100ms for typical models
- Total load time: I/O bound (disk speed)

---

## 14. Testing Requirements

### 14.1 Unit Tests

- Valid GGUF files (various versions)
- Invalid magic number
- Truncated files
- Oversized tensor counts
- String length overflows
- Hash mismatches
- Path traversal attempts

### 14.2 Property Tests

```rust
proptest! {
    #[test]
    fn test_gguf_parser_never_panics(bytes: Vec<u8>) {
        let loader = ModelLoader::new();
        let _ = loader.validate_gguf(&bytes);
        // Should never panic on any input
    }
}
```

### 14.3 Fuzz Testing

- GGUF parser with random byte sequences
- Header field mutations
- String length edge cases
- Tensor count edge cases

---

## 15. Traceability

**Code**: `bin/worker-orcd-crates/model-loader/src/lib.rs`  
**Tests**: `bin/worker-orcd-crates/model-loader/tests/`  
**Parent**: `bin/worker-orcd/.specs/00_worker-orcd.md` §4  
**Security Audit**: `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` Issue #19

---

## 16. Refinement Opportunities

### 16.1 Multi-Format Support

- Add support for SafeTensors format (alternative to GGUF)
- Add support for PyTorch .bin files (with strict validation)
- Define format detection heuristics (magic number registry)
- Add format conversion utilities (SafeTensors → GGUF)

### 16.2 Incremental Loading

- Support streaming model loading for large files (>100GB)
- Implement chunked hash verification (verify as you load)
- Add resume capability for interrupted loads
- Define checkpoint format for partial loads

### 16.3 Model Metadata Extraction

- Extract MCD (Model Capability Descriptor) from GGUF metadata
- Parse architecture-specific metadata (rope_freq_base, etc.)
- Validate metadata consistency (vocab_size matches tokenizer)
- Cache extracted metadata for faster subsequent loads

### 16.4 Signature Verification

- Define signature format (detached .sig files vs embedded)
- Support multiple signature algorithms (Ed25519, RSA-PSS)
- Implement signature chain verification (model → org → root CA)
- Add signature revocation checking (CRL or OCSP)

### 16.5 Advanced Validation

- Add tensor shape validation (verify dimensions match architecture)
- Implement quantization format validation (Q4_0, Q5_1, etc.)
- Add vocabulary validation (check for duplicate tokens)
- Verify embedding dimensions match model config

### 16.6 Performance Optimization

- Implement parallel hash computation (multi-threaded)
- Add mmap support for zero-copy validation
- Cache validation results (hash → validation status)
- Optimize GGUF parsing with SIMD (string length checks)

### 16.7 Observability

- Add detailed progress reporting for large loads
- Emit metrics for load times, hash times, validation times
- Add tracing spans for each validation step
- Implement validation profiling (identify slow steps)

### 16.8 Error Recovery

- Define retry strategy for transient I/O errors
- Add corruption detection and reporting
- Implement partial validation (skip corrupted sections if safe)
- Add repair suggestions for common issues

---

**End of Specification**
