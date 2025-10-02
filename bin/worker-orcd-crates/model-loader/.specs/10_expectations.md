# Model Loader — Consumer Expectations

**Status**: Draft  
**Purpose**: Documents what other crates expect from `model-loader`  
**Last Updated**: 2025-10-02

---

## 0. Overview

This document catalogs the expectations and dependencies that other worker-orcd crates have on `model-loader`. The model-loader is a **dumb component** — it validates and loads model bytes from disk or memory, nothing more.

**Core responsibility**: Take model bytes (from file or RAM), validate them (hash, GGUF format, security limits), return validated bytes ready for VRAM loading.

**Consuming crates**:
- `worker-api` — Commit endpoint (loads models on demand)
- `vram-residency` — Receives validated bytes for VRAM sealing
- `pool-managerd` — Stages models in RAM before worker commit
- `capability-matcher` — Extracts MCD from GGUF metadata (future)

---

## 1. Core Principle: Dumb Component

### 1.1 What model-loader IS (EXP-LOADER-1001)

**A pure validation and loading utility**:
- Reads model bytes from filesystem or accepts bytes from memory
- Validates GGUF format (magic number, header, bounds checking)
- Computes and verifies SHA-256 hash
- Validates file paths (no traversal)
- Returns validated bytes or error

**No business logic**:
- Does NOT decide which models to load
- Does NOT manage VRAM allocation
- Does NOT track loaded models
- Does NOT communicate with other services
- Does NOT have state beyond the current operation

### 1.2 What model-loader IS NOT (EXP-LOADER-1002)

**NOT a model manager**:
- Does NOT cache models
- Does NOT track what's loaded where
- Does NOT decide when to evict models

**NOT a VRAM manager**:
- Does NOT allocate VRAM
- Does NOT seal shards
- Does NOT verify residency

**NOT a catalog**:
- Does NOT resolve model references
- Does NOT download models
- Does NOT manage model metadata

---

## 2. Model Source Expectations

### 2.1 Two Input Modes (EXP-LOADER-2001)

**Mode 1: Load from filesystem**
```rust
pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>>
```

**Use case**: Worker loads model from local disk
- Model already staged by pool-managerd
- Path points to `/var/lib/llorch/models/...`
- Validates path, reads file, validates GGUF, returns bytes

**Mode 2: Validate from memory**
```rust
pub fn validate_bytes(&self, bytes: &[u8], expected_hash: Option<&str>) -> Result<()>
```

**Use case**: pool-managerd staged model in RAM, sends to worker
- Bytes already in memory (from download, staging, etc.)
- Validates GGUF format, hash, security limits
- Returns Ok(()) if valid, Err otherwise

### 2.2 pool-managerd Staging Pattern (EXP-LOADER-2002)

**Typical flow**:
1. pool-managerd downloads/resolves model → RAM
2. pool-managerd validates model locally (using model-loader)
3. pool-managerd sends bytes to worker via Commit endpoint
4. Worker validates bytes again (defense in depth)
5. Worker passes validated bytes to vram-residency for sealing

**Why validate twice?**
- Defense in depth (don't trust network)
- Worker is security boundary (must validate all inputs)
- pool-managerd validation is optimization (fail fast before network)

---

## 3. worker-api Expectations

### 3.1 Commit Endpoint Integration (EXP-API-3001)

**Required by**: `worker-api` (Commit endpoint)

**Expected usage**:
```rust
use model_loader::{ModelLoader, LoadRequest};

async fn commit_handler(
    Json(body): Json<CommitRequest>
) -> Result<Json<CommitResponse>> {
    let loader = ModelLoader::new();
    
    // Two paths: file or bytes
    let model_bytes = if let Some(path) = body.model_path {
        // Load from filesystem
        let request = LoadRequest {
            model_path: &path,
            expected_hash: body.expected_hash.as_deref(),
            max_size: 100_000_000_000, // 100GB
        };
        loader.load_and_validate(request)?
    } else if let Some(bytes) = body.model_bytes {
        // Validate from memory (pool-managerd staged it)
        loader.validate_bytes(&bytes, body.expected_hash.as_deref())?;
        bytes
    } else {
        return Err(WorkerError::InvalidRequest("No model source provided".into()));
    };
    
    // Pass validated bytes to vram-residency
    let shard = vram_manager.seal_model(&model_bytes, gpu_device)?;
    
    Ok(Json(CommitResponse {
        handle: shard.into(),
        sealed: true,
    }))
}
```

**Expectations**:
- **EXP-API-3001-R1**: Load from file path (with validation)
- **EXP-API-3001-R2**: Validate bytes from memory
- **EXP-API-3001-R3**: Return validated bytes ready for VRAM sealing
- **EXP-API-3001-R4**: Fail fast on invalid input (don't waste VRAM)
- **EXP-API-3001-R5**: Clear error messages for debugging

---

### 3.2 Error Handling (EXP-API-3002)

**Expected error mapping**:
```rust
match loader.load_and_validate(request) {
    Ok(bytes) => { /* proceed */ },
    Err(LoadError::Io(e)) => {
        // File not found, permission denied
        return Err(WorkerError::ModelNotFound(e.to_string()));
    },
    Err(LoadError::HashMismatch { expected, actual }) => {
        // Integrity violation
        return Err(WorkerError::IntegrityViolation(format!(
            "Hash mismatch: expected {}, got {}", expected, actual
        )));
    },
    Err(LoadError::TooLarge(actual, max)) => {
        // Model too large for VRAM
        return Err(WorkerError::ModelTooLarge { actual, max });
    },
    Err(LoadError::InvalidFormat(msg)) => {
        // Malformed GGUF
        return Err(WorkerError::InvalidModel(msg));
    },
    Err(LoadError::PathValidationFailed(msg)) => {
        // Path traversal attempt
        return Err(WorkerError::SecurityViolation(msg));
    },
}
```

---

## 4. vram-residency Expectations

### 4.1 Validated Bytes Contract (EXP-VRAM-4001)

**Required by**: `vram-residency` (seal_model)

**Expectation**: When vram-residency receives bytes from model-loader, they are:
1. ✅ Valid GGUF format (magic number, header, bounds checked)
2. ✅ Hash-verified (if expected_hash provided)
3. ✅ Within size limits (< max_size)
4. ✅ From validated path (no traversal)
5. ✅ Ready to copy to VRAM (no further validation needed)

**vram-residency does NOT re-validate**:
- Trusts model-loader validation
- Only computes digest for seal signature
- Focuses on VRAM allocation and sealing

**Example**:
```rust
impl VramManager {
    pub fn seal_model(
        &mut self,
        model_bytes: &[u8],  // Already validated by model-loader
        gpu_device: u32
    ) -> Result<SealedShard> {
        // No GGUF validation here — trust model-loader
        // Just allocate VRAM, copy bytes, compute seal
        
        let vram_ptr = cuda_malloc(model_bytes.len(), gpu_device)?;
        cuda_memcpy_host_to_device(vram_ptr, model_bytes)?;
        
        let digest = compute_sha256(model_bytes);
        let seal_signature = self.compute_seal_signature(&digest, ...);
        
        Ok(SealedShard {
            shard_id: generate_shard_id(),
            vram_ptr,
            digest,
            sealed: true,
            // ...
        })
    }
}
```

---

## 5. pool-managerd Expectations

### 5.1 Staging Validation (EXP-POOL-5001)

**Required by**: `pool-managerd` (model staging)

**Use case**: pool-managerd downloads model, validates before sending to worker

**Expected usage**:
```rust
use model_loader::ModelLoader;

async fn stage_model(model_ref: &str) -> Result<Vec<u8>> {
    // Download/resolve model
    let model_bytes = download_model(model_ref).await?;
    
    // Validate before sending to worker (fail fast)
    let loader = ModelLoader::new();
    loader.validate_bytes(&model_bytes, None)?;
    
    // Now safe to send to worker
    Ok(model_bytes)
}
```

**Expectations**:
- **EXP-POOL-5001-R1**: Validate bytes in memory (no file I/O)
- **EXP-POOL-5001-R2**: Fast validation (< 100ms for typical models)
- **EXP-POOL-5001-R3**: Fail fast on malformed models (before network send)
- **EXP-POOL-5001-R4**: Optional hash verification (if catalog provides hash)

---

### 5.2 File-Based Staging (EXP-POOL-5002)

**Alternative pattern**: pool-managerd stages to disk, worker loads from disk

**Flow**:
1. pool-managerd downloads model → `/var/lib/llorch/models/staging/<model-id>.gguf`
2. pool-managerd validates file (using model-loader)
3. pool-managerd tells worker: "load from /var/lib/llorch/models/staging/<model-id>.gguf"
4. Worker loads and validates (using model-loader)
5. Worker seals to VRAM

**Expected usage**:
```rust
// pool-managerd side
let staging_path = PathBuf::from("/var/lib/llorch/models/staging/model.gguf");
fs::write(&staging_path, &model_bytes)?;

let loader = ModelLoader::new();
let request = LoadRequest {
    model_path: &staging_path,
    expected_hash: Some(&expected_hash),
    max_size: 100_000_000_000,
};
loader.load_and_validate(request)?; // Validate before telling worker

// Tell worker to load from staging_path
```

---

## 6. capability-matcher Expectations (Future)

### 6.1 MCD Extraction (EXP-CAP-6001)

**Required by**: `capability-matcher` (future integration)

**Use case**: Extract Model Capability Descriptor from GGUF metadata

**Expected API** (post-M0):
```rust
pub fn extract_metadata(&self, bytes: &[u8]) -> Result<GgufMetadata>
```

**Returns**:
```rust
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv: HashMap<String, MetadataValue>,
}
```

**capability-matcher usage**:
```rust
let loader = ModelLoader::new();
let metadata = loader.extract_metadata(&model_bytes)?;

// Extract MCD fields from metadata
let mcd = ModelCapabilityDescriptor {
    model_id: metadata.get_string("general.name")?,
    positional: metadata.get_string("llama.rope.type")?,
    attention: metadata.get_string("llama.attention.type")?,
    // ...
};
```

**Expectations**:
- **EXP-CAP-6001-R1**: Parse GGUF metadata key-value pairs
- **EXP-CAP-6001-R2**: Return structured metadata (not raw bytes)
- **EXP-CAP-6001-R3**: Handle unknown keys gracefully (forward compatibility)
- **EXP-CAP-6001-R4**: Validate metadata types (string, int, float, etc.)

---

## 7. API Expectations Summary

### 7.1 Primary API (EXP-API-7001)

**Load from file**:
```rust
pub struct LoadRequest<'a> {
    pub model_path: &'a Path,
    pub expected_hash: Option<&'a str>,
    pub max_size: usize,
}

impl ModelLoader {
    pub fn new() -> Self;
    pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>>;
}
```

**Validate from memory**:
```rust
impl ModelLoader {
    pub fn validate_bytes(
        &self,
        bytes: &[u8],
        expected_hash: Option<&str>
    ) -> Result<()>;
}
```

**Future: Extract metadata**:
```rust
impl ModelLoader {
    pub fn extract_metadata(&self, bytes: &[u8]) -> Result<GgufMetadata>;
}
```

---

### 7.2 Error Types (EXP-API-7002)

**Expected error variants**:
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

**Error handling expectations**:
- Specific error types (not generic "failed to load")
- Actionable messages (tell operator what's wrong)
- No sensitive data in errors (no file contents)
- Distinguish retriable vs fatal errors

---

## 8. Security Expectations

### 8.1 Path Validation (EXP-SEC-8001)

**Required by**: All consumers

**Expectations**:
- **EXP-SEC-8001-R1**: Use `input-validation` crate for path validation
- **EXP-SEC-8001-R2**: Canonicalize paths (resolve `..`, symlinks)
- **EXP-SEC-8001-R3**: Validate against allowed root directory
- **EXP-SEC-8001-R4**: Reject paths outside allowed directory

**Example**:
```rust
use input_validation::validate_path;

impl ModelLoader {
    pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>> {
        let allowed_root = PathBuf::from("/var/lib/llorch/models");
        
        // Validate path (prevents traversal)
        let canonical_path = validate_path(request.model_path, &allowed_root)
            .map_err(|e| LoadError::PathValidationFailed(e.to_string()))?;
        
        // Now safe to read
        let bytes = fs::read(&canonical_path)?;
        // ...
    }
}
```

---

### 8.2 Hash Validation (EXP-SEC-8002)

**Required by**: All consumers

**Expectations**:
- **EXP-SEC-8002-R1**: Use `input-validation::validate_hex_string()` for hash format
- **EXP-SEC-8002-R2**: Compute SHA-256 digest of entire file
- **EXP-SEC-8002-R3**: Fail fast on hash mismatch (before GGUF parsing)
- **EXP-SEC-8002-R4**: Log hash for audit trail (even if not verified)

**Example**:
```rust
use input_validation::validate_hex_string;
use sha2::{Sha256, Digest};

impl ModelLoader {
    fn verify_hash(&self, bytes: &[u8], expected_hash: &str) -> Result<()> {
        // Validate hash format
        validate_hex_string(expected_hash, 64)
            .map_err(|e| LoadError::InvalidFormat(e.to_string()))?;
        
        // Compute actual hash
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let actual_hash = format!("{:x}", hasher.finalize());
        
        // Compare
        if actual_hash != expected_hash {
            return Err(LoadError::HashMismatch {
                expected: expected_hash.to_string(),
                actual: actual_hash,
            });
        }
        
        tracing::info!(
            hash = %actual_hash,
            "Model hash verified"
        );
        
        Ok(())
    }
}
```

---

### 8.3 GGUF Validation (EXP-SEC-8003)

**Required by**: All consumers

**Expectations**:
- **EXP-SEC-8003-R1**: Validate magic number (`0x46554747`)
- **EXP-SEC-8003-R2**: Validate header fields (version, tensor_count, etc.)
- **EXP-SEC-8003-R3**: Enforce security limits (MAX_TENSORS, MAX_STRING_LEN)
- **EXP-SEC-8003-R4**: Use bounds-checked reads (no buffer overflows)
- **EXP-SEC-8003-R5**: Validate all string lengths before allocation
- **EXP-SEC-8003-R6**: Validate tensor dimensions (no integer overflow)

**Security limits**:
```rust
const MAX_TENSORS: usize = 10_000;
const MAX_FILE_SIZE: usize = 100_000_000_000; // 100GB
const MAX_STRING_LEN: usize = 65536; // 64KB
const MAX_METADATA_PAIRS: usize = 1000;
```

---

## 9. Performance Expectations

### 9.1 Load Performance (EXP-PERF-9001)

**Expected performance**:
- Hash computation: < 1s per GB (I/O bound)
- GGUF validation: < 100ms for typical models
- Total load time: I/O bound (disk speed)
- Memory overhead: Minimal (stream where possible)

**Optimization expectations**:
- **EXP-PERF-9001-R1**: Stream hash computation (don't load entire file twice)
- **EXP-PERF-9001-R2**: Zero-copy validation where possible
- **EXP-PERF-9001-R3**: Early termination on invalid input
- **EXP-PERF-9001-R4**: No unnecessary allocations

---

### 9.2 Memory Expectations (EXP-PERF-9002)

**Memory usage**:
- Load from file: 1x model size (loaded into Vec<u8>)
- Validate from memory: 0x model size (validates in-place)
- Hash computation: Streaming (constant memory)
- GGUF parsing: Minimal overhead (< 1MB)

**Large model handling**:
- Models up to 100GB supported
- Progress logging for large loads (> 10GB)
- Async I/O support (future)

---

## 10. Testing Expectations

### 10.1 Test Fixtures (EXP-TEST-10001)

**Expected test helpers**:
```rust
#[cfg(test)]
pub mod test_utils {
    pub fn create_valid_gguf(size: usize) -> Vec<u8>;
    pub fn create_invalid_gguf() -> Vec<u8>;
    pub fn create_test_model_file(dir: &Path) -> PathBuf;
}
```

**Test cases expected**:
- Valid GGUF files (various versions)
- Invalid magic number
- Truncated files
- Oversized tensor counts
- String length overflows
- Hash mismatches
- Path traversal attempts
- Buffer overflow attempts

---

### 10.2 Property Testing (EXP-TEST-10002)

**Expected property tests**:
```rust
proptest! {
    #[test]
    fn gguf_parser_never_panics(bytes: Vec<u8>) {
        let loader = ModelLoader::new();
        let _ = loader.validate_bytes(&bytes, None);
        // Should never panic on any input
    }
}
```

---

## 11. Integration Patterns

### 11.1 worker-api Integration (EXP-INTEG-11001)

**Pattern**: Validate → Seal → Ready

```rust
// Commit endpoint
let model_bytes = loader.load_and_validate(request)?;
let shard = vram_manager.seal_model(&model_bytes, gpu_device)?;

// Ready endpoint
let handles = vram_manager.get_sealed_shards();
```

---

### 11.2 pool-managerd Integration (EXP-INTEG-11002)

**Pattern**: Download → Validate → Stage → Send

```rust
// Download model
let model_bytes = download_model(model_ref).await?;

// Validate before sending
loader.validate_bytes(&model_bytes, Some(&expected_hash))?;

// Send to worker
send_commit_request(&model_bytes).await?;
```

---

## 12. Configuration Expectations

### 12.1 Allowed Paths (EXP-CONFIG-12001)

**Expected configuration**:
```rust
pub struct ModelLoaderConfig {
    pub allowed_model_dir: PathBuf,  // Default: /var/lib/llorch/models
    pub max_model_size: usize,       // Default: 100GB
}
```

**Usage**:
```rust
let config = ModelLoaderConfig {
    allowed_model_dir: PathBuf::from("/var/lib/llorch/models"),
    max_model_size: 100_000_000_000,
};

let loader = ModelLoader::with_config(config);
```

---

## 13. Non-Expectations (What NOT to Expect)

### 13.1 NOT a Model Manager (EXP-NOT-13001)

**model-loader does NOT**:
- ❌ Track which models are loaded
- ❌ Cache loaded models
- ❌ Decide when to evict models
- ❌ Manage model lifecycle

**That's the job of**: pool-managerd, worker-api

---

### 13.2 NOT a Catalog (EXP-NOT-13002)

**model-loader does NOT**:
- ❌ Resolve model references (`hf:meta-llama/...`)
- ❌ Download models from HuggingFace
- ❌ Manage model metadata
- ❌ Track model versions

**That's the job of**: pool-managerd (model-catalog crate)

---

### 13.3 NOT a VRAM Manager (EXP-NOT-13003)

**model-loader does NOT**:
- ❌ Allocate VRAM
- ❌ Seal shards
- ❌ Verify residency
- ❌ Manage CUDA contexts

**That's the job of**: vram-residency

---

## 14. Implementation Priority

### Phase 1: M0 Essentials (Immediate)
1. ✅ `LoadRequest` struct
2. ✅ `load_and_validate()` — Load from file
3. ✅ `validate_bytes()` — Validate from memory
4. ✅ GGUF magic number validation
5. ✅ SHA-256 hash verification
6. ✅ Path validation (via input-validation)
7. ✅ `LoadError` enum with all variants
8. ✅ Basic unit tests

### Phase 2: Security Hardening (Next)
9. ⬜ Full GGUF header validation
10. ⬜ Bounds-checked parsing
11. ⬜ Security limits enforcement
12. ⬜ String length validation
13. ⬜ Tensor dimension validation
14. ⬜ Property tests
15. ⬜ Fuzz testing

### Phase 3: Advanced Features (Post-M0)
16. ⬜ `extract_metadata()` — MCD extraction
17. ⬜ Streaming hash computation
18. ⬜ Async I/O support
19. ⬜ Progress reporting for large loads
20. ⬜ Multi-format support (SafeTensors)

---

## 15. Open Questions

**Q1**: Should model-loader support streaming (load chunks at a time)?  
**A**: Defer to post-M0. Load entire file for M0 (simpler).

**Q2**: Should model-loader cache validation results?  
**A**: No. model-loader is stateless. Caching is pool-managerd's job.

**Q3**: Should model-loader support signature verification?  
**A**: Optional for M0. Add if time permits, otherwise post-M0.

**Q4**: Should model-loader extract MCD automatically?  
**A**: No. Separate method `extract_metadata()` for capability-matcher to call.

---

## 16. References

**Specifications**:
- `bin/worker-orcd-crates/model-loader/.specs/00_model-loader.md` — Main spec
- `bin/worker-orcd-crates/vram-residency/.specs/00_vram-residency.md` — Consumer
- `bin/worker-orcd-crates/api/.specs/00_api.md` — Consumer
- `bin/shared-crates/input-validation/.specs/00_input-validation.md` — Dependency

**Architecture**:
- `.docs/ARCHITECTURE_CHANGE_PLAN.md` — Phase 3, Task Group 4
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Issue #19

---

**End of Expectations Document**
