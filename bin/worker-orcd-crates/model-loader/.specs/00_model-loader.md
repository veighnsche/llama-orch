# Model Loader SPEC — Load Model from Location to VRAM (ML-1xxx)

**Status**: Draft  
**Applies to**: `bin/worker-orcd-crates/model-loader/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

Model loader is a **worker-side crate** that loads a model from a provided location (disk or RAM) directly into GPU VRAM.

**Responsibilities**:
- ✅ Load model from disk path or RAM segment
- ✅ Copy model bytes to VRAM
- ✅ Validate model format (GGUF)
- ✅ Report actual VRAM usage

**NOT Responsible For**:
- ❌ Downloading models (pool manager's model-cache does this)
- ❌ Model discovery (pool manager provides location)
- ❌ VRAM capacity planning (pool manager's gpu-inventory does this)
- ❌ Multiple models (worker is tied to ONE model)

**Parent spec**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

## 1. Core Responsibilities

### [ML-1001] Load from Location
Model loader MUST accept a model location and load it into VRAM. It MUST NOT download or fetch models from external sources.

### [ML-1002] VRAM-Only
Model loader MUST copy model bytes directly to VRAM. It MUST NOT keep model in host RAM (except for temporary staging during copy).

### [ML-1003] Single Model
Model loader MUST load ONE model per invocation. It MUST NOT support loading multiple models concurrently.

### [ML-1004] Format Validation
Model loader MUST validate model format before loading. It MUST support GGUF format for M0.

---

## 2. Model Location

### [ML-2010] Location Types
Model loader MUST accept two location types:

**1. Disk Path**:
```rust
ModelLocation::Disk {
    path: PathBuf,  // e.g., /path/to/model.gguf
}
```

**2. RAM-Staged** (Shared Memory):
```rust
ModelLocation::RamStaged {
    shared_mem_name: String,  // e.g., "llorch-model-llama7b"
    size_bytes: u64,           // Model size in bytes
}
```

### [ML-2011] Disk Path Loading
When loading from disk, model loader MUST:
1. Validate path exists and is readable
2. Open file with read-only permissions
3. Memory-map file (mmap) for efficient loading
4. Copy memory-mapped bytes to VRAM
5. Close file after copy completes

### [ML-2012] RAM-Staged Loading
When loading from RAM-staged shared memory, model loader MUST:
1. Open shared memory segment by name
2. Map shared memory into process address space
3. Copy shared memory bytes to VRAM
4. Unmap shared memory after copy completes

**Note**: RAM-staged loading is significantly faster than disk loading because model is already in RAM (staged by pool manager).

### [ML-2013] No Download
Model loader MUST NOT download models from:
- HTTP/HTTPS URLs
- Hugging Face
- S3 or other object storage
- Git repositories

These are pool manager responsibilities (model-cache crate).

---

## 3. GGUF Format Validation

### [ML-3010] GGUF Support
Model loader MUST support GGUF (GPT-Generated Unified Format) used by llama.cpp.

### [ML-3011] Format Detection
Model loader MUST validate GGUF magic bytes at file start:
```
Magic: "GGUF" (0x47475546)
Version: 3 (uint32)
```

### [ML-3012] Metadata Parsing
Model loader MUST parse GGUF metadata:
- `general.architecture` — Model architecture (e.g., "llama")
- `general.name` — Model name
- `llama.context_length` — Context window size
- `llama.embedding_length` — Embedding dimensions
- `llama.block_count` — Number of transformer blocks

### [ML-3013] Tensor Validation
Model loader MUST validate tensors:
- Tensor count within limits (max 10,000)
- Tensor names are valid UTF-8
- Tensor dimensions are reasonable
- Total tensor size matches file size

### [ML-3014] Unsupported Formats
Model loader SHOULD return clear error for unsupported formats:
- PyTorch (.pt, .pth)
- Safetensors (.safetensors)
- TensorFlow (.pb)

---

## 4. VRAM Loading

### [ML-4010] VRAM Allocation
Model loader MUST use `vram-policy` crate to allocate VRAM:
```rust
let vram_policy = VramPolicy::new(gpu_device)?;
let vram_bytes_used = vram_policy.load_model_to_vram(model_bytes)?;
```

### [ML-4011] Copy Strategy
Model loader MUST copy model bytes to VRAM efficiently:
- Use large block sizes (1MB chunks)
- Minimize CPU→GPU copies
- Free staging buffers immediately after copy

### [ML-4012] VRAM Usage Reporting
Model loader MUST return actual VRAM bytes used, not just model file size. VRAM usage includes:
- Model weights
- Metadata structures
- Alignment padding

### [ML-4013] Memory Safety
Model loader MUST use checked arithmetic and bounds checking. It MUST NOT use:
- `unwrap()` or `expect()`
- Unchecked array indexing
- Integer overflow arithmetic

---

## 5. API

### [ML-5010] Load Function
```rust
pub fn load_model(
    location: ModelLocation,
    gpu_device: u32,
) -> Result<LoadedModel, LoadError>
```

**Parameters**:
- `location` — Where to load model from (disk or RAM)
- `gpu_device` — CUDA device ID (0, 1, ...)

**Returns**:
```rust
pub struct LoadedModel {
    pub model_ref: String,        // Model identifier (from metadata)
    pub vram_bytes: u64,           // Actual VRAM used
    pub context_length: usize,     // Context window size
    pub architecture: String,      // e.g., "llama"
    // Internal: VRAM pointer, tensor map, etc.
}
```

### [ML-5011] Validation Function
```rust
pub fn validate_model_format(
    location: &ModelLocation,
) -> Result<ModelMetadata, LoadError>
```

Validates model format without loading to VRAM. Used for pre-flight checks.

**Returns**:
```rust
pub struct ModelMetadata {
    pub format: ModelFormat,       // GGUF
    pub architecture: String,
    pub name: String,
    pub context_length: usize,
    pub estimated_vram_bytes: u64,
}
```

---

## 6. Error Handling

### [ML-6010] Error Types
```rust
pub enum LoadError {
    InvalidLocation(String),       // Path doesn't exist or shared mem not found
    AccessDenied(String),          // Permission denied
    InvalidFormat(String),         // Not a valid GGUF file
    UnsupportedFormat(String),     // Recognized but unsupported format
    TensorCountExceeded,           // Too many tensors (>10k)
    InvalidMetadata(String),       // Malformed GGUF metadata
    InsufficientVram { needed: u64, available: u64 },
    VramAllocationFailed(String),  // CUDA malloc failed
    IoError(std::io::Error),       // File I/O error
}
```

### [ML-6011] Error Classification
Model loader MUST classify errors:
- **Retriable**: `InsufficientVram`, `VramAllocationFailed`, `IoError` (transient)
- **Fatal**: `InvalidFormat`, `UnsupportedFormat`, `InvalidMetadata`, `TensorCountExceeded`
- **Invalid Input**: `InvalidLocation`, `AccessDenied`

### [ML-6012] Error Messages
Model loader MUST provide actionable error messages:
```
"Invalid GGUF format: magic bytes expected 'GGUF', found '...'"
"Insufficient VRAM: need 26GB, have 20GB available on GPU 1"
"Model file not found: /path/to/model.gguf"
```

---

## 7. Performance

### [ML-7010] Load Time Target
Model loader SHOULD achieve:
- **Disk**: < 10 seconds for 7B model (14GB)
- **RAM-staged**: < 2 seconds for 7B model (14GB)

RAM staging is ~5x faster because model is pre-loaded in RAM by pool manager.

### [ML-7011] Memory Efficiency
Model loader MUST:
- Use streaming I/O (don't load entire file into RAM)
- Free staging buffers immediately after VRAM copy
- Use memory-mapped I/O for disk reads

### [ML-7012] Progress Reporting
Model loader SHOULD emit progress logs during load:
```
"Loading model from disk: 0%"
"Loading model from disk: 25% (3.5GB / 14GB)"
"Loading model from disk: 100% (14GB / 14GB)"
"Model loaded to VRAM: 14.2GB used"
```

---

## 8. Security

### [ML-8010] Input Validation
Model loader MUST validate all inputs:
- `location.path` — No path traversal, absolute or relative paths only
- `location.shared_mem_name` — Alphanumeric + dash/underscore only
- `gpu_device` — Within valid range (0 to num_gpus-1)

### [ML-8011] Path Traversal Prevention
Model loader MUST reject paths containing:
- `..` (parent directory)
- Symlinks to unexpected locations
- Absolute paths outside allowed directories (if sandbox is enabled)

### [ML-8012] Memory Safety
Model loader MUST use Rust's memory safety guarantees:
- No unsafe code without explicit justification
- Bounds checking on all array access
- Checked arithmetic (no overflows)

---

## 9. Observability

### [ML-9010] Structured Logging
Model loader MUST emit structured logs:
```rust
tracing::info!(
    model_ref = %model_metadata.name,
    gpu_device = gpu_device,
    location_type = "disk",
    vram_bytes = vram_bytes_used,
    load_time_ms = elapsed.as_millis(),
    "Model loaded successfully"
);
```

### [ML-9011] Audit Events
Model loader SHOULD emit audit events for security-relevant operations:
- Model load attempts (success/failure)
- Invalid format detected
- Path validation failures

### [ML-9012] Human Narration
Model loader SHOULD emit human-readable narration:
```
"Loading llama-7b from disk (/models/llama-7b.gguf) to GPU 0..."
"Model llama-7b loaded to VRAM: 14.2GB used (5.3s)"
```

---

## 10. Testing

### [ML-10010] Unit Tests
Model loader MUST have unit tests for:
- GGUF format validation (valid and invalid files)
- Disk path loading
- RAM-staged loading (mock shared memory)
- Error handling (insufficient VRAM, invalid format, etc.)
- Bounds checking (large tensors, malformed metadata)

### [ML-10011] Integration Tests
Model loader SHOULD have integration tests:
- Load real GGUF models (Qwen-0.5B, TinyLlama-1.1B)
- Measure actual VRAM usage
- Verify VRAM-only policy (no RAM fallback)

### [ML-10012] Property Tests
Model loader SHOULD use property-based testing:
- Random GGUF files (ensure no panics)
- Random model sizes (ensure no overflows)
- Fuzzing (malformed GGUF files)

---

## 11. Dependencies

### [ML-11010] Required Dependencies
```toml
[dependencies]
vram-policy = { path = "../vram-policy" }
thiserror.workspace = true
tracing.workspace = true
```

### [ML-11011] Optional Dependencies
```toml
memmap2 = "0.9"  # Memory-mapped file I/O
shared_memory = "0.12"  # Shared memory (RAM staging)
```

---

## 12. Traceability

**Code**: `bin/worker-orcd-crates/model-loader/src/`  
**Tests**: `bin/worker-orcd-crates/model-loader/tests/`  
**BDD**: `bin/worker-orcd-crates/model-loader/bdd/`  
**Parent**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

## 13. Data Flow

### Disk Loading Flow
```
1. Pool Manager:
   - Downloads model to /models/llama-7b.gguf (via model-cache)
   - Spawns worker: --model file:/models/llama-7b.gguf

2. Worker startup:
   - Calls model_loader::load_model(
       ModelLocation::Disk { path: "/models/llama-7b.gguf" },
       gpu_device: 0
     )

3. Model Loader:
   - Opens /models/llama-7b.gguf
   - Memory-maps file (14GB)
   - Validates GGUF format
   - Calls vram_policy.load_model_to_vram(mmap_bytes)
   - VRAM allocation: 14.2GB
   - Copies bytes to GPU 0 VRAM
   - Returns LoadedModel { vram_bytes: 14.2GB, ... }

4. Worker:
   - Reports to pool manager: "Ready, using 14.2GB"
```

### RAM-Staged Loading Flow
```
1. Pool Manager:
   - Downloads model to /models/llama-7b.gguf
   - Pre-stages in RAM: shared memory "llorch-model-llama7b"
   - Spawns worker: --model ram:llorch-model-llama7b

2. Worker startup:
   - Calls model_loader::load_model(
       ModelLocation::RamStaged {
         shared_mem_name: "llorch-model-llama7b",
         size_bytes: 14000000000
       },
       gpu_device: 0
     )

3. Model Loader:
   - Opens shared memory "llorch-model-llama7b"
   - Maps shared memory into process (14GB already in RAM!)
   - Validates GGUF format
   - Calls vram_policy.load_model_to_vram(shared_mem_bytes)
   - VRAM allocation: 14.2GB
   - Copies bytes to GPU 0 VRAM (FAST: RAM→VRAM)
   - Returns LoadedModel { vram_bytes: 14.2GB, ... }

4. Worker:
   - Reports to pool manager: "Ready, using 14.2GB"
```

**Key difference**: RAM-staged is ~5x faster because model is already in RAM.

---

## 14. Refinement Opportunities

### 14.1 Incremental Loading
- Load model in chunks (reduce peak RAM usage)
- Stream directly from disk to VRAM
- Reduce staging buffer size

### 14.2 Format Support
- Support safetensors format
- Support ONNX format
- Auto-detect format from magic bytes

### 14.3 Model Quantization
- Support on-the-fly quantization (FP16 → INT8)
- Reduce VRAM usage
- Trade accuracy for capacity

### 14.4 Multi-File Models
- Support models split across multiple files
- Load shards in parallel
- Merge tensors in VRAM

---

**End of Specification**
