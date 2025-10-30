# Hive Catalog Architecture Study

**Date:** Oct 30, 2025  
**Status:** COMPLETE ANALYSIS  
**Purpose:** Deep dive into artifact-catalog, model-catalog, and worker-catalog crates

---

## Summary

The Hive uses a **shared catalog abstraction** (TEAM-273) that provides a consistent pattern for managing both models and worker binaries. This architecture is elegant, well-tested, and extensible.

---

## Crate Structure

### 1. artifact-catalog (Shared Abstraction)

**Location:** `bin/25_rbee_hive_crates/artifact-catalog/`

**Purpose:** Shared traits and generic implementations for catalog and provisioning patterns.

**Key Components:**

```rust
// Core trait - implemented by ModelEntry, WorkerBinary, etc.
pub trait Artifact: Clone + Serialize + Deserialize {
    fn id(&self) -> &str;
    fn path(&self) -> &Path;
    fn size(&self) -> u64;
    fn status(&self) -> &ArtifactStatus;
    fn set_status(&mut self, status: ArtifactStatus);
    fn name(&self) -> &str { self.id() }
}

// Status enum
pub enum ArtifactStatus {
    Available,
    Downloading,
    Failed { error: String },
}

// Catalog trait - CRUD operations
pub trait ArtifactCatalog<T: Artifact> {
    fn add(&self, artifact: T) -> Result<()>;
    fn get(&self, id: &str) -> Result<T>;
    fn list(&self) -> Vec<T>;
    fn remove(&self, id: &str) -> Result<()>;
    fn contains(&self, id: &str) -> bool;
    fn len(&self) -> usize;
}

// Generic filesystem implementation
pub struct FilesystemCatalog<T: Artifact> {
    catalog_dir: PathBuf,
    _phantom: PhantomData<T>,
}

// Metadata wrapper
pub struct ArtifactMetadata<T> {
    pub artifact: T,
    pub added_at: DateTime<Utc>,
    pub last_accessed: Option<DateTime<Utc>>,
}
```

**Filesystem Layout:**
```
catalog_dir/
├── artifact-id-1/
│   └── metadata.json    # ArtifactMetadata<T> serialized
├── artifact-id-2/
│   └── metadata.json
└── ...
```

**Provisioner Pattern:**
```rust
// Vendor source trait
pub trait VendorSource: Send + Sync {
    async fn download(&self, id: &str, dest: &Path, job_id: &str) -> Result<u64>;
    fn supports(&self, id: &str) -> bool;
    fn name(&self) -> &str;
}

// Provisioner trait
pub trait ArtifactProvisioner<T: Artifact>: Send + Sync {
    async fn provision(&self, id: &str, job_id: &str) -> Result<T>;
    fn supports(&self, id: &str) -> bool;
}

// Multi-vendor implementation
pub struct MultiVendorProvisioner<T: Artifact> {
    vendors: Vec<Box<dyn VendorSource>>,
}
```

**Benefits:**
- ✅ Generic implementation (works for any Artifact type)
- ✅ Well-tested (comprehensive test suite)
- ✅ Metadata tracking (added_at, last_accessed)
- ✅ Multi-vendor support (HuggingFace, GitHub, local, etc.)

---

### 2. model-catalog (Concrete Implementation)

**Location:** `bin/25_rbee_hive_crates/model-catalog/`

**Purpose:** Model catalog for managing LLM model files (GGUF).

**Key Components:**

```rust
// Model entry - implements Artifact trait
pub struct ModelEntry {
    id: String,              // e.g., "meta-llama/Llama-2-7b"
    name: String,            // Human-readable name
    path: PathBuf,           // Absolute path to GGUF file
    size: u64,               // File size in bytes
    status: ArtifactStatus,  // Available, Downloading, Failed
    added_at: DateTime<Utc>, // When added to catalog
}

// Model catalog - wraps FilesystemCatalog
pub struct ModelCatalog {
    inner: FilesystemCatalog<ModelEntry>,
}

impl ModelCatalog {
    pub fn new() -> Result<Self> {
        // Uses ~/.cache/rbee/models/
    }
    
    pub fn with_dir(catalog_dir: PathBuf) -> Result<Self> {
        // Custom directory (for testing)
    }
    
    pub fn model_path(&self, model_id: &str) -> PathBuf {
        // Get path where model would be stored
    }
}

// Delegates to FilesystemCatalog
impl ArtifactCatalog<ModelEntry> for ModelCatalog {
    // ... delegates all methods to inner ...
}
```

**Storage:**
```
~/.cache/rbee/models/
├── meta-llama-Llama-2-7b/
│   ├── metadata.json
│   └── model.gguf
├── mistralai-Mistral-7B/
│   ├── metadata.json
│   └── model.gguf
└── ...
```

**Current Properties:**
- ✅ id, name, path, size
- ✅ status (Available, Downloading, Failed)
- ✅ added_at timestamp

**Missing Properties (TODO):**
- ⏳ architecture (llama, mistral, qwen, etc.)
- ⏳ quantization (Q4_K_M, Q8_0, F16, etc.)
- ⏳ parameter_count (1B, 3B, 7B, etc.)
- ⏳ context_length (2048, 4096, 8192, etc.)
- ⏳ gguf_version, tensor_count, kv_count
- ⏳ download_progress (bytes, speed, ETA)
- ⏳ checksum, verified

**GGUF Metadata Parsing:**
- Need to parse GGUF file headers to extract metadata
- GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Can use `gguf-rs` crate or implement custom parser

---

### 3. worker-catalog (Concrete Implementation)

**Location:** `bin/25_rbee_hive_crates/worker-catalog/`

**Purpose:** Worker catalog for managing worker binaries.

**Key Components:**

```rust
// Worker type enum
pub enum WorkerType {
    CpuLlm,      // CPU inference
    CudaLlm,     // NVIDIA GPU
    MetalLlm,    // Apple Metal
}

impl WorkerType {
    pub fn binary_name(&self) -> &str {
        match self {
            WorkerType::CpuLlm => "cpu-llm-worker-rbee",
            WorkerType::CudaLlm => "cuda-llm-worker-rbee",
            WorkerType::MetalLlm => "metal-llm-worker-rbee",
        }
    }
}

// Platform enum
pub enum Platform {
    Linux,
    MacOS,
    Windows,
}

impl Platform {
    pub fn current() -> Self { /* detect current platform */ }
    pub fn extension(&self) -> &str { /* "" or ".exe" */ }
}

// Worker binary entry - implements Artifact trait
pub struct WorkerBinary {
    id: String,              // e.g., "cpu-llm-worker-rbee-v0.1.0-linux"
    worker_type: WorkerType, // CpuLlm, CudaLlm, MetalLlm
    platform: Platform,      // Linux, MacOS, Windows
    path: PathBuf,           // Absolute path to binary
    size: u64,               // File size in bytes
    status: ArtifactStatus,  // Available, Downloading, Failed
    version: String,         // Semantic version (e.g., "0.1.0")
    added_at: DateTime<Utc>, // When added to catalog
}

// Worker catalog - wraps FilesystemCatalog
pub struct WorkerCatalog {
    inner: FilesystemCatalog<WorkerBinary>,
}

impl WorkerCatalog {
    pub fn new() -> Result<Self> {
        // Uses ~/.cache/rbee/workers/
    }
    
    pub fn find_by_type_and_platform(
        &self,
        worker_type: WorkerType,
        platform: Platform,
    ) -> Option<WorkerBinary> {
        // Find worker binary by type and platform
    }
}
```

**Storage:**
```
~/.cache/rbee/workers/
├── cpu-llm-worker-rbee-v0.1.0-linux/
│   ├── metadata.json
│   └── cpu-llm-worker-rbee
├── cuda-llm-worker-rbee-v0.1.0-linux/
│   ├── metadata.json
│   └── cuda-llm-worker-rbee
└── ...
```

**Critical (TEAM-277):**
- ❌ Hive does NOT install worker binaries
- ✅ WorkerCatalog is READ ONLY from Hive's perspective
- ✅ Hive discovers workers installed by Queen via SSH
- ✅ Hive only manages worker PROCESSES, not binaries

---

## Architecture Patterns

### 1. Trait-Based Abstraction

**Problem:** Need consistent interface for models and workers.

**Solution:** Shared `Artifact` trait with generic `FilesystemCatalog<T>`.

**Benefits:**
- ✅ Type safety (compiler enforces required methods)
- ✅ Code reuse (FilesystemCatalog works for any Artifact)
- ✅ Testability (test generic implementation once)
- ✅ Extensibility (easy to add new artifact types)

### 2. Filesystem-Based Storage

**Problem:** Need simple, reliable storage without database overhead.

**Solution:** JSON metadata files in subdirectories.

**Benefits:**
- ✅ Simple (no DB setup, just filesystem)
- ✅ Human-readable (JSON files can be inspected)
- ✅ Portable (works on any platform)
- ✅ Atomic (filesystem operations are atomic)

**Tradeoffs:**
- ⚠️ Not suitable for high-frequency updates
- ⚠️ No transactions (but we don't need them)
- ⚠️ No indexing (but catalog is small)

### 3. Metadata Wrapper

**Problem:** Need to track when artifacts were added/accessed.

**Solution:** `ArtifactMetadata<T>` wrapper.

**Benefits:**
- ✅ Automatic tracking (added_at, last_accessed)
- ✅ Non-invasive (doesn't modify artifact type)
- ✅ Extensible (can add more metadata fields)

### 4. Multi-Vendor Provisioning

**Problem:** Need to download artifacts from multiple sources.

**Solution:** `VendorSource` trait + `MultiVendorProvisioner`.

**Benefits:**
- ✅ Extensible (easy to add new vendors)
- ✅ Routing (provisioner finds correct vendor)
- ✅ Consistent (same interface for all vendors)

**Example Vendors:**
- `HuggingFaceVendor` - Download from HuggingFace Hub
- `GitHubReleaseVendor` - Download from GitHub releases
- `LocalBuildVendor` - Use locally built binaries
- `HttpVendor` - Download from arbitrary HTTP URL

---

## Integration with Hive

### Current Usage

**In job_router.rs:**
```rust
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,  // TEAM-268
    pub worker_catalog: Arc<WorkerCatalog>, // TEAM-274
}

// Model operations
Operation::ModelList(request) => {
    let models = state.model_catalog.list();
    // ... format and emit narration ...
}

Operation::ModelGet(request) => {
    let model = state.model_catalog.get(&request.id)?;
    // ... emit model details ...
}

Operation::ModelDelete(request) => {
    state.model_catalog.remove(&request.id)?;
    // ... emit success ...
}

// Worker operations
Operation::WorkerSpawn(request) => {
    let worker_binary = state.worker_catalog
        .find_by_type_and_platform(worker_type, Platform::current())
        .ok_or_else(|| anyhow!("Worker binary not found"))?;
    
    // Spawn worker process using daemon-lifecycle
    // ...
}
```

**In main.rs:**
```rust
// Initialize catalogs
let model_catalog = Arc::new(ModelCatalog::new()?);
let worker_catalog = Arc::new(WorkerCatalog::new()?);

// Create HTTP state
let state = HiveState {
    registry,
    model_catalog,
    worker_catalog,
};
```

---

## What's Missing

### 1. Enhanced ModelEntry

**Current:**
```rust
pub struct ModelEntry {
    id: String,
    name: String,
    path: PathBuf,
    size: u64,
    status: ArtifactStatus,
    added_at: DateTime<Utc>,
}
```

**Needed:**
```rust
pub struct ModelEntry {
    // ... existing fields ...
    
    // GGUF metadata
    architecture: String,        // "llama", "mistral", etc.
    quantization: String,        // "Q4_K_M", "Q8_0", etc.
    parameter_count: String,     // "1B", "3B", "7B", etc.
    context_length: u32,         // 2048, 4096, 8192, etc.
    gguf_version: u32,
    tensor_count: u32,
    kv_count: u32,
    
    // Download tracking
    download_progress: Option<DownloadProgress>,
    
    // Validation
    checksum: Option<String>,
    verified: bool,
}
```

### 2. Model Provisioner (TEAM-269)

**Needed:**
- `HuggingFaceVendor` - Download models from HuggingFace Hub
- `MultiVendorProvisioner` - Route to correct vendor
- Integration with `ModelCatalog`
- Progress tracking via narration

**Example:**
```rust
let provisioner = MultiVendorProvisioner::new(vec![
    Box::new(HuggingFaceVendor::new()),
]);

// Download model
let model = provisioner.provision("HF:meta-llama/Llama-2-7b", job_id).await?;

// Add to catalog
model_catalog.add(model)?;
```

### 3. GGUF Metadata Parser

**Needed:**
- Parse GGUF file headers
- Extract architecture, quantization, parameter count, etc.
- Update ModelEntry with parsed metadata

**Libraries:**
- `gguf-rs` (if exists)
- Or implement custom parser following GGUF spec

### 4. Download Progress Tracking

**Needed:**
```rust
pub struct DownloadProgress {
    bytes_downloaded: u64,
    total_bytes: u64,
    speed_bytes_per_sec: u64,
    eta_seconds: u64,
    started_at: DateTime<Utc>,
    source_url: String,
}
```

**Integration:**
- Update ModelEntry.status to Downloading
- Emit progress narration events
- Update ModelEntry.download_progress
- On completion: status = Available, download_progress = None

---

## Testing

### artifact-catalog Tests

**Coverage:**
- ✅ Add artifact
- ✅ Get artifact
- ✅ List artifacts
- ✅ Remove artifact
- ✅ Contains check
- ✅ Duplicate detection
- ✅ Metadata persistence

### model-catalog Tests

**Coverage:**
- ✅ CRUD operations
- ✅ Filesystem persistence
- ✅ Custom directory support

### worker-catalog Tests

**Coverage:**
- ✅ CRUD operations
- ✅ Find by type and platform
- ✅ Filesystem persistence

**Missing:**
- ⏳ Provisioner tests
- ⏳ Download progress tests
- ⏳ GGUF metadata parsing tests
- ⏳ Concurrent access tests

---

## Recommendations

### Immediate (High Priority)

1. **Enhance ModelEntry** - Add missing properties (architecture, quantization, etc.)
2. **GGUF Parser** - Implement metadata extraction from GGUF files
3. **Model Provisioner** - Implement HuggingFaceVendor (TEAM-269)
4. **Download Progress** - Add DownloadProgress struct and tracking

### Short-Term (Medium Priority)

5. **Worker Process Tracking** - Separate WorkerProcess from WorkerBinary
6. **Resource Monitoring** - Track CPU/memory/VRAM usage
7. **Health Checks** - Periodic worker health polling
8. **Checksum Validation** - Verify downloaded files

### Long-Term (Low Priority)

9. **Additional Vendors** - GitHubReleaseVendor, HttpVendor
10. **Artifact Metadata** - Track usage stats, last_accessed
11. **Batch Operations** - Download multiple models, spawn multiple workers
12. **Catalog Migrations** - Handle schema changes

---

## Summary

**The catalog architecture is solid:**
- ✅ Well-designed (trait-based, generic, extensible)
- ✅ Well-tested (comprehensive test coverage)
- ✅ Well-documented (clear code comments)
- ✅ Production-ready (used in Hive job_router)

**What's missing:**
- ⏳ Enhanced ModelEntry with GGUF metadata
- ⏳ Model provisioner (TEAM-269)
- ⏳ GGUF metadata parser
- ⏳ Download progress tracking

**Next steps:**
1. Add missing properties to ModelEntry
2. Implement GGUF metadata parser
3. Implement HuggingFaceVendor
4. Wire up MultiVendorProvisioner
5. Add download progress tracking
6. Update UI to show enhanced model properties

**The foundation is excellent. We just need to build on it.**
