# TEAM-273: Shared Artifact Catalog Abstraction

**Status:** ✅ COMPLETE  
**Date:** Oct 23, 2025  
**Effort:** 2-3 hours

---

## 🎯 Mission

Create shared abstractions for catalog and provisioning patterns to eliminate duplication between `model-catalog` and `worker-catalog`.

**Deliverables:**
1. ✅ `artifact-catalog` crate with generic traits
2. ✅ `FilesystemCatalog<T>` implementation
3. ✅ `VendorSource` and `ArtifactProvisioner` traits
4. ✅ Unit tests (6 tests)
5. ✅ Documentation

---

## 📁 Files Created

```
bin/25_rbee_hive_crates/artifact-catalog/
├── src/
│   ├── lib.rs          ← Module exports
│   ├── types.rs        ← Artifact trait, ArtifactStatus
│   ├── catalog.rs      ← ArtifactCatalog trait, FilesystemCatalog
│   └── provisioner.rs  ← VendorSource trait, ArtifactProvisioner trait
├── Cargo.toml          ← Dependencies
└── README.md           ← Documentation
```

**Total:** ~450 LOC

---

## 🏗️ Architecture

### Core Traits

**1. Artifact Trait**
```rust
pub trait Artifact: Clone + Serialize + Deserialize {
    fn id(&self) -> &str;
    fn path(&self) -> &Path;
    fn size(&self) -> u64;
    fn status(&self) -> &ArtifactStatus;
    fn set_status(&mut self, status: ArtifactStatus);
}
```

**2. ArtifactCatalog Trait**
```rust
pub trait ArtifactCatalog<T: Artifact> {
    fn add(&self, artifact: T) -> Result<()>;
    fn get(&self, id: &str) -> Result<T>;
    fn list(&self) -> Vec<T>;
    fn remove(&self, id: &str) -> Result<()>;
    fn contains(&self, id: &str) -> bool;
}
```

**3. VendorSource Trait**
```rust
#[async_trait]
pub trait VendorSource: Send + Sync {
    async fn download(&self, id: &str, dest: &Path, job_id: &str) -> Result<u64>;
    fn supports(&self, id: &str) -> bool;
    fn name(&self) -> &str;
}
```

### Concrete Implementation

**FilesystemCatalog<T>**
- Generic filesystem-based catalog
- Stores artifacts as JSON metadata files
- Each artifact in its own subdirectory
- Automatic last-accessed tracking

---

## 🔄 Usage Pattern

### Model Catalog (Example)

```rust
use artifact_catalog::{Artifact, ArtifactCatalog, FilesystemCatalog};

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    id: String,
    path: PathBuf,
    size: u64,
    status: ArtifactStatus,
}

impl Artifact for ModelEntry {
    fn id(&self) -> &str { &self.id }
    fn path(&self) -> &Path { &self.path }
    fn size(&self) -> u64 { self.size }
    fn status(&self) -> &ArtifactStatus { &self.status }
    fn set_status(&mut self, status: ArtifactStatus) { self.status = status; }
}

pub struct ModelCatalog {
    inner: FilesystemCatalog<ModelEntry>,
}

impl ArtifactCatalog<ModelEntry> for ModelCatalog {
    fn add(&self, model: ModelEntry) -> Result<()> {
        self.inner.add(model)
    }
    // ... delegate other methods
}
```

### Worker Catalog (Example)

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct WorkerBinary {
    id: String,
    path: PathBuf,
    size: u64,
    status: ArtifactStatus,
    worker_type: WorkerType, // cpu/cuda/metal
    platform: Platform,      // linux/mac/windows
}

impl Artifact for WorkerBinary {
    fn id(&self) -> &str { &self.id }
    fn path(&self) -> &Path { &self.path }
    fn size(&self) -> u64 { self.size }
    fn status(&self) -> &ArtifactStatus { &self.status }
    fn set_status(&mut self, status: ArtifactStatus) { self.status = status; }
}

pub struct WorkerCatalog {
    inner: FilesystemCatalog<WorkerBinary>,
}
```

---

## 📊 Benefits

### Code Reduction
- **Estimated savings:** 400-600 LOC
- **Model catalog:** Can remove ~200 LOC of catalog logic
- **Worker catalog:** Can remove ~200 LOC of catalog logic
- **Provisioners:** Shared vendor abstraction saves ~200 LOC

### Consistency
- Same patterns for all artifact types
- Consistent error handling
- Consistent metadata management
- Consistent filesystem layout

### Extensibility
- Easy to add new artifact types (datasets, configs, etc.)
- Easy to add new vendor sources
- Clear trait boundaries

### Maintainability
- Fix bugs once, benefit everywhere
- Shared test utilities
- Single source of truth

---

## 🧪 Testing

**Unit Tests:** 6 tests
- test_add_and_get
- test_add_duplicate
- test_list
- test_remove
- test_contains
- test_vendor_routing

**Test Coverage:**
- ✅ Catalog CRUD operations
- ✅ Duplicate detection
- ✅ Metadata persistence
- ✅ Vendor routing

```bash
cargo test --package rbee-hive-artifact-catalog
```

---

## 🔗 Integration

### TEAM-269: Model Provisioner
Can now use `artifact-catalog` as base:
```rust
use artifact_catalog::{FilesystemCatalog, VendorSource};

pub struct HuggingFaceVendor;
impl VendorSource for HuggingFaceVendor { /* ... */ }

pub struct ModelCatalog {
    inner: FilesystemCatalog<ModelEntry>,
}
```

### TEAM-273 (Worker Catalog)
Will use same pattern:
```rust
use artifact_catalog::{FilesystemCatalog, VendorSource};

pub struct GitHubReleaseVendor;
impl VendorSource for GitHubReleaseVendor { /* ... */ }

pub struct LocalBuildVendor;
impl VendorSource for LocalBuildVendor { /* ... */ }

pub struct WorkerCatalog {
    inner: FilesystemCatalog<WorkerBinary>,
}
```

---

## 📝 Key Design Decisions

### 1. Generic Over Concrete
Used Rust generics (`FilesystemCatalog<T: Artifact>`) instead of trait objects for:
- Zero-cost abstractions
- Type safety
- Better compiler optimizations

### 2. Filesystem-Based Storage
Chose filesystem over in-memory for:
- Persistence across restarts
- Easy inspection/debugging
- Standard tooling (ls, cat, etc.)

### 3. Metadata Files
Each artifact gets `metadata.json` for:
- Extensibility (add fields without breaking)
- Human-readable
- Easy migration

### 4. Vendor Abstraction
`VendorSource` trait allows:
- Multiple download sources (HuggingFace, GitHub, local)
- Easy testing (mock vendors)
- Clear separation of concerns

---

## ✅ Acceptance Criteria

- [x] Artifact trait defined
- [x] ArtifactCatalog trait defined
- [x] FilesystemCatalog implementation
- [x] VendorSource trait defined
- [x] ArtifactProvisioner trait defined
- [x] Unit tests passing (6 tests)
- [x] Documentation complete
- [x] README with usage examples

---

## 🎯 Next Steps

### TEAM-269 (Model Provisioner)
1. Update `model-catalog` to use `artifact-catalog`
2. Implement `HuggingFaceVendor`
3. Wire up `ModelProvisioner`

### TEAM-273 (Worker Catalog)
1. Create `worker-catalog` using `artifact-catalog`
2. Implement `GitHubReleaseVendor`
3. Implement `LocalBuildVendor`
4. Define worker binary structure (cpu/cuda/metal × linux/mac/windows)

---

## 📚 Reference

**Similar Patterns:**
- Rust's `std::io::Read` trait (generic I/O)
- Cargo's package registry abstraction
- Docker's image registry pattern

**Key Files:**
- `bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs` - Core implementation
- `bin/25_rbee_hive_crates/artifact-catalog/src/types.rs` - Trait definitions
- `bin/25_rbee_hive_crates/artifact-catalog/README.md` - Usage guide

---

**TEAM-273: Shared abstractions complete! 🎉**

**Impact:** Eliminates 400-600 LOC of duplication, provides consistent patterns for all artifact types.
