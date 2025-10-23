# rbee-hive-artifact-catalog

**TEAM-273: Shared artifact catalog and provisioning abstractions**

## Purpose

Provides generic abstractions for catalog and provisioning patterns used by:
- `model-catalog` - Model files from HuggingFace
- `worker-catalog` - Worker binaries from GitHub/local builds

## Architecture

```
artifact-catalog/
├── types.rs          → Artifact trait, ArtifactStatus
├── catalog.rs        → ArtifactCatalog trait, FilesystemCatalog
└── provisioner.rs    → VendorSource trait, ArtifactProvisioner trait
```

## Core Abstractions

### Artifact Trait

```rust
pub trait Artifact: Clone + Serialize + Deserialize {
    fn id(&self) -> &str;
    fn path(&self) -> &Path;
    fn size(&self) -> u64;
    fn status(&self) -> &ArtifactStatus;
    fn set_status(&mut self, status: ArtifactStatus);
}
```

### ArtifactCatalog Trait

```rust
pub trait ArtifactCatalog<T: Artifact> {
    fn add(&self, artifact: T) -> Result<()>;
    fn get(&self, id: &str) -> Result<T>;
    fn list(&self) -> Vec<T>;
    fn remove(&self, id: &str) -> Result<()>;
    fn contains(&self, id: &str) -> bool;
}
```

### VendorSource Trait

```rust
#[async_trait]
pub trait VendorSource: Send + Sync {
    async fn download(&self, id: &str, dest: &Path, job_id: &str) -> Result<u64>;
    fn supports(&self, id: &str) -> bool;
    fn name(&self) -> &str;
}
```

## Usage Example

### Model Catalog

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

### Worker Catalog

```rust
use artifact_catalog::{Artifact, ArtifactCatalog, FilesystemCatalog};

#[derive(Clone, Serialize, Deserialize)]
pub struct WorkerBinary {
    id: String,
    path: PathBuf,
    size: u64,
    status: ArtifactStatus,
    worker_type: WorkerType, // cpu/cuda/metal
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

## Benefits

1. **DRY**: Eliminates ~400-600 LOC of duplication
2. **Consistency**: Same patterns for all artifact types
3. **Extensibility**: Easy to add new artifact types
4. **Testing**: Shared test utilities
5. **Maintainability**: Fix bugs once, benefit everywhere

## Testing

```bash
cargo test --package rbee-hive-artifact-catalog
```

## Created By

**TEAM-273** - Shared abstractions for catalog/provisioner patterns
