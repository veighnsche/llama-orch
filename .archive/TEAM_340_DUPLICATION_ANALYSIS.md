# TEAM-340: Code Duplication Analysis

**Status:** 📋 ANALYSIS COMPLETE  
**Date:** Oct 27, 2025

## Executive Summary

**Found:** 3 categories of duplication across catalog and daemon-lifecycle crates

1. ✅ **GOOD DUPLICATION** - Mock test artifacts (intentional, isolated)
2. ⚠️ **MINOR DUPLICATION** - Catalog wrapper pattern (acceptable, ~50 LOC per crate)
3. ❌ **NO DUPLICATION** - Types are properly abstracted

## Detailed Analysis

### 1. Mock Test Artifacts (✅ GOOD - Keep as is)

**Location:** Test modules in multiple files

**Duplication:**
```rust
// artifact-catalog/src/catalog.rs (lines 169-206)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestArtifact {
    id: String,
    path: PathBuf,
    size: u64,
    status: ArtifactStatus,
}

impl Artifact for TestArtifact { /* ... */ }

// artifact-catalog/src/provisioner.rs (lines 95-123)
#[derive(Clone, Serialize, Deserialize)]
struct MockArtifact {
    id: String,
    path: PathBuf,
    size: u64,
    status: ArtifactStatus,
}

impl Artifact for MockArtifact { /* ... */ }
```

**Analysis:**
- **Purpose:** Test fixtures for unit tests
- **Scope:** `#[cfg(test)]` modules only
- **Impact:** ~40 LOC per file (80 LOC total)
- **Verdict:** ✅ **KEEP** - Test isolation is more important than DRY

**Why keep it:**
1. Test code should be self-contained
2. Each test module tests different functionality
3. Changing one test shouldn't affect others
4. Mock types are simple (4 fields)
5. No runtime cost (compiled out in release)

---

### 2. Catalog Wrapper Pattern (⚠️ MINOR - Acceptable)

**Location:** model-catalog and worker-catalog

**Pattern:**
```rust
// model-catalog/src/lib.rs (42 LOC)
pub struct ModelCatalog {
    inner: FilesystemCatalog<ModelEntry>,
}

impl ModelCatalog {
    pub fn new() -> Result<Self> {
        let catalog_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("rbee")
            .join("models");
        
        let inner = FilesystemCatalog::new(catalog_dir)?;
        Ok(Self { inner })
    }
    
    pub fn with_dir(catalog_dir: PathBuf) -> Result<Self> {
        let inner = FilesystemCatalog::new(catalog_dir)?;
        Ok(Self { inner })
    }
    
    pub fn model_path(&self, model_id: &str) -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("rbee")
            .join("models")
            .join(model_id)
    }
}

// Delegate to FilesystemCatalog
impl ArtifactCatalog<ModelEntry> for ModelCatalog {
    fn add(&self, model: ModelEntry) -> Result<()> {
        self.inner.add(model)
    }
    // ... 5 more delegation methods
}

// worker-catalog/src/lib.rs (71 LOC - SAME PATTERN)
pub struct WorkerCatalog {
    inner: FilesystemCatalog<WorkerBinary>,
}

impl WorkerCatalog {
    pub fn new() -> Result<Self> {
        let catalog_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("rbee")
            .join("workers");  // ← Only difference!
        
        let inner = FilesystemCatalog::new(catalog_dir)?;
        Ok(Self { inner })
    }
    
    pub fn with_dir(catalog_dir: PathBuf) -> Result<Self> {
        let inner = FilesystemCatalog::new(catalog_dir)?;
        Ok(Self { inner })
    }
    
    pub fn worker_path(&self, worker_id: &str) -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("rbee")
            .join("workers")  // ← Only difference!
            .join(worker_id)
    }
}

// Delegate to FilesystemCatalog (IDENTICAL)
impl ArtifactCatalog<WorkerBinary> for WorkerCatalog {
    fn add(&self, worker: WorkerBinary) -> Result<()> {
        self.inner.add(worker)
    }
    // ... 5 more delegation methods
}
```

**Duplication Metrics:**
- **Common code:** ~50 LOC per catalog
- **Total duplication:** ~100 LOC
- **Differences:** Only subdirectory name ("models" vs "workers")

**Analysis:**

**Could we consolidate?**

Option 1: Generic catalog with subdirectory parameter
```rust
pub struct TypedCatalog<T: Artifact> {
    inner: FilesystemCatalog<T>,
    subdir: &'static str,
}

impl<T: Artifact> TypedCatalog<T> {
    pub fn new(subdir: &'static str) -> Result<Self> {
        let catalog_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("rbee")
            .join(subdir);
        
        let inner = FilesystemCatalog::new(catalog_dir)?;
        Ok(Self { inner, subdir })
    }
}

// Usage
pub type ModelCatalog = TypedCatalog<ModelEntry>;
pub type WorkerCatalog = TypedCatalog<WorkerBinary>;

let models = ModelCatalog::new("models")?;
let workers = WorkerCatalog::new("workers")?;
```

**Verdict:** ⚠️ **ACCEPTABLE AS-IS**

**Why NOT consolidate:**
1. **Type safety** - Each catalog has domain-specific methods:
   - `ModelCatalog::model_path()`
   - `WorkerCatalog::worker_path()` + `find_by_type_and_platform()`
2. **Clarity** - Explicit types are easier to understand than generics
3. **Small scope** - Only 50 LOC per catalog
4. **Future divergence** - Catalogs may add different methods
5. **Pre-1.0** - Consolidation would be a breaking change later

**If we consolidate (future):**
- Save ~80 LOC (delegation methods)
- Lose type-specific methods (need to move to trait extensions)
- More complex API (generic parameters everywhere)
- ROI: Not worth it for 80 LOC

---

### 3. Type Abstractions (❌ NO DUPLICATION - Well designed)

**Shared Types (artifact-catalog):**
```rust
// artifact-catalog/src/types.rs
pub enum ArtifactStatus {
    Available,
    Downloading,
    Failed { error: String },
}

pub trait Artifact: Clone + Serialize + for<'de> Deserialize<'de> {
    fn id(&self) -> &str;
    fn path(&self) -> &Path;
    fn size(&self) -> u64;
    fn status(&self) -> &ArtifactStatus;
    fn set_status(&mut self, status: ArtifactStatus);
    fn name(&self) -> &str { self.id() }
}

pub struct ArtifactMetadata<T> {
    pub artifact: T,
    pub added_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: Option<chrono::DateTime<chrono::Utc>>,
}
```

**Concrete Implementations:**
```rust
// model-catalog/src/types.rs
pub type ModelStatus = ArtifactStatus;  // ✅ Type alias, not duplication

pub struct ModelEntry {
    id: String,
    name: String,
    path: PathBuf,
    size: u64,
    status: ArtifactStatus,  // ✅ Uses shared type
    added_at: chrono::DateTime<chrono::Utc>,
}

impl Artifact for ModelEntry { /* ... */ }  // ✅ Implements shared trait

// worker-catalog/src/types.rs
pub type WorkerStatus = ArtifactStatus;  // ✅ Type alias, not duplication

pub struct WorkerBinary {
    id: String,
    worker_type: WorkerType,  // ← Domain-specific field
    platform: Platform,       // ← Domain-specific field
    path: PathBuf,
    size: u64,
    status: ArtifactStatus,   // ✅ Uses shared type
    version: String,          // ← Domain-specific field
    added_at: chrono::DateTime<chrono::Utc>,
}

impl Artifact for WorkerBinary { /* ... */ }  // ✅ Implements shared trait
```

**Analysis:**
- ✅ **Perfect abstraction** - Shared behavior in trait, domain-specific fields in structs
- ✅ **No duplication** - Both use `ArtifactStatus` from shared crate
- ✅ **Type aliases** - `ModelStatus`/`WorkerStatus` are just aliases (zero cost)
- ✅ **Domain-specific** - Each struct has unique fields (WorkerType, Platform, version)

**Verdict:** ❌ **NO DUPLICATION** - This is textbook trait abstraction

---

### 4. Daemon Lifecycle (❌ NO DUPLICATION FOUND)

**Checked:**
- `daemon-lifecycle/src/start.rs` - `HttpDaemonConfig` (inline, not duplicated)
- `daemon-lifecycle/src/lib.rs` - `SshConfig` (unique to this crate)
- `daemon-lifecycle/src/utils/` - SSH/poll utils (unique implementations)

**Analysis:**
- ✅ All types are unique to daemon-lifecycle
- ✅ No overlap with catalog crates
- ✅ Utils are properly modularized

**Verdict:** ❌ **NO DUPLICATION**

---

## Summary Table

| Category | Location | LOC | Verdict | Action |
|----------|----------|-----|---------|--------|
| Mock test artifacts | `#[cfg(test)]` modules | ~80 | ✅ GOOD | Keep (test isolation) |
| Catalog wrapper pattern | model/worker-catalog | ~100 | ⚠️ MINOR | Keep (type safety > DRY) |
| Type abstractions | artifact-catalog trait | 0 | ✅ PERFECT | Keep (textbook design) |
| Daemon lifecycle | daemon-lifecycle | 0 | ✅ UNIQUE | Keep (no duplication) |

**Total Duplication:** ~180 LOC (all acceptable)

---

## Recommendations

### ✅ Keep Everything As-Is

**Rationale:**
1. **Test duplication** - Intentional for test isolation
2. **Catalog wrappers** - Type safety > 100 LOC savings
3. **Type abstractions** - Already perfect (trait-based)
4. **Daemon lifecycle** - No duplication found

### 🔮 Future Consolidation (Post-1.0)

**IF** catalog wrappers diverge significantly:
- Keep separate (domain-specific behavior)

**IF** catalog wrappers stay identical:
- Consider `TypedCatalog<T>` generic wrapper
- ROI: ~80 LOC savings
- Cost: More complex API, breaking change

**Decision:** Wait until 1.0 to evaluate

---

## Architecture Validation

### ✅ Proper Abstraction Layers

```
┌─────────────────────────────────────────────┐
│ artifact-catalog (shared abstractions)      │
│ - Artifact trait                            │
│ - ArtifactStatus enum                       │
│ - FilesystemCatalog<T>                      │
│ - ArtifactProvisioner trait                 │
└─────────────────────────────────────────────┘
                    ▲
                    │ implements
        ┌───────────┴───────────┐
        │                       │
┌───────────────┐       ┌───────────────┐
│ model-catalog │       │ worker-catalog│
│ - ModelEntry  │       │ - WorkerBinary│
│ - ModelCatalog│       │ - WorkerCatalog│
└───────────────┘       └───────────────┘
```

**Design Principles:**
1. ✅ **Shared behavior** → Trait (Artifact)
2. ✅ **Shared state** → Enum (ArtifactStatus)
3. ✅ **Shared implementation** → Generic (FilesystemCatalog<T>)
4. ✅ **Domain-specific** → Concrete structs (ModelEntry, WorkerBinary)

**This is CORRECT architecture.**

---

## Conclusion

**No action needed.** The codebase has:
- ✅ Proper trait abstractions
- ✅ Acceptable test duplication (isolation)
- ✅ Acceptable wrapper duplication (type safety)
- ✅ No type duplication (shared via traits)

**Total unnecessary duplication:** 0 LOC

**Pre-1.0 verdict:** Ship it. This is clean code.

---

**Files Analyzed:**
- `bin/25_rbee_hive_crates/artifact-catalog/src/catalog.rs` (274 LOC)
- `bin/25_rbee_hive_crates/artifact-catalog/src/lib.rs` (21 LOC)
- `bin/25_rbee_hive_crates/artifact-catalog/src/provisioner.rs` (159 LOC)
- `bin/25_rbee_hive_crates/artifact-catalog/src/types.rs` (70 LOC)
- `bin/25_rbee_hive_crates/model-catalog/src/lib.rs` (124 LOC)
- `bin/25_rbee_hive_crates/model-catalog/src/types.rs` (82 LOC)
- `bin/25_rbee_hive_crates/worker-catalog/src/lib.rs` (146 LOC)
- `bin/25_rbee_hive_crates/worker-catalog/src/types.rs` (162 LOC)
- `bin/99_shared_crates/daemon-lifecycle/src/lib.rs` (124 LOC)
- `bin/99_shared_crates/daemon-lifecycle/src/start.rs` (290 LOC)

**Total analyzed:** ~1,452 LOC
