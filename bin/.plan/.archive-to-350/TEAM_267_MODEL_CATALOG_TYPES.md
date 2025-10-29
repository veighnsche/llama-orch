# TEAM-267: Model Catalog Types & Storage

**Phase:** 1 of 9  
**Estimated Effort:** 20-24 hours  
**Prerequisites:** None (first phase!)  
**Blocks:** TEAM-268 (Model Catalog Operations)

---

## 🎯 Mission

Implement the foundational types and storage layer for the model catalog. This crate tracks which models are available on the hive by **scanning the filesystem**.

**Storage Strategy:**
- Models stored in: `~/.cache/rbee/models/` (Linux/Mac) or `%LOCALAPPDATA%\rbee\models\` (Windows)
- Each model has a directory: `~/.cache/rbee/models/{model-id}/`
- Metadata stored as: `{model-id}/metadata.yaml`
- Catalog is **filesystem-based** - scans directory on demand

**Deliverables:**
1. ✅ ModelEntry struct with all required fields
2. ✅ ModelCatalog struct that scans filesystem
3. ✅ Metadata YAML read/write functions
4. ✅ Basic CRUD operations (add, get, remove, list)
5. ✅ Unit tests for all operations
6. ✅ Public API documentation

---

## 📁 Files to Modify

```
bin/25_rbee_hive_crates/model-catalog/
├── src/
│   ├── lib.rs          ← Update module exports
│   ├── types.rs        ← Implement ModelEntry and related types
│   └── catalog.rs      ← Implement ModelCatalog struct
├── Cargo.toml          ← Add dependencies
└── README.md           ← Document public API
```

---

## 🏗️ Implementation Guide

### Step 1: Define Types (types.rs)

**Create ModelEntry:**

```rust
// TEAM-267: Model catalog entry
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelEntry {
    /// Unique model ID (e.g., "meta-llama/Llama-2-7b-chat-hf")
    pub id: String,
    
    /// Display name
    pub name: String,
    
    /// Local path where model files are stored
    pub path: PathBuf,
    
    /// Model size in bytes
    pub size_bytes: u64,
    
    /// When the model was added to catalog
    pub added_at: chrono::DateTime<chrono::Utc>,
    
    /// Model status
    pub status: ModelStatus,
    
    /// Optional metadata
    pub metadata: Option<ModelMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    /// Model is fully downloaded and ready
    Ready,
    
    /// Model is currently being downloaded
    Downloading { progress: f32 },
    
    /// Model download failed
    Failed { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    /// Model architecture (e.g., "llama", "mistral")
    pub architecture: Option<String>,
    
    /// Parameter count (e.g., "7B", "13B")
    pub parameters: Option<String>,
    
    /// Quantization (e.g., "Q4_K_M", "Q5_K_S")
    pub quantization: Option<String>,
    
    /// HuggingFace repo URL
    pub source_url: Option<String>,
}

impl ModelEntry {
    pub fn new(id: String, name: String, path: PathBuf, size_bytes: u64) -> Self {
        Self {
            id,
            name,
            path,
            size_bytes,
            added_at: chrono::Utc::now(),
            status: ModelStatus::Ready,
            metadata: None,
        }
    }
    
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
    
    pub fn is_ready(&self) -> bool {
        matches!(self.status, ModelStatus::Ready)
    }
}
```

### Step 2: Implement ModelCatalog (catalog.rs)

**Create filesystem-based catalog:**

```rust
// TEAM-267: Model catalog implementation (filesystem-based)
use crate::types::{ModelEntry, ModelStatus, ModelMetadata};
use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::fs;

/// Filesystem-based model catalog
#[derive(Clone)]
pub struct ModelCatalog {
    models_dir: PathBuf,
}

impl ModelCatalog {
    /// Create a new catalog pointing to the models directory
    pub fn new() -> Result<Self> {
        let models_dir = Self::get_models_dir()?;
        fs::create_dir_all(&models_dir)?;
        
        Ok(Self { models_dir })
    }
    
    /// Get the platform-specific models directory
    fn get_models_dir() -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow!("Cannot determine cache directory"))?;
        
        Ok(cache_dir.join("rbee").join("models"))
    }
    
    /// Get path for a specific model
    fn model_path(&self, id: &str) -> PathBuf {
        self.models_dir.join(id)
    }
    
    /// Get metadata file path for a model
    fn metadata_path(&self, id: &str) -> PathBuf {
        self.model_path(id).join("metadata.yaml")
    }
    
    /// Read metadata from YAML file
    fn read_metadata(&self, id: &str) -> Result<ModelEntry> {
        let metadata_path = self.metadata_path(id);
        
        if !metadata_path.exists() {
            return Err(anyhow!("Metadata file not found for model '{}'", id));
        }
        
        let content = fs::read_to_string(&metadata_path)?;
        let entry: ModelEntry = serde_yaml::from_str(&content)?;
        Ok(entry)
    }
    
    /// Write metadata to YAML file
    fn write_metadata(&self, entry: &ModelEntry) -> Result<()> {
        let metadata_path = self.metadata_path(&entry.id);
        let model_dir = self.model_path(&entry.id);
        
        fs::create_dir_all(&model_dir)?;
        
        let content = serde_yaml::to_string(entry)?;
        fs::write(&metadata_path, content)?;
        Ok(())
    }
    
    /// Add a model to the catalog (creates directory and metadata file)
    pub fn add(&self, model: ModelEntry) -> Result<()> {
        let model_dir = self.model_path(&model.id);
        
        if model_dir.exists() {
            return Err(anyhow!("Model '{}' already exists in catalog", model.id));
        }
        
        self.write_metadata(&model)?;
        Ok(())
    }
    
    /// Get a model by ID (reads from filesystem)
    pub fn get(&self, id: &str) -> Result<ModelEntry> {
        self.read_metadata(id)
    }
    
    /// Remove a model from the catalog (deletes directory)
    pub fn remove(&self, id: &str) -> Result<ModelEntry> {
        let entry = self.read_metadata(id)?;
        let model_dir = self.model_path(id);
        
        if model_dir.exists() {
            fs::remove_dir_all(&model_dir)?;
        }
        
        Ok(entry)
    }
    
    /// List all models (scans filesystem)
    pub fn list(&self) -> Vec<ModelEntry> {
        let mut models = Vec::new();
        
        if let Ok(entries) = fs::read_dir(&self.models_dir) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        if let Some(model_id) = entry.file_name().to_str() {
                            if let Ok(model_entry) = self.read_metadata(model_id) {
                                models.push(model_entry);
                            }
                        }
                    }
                }
            }
        }
        
        models
    }
    
    /// List models by status
    pub fn list_by_status(&self, status_filter: fn(&ModelStatus) -> bool) -> Vec<ModelEntry> {
        self.list()
            .into_iter()
            .filter(|m| status_filter(&m.status))
            .collect()
    }
    
    /// Update model status (rewrites metadata file)
    pub fn update_status(&self, id: &str, status: ModelStatus) -> Result<()> {
        let mut entry = self.read_metadata(id)?;
        entry.status = status;
        self.write_metadata(&entry)?;
        Ok(())
    }
    
    /// Check if model exists (checks filesystem)
    pub fn contains(&self, id: &str) -> bool {
        self.model_path(id).exists()
    }
    
    /// Get catalog size (number of models)
    pub fn len(&self) -> usize {
        self.list().len()
    }
    
    /// Check if catalog is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::new()
    }
}
```

### Step 3: Update lib.rs

```rust
// TEAM-267: Model catalog crate
#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-hive-model-catalog
//!
//! Model catalog management for tracking available models

pub mod catalog;
pub mod types;

// Re-export main types
pub use catalog::ModelCatalog;
pub use types::{ModelEntry, ModelMetadata, ModelStatus};
```

### Step 4: Add Dependencies (Cargo.toml)

```toml
[dependencies]
anyhow = "1.0"
chrono = { version = "0.4", features = ["serde"] }
dirs = "5.0"  # For cross-platform cache directory
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"  # For metadata.yaml files
```

### Step 5: Write Unit Tests

**Add to catalog.rs:**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_catalog_add_get() {
        let catalog = ModelCatalog::new();
        let model = ModelEntry::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            PathBuf::from("/tmp/models/test"),
            1024 * 1024 * 100, // 100 MB
        );
        
        catalog.add(model.clone()).unwrap();
        let retrieved = catalog.get("test-model").unwrap();
        
        assert_eq!(retrieved.id, "test-model");
        assert_eq!(retrieved.name, "Test Model");
    }
    
    #[test]
    fn test_catalog_duplicate_add() {
        let catalog = ModelCatalog::new();
        let model = ModelEntry::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            PathBuf::from("/tmp/models/test"),
            1024,
        );
        
        catalog.add(model.clone()).unwrap();
        let result = catalog.add(model);
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }
    
    #[test]
    fn test_catalog_remove() {
        let catalog = ModelCatalog::new();
        let model = ModelEntry::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            PathBuf::from("/tmp/models/test"),
            1024,
        );
        
        catalog.add(model).unwrap();
        assert_eq!(catalog.len(), 1);
        
        let removed = catalog.remove("test-model").unwrap();
        assert_eq!(removed.id, "test-model");
        assert_eq!(catalog.len(), 0);
    }
    
    #[test]
    fn test_catalog_list() {
        let catalog = ModelCatalog::new();
        
        for i in 0..3 {
            let model = ModelEntry::new(
                format!("model-{}", i),
                format!("Model {}", i),
                PathBuf::from(format!("/tmp/models/model-{}", i)),
                1024,
            );
            catalog.add(model).unwrap();
        }
        
        let models = catalog.list();
        assert_eq!(models.len(), 3);
    }
    
    #[test]
    fn test_catalog_update_status() {
        let catalog = ModelCatalog::new();
        let model = ModelEntry::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            PathBuf::from("/tmp/models/test"),
            1024,
        );
        
        catalog.add(model).unwrap();
        
        catalog
            .update_status("test-model", ModelStatus::Downloading { progress: 0.5 })
            .unwrap();
        
        let updated = catalog.get("test-model").unwrap();
        assert!(matches!(updated.status, ModelStatus::Downloading { .. }));
    }
}
```

---

## ✅ Acceptance Criteria

- [ ] ModelEntry struct defined with all fields
- [ ] ModelStatus enum with Ready/Downloading/Failed variants
- [ ] ModelMetadata struct for optional metadata
- [ ] ModelCatalog struct with Arc<Mutex<HashMap>>
- [ ] add() method working
- [ ] get() method working
- [ ] remove() method working
- [ ] list() method working
- [ ] update_status() method working
- [ ] Unit tests passing (5+ tests)
- [ ] `cargo check --package rbee-hive-model-catalog` passes
- [ ] `cargo test --package rbee-hive-model-catalog` passes
- [ ] Public API documented in README.md

---

## 🧪 Testing Commands

```bash
# Check compilation
cargo check --package rbee-hive-model-catalog

# Run unit tests
cargo test --package rbee-hive-model-catalog

# Run with output
cargo test --package rbee-hive-model-catalog -- --nocapture

# Check documentation
cargo doc --package rbee-hive-model-catalog --open
```

---

## 📝 Handoff Checklist

Create `TEAM_267_HANDOFF.md` with:

- [ ] What was implemented (list all functions)
- [ ] What tests were added
- [ ] Any deviations from this guide
- [ ] Known issues or limitations
- [ ] Notes for TEAM-268

---

## 🎯 Success Metrics

**Code:**
- ModelEntry: ~50 lines
- ModelCatalog: ~150 lines
- Tests: ~100 lines
- Total: ~300 lines

**Tests:**
- Unit tests: 5+ passing
- Coverage: All public methods tested

**Compilation:**
- ✅ `cargo check` passes
- ✅ `cargo test` passes
- ✅ No warnings

---

## 📚 Reference Implementations

Look at these for patterns:
- `bin/15_queen_rbee_crates/worker-registry/src/lib.rs` (similar registry pattern)
- `bin/99_shared_crates/job-server/src/lib.rs` (Arc<Mutex<>> usage)

---

## 🚨 Common Issues

### Issue 1: Mutex Poisoning

If a thread panics while holding the lock, the Mutex becomes "poisoned".

**Solution:** Use `.unwrap()` for now (acceptable in v0.1.0). Later, use `.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?`.

### Issue 2: Clone Performance

Cloning ModelEntry on every get() might be slow for large catalogs.

**Solution:** Acceptable for v0.1.0. Optimize later if needed (return Arc<ModelEntry>).

---

## 🎓 Learning Resources

- **Arc<Mutex<>>:** https://doc.rust-lang.org/book/ch16-03-shared-state.html
- **HashMap:** https://doc.rust-lang.org/std/collections/struct.HashMap.html
- **chrono:** https://docs.rs/chrono/

---

**TEAM-267: You're the foundation! Make it solid! 🏗️**
