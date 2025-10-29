# Storage Architecture for rbee-hive

**Created by:** TEAM-266 (updated)  
**Date:** Oct 23, 2025

---

## 🎯 Overview

rbee-hive uses **filesystem-based storage** for models and workers. The catalog is built by scanning directories, not by maintaining in-memory state.

---

## 📁 Directory Structure

### Models Directory

**Location (cross-platform):**
- **Linux/Mac:** `~/.cache/rbee/models/`
- **Windows:** `%LOCALAPPDATA%\rbee\models\`

**Structure:**
```
~/.cache/rbee/models/
├── meta-llama/
│   └── Llama-2-7b-chat-hf/
│       ├── metadata.yaml          ← Model metadata
│       ├── model.safetensors      ← Model weights
│       ├── tokenizer.json         ← Tokenizer
│       └── config.json            ← Model config
│
├── mistralai/
│   └── Mistral-7B-Instruct-v0.2/
│       ├── metadata.yaml
│       ├── model.safetensors
│       └── ...
│
└── TheBloke/
    └── Llama-2-7B-Chat-GGUF/
        ├── metadata.yaml
        ├── llama-2-7b-chat.Q4_K_M.gguf
        └── ...
```

### Worker Binaries

**Development:**
- `./target/debug/llama-worker` (debug builds)
- `./target/release/llama-worker` (release builds)

**Production (search order):**
1. `./target/release/llama-worker`
2. `/usr/local/bin/llama-worker`
3. `~/.local/bin/llama-worker`
4. System PATH

**Rationale:** Using `./target/` during development allows for easier rebuilds without installation.

---

## 📄 Metadata Format

### metadata.yaml

Each model directory contains a `metadata.yaml` file with the following structure:

```yaml
id: "meta-llama/Llama-2-7b-chat-hf"
name: "Llama 2 7B Chat"
path: "/home/user/.cache/rbee/models/meta-llama/Llama-2-7b-chat-hf"
size_bytes: 13476839424
added_at: "2025-10-23T10:15:30Z"
status: "Ready"
metadata:
  architecture: "llama"
  parameters: "7B"
  quantization: null
  source_url: "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
```

**Fields:**
- `id`: Unique model identifier (HuggingFace repo format)
- `name`: Human-readable display name
- `path`: Full path to model directory
- `size_bytes`: Total size of model files
- `added_at`: ISO 8601 timestamp when model was added
- `status`: One of `Ready`, `Downloading`, or `Failed`
- `metadata`: Optional metadata (architecture, parameters, quantization, source URL)

---

## 🔄 Catalog Operations

### ModelCatalog (Filesystem-Based)

The `ModelCatalog` does **NOT** maintain in-memory state. Instead, it scans the filesystem on demand.

#### List Models
```rust
pub fn list(&self) -> Vec<ModelEntry> {
    // 1. Scan ~/.cache/rbee/models/ directory
    // 2. For each subdirectory, read metadata.yaml
    // 3. Return all valid ModelEntry structs
}
```

#### Get Model
```rust
pub fn get(&self, id: &str) -> Result<ModelEntry> {
    // 1. Construct path: ~/.cache/rbee/models/{id}/metadata.yaml
    // 2. Read and parse YAML file
    // 3. Return ModelEntry
}
```

#### Add Model
```rust
pub fn add(&self, model: ModelEntry) -> Result<()> {
    // 1. Create directory: ~/.cache/rbee/models/{id}/
    // 2. Write metadata.yaml with model info
    // 3. Return success
}
```

#### Remove Model
```rust
pub fn remove(&self, id: &str) -> Result<ModelEntry> {
    // 1. Read metadata.yaml to get ModelEntry
    // 2. Delete entire directory: ~/.cache/rbee/models/{id}/
    // 3. Return deleted ModelEntry
}
```

#### Update Status
```rust
pub fn update_status(&self, id: &str, status: ModelStatus) -> Result<()> {
    // 1. Read metadata.yaml
    // 2. Update status field
    // 3. Write metadata.yaml back
}
```

---

## 🔐 Permissions

### Models Directory

**Required permissions:**
- **Read:** List models, get model info
- **Write:** Download models, update metadata
- **Delete:** Remove models

**Owner:** rbee-hive process (typically the user running rbee-hive)

**Permissions:** `0755` (rwxr-xr-x) for directories, `0644` (rw-r--r--) for files

### Worker Binaries

**Development:**
- Binaries in `./target/` inherit project permissions
- No special permissions needed

**Production:**
- Binaries should be executable: `0755` (rwxr-xr-x)
- Owned by user or system (depending on installation)

---

## 🚀 Benefits of Filesystem-Based Catalog

### 1. **Persistence**
- No need for SQLite or other database
- Models survive process restarts automatically
- Metadata is human-readable and editable

### 2. **Simplicity**
- No schema migrations
- No database corruption issues
- Easy to inspect and debug

### 3. **Cross-Platform**
- Works on Linux, Mac, Windows
- Uses standard cache directories
- No platform-specific database drivers

### 4. **Transparency**
- Users can see exactly what's stored
- Easy to manually add/remove models
- Clear separation between model files and metadata

### 5. **Compatibility**
- Can integrate with external tools
- Easy to backup (just copy directory)
- Can share models between systems

---

## 🔄 Integration with Model Provisioner

### Download Flow

1. **ModelProvisioner receives download request**
   ```rust
   download_model(job_id, model_id)
   ```

2. **Create metadata with Downloading status**
   ```rust
   let model = ModelEntry::new(model_id, ...);
   model.status = ModelStatus::Downloading { progress: 0.0 };
   catalog.add(model)?;  // Creates directory + metadata.yaml
   ```

3. **Download model files from HuggingFace**
   ```rust
   // Download to ~/.cache/rbee/models/{model_id}/
   download_files_from_hf(model_id, &model_path).await?;
   ```

4. **Update metadata to Ready**
   ```rust
   catalog.update_status(model_id, ModelStatus::Ready)?;
   // Rewrites metadata.yaml with new status
   ```

5. **Model now appears in catalog**
   ```rust
   let models = catalog.list();  // Scans filesystem, includes new model
   ```

---

## 🛠️ Implementation Details

### Cross-Platform Paths

Using the `dirs` crate for cross-platform directory resolution:

```rust
use dirs;

// Get cache directory
let cache_dir = dirs::cache_dir()
    .ok_or_else(|| anyhow!("Cannot determine cache directory"))?;

// Construct models directory
let models_dir = cache_dir.join("rbee").join("models");

// Result:
// Linux:   /home/user/.cache/rbee/models
// Mac:     /Users/user/Library/Caches/rbee/models
// Windows: C:\Users\user\AppData\Local\rbee\models
```

### YAML Serialization

Using `serde_yaml` for metadata files:

```rust
use serde_yaml;

// Write metadata
let yaml = serde_yaml::to_string(&model_entry)?;
fs::write(&metadata_path, yaml)?;

// Read metadata
let content = fs::read_to_string(&metadata_path)?;
let model_entry: ModelEntry = serde_yaml::from_str(&content)?;
```

### Error Handling

```rust
// Missing metadata file
if !metadata_path.exists() {
    return Err(anyhow!("Model '{}' not found in catalog", id));
}

// Corrupted metadata file
match serde_yaml::from_str(&content) {
    Ok(entry) => Ok(entry),
    Err(e) => Err(anyhow!("Failed to parse metadata for '{}': {}", id, e)),
}
```

---

## 🧪 Testing Considerations

### Unit Tests

```rust
#[test]
fn test_catalog_filesystem_operations() {
    // Use temporary directory for tests
    let temp_dir = tempfile::tempdir().unwrap();
    let catalog = ModelCatalog::with_dir(temp_dir.path()).unwrap();
    
    // Test add/get/remove
    let model = ModelEntry::new(...);
    catalog.add(model.clone()).unwrap();
    
    // Verify metadata.yaml exists
    let metadata_path = temp_dir.path()
        .join(&model.id)
        .join("metadata.yaml");
    assert!(metadata_path.exists());
    
    // Verify get works
    let retrieved = catalog.get(&model.id).unwrap();
    assert_eq!(retrieved.id, model.id);
    
    // Verify remove works
    catalog.remove(&model.id).unwrap();
    assert!(!metadata_path.exists());
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_download_creates_metadata() {
    let catalog = Arc::new(ModelCatalog::new().unwrap());
    let provisioner = ModelProvisioner::new(catalog.clone()).unwrap();
    
    // Download model
    provisioner.download_model("job-123", "test-model").await.unwrap();
    
    // Verify metadata exists
    let model = catalog.get("test-model").unwrap();
    assert_eq!(model.status, ModelStatus::Ready);
    
    // Verify files exist
    assert!(model.path.exists());
    assert!(model.path.join("metadata.yaml").exists());
}
```

---

## 📊 Performance Considerations

### Filesystem Scanning

**Concern:** Scanning filesystem on every `list()` call might be slow.

**Mitigation:**
1. Cache directory is typically small (10-100 models)
2. Filesystem metadata is cached by OS
3. Only reads metadata.yaml, not model files
4. Can add in-memory cache later if needed

**Benchmark target:** `list()` should complete in <10ms for 100 models

### Metadata File Size

**Concern:** Writing metadata.yaml on every status update.

**Mitigation:**
1. YAML files are small (~500 bytes)
2. Status updates are infrequent
3. Filesystem writes are buffered by OS

**Benchmark target:** `update_status()` should complete in <1ms

---

## 🔮 Future Enhancements

### Optional In-Memory Cache

```rust
pub struct ModelCatalog {
    models_dir: PathBuf,
    cache: Arc<Mutex<Option<HashMap<String, ModelEntry>>>>,
    cache_ttl: Duration,
}

impl ModelCatalog {
    pub fn list(&self) -> Vec<ModelEntry> {
        // Check cache first
        if let Some(cached) = self.get_cached() {
            return cached;
        }
        
        // Scan filesystem
        let models = self.scan_filesystem();
        
        // Update cache
        self.update_cache(models.clone());
        
        models
    }
}
```

### Watch for External Changes

```rust
use notify::{Watcher, RecursiveMode};

// Watch models directory for external changes
let watcher = notify::recommended_watcher(|event| {
    // Invalidate cache when directory changes
    catalog.invalidate_cache();
})?;

watcher.watch(&models_dir, RecursiveMode::Recursive)?;
```

---

## ✅ Summary

**Storage Strategy:**
- ✅ Filesystem-based catalog (no database)
- ✅ Cross-platform cache directories
- ✅ YAML metadata files
- ✅ Scans directory on demand
- ✅ Simple, transparent, maintainable

**Locations:**
- ✅ Models: `~/.cache/rbee/models/`
- ✅ Workers: `./target/` (dev) or system PATH (prod)

**Benefits:**
- ✅ No database complexity
- ✅ Human-readable metadata
- ✅ Easy to inspect and debug
- ✅ Survives process restarts
- ✅ Cross-platform compatible

---

**This architecture is ready for implementation in Phase 1 (TEAM-267)!**
