# Component: Model Catalog (SQLite)

**Location:** `bin/shared-crates/model-catalog/src/lib.rs`  
**Type:** Persistent storage  
**Language:** Rust  
**Created by:** TEAM-030  
**Status:** ✅ IMPLEMENTED

## Overview

Persistent SQLite database tracking downloaded models for a single rbee-hive. Stores local paths, sizes, metadata. Survives hive restarts.

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS models (
    reference TEXT NOT NULL,        -- Model reference (e.g., "TinyLlama-1.1B")
    provider TEXT NOT NULL,         -- Provider (e.g., "hf", "local")
    local_path TEXT NOT NULL,       -- Full path to model file
    size_bytes INTEGER,             -- File size in bytes
    downloaded_at INTEGER,          -- Unix timestamp
    PRIMARY KEY (reference, provider)
)
```

## Data Model

```rust
pub struct ModelInfo {
    pub reference: String,          // "TinyLlama-1.1B-Chat-v1.0"
    pub provider: String,           // "hf" (HuggingFace)
    pub local_path: String,         // "/path/to/model.gguf"
    pub size_bytes: u64,            // File size
    pub downloaded_at: i64,         // Unix timestamp
}
```

## API Methods

```rust
// Create/open catalog
pub async fn new(db_path: &Path) -> Result<Self>

// Register downloaded model
pub async fn register_model(&self, model: &ModelInfo) -> Result<()>

// Find model by reference and provider
pub async fn find_model(&self, reference: &str, provider: &str) -> Result<Option<ModelInfo>>

// List all models
pub async fn list_models(&self) -> Result<Vec<ModelInfo>>

// Remove model from catalog
pub async fn remove_model(&self, reference: &str, provider: &str) -> Result<bool>

// Get total disk usage
pub async fn get_total_size(&self) -> Result<u64>
```

## Storage Location

**Default:** `.rbee/models/catalog.db` (relative to hive working directory)  
**Override:** Via constructor parameter

## Lifecycle

### 1. Initialization
```rust
// Create catalog on hive startup
let catalog = ModelCatalog::new(Path::new(".rbee/models/catalog.db")).await?;
```

### 2. Model Download
```rust
// Check if model exists
if catalog.find_model("TinyLlama", "hf").await?.is_none() {
    // Download model
    let path = download_model("hf:TinyLlama").await?;
    
    // Register in catalog
    catalog.register_model(&ModelInfo {
        reference: "TinyLlama".to_string(),
        provider: "hf".to_string(),
        local_path: path.to_string_lossy().to_string(),
        size_bytes: get_file_size(&path)?,
        downloaded_at: now_unix(),
    }).await?;
}
```

### 3. Worker Spawn
```rust
// Find model path from catalog
let model_info = catalog.find_model("TinyLlama", "hf").await?
    .ok_or("Model not found")?;

// Use local_path to spawn worker
spawn_worker(&model_info.local_path).await?;
```

### 4. Cleanup
```rust
// Remove model from catalog (doesn't delete file)
catalog.remove_model("TinyLlama", "hf").await?;

// Or delete file and remove from catalog
std::fs::remove_file(&model_info.local_path)?;
catalog.remove_model("TinyLlama", "hf").await?;
```

## Integration Points

### Model Provisioner
```rust
// bin/rbee-hive/src/provisioner/catalog.rs
pub async fn find_local_model(&self, model_ref: &str) -> Result<Option<PathBuf>> {
    let (provider, reference) = parse_model_ref(model_ref)?;
    
    if let Some(model) = self.catalog.find_model(&reference, &provider).await? {
        Ok(Some(PathBuf::from(model.local_path)))
    } else {
        Ok(None)
    }
}
```

### Worker Spawn
```rust
// bin/rbee-hive/src/http/workers.rs
let model_path = match state.model_catalog
    .find_model(&reference, &provider).await?
{
    Some(model_info) => model_info.local_path,
    None => {
        // Download and register
        let path = state.provisioner.download_model(&reference, &provider).await?;
        state.model_catalog.register_model(&ModelInfo { ... }).await?;
        path
    }
};
```

## Concurrency

**Thread-Safe:** Uses SQLite with async/await via `sqlx`

```rust
pub struct ModelCatalog {
    pool: SqlitePool,  // Connection pool
}
```

- Multiple readers allowed
- Writes are serialized by SQLite
- Connection pooling for concurrent access

## Maturity Assessment

**Status:** ✅ **PRODUCTION READY**

**Strengths:**
- ✅ Persistent storage (survives restarts)
- ✅ Complete CRUD operations
- ✅ Async/await support via sqlx
- ✅ Connection pooling
- ✅ Composite primary key (reference + provider)
- ✅ Size tracking for disk usage monitoring

**Limitations:**
- ⚠️ No model verification (checksums, signatures)
- ⚠️ No version tracking (can't have multiple versions)
- ⚠️ No metadata (architecture, quantization, context length)
- ⚠️ No tags/categories
- ⚠️ No usage statistics (download count, last used)
- ⚠️ Doesn't track model file changes (if file deleted externally)

**Recommended Improvements:**
1. Add checksum verification (SHA256)
2. Add version support (reference + version as key)
3. Add metadata fields (architecture, quant, context_len)
4. Add tags/categories for organization
5. Add usage statistics
6. Add periodic file existence checks
7. Add model description/notes field

## Comparison: Model Catalog vs Beehive Registry

| Feature | Model Catalog | Beehive Registry |
|---------|--------------|------------------|
| **Scope** | Models on this hive | Hives in network |
| **Storage** | SQLite | SQLite |
| **Persistence** | ✅ Yes | ✅ Yes |
| **Per-hive** | ✅ Yes | ❌ No (global) |
| **Purpose** | Track downloads | Track nodes |
| **Credentials** | ❌ No | ✅ Yes (SSH) |

## Testing

```bash
# Unit tests
cargo test -p model-catalog

# Integration test
# 1. Start rbee-hive
# 2. Download model
curl -X POST http://localhost:8081/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_ref": "hf:TheBloke/TinyLlama"}'

# 3. Check catalog
sqlite3 .rbee/models/catalog.db "SELECT * FROM models;"
```

## Related Components

- **Model Provisioner** - Downloads models, registers in catalog
- **Worker Spawn** - Reads catalog to find model paths
- **Download Tracker** - Tracks download progress (separate from catalog)

---

**Created by:** TEAM-030  
**Last Updated:** 2025-10-18  
**Maturity:** ✅ Production Ready (with noted limitations)
