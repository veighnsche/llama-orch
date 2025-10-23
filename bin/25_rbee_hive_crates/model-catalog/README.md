# rbee-hive-model-catalog

**Status:** ✅ IMPLEMENTED (TEAM-267)  
**Purpose:** Filesystem-based model catalog management

## Overview

Manages the catalog of available LLM models on this rbee-hive using a filesystem-based approach. Each model has a directory with YAML metadata, and the catalog scans the filesystem on demand.

## Architecture

### Storage Strategy

- **Location:** Platform-specific cache directory
  - Linux/Mac: `~/.cache/rbee/models/`
  - Windows: `%LOCALAPPDATA%\rbee\models\`
- **Structure:** Each model has its own directory
  - Directory: `{cache}/rbee/models/{model-id}/`
  - Metadata: `{model-id}/metadata.yaml`
- **No in-memory cache:** Catalog scans filesystem on each operation

### Why Filesystem-Based?

1. **Simplicity:** No need for database setup or migration
2. **Transparency:** Users can inspect models directly
3. **Reliability:** Filesystem is the source of truth
4. **Cross-platform:** Works on Linux, Mac, Windows

## Public API

### ModelCatalog

Main catalog struct for managing models:

```rust
pub struct ModelCatalog {
    // Internal: models_dir path
}

impl ModelCatalog {
    pub fn new() -> Result<Self>;
    pub fn add(&self, model: ModelEntry) -> Result<()>;
    pub fn get(&self, id: &str) -> Result<ModelEntry>;
    pub fn remove(&self, id: &str) -> Result<ModelEntry>;
    pub fn list(&self) -> Vec<ModelEntry>;
    pub fn list_by_status(&self, filter: fn(&ModelStatus) -> bool) -> Vec<ModelEntry>;
    pub fn update_status(&self, id: &str, status: ModelStatus) -> Result<()>;
    pub fn contains(&self, id: &str) -> bool;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

### ModelEntry

Represents a single model in the catalog:

```rust
pub struct ModelEntry {
    pub id: String,                              // e.g., "meta-llama/Llama-2-7b-chat-hf"
    pub name: String,                            // Display name
    pub path: PathBuf,                           // Local path to model files
    pub size_bytes: u64,                         // Model size
    pub added_at: DateTime<Utc>,                 // When added to catalog
    pub status: ModelStatus,                     // Current status
    pub metadata: Option<ModelMetadata>,         // Optional metadata
}

impl ModelEntry {
    pub fn new(id: String, name: String, path: PathBuf, size_bytes: u64) -> Self;
    pub fn with_metadata(self, metadata: ModelMetadata) -> Self;
    pub fn is_ready(&self) -> bool;
}
```

### ModelStatus

Model availability status:

```rust
pub enum ModelStatus {
    Ready,                              // Fully downloaded and ready
    Downloading { progress: f32 },      // Currently downloading (0.0-1.0)
    Failed { error: String },           // Download failed
}
```

### ModelMetadata

Optional model metadata:

```rust
pub struct ModelMetadata {
    pub architecture: Option<String>,    // e.g., "llama", "mistral"
    pub parameters: Option<String>,      // e.g., "7B", "13B"
    pub quantization: Option<String>,    // e.g., "Q4_K_M", "Q5_K_S"
    pub source_url: Option<String>,      // HuggingFace repo URL
}
```

## Usage Examples

### Basic Operations

```rust
use rbee_hive_model_catalog::{ModelCatalog, ModelEntry};
use std::path::PathBuf;

// Create catalog
let catalog = ModelCatalog::new()?;

// Add a model
let model = ModelEntry::new(
    "meta-llama/Llama-2-7b-chat-hf".to_string(),
    "Llama 2 7B Chat".to_string(),
    PathBuf::from("/path/to/model"),
    7_000_000_000,
);
catalog.add(model)?;

// List all models
let models = catalog.list();
for model in models {
    println!("{}: {} bytes", model.name, model.size_bytes);
}

// Get a specific model
let model = catalog.get("meta-llama/Llama-2-7b-chat-hf")?;
println!("Status: {:?}", model.status);

// Remove a model
catalog.remove("meta-llama/Llama-2-7b-chat-hf")?;
```

### Status Management

```rust
use rbee_hive_model_catalog::{ModelStatus, ModelCatalog};

let catalog = ModelCatalog::new()?;

// Update status to downloading
catalog.update_status(
    "model-id",
    ModelStatus::Downloading { progress: 0.5 }
)?;

// List only ready models
let ready_models = catalog.list_by_status(|s| matches!(s, ModelStatus::Ready));
println!("Ready models: {}", ready_models.len());
```

### With Metadata

```rust
use rbee_hive_model_catalog::{ModelEntry, ModelMetadata};
use std::path::PathBuf;

let model = ModelEntry::new(
    "meta-llama/Llama-2-7b-chat-hf".to_string(),
    "Llama 2 7B Chat".to_string(),
    PathBuf::from("/path/to/model"),
    7_000_000_000,
).with_metadata(ModelMetadata {
    architecture: Some("llama".to_string()),
    parameters: Some("7B".to_string()),
    quantization: Some("Q4_K_M".to_string()),
    source_url: Some("https://huggingface.co/meta-llama/Llama-2-7b-chat-hf".to_string()),
});

catalog.add(model)?;
```

## Implementation Details

### YAML Metadata Format

Each model's `metadata.yaml` contains:

```yaml
id: "meta-llama/Llama-2-7b-chat-hf"
name: "Llama 2 7B Chat"
path: "/home/user/.cache/rbee/models/meta-llama/Llama-2-7b-chat-hf"
size_bytes: 7000000000
added_at: "2025-10-23T14:30:00Z"
status: Ready
metadata:
  architecture: "llama"
  parameters: "7B"
  quantization: "Q4_K_M"
  source_url: "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
```

### Error Handling

All operations return `Result<T, anyhow::Error>`:

- **add():** Fails if model already exists
- **get():** Fails if model not found or metadata invalid
- **remove():** Fails if model not found
- **update_status():** Fails if model not found

### Thread Safety

`ModelCatalog` is `Clone` and safe to share across threads. Each operation reads/writes to the filesystem directly.

## Testing

Run unit tests:

```bash
cargo test --package rbee-hive-model-catalog
```

Run with output:

```bash
cargo test --package rbee-hive-model-catalog -- --nocapture
```

## Implementation Status

- [x] Core functionality (TEAM-267)
- [x] Unit tests (9 tests)
- [x] Documentation
- [ ] Integration with model-provisioner (TEAM-269)
- [ ] Integration with job_router (TEAM-273)

## Dependencies

- `anyhow` - Error handling
- `chrono` - Timestamps with serde support
- `dirs` - Cross-platform cache directory
- `serde` - Serialization
- `serde_yaml` - YAML metadata files
- `tempfile` (dev) - Isolated test directories

## Next Steps (TEAM-268)

TEAM-268 will implement the operation handlers:
- `execute_model_list()` - List all models
- `execute_model_get()` - Get model details
- `execute_model_delete()` - Remove a model

These will use the `ModelCatalog` API and emit narration events for SSE streaming.

## Notes

- Originally created by TEAM-135 as a stub
- Implemented by TEAM-267 as part of rbee-hive implementation plan
- Moved from shared-crates to rbee-hive-crates (hive-specific)
- Filesystem-based approach chosen for simplicity and transparency
