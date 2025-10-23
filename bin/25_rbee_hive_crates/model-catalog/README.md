# rbee-hive-model-catalog

**TEAM-267: Model catalog for managing LLM model files**

## Purpose

Manages model files downloaded from HuggingFace or other sources.
Built on top of `artifact-catalog` for consistency with worker-catalog.

## Storage

Models are stored in:
- **Linux/Mac:** `~/.cache/rbee/models/`
- **Windows:** `%LOCALAPPDATA%\rbee\models\`

## Usage

```rust
use rbee_hive_model_catalog::{ModelCatalog, ModelEntry};
use rbee_hive_artifact_catalog::ArtifactCatalog;

// Create catalog
let catalog = ModelCatalog::new()?;

// Add a model
let model = ModelEntry::new(
    "meta-llama/Llama-2-7b".to_string(),
    "Llama 2 7B".to_string(),
    PathBuf::from("/path/to/model"),
    8_000_000_000, // 8 GB
);
catalog.add(model)?;

// List models
let models = catalog.list();

// Get a model
let model = catalog.get("meta-llama/Llama-2-7b")?;

// Remove a model
catalog.remove("meta-llama/Llama-2-7b")?;
```

## Architecture

Uses `artifact-catalog` as base:
- `ModelEntry` implements `Artifact` trait
- `ModelCatalog` delegates to `FilesystemCatalog<ModelEntry>`
- Consistent with `worker-catalog` pattern

## Testing

```bash
cargo test --package rbee-hive-model-catalog
```
