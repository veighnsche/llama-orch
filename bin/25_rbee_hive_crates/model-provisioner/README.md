# Model Provisioner - HuggingFace Integration

**Status:** ✅ COMPLETE - Downloads models from HuggingFace using official `hf-hub` Rust crate

## Overview

This crate provides model provisioning functionality for rbee-hive, downloading GGUF models from HuggingFace Hub using the official `hf-hub` Rust crate (the same library used by Candle).

## Architecture

```
ModelProvisioner
    ↓
HuggingFaceVendor (implements VendorSource)
    ↓
hf-hub crate (official HuggingFace Rust client)
    ↓
Downloads to ~/.cache/rbee/models/{model_id}/
```

## Key Components

### 1. HuggingFaceVendor

Implements the `VendorSource` trait from `artifact-catalog`.

**Supported ID Formats:**
- `meta-llama/Llama-2-7b-chat-hf` - Standard HF repo
- `TheBloke/Llama-2-7B-Chat-GGUF` - GGUF-specific repos
- `meta-llama/Llama-2-7b:model-Q4_K_M.gguf` - Explicit filename

**Features:**
- Automatic GGUF file detection (tries common quantizations: Q4_K_M, Q5_K_M, Q4_0, etc.)
- Uses HuggingFace cache (~/.cache/huggingface/)
- Supports custom cache directories
- Narration integration for progress visibility

### 2. ModelProvisioner

Implements the `ArtifactProvisioner<ModelEntry>` trait.

**Features:**
- Downloads models from HuggingFace
- Creates `ModelEntry` artifacts
- Adds to model catalog
- Handles filesystem storage

## Usage

### Basic Example

```rust
use rbee_hive_model_provisioner::ModelProvisioner;
use rbee_hive_artifact_catalog::ArtifactProvisioner;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provisioner = ModelProvisioner::new()?;
    
    // Download model from HuggingFace
    let model = provisioner.provision(
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "job-123"
    ).await?;
    
    println!("Downloaded: {} ({} bytes)", model.id(), model.size());
    Ok(())
}
```

### With Explicit Filename

```rust
// Specify exact GGUF file
let model = provisioner.provision(
    "TheBloke/Llama-2-7B-Chat-GGUF:llama-2-7b-chat.Q4_K_M.gguf",
    "job-123"
).await?;
```

### Integration with rbee-hive

The provisioner is automatically initialized in `rbee-hive` and used by the `ModelDownload` operation:

```rust
// In job_router.rs
Operation::ModelDownload(request) => {
    // Check if model already exists
    if state.model_catalog.contains(&model) {
        return Err(anyhow::anyhow!("Model '{}' already exists", model));
    }
    
    // Provision model from HuggingFace
    let model_entry = state.model_provisioner.provision(&model, &job_id).await?;
    
    // Add to catalog
    state.model_catalog.add(model_entry)?;
}
```

## Dependencies

### Core Dependencies

- **hf-hub** (v0.3) - Official HuggingFace Rust client
  - Same library used by Candle
  - Supports tokio async runtime
  - Automatic caching
  - No Python dependencies

- **async-trait** - Async trait support
- **anyhow** - Error handling
- **tokio** - Async runtime

### Internal Dependencies

- **rbee-hive-artifact-catalog** - Shared provisioner abstractions
- **rbee-hive-model-catalog** - ModelEntry type
- **observability-narration-core** - Progress narration

## Storage

Models are stored in:
- **Linux/Mac:** `~/.cache/rbee/models/{model_id}/model.gguf`
- **Windows:** `%LOCALAPPDATA%\rbee\models\{model_id}\model.gguf`

Model IDs are sanitized for filesystem (e.g., `meta-llama/Llama-2-7b` → `meta-llama-Llama-2-7b`).

## Narration Events

The provisioner emits narration events for SSE streaming:

- `hf_download_start` - Download initiated
- `hf_find_gguf` - Searching for GGUF file
- `hf_download_file` - Downloading specific file
- `hf_download_cached` - File cached by hf-hub
- `hf_download_complete` - Download finished

## Testing

```bash
# Run unit tests
cargo test --package rbee-hive-model-provisioner

# Test with real download (requires network)
cargo test --package rbee-hive-model-provisioner -- --ignored
```

## Why hf-hub?

1. **Official Rust client** from HuggingFace team
2. **Same library as Candle** - proven compatibility
3. **No Python dependencies** - pure Rust
4. **Automatic caching** - reuses downloads
5. **Active maintenance** - regularly updated
6. **Tokio async support** - integrates with rbee architecture

## Alternatives Considered

❌ **huggingface-cli** - Deprecated, Python-based
❌ **Manual HTTP downloads** - Reinventing the wheel
❌ **Git LFS** - Overcomplicated for GGUF files
✅ **hf-hub** - Official, Rust-native, proven

## Future Enhancements

- [ ] Download progress tracking (percentage, speed)
- [ ] GGUF metadata parsing (quantization, parameters)
- [ ] Resume interrupted downloads
- [ ] Parallel downloads for multi-file models
- [ ] Custom HuggingFace token support (private repos)
- [ ] Model verification (checksums)

## Integration Status

✅ **Backend:**
- HuggingFaceVendor implemented
- ModelProvisioner implemented
- Wired into job_router.rs
- ModelDownload operation functional

✅ **Frontend:**
- useModels() hook ready
- ModelManagement component ready
- Download button placeholder (needs dialog)

⚠️ **Missing:**
- Download progress UI
- Model download dialog
- Error handling UI
- Cancel download functionality

## Example: Download Llama-2-7B

```bash
# Using rbee-keeper CLI (when implemented)
./rbee model download TheBloke/Llama-2-7B-Chat-GGUF

# Using curl (direct API)
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "model_download",
    "hive_id": "localhost",
    "model": "TheBloke/Llama-2-7B-Chat-GGUF"
  }'

# Then connect to SSE stream to watch progress
curl -N http://localhost:7835/v1/jobs/{job_id}/stream
```

## Troubleshooting

### Error: "Could not find GGUF file in repository"

**Solution:** Specify the exact filename:
```rust
provisioner.provision(
    "repo/model:exact-filename.gguf",
    "job-123"
).await?
```

### Error: "Model already exists"

**Solution:** Delete the existing model first:
```rust
model_catalog.remove("model-id")?;
```

### Slow downloads

**Cause:** HuggingFace CDN may be slow for large models.

**Solution:** 
- Use a mirror (if available)
- Download during off-peak hours
- Check network connection

## License

GPL-3.0-or-later

## Credits

- **hf-hub** - HuggingFace team
- **Candle** - HuggingFace ML framework for Rust
- **rbee** - Built with ❤️ for private LLM hosting
