# model-provisioner

**Resolves model references to local paths and manages model catalog**

`libs/provisioners/model-provisioner` — Resolves model references, verifies digests, and registers models in catalog.

---

## What This Library Does

model-provisioner provides **model lifecycle management** for llama-orch:

- **Model resolution** — Resolve `model_ref` to local file path
- **Digest verification** — Verify SHA256 checksums
- **Catalog integration** — Register/update models in catalog-core
- **Handoff files** — Emit metadata for engine-provisioner
- **Offline-first** — Prefer local files, no network I/O by default
- **Future: HuggingFace** — Optional `hf:` scheme support (feature-gated)

**Used by**: `engine-provisioner` to resolve model paths

---

## Usage

### Resolve Model

```rust
use model_provisioner::{ModelProvisioner, ModelRef};

let provisioner = ModelProvisioner::new(catalog);

// Resolve local file
let resolved = provisioner.ensure_present(
    ModelRef::Local("/models/llama-3.1-8b.gguf".into())
).await?;

println!("Model ID: {}", resolved.id);
println!("Local path: {}", resolved.local_path);
```

### With Digest Verification

```rust
use model_provisioner::{ModelProvisioner, ModelRef, Digest};

let digest = Digest {
    algo: "sha256".to_string(),
    value: "abc123...".to_string(),
};

let resolved = provisioner.ensure_present_with_digest(
    ModelRef::Local("/models/llama-3.1-8b.gguf".into()),
    Some(digest),
).await?;
```

### Handoff to Engine Provisioner

```rust
use model_provisioner::provision_from_config_to_default_handoff;

let resolved = provision_from_config_to_default_handoff(
    "/etc/llorch/model.yaml",
    std::env::temp_dir(),
)?;

println!("Handoff written to .runtime/engines/llamacpp.json");
```

---

## Model References

### Supported Schemes

#### Local Files

```rust
// Absolute path
ModelRef::Local("/models/llama-3.1-8b.gguf".into())

// Relative path (resolved from working directory)
ModelRef::Local("./models/llama-3.1-8b.gguf".into())
```

#### HuggingFace (Future)

```rust
// Feature-gated, requires huggingface-cli
ModelRef::HuggingFace {
    org: "meta-llama".to_string(),
    repo: "Llama-3.1-8B-Instruct".to_string(),
    path: Some("gguf/q4_k_m.gguf".to_string()),
}
```

---

## Handoff File Format

### Location

`.runtime/engines/llamacpp.json`

### Format

```json
{
  "model": {
    "id": "local:/models/llama-3.1-8b-instruct-q4_k_m.gguf",
    "path": "/models/llama-3.1-8b-instruct-q4_k_m.gguf"
  }
}
```

### Fields

- **id** — Normalized model identifier
- **path** — Absolute local file path

---

## Digest Verification

### Configuration

```yaml
model_ref: "file:/models/llama-3.1-8b.gguf"
expected_digest:
  algo: sha256
  value: "abc123def456..."
strict_verification: true
```

### Verification Flow

1. Resolve model path
2. Compute SHA256 digest
3. Compare with expected digest
4. Record verification result in catalog
5. Fail if mismatch and `strict_verification: true`

---

## Catalog Integration

### Model Registration

```rust
// Register model in catalog
provisioner.register_model(
    "local:/models/llama-3.1-8b.gguf",
    "/models/llama-3.1-8b.gguf",
    Some(digest),
).await?;
```

### Lifecycle States

- **Active** — Model is available and verified
- **Pending** — Model is being downloaded/verified
- **Failed** — Verification failed

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p model-provisioner -- --nocapture

# Run specific test
cargo test -p model-provisioner -- test_resolve_local --nocapture
```

---

## Dependencies

### Internal

- `catalog-core` — Model catalog and registry

### External

- `tokio` — Async runtime
- `serde` — Serialization
- `serde_json` — JSON handoff files
- `sha2` — SHA256 digest computation

---

## Configuration

### Input (YAML)

```yaml
model_ref: "file:/models/TinyLlama-1.1B-Chat-v1.0-q4_k_m.gguf"
expected_digest:
  algo: sha256
  value: "abc123..."
strict_verification: true
```

### Output (Handoff JSON)

```json
{
  "model": {
    "id": "local:/models/TinyLlama-1.1B-Chat-v1.0-q4_k_m.gguf",
    "path": "/models/TinyLlama-1.1B-Chat-v1.0-q4_k_m.gguf"
  }
}
```

---

## HuggingFace Support (Future)

### Installation

On Arch/CachyOS:

```bash
sudo pacman -S python-huggingface-hub
```

### Usage

```rust
// Feature-gated, requires huggingface-cli
let resolved = provisioner.ensure_present(
    ModelRef::HuggingFace {
        org: "meta-llama".to_string(),
        repo: "Llama-3.1-8B-Instruct".to_string(),
        path: Some("gguf/q4_k_m.gguf".to_string()),
    }
).await?;
```

If `huggingface-cli` is not installed, returns error:

```
Error: huggingface-cli not found. Install with:
  sudo pacman -S python-huggingface-hub
Or use a local file: path instead.
```

---

## Known Limitations

### MVP Scope

- ✅ **Local file resolution** — Absolute and relative paths
- ✅ **Digest verification** — SHA256 checksums
- ✅ **Catalog integration** — Register and update models
- ❌ **HuggingFace support** — Not implemented (feature-gated)
- ❌ **LRU cache** — No automatic eviction
- ❌ **GGUF parsing** — No header parsing for metadata

### Future Work

- Native HuggingFace fetcher (no shell-outs)
- LRU cache accounting and eviction
- GGUF header parsing for `ctx_max` and tokenizer info
- Provenance bundle linking verification outcomes

---

## Specifications

Implements requirements from `.specs/00_llama-orch.md`:
- Model resolution
- Digest verification
- Catalog integration
- Handoff file format

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
