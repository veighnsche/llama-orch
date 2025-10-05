# Adapter API Documentation

**Purpose**: Complete API reference for the adapter pattern  
**Owner**: Foundation-Alpha  
**Status**: FT-037

---

## Overview

The adapter pattern provides a unified interface for all model architectures (Llama, GPT, etc.). This document describes the complete API for creating, configuring, and using adapters.

---

## Quick Start

```rust
use worker_orcd::models::{AdapterFactory, AdapterForwardConfig};

// Create adapter (auto-detect architecture)
let adapter = AdapterFactory::from_gguf("model.gguf")?;

// Configure generation
let config = AdapterForwardConfig {
    is_prefill: true,
    batch_size: 1,
    seq_len: input_ids.len(),
    cache_len: 0,
    temperature: 0.7,
    seed: 42,
};

// Generate tokens
let output = adapter.generate(&input_ids, 50, &config)?;
```

---

## Core Types

### `LlamaModelAdapter`

Unified adapter for all model types.

```rust
pub struct LlamaModelAdapter { /* private fields */ }
```

**Thread Safety**: `Send` but not `Sync`. Use one adapter per thread.

**Methods**:

#### `model_type() -> ModelType`

Get the model type.

```rust
let model_type = adapter.model_type();
match model_type {
    ModelType::Qwen2_5 => println!("Qwen 2.5"),
    ModelType::Phi3 => println!("Phi-3"),
    ModelType::GPT2 => println!("GPT-2"),
    _ => println!("Other"),
}
```

#### `vocab_size() -> Result<usize, AdapterError>`

Get vocabulary size.

```rust
let vocab_size = adapter.vocab_size()?;
println!("Vocabulary: {} tokens", vocab_size);
```

#### `hidden_dim() -> Result<usize, AdapterError>`

Get hidden dimension.

```rust
let hidden_dim = adapter.hidden_dim()?;
println!("Hidden dimension: {}", hidden_dim);
```

#### `num_layers() -> Result<usize, AdapterError>`

Get number of transformer layers.

```rust
let num_layers = adapter.num_layers()?;
println!("Layers: {}", num_layers);
```

#### `vram_usage() -> Result<usize, AdapterError>`

Get total VRAM usage in bytes.

```rust
let vram_bytes = adapter.vram_usage()?;
println!("VRAM: {} MB", vram_bytes / (1024 * 1024));
```

#### `prefill(input_ids: &[u32], config: &AdapterForwardConfig) -> Result<Vec<u32>, AdapterError>`

Process full prompt (prefill phase).

```rust
let input_ids = vec![1, 2, 3, 4, 5];
let config = AdapterForwardConfig {
    is_prefill: true,
    batch_size: 1,
    seq_len: input_ids.len(),
    cache_len: 0,
    temperature: 1.0,
    seed: 42,
};

let logits = adapter.prefill(&input_ids, &config)?;
```

#### `decode(input_id: u32, config: &AdapterForwardConfig) -> Result<u32, AdapterError>`

Generate single token (decode phase).

```rust
let config = AdapterForwardConfig {
    is_prefill: false,
    batch_size: 1,
    seq_len: 1,
    cache_len: prompt_len,
    temperature: 0.7,
    seed: 42,
};

let next_token = adapter.decode(last_token, &config)?;
```

#### `generate(input_ids: &[u32], max_tokens: usize, config: &AdapterForwardConfig) -> Result<Vec<u32>, AdapterError>`

Generate tokens autoregressively.

```rust
let input_ids = vec![1, 2, 3];
let config = AdapterForwardConfig {
    is_prefill: true,
    batch_size: 1,
    seq_len: input_ids.len(),
    cache_len: 0,
    temperature: 0.7,
    seed: 42,
};

let output = adapter.generate(&input_ids, 50, &config)?;
// output.len() == input_ids.len() + 50
```

---

### `AdapterFactory`

Factory for creating adapters with automatic architecture detection.

```rust
pub struct AdapterFactory;
```

**Methods**:

#### `from_gguf(path: &str) -> Result<LlamaModelAdapter, FactoryError>`

Create adapter from GGUF file (auto-detect architecture).

```rust
let adapter = AdapterFactory::from_gguf("qwen-2.5-0.5b.gguf")?;
```

**Architecture Detection**:
- Parses GGUF metadata
- Detects architecture ("llama", "gpt", etc.)
- Detects model variant (Qwen, Phi-3, GPT-2, etc.)
- Creates appropriate adapter

#### `from_gguf_with_arch(path: &str, arch: Architecture) -> Result<LlamaModelAdapter, FactoryError>`

Create adapter with explicit architecture.

```rust
use worker_orcd::models::Architecture;

let adapter = AdapterFactory::from_gguf_with_arch(
    "model.gguf",
    Architecture::Llama
)?;
```

#### `from_gguf_with_arch_str(path: &str, arch_str: &str) -> Result<LlamaModelAdapter, FactoryError>`

Create adapter with architecture string.

```rust
let adapter = AdapterFactory::from_gguf_with_arch_str("model.gguf", "llama")?;
```

**Supported Architectures**:
- `"llama"` → Llama architecture (Qwen, Phi-3, Llama 2/3)
- `"gpt"` → GPT architecture (GPT-2, GPT-3)

#### `default_for_testing() -> Result<LlamaModelAdapter, FactoryError>`

Create default adapter for testing (Qwen 0.5B).

```rust
let adapter = AdapterFactory::default_for_testing()?;
```

---

### `AdapterForwardConfig`

Configuration for forward pass.

```rust
pub struct AdapterForwardConfig {
    pub is_prefill: bool,
    pub batch_size: usize,
    pub seq_len: usize,
    pub cache_len: usize,
    pub temperature: f32,
    pub seed: u32,
}
```

**Fields**:

- **`is_prefill`**: `true` for prefill phase, `false` for decode
- **`batch_size`**: Number of sequences (usually 1)
- **`seq_len`**: Sequence length (prompt length for prefill, 1 for decode)
- **`cache_len`**: KV cache length (0 for prefill, grows during decode)
- **`temperature`**: Sampling temperature (0.0 = greedy, 1.0 = normal, >1.0 = creative)
- **`seed`**: Random seed for reproducibility

**Example**:

```rust
// Prefill configuration
let prefill_config = AdapterForwardConfig {
    is_prefill: true,
    batch_size: 1,
    seq_len: prompt_ids.len(),
    cache_len: 0,
    temperature: 0.7,
    seed: 42,
};

// Decode configuration
let decode_config = AdapterForwardConfig {
    is_prefill: false,
    batch_size: 1,
    seq_len: 1,
    cache_len: prompt_ids.len() + generated_tokens,
    temperature: 0.7,
    seed: 42,
};
```

---

### `ModelType`

Model type enumeration.

```rust
pub enum ModelType {
    Qwen2_5,
    Phi3,
    Llama2,
    Llama3,
    GPT2,
    GPT3,
}
```

---

### `Architecture`

Architecture type.

```rust
pub enum Architecture {
    Llama,  // Llama-family (Qwen, Phi-3, Llama 2/3)
    GPT,    // GPT-family (GPT-2, GPT-3)
}
```

**Methods**:

#### `from_str(s: &str) -> Result<Self, FactoryError>`

Parse architecture from string.

```rust
let arch = Architecture::from_str("llama")?;
```

---

## Error Types

### `AdapterError`

Errors from adapter operations.

```rust
pub enum AdapterError {
    ModelNotLoaded,
    InvalidModelType(ModelType),
    ForwardPassFailed(String),
    UnsupportedOperation(ModelType),
}
```

**Handling**:

```rust
match adapter.generate(&input_ids, 50, &config) {
    Ok(output) => println!("Generated {} tokens", output.len()),
    Err(AdapterError::ModelNotLoaded) => eprintln!("Model not loaded"),
    Err(AdapterError::ForwardPassFailed(msg)) => eprintln!("Forward pass failed: {}", msg),
    Err(e) => eprintln!("Error: {:?}", e),
}
```

### `FactoryError`

Errors from factory operations.

```rust
pub enum FactoryError {
    UnknownArchitecture(String),
    ModelLoadingFailed(String),
    GGUFParsingFailed(String),
    UnsupportedVariant(String),
    ConfigurationError(String),
}
```

---

## Usage Patterns

### Pattern 1: Simple Generation

```rust
use worker_orcd::models::{AdapterFactory, AdapterForwardConfig};

let adapter = AdapterFactory::from_gguf("model.gguf")?;

let input_ids = vec![1, 2, 3];
let config = AdapterForwardConfig {
    is_prefill: true,
    batch_size: 1,
    seq_len: input_ids.len(),
    cache_len: 0,
    temperature: 0.7,
    seed: 42,
};

let output = adapter.generate(&input_ids, 50, &config)?;
```

### Pattern 2: Prefill + Decode Loop

```rust
// Prefill
let prefill_config = AdapterForwardConfig {
    is_prefill: true,
    batch_size: 1,
    seq_len: prompt_ids.len(),
    cache_len: 0,
    temperature: 0.7,
    seed: 42,
};

let mut output = adapter.prefill(&prompt_ids, &prefill_config)?;

// Decode loop
for i in 0..max_tokens {
    let decode_config = AdapterForwardConfig {
        is_prefill: false,
        batch_size: 1,
        seq_len: 1,
        cache_len: prompt_ids.len() + i,
        temperature: 0.7,
        seed: 42,
    };
    
    let next_token = adapter.decode(output.last().unwrap(), &decode_config)?;
    output.push(next_token);
    
    if next_token == eos_token {
        break;
    }
}
```

### Pattern 3: Model Comparison

```rust
let models = vec![
    AdapterFactory::from_gguf("qwen-2.5-0.5b.gguf")?,
    AdapterFactory::from_gguf("phi-3-mini.gguf")?,
    AdapterFactory::from_gguf("gpt2-small.gguf")?,
];

for adapter in &models {
    println!("Model: {:?}", adapter.model_type());
    println!("Vocab: {}", adapter.vocab_size()?);
    println!("VRAM: {} MB", adapter.vram_usage()? / (1024 * 1024));
    
    let output = adapter.generate(&input_ids, 50, &config)?;
    println!("Generated: {} tokens\n", output.len());
}
```

### Pattern 4: Temperature Sweep

```rust
let adapter = AdapterFactory::from_gguf("model.gguf")?;
let input_ids = vec![1, 2, 3];

for temp in [0.1, 0.5, 0.7, 1.0, 1.5, 2.0] {
    let config = AdapterForwardConfig {
        temperature: temp,
        // ... other fields
    };
    
    let output = adapter.generate(&input_ids, 20, &config)?;
    println!("Temperature {}: {:?}", temp, output);
}
```

---

## Best Practices

### DO ✅

1. **Use the factory**: Always create adapters via `AdapterFactory`
2. **Handle errors**: Never unwrap in production code
3. **Reuse adapters**: Create once, use many times
4. **Set seeds**: Use consistent seeds for reproducibility
5. **Check VRAM**: Query `vram_usage()` before loading

### DON'T ❌

1. **Don't bypass factory**: Don't create adapters directly
2. **Don't ignore errors**: Handle all error cases
3. **Don't share adapters**: One adapter per thread
4. **Don't hardcode types**: Use `model_type()` to check
5. **Don't skip validation**: Validate inputs before generation

---

## Performance Considerations

### VRAM Usage

```rust
let adapter = AdapterFactory::from_gguf("model.gguf")?;
let vram_mb = adapter.vram_usage()? / (1024 * 1024);

if vram_mb > available_vram_mb {
    return Err("Insufficient VRAM");
}
```

### Batch Size

Currently only `batch_size = 1` is supported. Future versions will support batching.

### Context Length

Longer contexts use more VRAM for KV cache:

```
KV Cache VRAM = 2 × num_layers × num_kv_heads × head_dim × context_len × 2 bytes
```

---

## Examples

See:
- `tests/adapter_factory_integration.rs` - Factory pattern examples
- `tests/llama_integration_suite.rs` - Llama adapter examples
- `tests/gpt_integration.rs` - GPT adapter examples
- `docs/ADAPTER_PATTERN_GUIDE.md` - Detailed guide

---

## Changelog

### v0.1.0 (2025-10-05)
- Initial adapter pattern implementation
- Support for Qwen, Phi-3, GPT-2
- Automatic architecture detection
- Factory pattern
- GGUF metadata parsing

---

**Last Updated**: 2025-10-05  
**API Version**: 0.1.0  
**Status**: Stable

---
Built by Foundation-Alpha 🏗️
