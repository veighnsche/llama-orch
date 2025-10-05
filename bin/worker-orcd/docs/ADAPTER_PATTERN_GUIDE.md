# Adapter Pattern Guide

**Purpose**: Guide for using and extending the `LlamaInferenceAdapter`  
**Audience**: Model implementation teams (Llama-Beta, GPT-Gamma)  
**Owner**: Foundation-Alpha

---

## Overview

The `LlamaInferenceAdapter` provides a **unified interface** for all Llama-family models (Qwen, Phi-3, Llama 2/3, GPT-2/3). It abstracts model-specific differences behind a consistent API, enabling:

1. **Model-agnostic code**: Write once, run on any model
2. **Easy model switching**: Change models without code changes
3. **Consistent testing**: Same test patterns for all models
4. **Clear boundaries**: Separation between model logic and inference logic

**Location**: `src/models/adapter.rs`

---

## Architecture

### Design Principles

1. **Enum dispatch**: Use `ModelType` enum to route to correct implementation
2. **Trait-free**: No trait overhead, direct function calls
3. **Zero-cost abstraction**: Compiles to direct calls (no vtable)
4. **Fail-fast**: Return errors immediately, don't hide issues

### Components

```
LlamaInferenceAdapter
├── ModelType (enum)
│   ├── Qwen2_5
│   ├── Phi3
│   ├── Llama2
│   ├── Llama3
│   └── (future: GPT2, GPT3)
│
├── Model Storage (Option<T>)
│   ├── qwen_model: Option<QwenModel>
│   ├── phi3_model: Option<Phi3Model>
│   └── (future: gpt_model, llama_model)
│
└── Unified API
    ├── vocab_size() -> usize
    ├── hidden_dim() -> usize
    ├── num_layers() -> usize
    ├── vram_usage() -> usize
    ├── prefill(input_ids, config) -> Vec<u32>
    ├── decode(input_id, config) -> u32
    └── generate(input_ids, max_tokens, config) -> Vec<u32>
```

---

## Using the Adapter

### Basic Usage

```rust
use worker_orcd::models::{
    LlamaInferenceAdapter, ModelType, AdapterForwardConfig,
    qwen::{QwenConfig, QwenWeightLoader},
};

// 1. Load model
let config = QwenConfig::qwen2_5_0_5b();
let model = QwenWeightLoader::load_to_vram("model.gguf", &config)?;

// 2. Create adapter
let adapter = LlamaInferenceAdapter::new_qwen(model);

// 3. Configure forward pass
let fwd_config = AdapterForwardConfig {
    is_prefill: true,
    batch_size: 1,
    seq_len: input_ids.len(),
    cache_len: 0,
    temperature: 0.7,
    seed: 42,
};

// 4. Generate tokens
let output_ids = adapter.generate(&input_ids, 50, &fwd_config)?;
```

### Model Information

```rust
// Query model properties
let vocab_size = adapter.vocab_size()?;
let hidden_dim = adapter.hidden_dim()?;
let num_layers = adapter.num_layers()?;
let vram_bytes = adapter.vram_usage()?;

println!("Model: {:?}", adapter.model_type());
println!("Vocab: {}, Hidden: {}, Layers: {}", 
         vocab_size, hidden_dim, num_layers);
println!("VRAM: {} MB", vram_bytes / (1024 * 1024));
```

### Prefill and Decode

```rust
// Prefill: Process full prompt
let prompt_ids = vec![1, 2, 3, 4, 5];
let prefill_config = AdapterForwardConfig {
    is_prefill: true,
    batch_size: 1,
    seq_len: prompt_ids.len(),
    cache_len: 0,
    temperature: 1.0,
    seed: 42,
};

let logits = adapter.prefill(&prompt_ids, &prefill_config)?;

// Decode: Generate one token at a time
let mut generated = Vec::new();
for i in 0..max_tokens {
    let decode_config = AdapterForwardConfig {
        is_prefill: false,
        batch_size: 1,
        seq_len: 1,
        cache_len: prompt_ids.len() + i,
        temperature: 0.7,
        seed: 42,
    };
    
    let next_token = adapter.decode(last_token, &decode_config)?;
    generated.push(next_token);
    last_token = next_token;
}
```

---

## Extending the Adapter

### Adding a New Model Type

Follow these steps to add support for a new model (e.g., GPT-2):

#### Step 1: Add Model Type Variant

```rust
// In src/models/adapter.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Qwen2_5,
    Phi3,
    Llama2,
    Llama3,
    GPT2,      // Add this
}
```

#### Step 2: Add Model Storage

```rust
// In src/models/adapter.rs
pub struct LlamaInferenceAdapter {
    model_type: ModelType,
    qwen_model: Option<QwenModel>,
    phi3_model: Option<Phi3Model>,
    gpt2_model: Option<GPT2Model>,  // Add this
}
```

#### Step 3: Add Constructor

```rust
// In src/models/adapter.rs
impl LlamaInferenceAdapter {
    /// Create adapter for GPT-2 model
    pub fn new_gpt2(model: GPT2Model) -> Self {
        Self {
            model_type: ModelType::GPT2,
            qwen_model: None,
            phi3_model: None,
            gpt2_model: Some(model),
        }
    }
}
```

#### Step 4: Implement Query Methods

```rust
// In src/models/adapter.rs
pub fn vocab_size(&self) -> Result<usize, AdapterError> {
    match self.model_type {
        ModelType::Qwen2_5 => { /* ... */ }
        ModelType::Phi3 => { /* ... */ }
        ModelType::GPT2 => {
            self.gpt2_model
                .as_ref()
                .map(|m| m.config.vocab_size)
                .ok_or(AdapterError::ModelNotLoaded)
        }
        _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
    }
}

// Repeat for: hidden_dim(), num_layers(), vram_usage()
```

#### Step 5: Implement Forward Pass Methods

```rust
// In src/models/adapter.rs
pub fn prefill(
    &self,
    input_ids: &[u32],
    config: &AdapterForwardConfig,
) -> Result<Vec<u32>, AdapterError> {
    match self.model_type {
        ModelType::Qwen2_5 => { /* ... */ }
        ModelType::Phi3 => { /* ... */ }
        ModelType::GPT2 => {
            let model = self.gpt2_model
                .as_ref()
                .ok_or(AdapterError::ModelNotLoaded)?;
            
            GPT2Forward::prefill(model, input_ids, &config.to_gpt2_config())
                .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
        }
        _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
    }
}

// Repeat for: decode(), generate()
```

#### Step 6: Add Config Conversion

```rust
// In src/models/adapter.rs
impl AdapterForwardConfig {
    /// Convert to GPT-2-specific config
    fn to_gpt2_config(&self) -> GPT2ForwardConfig {
        GPT2ForwardConfig {
            is_prefill: self.is_prefill,
            batch_size: self.batch_size,
            seq_len: self.seq_len,
            cache_len: self.cache_len,
            temperature: self.temperature,
            seed: self.seed,
        }
    }
}
```

#### Step 7: Add Tests

```rust
// In src/models/adapter.rs (tests module)
#[test]
fn test_adapter_gpt2() {
    let config = GPT2Config::gpt2_small();
    let model = GPT2WeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_gpt2(model);
    
    assert_eq!(adapter.model_type(), ModelType::GPT2);
    assert_eq!(adapter.vocab_size().unwrap(), 50257);
    assert_eq!(adapter.hidden_dim().unwrap(), 768);
    assert_eq!(adapter.num_layers().unwrap(), 12);
}
```

---

## Model Implementation Requirements

To integrate with the adapter, your model must provide:

### 1. Configuration Struct

```rust
#[derive(Debug, Clone)]
pub struct YourModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    // ... other config fields
}
```

### 2. Model Struct

```rust
#[derive(Debug)]
pub struct YourModel {
    pub config: YourModelConfig,
    pub total_vram_bytes: usize,
    // ... weight tensors
}
```

### 3. Weight Loader

```rust
pub struct YourWeightLoader;

impl YourWeightLoader {
    pub fn load_to_vram(
        path: &str,
        config: &YourModelConfig,
    ) -> Result<YourModel, YourError> {
        // Load GGUF file
        // Allocate VRAM
        // Copy weights to GPU
        // Return model
    }
    
    pub fn calculate_vram_usage(config: &YourModelConfig) -> usize {
        // Calculate total VRAM needed
    }
}
```

### 4. Forward Pass Implementation

```rust
#[derive(Debug, Clone)]
pub struct YourForwardConfig {
    pub is_prefill: bool,
    pub batch_size: usize,
    pub seq_len: usize,
    pub cache_len: usize,
    pub temperature: f32,
    pub seed: u32,
}

pub struct YourForward;

impl YourForward {
    pub fn prefill(
        model: &YourModel,
        input_ids: &[u32],
        config: &YourForwardConfig,
    ) -> Result<Vec<u32>, YourError> {
        // Implement prefill logic
    }
    
    pub fn decode(
        model: &YourModel,
        input_id: u32,
        config: &YourForwardConfig,
    ) -> Result<u32, YourError> {
        // Implement decode logic
    }
    
    pub fn generate(
        model: &YourModel,
        input_ids: &[u32],
        max_tokens: usize,
        config: &YourForwardConfig,
    ) -> Result<Vec<u32>, YourError> {
        // Implement generation logic
    }
}
```

---

## Configuration

### AdapterForwardConfig Fields

```rust
pub struct AdapterForwardConfig {
    /// True for prefill, false for decode
    pub is_prefill: bool,
    
    /// Batch size (usually 1 for now)
    pub batch_size: usize,
    
    /// Sequence length (prompt length for prefill, 1 for decode)
    pub seq_len: usize,
    
    /// KV cache length (0 for prefill, grows during decode)
    pub cache_len: usize,
    
    /// Sampling temperature (0.0 = greedy, 1.0 = normal, >1.0 = creative)
    pub temperature: f32,
    
    /// Random seed for reproducibility
    pub seed: u32,
}
```

### Temperature Guidelines

- **0.0**: Greedy decoding (always pick most likely token)
- **0.1-0.3**: Very focused, deterministic
- **0.5-0.7**: Balanced, good for most tasks
- **0.8-1.0**: Creative, diverse outputs
- **1.5-2.0**: Very creative, may be incoherent

---

## Error Handling

### AdapterError Types

```rust
pub enum AdapterError {
    /// Model not loaded (internal error)
    ModelNotLoaded,
    
    /// Invalid model type for operation
    InvalidModelType(ModelType),
    
    /// Forward pass failed
    ForwardPassFailed(String),
    
    /// Operation not supported for this model
    UnsupportedOperation(ModelType),
}
```

### Error Handling Pattern

```rust
match adapter.generate(&input_ids, 50, &config) {
    Ok(output) => {
        println!("Generated {} tokens", output.len());
    }
    Err(AdapterError::ModelNotLoaded) => {
        eprintln!("Model not loaded - internal error");
    }
    Err(AdapterError::ForwardPassFailed(msg)) => {
        eprintln!("Forward pass failed: {}", msg);
    }
    Err(e) => {
        eprintln!("Unexpected error: {:?}", e);
    }
}
```

---

## Testing Patterns

### Unit Tests

```rust
#[test]
fn test_adapter_creation() {
    let config = YourConfig::default();
    let model = YourWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_your_model(model);
    
    assert_eq!(adapter.model_type(), ModelType::YourModel);
}

#[test]
fn test_adapter_properties() {
    let adapter = create_test_adapter();
    
    assert!(adapter.vocab_size().is_ok());
    assert!(adapter.hidden_dim().is_ok());
    assert!(adapter.num_layers().is_ok());
    assert!(adapter.vram_usage().is_ok());
}

#[test]
fn test_adapter_generation() {
    let adapter = create_test_adapter();
    let input_ids = vec![1, 2, 3];
    let config = AdapterForwardConfig { /* ... */ };
    
    let output = adapter.generate(&input_ids, 10, &config).unwrap();
    assert_eq!(output.len(), input_ids.len() + 10);
}
```

### Integration Tests

```rust
// In tests/your_model_integration.rs
#[test]
fn test_full_pipeline() {
    // 1. Load model
    let config = YourConfig::default();
    let model = YourWeightLoader::load_to_vram("model.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_your_model(model);
    
    // 2. Create tokenizer
    let tokenizer = create_tokenizer();
    
    // 3. Encode prompt
    let prompt = "Hello, world!";
    let input_ids = tokenizer.encode(prompt).unwrap();
    
    // 4. Generate
    let config = AdapterForwardConfig { /* ... */ };
    let output_ids = adapter.generate(&input_ids, 20, &config).unwrap();
    
    // 5. Decode
    let output_text = tokenizer.decode(&output_ids).unwrap();
    
    // 6. Verify
    assert!(!output_text.is_empty());
    assert!(output_ids.len() > input_ids.len());
}
```

---

## Best Practices

### DO ✅

1. **Use the adapter for all inference**: Don't call model-specific code directly
2. **Handle all errors**: Never unwrap in production code
3. **Test with multiple models**: Ensure your code works with all model types
4. **Document model-specific behavior**: Note differences in comments
5. **Use consistent configuration**: Same config format for all models

### DON'T ❌

1. **Don't bypass the adapter**: Always use the unified interface
2. **Don't assume model type**: Use `model_type()` to check
3. **Don't ignore errors**: Handle all error cases
4. **Don't hardcode model parameters**: Query via adapter methods
5. **Don't mix model-specific code**: Keep it isolated in model modules

---

## Examples

### Example 1: Model-Agnostic Generation

```rust
fn generate_text(
    adapter: &LlamaInferenceAdapter,
    prompt: &str,
    max_tokens: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    // Works with ANY model type
    let tokenizer = create_tokenizer(adapter.model_type())?;
    let input_ids = tokenizer.encode(prompt)?;
    
    let config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: input_ids.len(),
        cache_len: 0,
        temperature: 0.7,
        seed: 42,
    };
    
    let output_ids = adapter.generate(&input_ids, max_tokens, &config)?;
    let output_text = tokenizer.decode(&output_ids)?;
    
    Ok(output_text)
}
```

### Example 2: Model Comparison

```rust
fn compare_models() {
    let qwen = load_qwen_adapter();
    let phi3 = load_phi3_adapter();
    
    let prompt_ids = vec![1, 2, 3];
    let config = AdapterForwardConfig { /* ... */ };
    
    let qwen_output = qwen.generate(&prompt_ids, 10, &config).unwrap();
    let phi3_output = phi3.generate(&prompt_ids, 10, &config).unwrap();
    
    println!("Qwen: {:?}", qwen_output);
    println!("Phi-3: {:?}", phi3_output);
}
```

### Example 3: Temperature Sweep

```rust
fn temperature_sweep(adapter: &LlamaInferenceAdapter) {
    let input_ids = vec![1, 2, 3];
    
    for temp in [0.1, 0.5, 0.7, 1.0, 1.5, 2.0] {
        let config = AdapterForwardConfig {
            temperature: temp,
            // ... other fields
        };
        
        let output = adapter.generate(&input_ids, 20, &config).unwrap();
        println!("Temp {}: {:?}", temp, output);
    }
}
```

---

## FAQ

### Q: Why not use traits?

**A**: Enum dispatch is faster (no vtable), simpler (no trait bounds), and more explicit (clear which models are supported).

### Q: Can I add my own model without modifying adapter.rs?

**A**: No, you must modify the adapter to add a new model type. This is intentional - we want a single source of truth for supported models.

### Q: What if my model has unique requirements?

**A**: Add model-specific fields to `AdapterForwardConfig` or create a model-specific config type and add a conversion method.

### Q: How do I handle model-specific optimizations?

**A**: Implement them in your model's forward pass. The adapter just routes to your code.

### Q: Can I use the adapter with multiple models simultaneously?

**A**: Yes, create multiple adapter instances. Each adapter owns one model.

---

## Troubleshooting

### Error: ModelNotLoaded

**Cause**: Internal error - model field is None  
**Fix**: This shouldn't happen. File a bug report.

### Error: UnsupportedOperation

**Cause**: Operation not implemented for this model type  
**Fix**: Implement the operation for your model type or use a different model.

### Error: ForwardPassFailed

**Cause**: Model-specific error during forward pass  
**Fix**: Check the error message for details. Debug your model's forward pass implementation.

---

## References

- **Source**: `src/models/adapter.rs`
- **Tests**: `tests/adapter_integration.rs`, `tests/llama_integration_suite.rs`
- **Examples**: `tests/qwen_integration.rs`, `tests/phi3_integration.rs`

---

**Last Updated**: 2025-10-05  
**Maintainer**: Foundation-Alpha  
**Status**: Complete

---
Built by Foundation-Alpha 🏗️
