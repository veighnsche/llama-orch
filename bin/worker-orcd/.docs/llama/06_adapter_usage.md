# LlamaInferenceAdapter Usage

**Component**: Adapter Pattern  
**Story**: LT-033  
**Spec**: M0-W-1214, FT-071

---

## Overview

The `LlamaInferenceAdapter` provides a unified interface for all Llama-family models. It abstracts model-specific differences and enables polymorphic usage across Qwen, Phi-3, and future Llama variants.

---

## Architecture

### Adapter Pattern

```
┌─────────────────────────────────────┐
│ LlamaInferenceAdapter               │
│  - Unified interface                │
│  - Model polymorphism               │
│  - Type-safe dispatch               │
└─────────────────────────────────────┘
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ QwenModel   │  │ Phi3Model   │  │ Llama3Model │
│  - 24 layers│  │  - 32 layers│  │  (future)   │
│  - GQA      │  │  - MHA      │  │             │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## Creating Adapters

### From Qwen Model

```rust
use worker_orcd::models::{
    LlamaInferenceAdapter,
    qwen::{QwenConfig, QwenWeightLoader},
};

let config = QwenConfig::qwen2_5_0_5b();
let model = QwenWeightLoader::load_to_vram("qwen2.5-0.5b.gguf", &config)?;
let adapter = LlamaInferenceAdapter::new_qwen(model);

assert_eq!(adapter.model_type(), ModelType::Qwen2_5);
```

### From Phi-3 Model

```rust
use worker_orcd::models::{
    LlamaInferenceAdapter,
    phi3::{Phi3Config, Phi3WeightLoader},
};

let config = Phi3Config::phi3_mini_4k();
let model = Phi3WeightLoader::load_to_vram("phi-3-mini-4k.gguf", &config)?;
let adapter = LlamaInferenceAdapter::new_phi3(model);

assert_eq!(adapter.model_type(), ModelType::Phi3);
```

---

## Unified Interface

### Configuration

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

### Methods

```rust
impl LlamaInferenceAdapter {
    // Model information
    pub fn model_type(&self) -> ModelType;
    pub fn vocab_size(&self) -> Result<usize, AdapterError>;
    pub fn hidden_dim(&self) -> Result<usize, AdapterError>;
    pub fn num_layers(&self) -> Result<usize, AdapterError>;
    pub fn vram_usage(&self) -> Result<usize, AdapterError>;
    
    // Inference
    pub fn prefill(&self, input_ids: &[u32], config: &AdapterForwardConfig) 
        -> Result<Vec<u32>, AdapterError>;
    
    pub fn decode(&self, input_id: u32, config: &AdapterForwardConfig) 
        -> Result<u32, AdapterError>;
    
    pub fn generate(&self, input_ids: &[u32], max_tokens: usize, config: &AdapterForwardConfig) 
        -> Result<Vec<u32>, AdapterError>;
}
```

---

## Usage Examples

### Basic Generation

```rust
use worker_orcd::models::{LlamaInferenceAdapter, AdapterForwardConfig};

// Works with any model
fn generate_text(adapter: &LlamaInferenceAdapter, prompt_ids: &[u32]) -> Vec<u32> {
    let config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: prompt_ids.len(),
        cache_len: 0,
        temperature: 0.7,
        seed: 42,
    };
    
    adapter.generate(prompt_ids, 30, &config).unwrap()
}

// Use with Qwen
let qwen_adapter = LlamaInferenceAdapter::new_qwen(qwen_model);
let output = generate_text(&qwen_adapter, &input_ids);

// Use with Phi-3
let phi3_adapter = LlamaInferenceAdapter::new_phi3(phi3_model);
let output = generate_text(&phi3_adapter, &input_ids);
```

### Model Information

```rust
fn print_model_info(adapter: &LlamaInferenceAdapter) {
    println!("Model type: {:?}", adapter.model_type());
    println!("Vocab size: {}", adapter.vocab_size().unwrap());
    println!("Hidden dim: {}", adapter.hidden_dim().unwrap());
    println!("Layers: {}", adapter.num_layers().unwrap());
    println!("VRAM: {} MB", adapter.vram_usage().unwrap() / (1024 * 1024));
}

print_model_info(&qwen_adapter);
// Output:
// Model type: Qwen2_5
// Vocab size: 151936
// Hidden dim: 896
// Layers: 24
// VRAM: 1300 MB

print_model_info(&phi3_adapter);
// Output:
// Model type: Phi3
// Vocab size: 32064
// Hidden dim: 3072
// Layers: 32
// VRAM: 7500 MB
```

### Prefill + Decode Loop

```rust
fn generate_with_streaming(
    adapter: &LlamaInferenceAdapter,
    input_ids: &[u32],
    max_tokens: usize,
) -> Vec<u32> {
    // Prefill
    let mut config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: input_ids.len(),
        cache_len: 0,
        temperature: 0.7,
        seed: 42,
    };
    
    let prefill_output = adapter.prefill(input_ids, &config).unwrap();
    let mut output_ids = prefill_output.clone();
    
    // Decode loop
    config.is_prefill = false;
    config.seq_len = 1;
    
    for i in 0..max_tokens {
        config.cache_len = input_ids.len() + i;
        
        let last_token = *output_ids.last().unwrap();
        let next_token = adapter.decode(last_token, &config).unwrap();
        
        output_ids.push(next_token);
        
        // Check for EOS
        if next_token == 1 {  // EOS token
            break;
        }
    }
    
    output_ids
}
```

---

## Polymorphic Usage

### Generic Function

```rust
fn benchmark_model<T: Into<LlamaInferenceAdapter>>(
    model: T,
    test_prompts: &[Vec<u32>],
) -> f64 {
    let adapter: LlamaInferenceAdapter = model.into();
    let start = std::time::Instant::now();
    
    for prompt in test_prompts {
        let config = AdapterForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: prompt.len(),
            cache_len: 0,
            temperature: 1.0,
            seed: 42,
        };
        
        adapter.generate(prompt, 10, &config).unwrap();
    }
    
    start.elapsed().as_secs_f64()
}

// Use with any model
let qwen_time = benchmark_model(qwen_adapter, &test_prompts);
let phi3_time = benchmark_model(phi3_adapter, &test_prompts);

println!("Qwen: {:.2}s, Phi-3: {:.2}s", qwen_time, phi3_time);
```

### Model Registry

```rust
use std::collections::HashMap;

pub struct ModelRegistry {
    models: HashMap<String, LlamaInferenceAdapter>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }
    
    pub fn register(&mut self, name: String, adapter: LlamaInferenceAdapter) {
        self.models.insert(name, adapter);
    }
    
    pub fn get(&self, name: &str) -> Option<&LlamaInferenceAdapter> {
        self.models.get(name)
    }
    
    pub fn list_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

// Usage
let mut registry = ModelRegistry::new();
registry.register("qwen-0.5b".to_string(), qwen_adapter);
registry.register("phi3-4k".to_string(), phi3_adapter);

for model_name in registry.list_models() {
    let adapter = registry.get(&model_name).unwrap();
    println!("{}: {} layers", model_name, adapter.num_layers().unwrap());
}
```

---

## Model Switching

### Load Multiple Models

```rust
// Load both models
let qwen_config = QwenConfig::qwen2_5_0_5b();
let qwen_model = QwenWeightLoader::load_to_vram("qwen2.5-0.5b.gguf", &qwen_config)?;
let qwen_adapter = LlamaInferenceAdapter::new_qwen(qwen_model);

let phi3_config = Phi3Config::phi3_mini_4k();
let phi3_model = Phi3WeightLoader::load_to_vram("phi-3-mini-4k.gguf", &phi3_config)?;
let phi3_adapter = LlamaInferenceAdapter::new_phi3(phi3_model);

// Switch between models
let current_adapter = if use_small_model {
    &qwen_adapter
} else {
    &phi3_adapter
};

let output = current_adapter.generate(&input_ids, 20, &config)?;
```

---

## Error Handling

### Adapter Errors

```rust
#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("Model not loaded")]
    ModelNotLoaded,
    
    #[error("Invalid model type: {0:?}")]
    InvalidModelType(ModelType),
    
    #[error("Forward pass failed: {0}")]
    ForwardPassFailed(String),
    
    #[error("Unsupported operation for model type: {0:?}")]
    UnsupportedOperation(ModelType),
}
```

### Error Handling Example

```rust
match adapter.generate(&input_ids, 30, &config) {
    Ok(output_ids) => {
        println!("Generated {} tokens", output_ids.len());
    }
    Err(AdapterError::ModelNotLoaded) => {
        eprintln!("Error: Model not loaded");
    }
    Err(AdapterError::ForwardPassFailed(msg)) => {
        eprintln!("Forward pass failed: {}", msg);
    }
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
    }
}
```

---

## Performance Comparison

### Benchmark

```rust
fn compare_models() {
    let qwen_adapter = create_qwen_adapter();
    let phi3_adapter = create_phi3_adapter();
    
    let test_prompt = vec![1, 2, 3, 4, 5];
    let config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: test_prompt.len(),
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };
    
    // Benchmark Qwen
    let start = std::time::Instant::now();
    qwen_adapter.generate(&test_prompt, 100, &config).unwrap();
    let qwen_time = start.elapsed();
    
    // Benchmark Phi-3
    let start = std::time::Instant::now();
    phi3_adapter.generate(&test_prompt, 100, &config).unwrap();
    let phi3_time = start.elapsed();
    
    println!("Qwen: {:.2}s ({:.1} tokens/sec)", 
             qwen_time.as_secs_f64(), 
             100.0 / qwen_time.as_secs_f64());
    println!("Phi-3: {:.2}s ({:.1} tokens/sec)", 
             phi3_time.as_secs_f64(), 
             100.0 / phi3_time.as_secs_f64());
}
```

**Expected Output**:
```
Qwen: 10.0s (10.0 tokens/sec)
Phi-3: 15.0s (6.7 tokens/sec)
```

---

## Testing

### Unit Tests

```rust
#[test]
fn test_adapter_qwen() {
    let config = QwenConfig::qwen2_5_0_5b();
    let model = QwenWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_qwen(model);
    
    assert_eq!(adapter.model_type(), ModelType::Qwen2_5);
    assert_eq!(adapter.vocab_size().unwrap(), 151936);
}

#[test]
fn test_adapter_phi3() {
    let config = Phi3Config::phi3_mini_4k();
    let model = Phi3WeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_phi3(model);
    
    assert_eq!(adapter.model_type(), ModelType::Phi3);
    assert_eq!(adapter.vocab_size().unwrap(), 32064);
}

#[test]
fn test_adapter_polymorphism() {
    let qwen_adapter = create_qwen_adapter();
    let phi3_adapter = create_phi3_adapter();
    
    let input_ids = vec![1, 2, 3];
    let config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: 3,
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };
    
    // Both should work with same code
    let qwen_output = qwen_adapter.generate(&input_ids, 5, &config).unwrap();
    let phi3_output = phi3_adapter.generate(&input_ids, 5, &config).unwrap();
    
    assert_eq!(qwen_output.len(), 8);
    assert_eq!(phi3_output.len(), 8);
}
```

---

## Best Practices

### 1. Use Adapter for All Model Interactions

```rust
// Good: Use adapter
let adapter = LlamaInferenceAdapter::new_qwen(model);
let output = adapter.generate(&input_ids, 30, &config)?;

// Avoid: Direct model access
let output = QwenForward::generate(&model, &input_ids, 30, &config)?;
```

### 2. Handle Errors Gracefully

```rust
match adapter.generate(&input_ids, 30, &config) {
    Ok(output) => process_output(output),
    Err(e) => {
        log::error!("Generation failed: {}", e);
        fallback_response()
    }
}
```

### 3. Reuse Configuration

```rust
let config = AdapterForwardConfig {
    is_prefill: true,
    batch_size: 1,
    seq_len: input_ids.len(),
    cache_len: 0,
    temperature: 0.7,
    seed: 42,
};

// Reuse for multiple generations
for prompt in prompts {
    let output = adapter.generate(prompt, 30, &config)?;
    process(output);
}
```

---

## Future Extensions

### Planned Features

1. **Llama 3 Support**: Add Llama3Model variant
2. **Batching**: Support batch_size > 1
3. **Streaming**: Real-time token streaming
4. **Quantization**: INT8/INT4 weight support
5. **Model Unloading**: Free VRAM when done

---

## References

- **Story**: LT-033 (Adapter Pattern)
- **Spec**: `bin/.specs/01_M0_worker_orcd.md` Section 6.12
- **Test Coverage**: 4+ integration tests

---

**Status**: Implemented  
**Pattern**: Adapter  
**Models Supported**: Qwen2.5, Phi-3
