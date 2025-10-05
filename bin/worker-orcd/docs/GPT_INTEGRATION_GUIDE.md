# GPT Integration Guide

**Purpose**: Guide for GPT-Gamma team to integrate GPT-2/GPT-3 models  
**Audience**: GPT-Gamma team  
**Owner**: Foundation-Alpha

---

## Overview

This guide helps the GPT-Gamma team integrate GPT-2 and GPT-3 models with the Foundation layer. GPT models differ from Llama-family models in several key ways:

**Key Differences**:
- **Normalization**: LayerNorm (not RMSNorm)
- **Activation**: GELU (not SiLU)
- **Attention**: MHA only (no GQA)
- **Positional Encoding**: Absolute embeddings (not RoPE)

---

## Architecture Overview

### GPT-2 Architecture

```
Input Tokens
    ↓
Token Embeddings + Positional Embeddings
    ↓
┌─────────────────────────────────┐
│ Transformer Block (×12-48)      │
│  ├─ LayerNorm                   │
│  ├─ Multi-Head Attention (MHA)  │
│  ├─ Residual Connection         │
│  ├─ LayerNorm                   │
│  ├─ FFN (Linear → GELU → Linear)│
│  └─ Residual Connection         │
└─────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
LM Head (project to vocabulary)
    ↓
Output Logits
```

### Model Sizes

| Model | Params | Layers | Hidden | Heads | FFN | Vocab |
|-------|--------|--------|--------|-------|-----|-------|
| GPT-2 Small | 117M | 12 | 768 | 12 | 3072 | 50257 |
| GPT-2 Medium | 345M | 24 | 1024 | 16 | 4096 | 50257 |
| GPT-2 Large | 774M | 36 | 1280 | 20 | 5120 | 50257 |
| GPT-2 XL | 1.5B | 48 | 1600 | 25 | 6400 | 50257 |

---

## Integration Steps

### Step 1: Implement CUDA Kernels

#### 1.1 LayerNorm Kernel

**File**: `cuda/kernels/layernorm.cu`

```cuda
// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
__global__ void layernorm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int hidden_dim,
    float eps
) {
    // TODO(GPT-Gamma): Implement LayerNorm
    // 1. Compute mean and variance
    // 2. Normalize
    // 3. Apply affine transformation
}
```

**FFI Binding**: `src/cuda_ffi/layernorm.rs`

```rust
pub fn layernorm(
    input: &SafeCudaPtr,
    gamma: &SafeCudaPtr,
    beta: &SafeCudaPtr,
    output: &mut SafeCudaPtr,
    batch_size: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<(), CudaError> {
    // Call CUDA kernel
}
```

#### 1.2 GELU Activation Kernel

**File**: `cuda/kernels/gelu.cu`

```cuda
// GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__global__ void gelu_kernel(
    const float* input,
    float* output,
    int size
) {
    // TODO(GPT-Gamma): Implement GELU
}
```

**FFI Binding**: `src/cuda_ffi/gelu.rs`

```rust
pub fn gelu(
    input: &SafeCudaPtr,
    output: &mut SafeCudaPtr,
    size: usize,
) -> Result<(), CudaError> {
    // Call CUDA kernel
}
```

#### 1.3 MHA (Multi-Head Attention) Kernel

**File**: `cuda/kernels/mha.cu`

```cuda
// MHA: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
__global__ void mha_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // TODO(GPT-Gamma): Implement MHA
    // 1. Compute attention scores
    // 2. Apply softmax
    // 3. Multiply by values
}
```

**FFI Binding**: `src/cuda_ffi/mha.rs`

```rust
pub fn mha(
    q: &SafeCudaPtr,
    k: &SafeCudaPtr,
    v: &SafeCudaPtr,
    output: &mut SafeCudaPtr,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<(), CudaError> {
    // Call CUDA kernel
}
```

---

### Step 2: Implement Weight Loading

**File**: `src/models/gpt.rs` (already created by Foundation)

```rust
impl GPTWeightLoader {
    pub fn load_to_vram(path: &str, config: &GPTConfig) -> Result<GPTModel, GPTError> {
        // 1. Parse GGUF file
        let gguf = parse_gguf_file(path)?;
        
        // 2. Validate metadata
        validate_gpt_metadata(&gguf, config)?;
        
        // 3. Allocate VRAM
        let ctx = CudaContext::new(0)?;
        let token_embeddings = allocate_and_load(&ctx, &gguf, "token_embd.weight")?;
        let position_embeddings = allocate_and_load(&ctx, &gguf, "position_embd.weight")?;
        
        // 4. Load layers
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            layers.push(load_gpt_layer(&ctx, &gguf, i)?);
        }
        
        // 5. Load final LayerNorm and LM head
        let final_ln = load_layernorm(&ctx, &gguf, "ln_f")?;
        let lm_head = allocate_and_load(&ctx, &gguf, "lm_head.weight")?;
        
        Ok(GPTModel {
            config: config.clone(),
            total_vram_bytes: calculate_total_vram(),
            token_embeddings,
            position_embeddings,
            layers,
            final_ln,
            lm_head,
        })
    }
}
```

---

### Step 3: Implement Forward Pass

**File**: `src/models/gpt.rs`

```rust
impl GPTForward {
    pub fn prefill(
        model: &GPTModel,
        input_ids: &[u32],
        config: &GPTForwardConfig,
    ) -> Result<Vec<u32>, GPTError> {
        let seq_len = input_ids.len();
        
        // 1. Embed tokens
        let mut hidden = embed_tokens(model, input_ids)?;
        
        // 2. Add positional embeddings
        add_positional_embeddings(&mut hidden, seq_len)?;
        
        // 3. Run through transformer layers
        for layer in &model.layers {
            hidden = gpt_layer_forward(layer, &hidden, config)?;
        }
        
        // 4. Apply final LayerNorm
        hidden = layernorm(&hidden, &model.final_ln)?;
        
        // 5. Project to vocabulary
        let logits = matmul(&hidden, &model.lm_head)?;
        
        // 6. Sample tokens
        let output_ids = sample_tokens(&logits, config)?;
        
        Ok(output_ids)
    }
}

fn gpt_layer_forward(
    layer: &GPTLayer,
    hidden: &SafeCudaPtr,
    config: &GPTForwardConfig,
) -> Result<SafeCudaPtr, GPTError> {
    // 1. LayerNorm
    let ln1_out = layernorm(hidden, &layer.ln1)?;
    
    // 2. Multi-Head Attention
    let attn_out = mha(&ln1_out, &layer.attn, config)?;
    
    // 3. Residual connection
    let hidden = add(hidden, &attn_out)?;
    
    // 4. LayerNorm
    let ln2_out = layernorm(&hidden, &layer.ln2)?;
    
    // 5. FFN
    let ffn_out = gpt_ffn(&ln2_out, &layer.ffn)?;
    
    // 6. Residual connection
    let hidden = add(&hidden, &ffn_out)?;
    
    Ok(hidden)
}

fn gpt_ffn(
    input: &SafeCudaPtr,
    ffn: &GPTFFN,
) -> Result<SafeCudaPtr, GPTError> {
    // 1. Up projection
    let up = matmul(input, &ffn.up_proj)?;
    
    // 2. GELU activation
    let activated = gelu(&up)?;
    
    // 3. Down projection
    let down = matmul(&activated, &ffn.down_proj)?;
    
    Ok(down)
}
```

---

### Step 4: Extend Adapter Pattern

**File**: `src/models/adapter.rs`

```rust
// 1. Add GPT2 and GPT3 to ModelType enum
pub enum ModelType {
    Qwen2_5,
    Phi3,
    Llama2,
    Llama3,
    GPT2,      // Add this
    GPT3,      // Add this
}

// 2. Add GPT model storage
pub struct LlamaModelAdapter {
    model_type: ModelType,
    qwen_model: Option<QwenModel>,
    phi3_model: Option<Phi3Model>,
    gpt_model: Option<GPTModel>,  // Add this
}

// 3. Add constructor
impl LlamaModelAdapter {
    pub fn new_gpt2(model: GPTModel) -> Self {
        Self {
            model_type: ModelType::GPT2,
            qwen_model: None,
            phi3_model: None,
            gpt_model: Some(model),
        }
    }
}

// 4. Implement query methods
pub fn vocab_size(&self) -> Result<usize, AdapterError> {
    match self.model_type {
        // ... existing cases
        ModelType::GPT2 | ModelType::GPT3 => {
            self.gpt_model
                .as_ref()
                .map(|m| m.config.vocab_size)
                .ok_or(AdapterError::ModelNotLoaded)
        }
        _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
    }
}

// 5. Implement forward pass methods
pub fn prefill(
    &self,
    input_ids: &[u32],
    config: &AdapterForwardConfig,
) -> Result<Vec<u32>, AdapterError> {
    match self.model_type {
        // ... existing cases
        ModelType::GPT2 | ModelType::GPT3 => {
            let model = self.gpt_model.as_ref().ok_or(AdapterError::ModelNotLoaded)?;
            GPTForward::prefill(model, input_ids, &config.to_gpt_config())
                .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
        }
        _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
    }
}
```

---

### Step 5: Add Integration Tests

**File**: `tests/gpt_integration.rs` (already created by Foundation)

Implement the ignored tests:
- `test_gpt_layernorm_kernel`
- `test_gpt_gelu_kernel`
- `test_gpt_mha_kernel`
- `test_gpt_positional_embeddings`
- `test_gpt2_full_pipeline`

---

## FFI Interface Patterns

### Memory Safety

```rust
// ✅ GOOD: Use SafeCudaPtr
pub fn process_tensor(input: &SafeCudaPtr, output: &mut SafeCudaPtr) -> Result<(), CudaError> {
    // Bounds checking automatic
    // Cleanup automatic (Drop)
}

// ❌ BAD: Raw pointers
pub fn process_tensor(input: *const f32, output: *mut f32) {
    // No bounds checking
    // Manual cleanup required
}
```

### Error Handling

```rust
// ✅ GOOD: Propagate errors
pub fn layernorm(...) -> Result<(), CudaError> {
    let result = unsafe { cuda_layernorm(...) };
    if result != 0 {
        return Err(CudaError::KernelLaunchFailed("LayerNorm failed".to_string()));
    }
    Ok(())
}

// ❌ BAD: Panic on error
pub fn layernorm(...) {
    unsafe { cuda_layernorm(...) };
    // What if it fails?
}
```

### Ownership

```rust
// ✅ GOOD: Clear ownership
pub struct GPTModel {
    token_embeddings: SafeCudaPtr,  // Owned
    layers: Vec<GPTLayer>,          // Owned
}

impl Drop for GPTModel {
    fn drop(&mut self) {
        // Automatic cleanup via SafeCudaPtr Drop
    }
}

// ❌ BAD: Unclear ownership
pub struct GPTModel {
    token_embeddings: *mut c_void,  // Who owns this?
}
```

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_layernorm_kernel() {
    let input = create_test_tensor(vec![1.0, 2.0, 3.0]);
    let gamma = create_test_tensor(vec![1.0; 3]);
    let beta = create_test_tensor(vec![0.0; 3]);
    let mut output = allocate_tensor(3);
    
    layernorm(&input, &gamma, &beta, &mut output, 1, 3, 1e-5).unwrap();
    
    let result = read_tensor(&output);
    assert_close(&result, &[-1.224, 0.0, 1.224], 1e-3);
}
```

### Integration Tests

```rust
#[test]
fn test_gpt2_full_pipeline() {
    let config = GPTConfig::gpt2_small();
    let model = GPTWeightLoader::load_to_vram("gpt2.gguf", &config).unwrap();
    let adapter = LlamaModelAdapter::new_gpt2(model);
    
    let tokenizer = create_gpt2_tokenizer();
    let prompt = "Hello, world!";
    let input_ids = tokenizer.encode(prompt).unwrap();
    
    let fwd_config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: input_ids.len(),
        cache_len: 0,
        temperature: 0.7,
        seed: 42,
    };
    
    let output_ids = adapter.generate(&input_ids, 20, &fwd_config).unwrap();
    let output_text = tokenizer.decode(&output_ids).unwrap();
    
    assert!(!output_text.is_empty());
    assert!(output_text.starts_with("Hello, world!"));
}
```

---

## Performance Expectations

### GPT-2 Small (117M)

- **VRAM**: ~500 MB
- **Prefill**: ~10ms for 512 tokens
- **Decode**: ~5ms per token
- **Throughput**: ~200 tokens/second

### GPT-2 XL (1.5B)

- **VRAM**: ~6 GB
- **Prefill**: ~50ms for 512 tokens
- **Decode**: ~20ms per token
- **Throughput**: ~50 tokens/second

---

## Common Issues

### Issue: LayerNorm numerical instability

**Symptom**: NaN or Inf in output

**Fix**:
```cuda
// Use higher precision for variance calculation
float var = 0.0f;
for (int i = 0; i < hidden_dim; i++) {
    float diff = input[i] - mean;
    var += diff * diff;
}
var = var / hidden_dim + eps;  // Add epsilon before sqrt
float std = sqrtf(var);
```

### Issue: GELU approximation accuracy

**Symptom**: Output differs from reference

**Fix**:
```cuda
// Use accurate GELU formula
float x3 = x * x * x;
float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
float tanh_inner = tanhf(inner);
float gelu = 0.5f * x * (1.0f + tanh_inner);
```

### Issue: MHA memory usage

**Symptom**: VRAM exhausted during attention

**Fix**:
```rust
// Use chunked attention for long sequences
fn chunked_mha(q, k, v, chunk_size) {
    for chunk in 0..seq_len/chunk_size {
        // Process chunk
    }
}
```

---

## Checklist

### Kernel Implementation
- [ ] LayerNorm kernel implemented
- [ ] GELU kernel implemented
- [ ] MHA kernel implemented
- [ ] Positional embedding implemented
- [ ] All kernels tested

### Weight Loading
- [ ] GGUF parser working
- [ ] Metadata extraction working
- [ ] VRAM allocation working
- [ ] Weight transfer working
- [ ] Validation working

### Forward Pass
- [ ] Prefill implemented
- [ ] Decode implemented
- [ ] Generate implemented
- [ ] KV cache working
- [ ] Sampling working

### Adapter Integration
- [ ] ModelType extended
- [ ] Constructor added
- [ ] Query methods implemented
- [ ] Forward pass methods implemented
- [ ] Config conversion added

### Testing
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Full pipeline test passing
- [ ] Performance validated
- [ ] Memory validated

---

## Support

**Contact**: Foundation-Alpha  
**Files**: 
- `src/models/gpt.rs` (skeleton provided)
- `tests/gpt_integration.rs` (template provided)
- `docs/ADAPTER_PATTERN_GUIDE.md` (reference)
- `docs/INTEGRATION_CHECKLIST.md` (checklist)

**Questions**: Update `execution/day-tracker.md` with issues

---

**Last Updated**: 2025-10-05  
**Maintainer**: Foundation-Alpha  
**Status**: Ready for GPT-Gamma team

---
Built by Foundation-Alpha 🏗️
