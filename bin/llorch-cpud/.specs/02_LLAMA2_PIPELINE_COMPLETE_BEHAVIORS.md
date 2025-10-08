# Llama-2 Inference Pipeline - Complete Behavioral Specification

**Created by:** TEAM-008  
**Date:** 2025-10-08  
**Model:** Llama-2 7B (7 billion parameters)  
**Format:** GGUF Q8_0  
**Configuration:** CPU/GPU, Q8_0 quantization, Temperature=0 (deterministic argmax)  
**Scope:** INFERENCE ONLY (training is out of scope)

This document specifies ALL behaviors in the Llama-2 inference pipeline in code flow order.

---

## Executive Summary

This specification replaces `01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md` as part of the **Foundation Reset** strategic pivot. We are moving from GPT-2 (educational, outdated) to Llama-2 (production-ready, modern architecture).

**Why Llama-2:**
- ✅ Modern architecture (RoPE, RMSNorm, SwiGLU)
- ✅ Works for 50+ models (Llama-3, Mistral, Qwen, CodeLlama)
- ✅ Production-ready (7B = sweet spot for validation)
- ✅ GGUF native (industry standard format)
- ✅ Commercial viability from day 1

**Key Architectural Differences from GPT-2:**

| Component | GPT-2 | Llama-2 |
|-----------|-------|---------|
| Position Encoding | Absolute (learned) | RoPE (rotary) |
| Normalization | LayerNorm | RMSNorm |
| Activation | GELU | SwiGLU |
| Attention | Standard MHA | Standard MHA (GQA-ready) |
| Bias Terms | Yes (everywhere) | No (no bias) |
| Vocab Size | 50257 | 32000 |
| Layers | 12 (base) / 24 (medium) | 32 |
| Hidden Size | 768 / 1024 | 4096 |

---

## RFC 2119 Keywords

- **MUST**: Absolute requirement
- **MUST NOT**: Absolute prohibition
- **SHOULD**: Recommended but not required
- **SHOULD NOT**: Not recommended but not prohibited
- **MAY**: Optional

---

## Model Configuration (Llama-2 7B)

```rust
pub struct Llama2Config {
    // Vocabulary and embeddings
    pub vocab_size: usize,              // 32000
    pub hidden_size: usize,              // 4096 (d_model)
    pub head_dim: usize,                 // 128 (hidden_size / num_heads)
    
    // Architecture
    pub num_layers: usize,               // 32
    pub num_heads: usize,                // 32 (for queries)
    pub num_kv_heads: usize,             // 32 (same as num_heads for Llama-2)
    pub intermediate_size: usize,        // 11008 (FFN hidden size)
    
    // Position encoding
    pub max_position_embeddings: usize,  // 4096 (context length)
    pub rope_theta: f32,                 // 10000.0 (RoPE base frequency)
    
    // Normalization
    pub rms_norm_eps: f32,               // 1e-5 (epsilon for RMSNorm)
    
    // Quantization (GGUF specific)
    pub quantization: QuantizationType,  // Q8_0 for our foundation model
}
```

**Critical Notes:**
- `num_kv_heads == num_heads` for Llama-2 (standard MHA)
- Llama-3 uses GQA where `num_kv_heads < num_heads`
- No bias terms anywhere (all Linear layers are bias=false)
- `head_dim = hidden_size / num_heads = 4096 / 32 = 128`

---

## Phase 1: Model Initialization

### 1.1 GGUF File Loading

**Requirement ID:** LLAMA2-INIT-001  
**Priority:** CRITICAL

The implementation MUST load Llama-2 weights from GGUF format.

**GGUF Structure:**
```
GGUF File:
├── Header (magic number, version, metadata count, tensor count)
├── Metadata (key-value pairs)
│   ├── general.architecture = "llama"
│   ├── general.name = "llama-2-7b"
│   ├── llama.context_length = 4096
│   ├── llama.embedding_length = 4096
│   ├── llama.block_count = 32
│   ├── llama.attention.head_count = 32
│   ├── llama.attention.head_count_kv = 32
│   ├── llama.rope.dimension_count = 128
│   ├── llama.rope.freq_base = 10000.0
│   ├── llama.attention.layer_norm_rms_epsilon = 1e-5
│   └── ... (more metadata)
├── Tensor Info (name, dimensions, type, offset)
└── Tensor Data (raw weight bytes)
```

**Implementation Requirements:**
1. MUST parse GGUF header (magic: "GGUF", version: 3)
2. MUST extract all metadata key-value pairs
3. MUST validate `general.architecture == "llama"`
4. MUST load all tensor metadata (names, shapes, types, offsets)
5. MUST support Q8_0 quantization type
6. SHOULD support F16 and F32 for future compatibility
7. MUST validate tensor shapes match expected config

**Validation:**
- Compare loaded metadata with expected Llama2Config
- Verify all required tensors are present
- Check tensor shapes match architecture

---

### 1.2 Weight Tensor Names (GGUF Convention)

**Requirement ID:** LLAMA2-INIT-002  
**Priority:** CRITICAL

The implementation MUST map GGUF tensor names to model components.

**Tensor Naming Convention:**
```
token_embd.weight                           # [vocab_size, hidden_size] = [32000, 4096]

blk.{layer}.attn_norm.weight                # [hidden_size] = [4096]
blk.{layer}.attn_q.weight                   # [hidden_size, hidden_size] = [4096, 4096]
blk.{layer}.attn_k.weight                   # [hidden_size, hidden_size] = [4096, 4096]
blk.{layer}.attn_v.weight                   # [hidden_size, hidden_size] = [4096, 4096]
blk.{layer}.attn_output.weight              # [hidden_size, hidden_size] = [4096, 4096]

blk.{layer}.ffn_norm.weight                 # [hidden_size] = [4096]
blk.{layer}.ffn_gate.weight                 # [hidden_size, intermediate_size] = [4096, 11008]
blk.{layer}.ffn_up.weight                   # [hidden_size, intermediate_size] = [4096, 11008]
blk.{layer}.ffn_down.weight                 # [intermediate_size, hidden_size] = [11008, 4096]

output_norm.weight                          # [hidden_size] = [4096]
output.weight                               # [hidden_size, vocab_size] = [4096, 32000]
```

**Key Differences from GPT-2:**
- No bias terms (GPT-2 has `.bias` for every `.weight`)
- Separate Q, K, V projections (GPT-2 often fuses them)
- Three FFN weights (gate, up, down) instead of two (c_fc, c_proj)
- RMSNorm weights (single weight vector, no bias)

---

### 1.3 RoPE Precomputation

**Requirement ID:** LLAMA2-INIT-003  
**Priority:** HIGH

The implementation SHOULD precompute RoPE (Rotary Position Embeddings) frequencies.

**RoPE Frequency Computation:**
```rust
fn precompute_rope_freqs(
    head_dim: usize,      // 128
    max_seq_len: usize,   // 4096
    theta: f32,           // 10000.0
) -> (Vec<f32>, Vec<f32>) {
    // Compute frequency for each dimension pair
    let freqs: Vec<f32> = (0..head_dim/2)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();
    
    // Precompute cos and sin for all positions
    let mut cos_cache = Vec::with_capacity(max_seq_len * head_dim / 2);
    let mut sin_cache = Vec::with_capacity(max_seq_len * head_dim / 2);
    
    for pos in 0..max_seq_len {
        for &freq in &freqs {
            let angle = pos as f32 * freq;
            cos_cache.push(angle.cos());
            sin_cache.push(angle.sin());
        }
    }
    
    (cos_cache, sin_cache)
}
```

**Behavior:**
- Precomputation is OPTIONAL but RECOMMENDED for performance
- Alternative: compute on-the-fly during attention
- Cache shape: `[max_seq_len, head_dim/2]`

---

## Phase 2: Input Processing

### 2.1 Tokenization

**Requirement ID:** LLAMA2-INPUT-001  
**Priority:** CRITICAL

The implementation MUST tokenize input text using Llama-2 tokenizer.

**Tokenizer:**
- Type: SentencePiece BPE
- Vocab size: 32000
- Special tokens:
  - BOS (Beginning of Sequence): token_id = 1
  - EOS (End of Sequence): token_id = 2
  - UNK (Unknown): token_id = 0

**Standard Test Input:**
```
Prompt: "Hello"
Tokens: [1, 15043]  # [BOS, "Hello"]
```

**Behavior:**
- MUST prepend BOS token (id=1) to all prompts
- MUST NOT append EOS during inference (only during generation when stop condition met)
- MUST handle unknown tokens with UNK (id=0)

---

### 2.2 Token Embedding

**Requirement ID:** LLAMA2-INPUT-002  
**Priority:** CRITICAL

The implementation MUST embed input tokens using the token embedding matrix.

**Operation:**
```rust
fn embed_tokens(
    token_ids: &[u32],           // [seq_len]
    embedding_weight: &Tensor,   // [vocab_size, hidden_size] = [32000, 4096]
) -> Tensor {
    // Shape: [seq_len, hidden_size]
    // Each token_id indexes into embedding_weight
    // Result: embedding_weight[token_ids[i]] for each i
}
```

**Example:**
```
Input: [1, 15043]  # [BOS, "Hello"]
Output shape: [2, 4096]
Output[0] = embedding_weight[1]      # BOS embedding
Output[1] = embedding_weight[15043]  # "Hello" embedding
```

**Critical Notes:**
- NO position embeddings added here (unlike GPT-2)
- Position information added via RoPE in attention
- NO scaling factor applied (unlike some models)

---

## Phase 3: Transformer Blocks (32 layers)

### 3.1 Block Structure

**Requirement ID:** LLAMA2-BLOCK-001  
**Priority:** CRITICAL

Each transformer block MUST follow this exact structure:

```rust
fn transformer_block(
    x: &Tensor,              // [batch, seq_len, hidden_size]
    layer_idx: usize,
    cache: &mut KVCache,
) -> Tensor {
    // Pre-norm architecture (normalize BEFORE sublayer)
    
    // 1. Attention sublayer with residual
    let normed = rms_norm(x, attn_norm_weight, eps);
    let attn_out = attention(normed, layer_idx, cache);
    let x = x + attn_out;  // Residual connection
    
    // 2. FFN sublayer with residual
    let normed = rms_norm(&x, ffn_norm_weight, eps);
    let ffn_out = swiglu_ffn(normed, gate_weight, up_weight, down_weight);
    let x = x + ffn_out;  // Residual connection
    
    x
}
```

**Key Points:**
- Pre-norm (RMSNorm before sublayer, not after)
- Residual connections around both attention and FFN
- No dropout (inference only)
- No bias terms anywhere

---

### 3.2 RMSNorm (Root Mean Square Normalization)

**Requirement ID:** LLAMA2-NORM-001  
**Priority:** CRITICAL

The implementation MUST use RMSNorm instead of LayerNorm.

**Algorithm:**
```rust
fn rms_norm(
    x: &Tensor,        // [batch, seq_len, hidden_size]
    weight: &Tensor,   // [hidden_size]
    eps: f32,          // 1e-5
) -> Tensor {
    // Compute RMS (Root Mean Square) over last dimension
    let rms = (x.pow(2).mean(dim=-1, keepdim=true) + eps).sqrt();
    
    // Normalize and scale
    let normalized = x / rms;
    let output = normalized * weight;  // Element-wise multiply
    
    output
}
```

**Differences from LayerNorm:**
- NO mean subtraction (only RMS normalization)
- NO bias term (only weight)
- Simpler and faster than LayerNorm
- Same output shape as input

**Validation Checkpoint:**
- **Checkpoint 1:** RMSNorm output for first token, first layer
- Tolerance: 1e-5
- Critical: Errors propagate to all downstream layers

---

### 3.3 Attention Mechanism

**Requirement ID:** LLAMA2-ATTN-001  
**Priority:** CRITICAL

The implementation MUST implement multi-head attention with RoPE.

**Full Attention Algorithm:**
```rust
fn attention(
    x: &Tensor,              // [batch, seq_len, hidden_size]
    layer_idx: usize,
    cache: &mut KVCache,
) -> Tensor {
    let (batch, seq_len, hidden_size) = x.shape();
    let num_heads = 32;
    let head_dim = 128;  // hidden_size / num_heads
    
    // 1. Project to Q, K, V (no bias)
    let q = x.matmul(&wq);  // [batch, seq_len, hidden_size]
    let k = x.matmul(&wk);  // [batch, seq_len, hidden_size]
    let v = x.matmul(&wv);  // [batch, seq_len, hidden_size]
    
    // 2. Reshape to multi-head format
    let q = q.reshape([batch, seq_len, num_heads, head_dim])
             .transpose(1, 2);  // [batch, num_heads, seq_len, head_dim]
    let k = k.reshape([batch, seq_len, num_heads, head_dim])
             .transpose(1, 2);  // [batch, num_heads, seq_len, head_dim]
    let v = v.reshape([batch, seq_len, num_heads, head_dim])
             .transpose(1, 2);  // [batch, num_heads, seq_len, head_dim]
    
    // 3. Apply RoPE to Q and K
    let (q, k) = apply_rope(q, k, cache.current_position);
    
    // 4. Update KV cache
    cache.update(layer_idx, k, v);
    let k_cached = cache.get_k(layer_idx);  // [batch, num_heads, total_len, head_dim]
    let v_cached = cache.get_v(layer_idx);  // [batch, num_heads, total_len, head_dim]
    
    // 5. Compute attention scores
    let scores = q.matmul(&k_cached.transpose(-2, -1));  // [batch, num_heads, seq_len, total_len]
    let scores = scores / (head_dim as f32).sqrt();      // Scale
    
    // 6. Apply causal mask (only for prompt processing, not for generation)
    if seq_len > 1 {
        let mask = create_causal_mask(seq_len, total_len);
        scores = scores + mask;  // mask has -inf for future positions
    }
    
    // 7. Softmax and apply to values
    let attn_weights = softmax(scores, dim=-1);
    let output = attn_weights.matmul(&v_cached);  // [batch, num_heads, seq_len, head_dim]
    
    // 8. Reshape and project output
    let output = output.transpose(1, 2)
                       .reshape([batch, seq_len, hidden_size]);
    let output = output.matmul(&wo);  // Output projection (no bias)
    
    output
}
```

**Validation Checkpoints:**
- **Checkpoint 2:** QKV projection output
- **Checkpoint 3:** After RoPE application
- **Checkpoint 4:** Attention scores
- **Checkpoint 5:** Attention output

---

### 3.4 RoPE (Rotary Position Embeddings)

**Requirement ID:** LLAMA2-ROPE-001  
**Priority:** CRITICAL

The implementation MUST apply RoPE to queries and keys.

**RoPE Algorithm:**
```rust
fn apply_rope(
    q: &Tensor,  // [batch, num_heads, seq_len, head_dim]
    k: &Tensor,  // [batch, num_heads, seq_len, head_dim]
    position_offset: usize,  // Starting position in sequence
) -> (Tensor, Tensor) {
    let (batch, num_heads, seq_len, head_dim) = q.shape();
    
    // For each position in the sequence
    for pos in 0..seq_len {
        let absolute_pos = position_offset + pos;
        
        // Apply rotation to each head
        for head in 0..num_heads {
            // Process pairs of dimensions
            for i in 0..(head_dim/2) {
                let freq = 1.0 / (10000.0_f32).powf((2 * i) as f32 / head_dim as f32);
                let angle = absolute_pos as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();
                
                // Rotate pair (i, i + head_dim/2)
                let q_i = q[batch][head][pos][i];
                let q_j = q[batch][head][pos][i + head_dim/2];
                q[batch][head][pos][i] = q_i * cos_val - q_j * sin_val;
                q[batch][head][pos][i + head_dim/2] = q_i * sin_val + q_j * cos_val;
                
                // Same for K
                let k_i = k[batch][head][pos][i];
                let k_j = k[batch][head][pos][i + head_dim/2];
                k[batch][head][pos][i] = k_i * cos_val - k_j * sin_val;
                k[batch][head][pos][i + head_dim/2] = k_i * sin_val + k_j * cos_val;
            }
        }
    }
    
    (q, k)
}
```

**Key Properties:**
- Applied to Q and K, NOT to V
- Position-dependent rotation
- Enables relative position encoding
- No learned parameters

**Validation:**
- **Checkpoint 3:** After RoPE, compare with llama.cpp output
- Tolerance: 1e-5

---

### 3.5 KV Cache

**Requirement ID:** LLAMA2-CACHE-001  
**Priority:** CRITICAL

The implementation MUST maintain KV cache for autoregressive generation.

**Cache Structure:**
```rust
struct KVCache {
    // For each layer: [batch, num_heads, max_seq_len, head_dim]
    k_cache: Vec<Tensor>,  // 32 layers
    v_cache: Vec<Tensor>,  // 32 layers
    current_position: usize,
    max_seq_len: usize,
}

impl KVCache {
    fn update(&mut self, layer_idx: usize, k: Tensor, v: Tensor) {
        // Append new K, V to cache at current_position
        let seq_len = k.shape()[2];
        self.k_cache[layer_idx][:, :, self.current_position:self.current_position+seq_len, :] = k;
        self.v_cache[layer_idx][:, :, self.current_position:self.current_position+seq_len, :] = v;
    }
    
    fn get_k(&self, layer_idx: usize) -> Tensor {
        // Return K cache up to current_position
        self.k_cache[layer_idx][:, :, :self.current_position, :]
    }
    
    fn get_v(&self, layer_idx: usize) -> Tensor {
        // Return V cache up to current_position
        self.v_cache[layer_idx][:, :, :self.current_position, :]
    }
}
```

**Behavior:**
- Prompt processing: Cache all tokens at once
- Token generation: Cache one token at a time
- Position tracking: `current_position` increments after each token

---

### 3.6 SwiGLU FFN (Feed-Forward Network)

**Requirement ID:** LLAMA2-FFN-001  
**Priority:** CRITICAL

The implementation MUST use SwiGLU activation in FFN.

**SwiGLU Algorithm:**
```rust
fn swiglu_ffn(
    x: &Tensor,          // [batch, seq_len, hidden_size] = [batch, seq_len, 4096]
    gate_weight: &Tensor,  // [hidden_size, intermediate_size] = [4096, 11008]
    up_weight: &Tensor,    // [hidden_size, intermediate_size] = [4096, 11008]
    down_weight: &Tensor,  // [intermediate_size, hidden_size] = [11008, 4096]
) -> Tensor {
    // 1. Gate projection with SiLU activation
    let gate = x.matmul(&gate_weight);  // [batch, seq_len, 11008]
    let gate = silu(gate);              // SiLU(x) = x * sigmoid(x)
    
    // 2. Up projection (no activation)
    let up = x.matmul(&up_weight);      // [batch, seq_len, 11008]
    
    // 3. Element-wise multiply (gating)
    let hidden = gate * up;             // [batch, seq_len, 11008]
    
    // 4. Down projection
    let output = hidden.matmul(&down_weight);  // [batch, seq_len, 4096]
    
    output
}

fn silu(x: &Tensor) -> Tensor {
    // SiLU (Swish) activation: x * sigmoid(x)
    x * sigmoid(x)
}
```

**Differences from GPT-2 FFN:**
- GPT-2: `GELU(x @ W1 + b1) @ W2 + b2`
- Llama-2: `(SiLU(x @ Wgate) * (x @ Wup)) @ Wdown`
- Three weight matrices instead of two
- No bias terms
- Gating mechanism (element-wise multiply)

**Validation:**
- **Checkpoint 6:** FFN output for first token, first layer
- Tolerance: 1e-4

---

## Phase 4: Output Processing

### 4.1 Final RMSNorm

**Requirement ID:** LLAMA2-OUTPUT-001  
**Priority:** CRITICAL

After all 32 transformer blocks, MUST apply final RMSNorm.

```rust
fn final_norm(x: &Tensor) -> Tensor {
    rms_norm(x, output_norm_weight, eps)
}
```

---

### 4.2 LM Head (Language Model Head)

**Requirement ID:** LLAMA2-OUTPUT-002  
**Priority:** CRITICAL

The implementation MUST project to vocabulary logits.

```rust
fn lm_head(
    x: &Tensor,           // [batch, seq_len, hidden_size] = [batch, seq_len, 4096]
    output_weight: &Tensor,  // [hidden_size, vocab_size] = [4096, 32000]
) -> Tensor {
    // Project to vocabulary
    let logits = x.matmul(&output_weight);  // [batch, seq_len, vocab_size]
    logits
}
```

**Note:** Unlike GPT-2, Llama-2 does NOT tie embeddings with output weights by default in GGUF models. They are separate tensors.

**Validation:**
- **Checkpoint 8:** Full logits after all 32 layers
- Tolerance: 1e-3 (accumulated error)

---

## Phase 5: Sampling

### 5.1 Greedy Sampling (Temperature = 0)

**Requirement ID:** LLAMA2-SAMPLE-001  
**Priority:** CRITICAL

For deterministic validation, MUST use greedy sampling.

```rust
fn greedy_sample(logits: &Tensor) -> u32 {
    // logits shape: [batch, seq_len, vocab_size]
    // Select last token: [batch, vocab_size]
    let last_logits = logits[:, -1, :];
    
    // Argmax
    let token_id = last_logits.argmax(dim=-1);
    token_id
}
```

**Validation:**
- **Checkpoint 10:** Argmax sampling determinism
- Tolerance: Exact match

---

### 5.2 Temperature Sampling

**Requirement ID:** LLAMA2-SAMPLE-002  
**Priority:** HIGH

For stochastic generation, MUST support temperature sampling.

```rust
fn temperature_sample(
    logits: &Tensor,
    temperature: f32,
    top_p: Option<f32>,
) -> u32 {
    let last_logits = logits[:, -1, :];
    
    // Apply temperature
    let scaled_logits = last_logits / temperature;
    
    // Softmax to probabilities
    let probs = softmax(scaled_logits, dim=-1);
    
    // Optional: Top-p (nucleus) sampling
    if let Some(p) = top_p {
        probs = apply_top_p(probs, p);
    }
    
    // Sample from distribution
    let token_id = sample_from_distribution(probs);
    token_id
}
```

**Validation:**
- **Checkpoint 11:** Softmax probabilities
- Tolerance: 1e-6

---

## Phase 6: Autoregressive Generation Loop

### 6.1 Generation Loop

**Requirement ID:** LLAMA2-GEN-001  
**Priority:** CRITICAL

The implementation MUST generate tokens autoregressively.

```rust
fn generate(
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    temperature: f32,
) -> Vec<u32> {
    let mut tokens = prompt_tokens.to_vec();
    let mut cache = KVCache::new(num_layers=32, max_seq_len=4096);
    
    // 1. Process prompt (all tokens at once)
    let prompt_tensor = embed_tokens(&tokens);
    let mut x = prompt_tensor;
    
    for layer_idx in 0..32 {
        x = transformer_block(&x, layer_idx, &mut cache);
    }
    
    x = final_norm(&x);
    let logits = lm_head(&x);
    let next_token = sample(logits, temperature);
    tokens.push(next_token);
    cache.current_position = tokens.len();
    
    // 2. Generate tokens one at a time
    for _ in 0..max_new_tokens-1 {
        if tokens.last() == Some(&EOS_TOKEN) {
            break;
        }
        
        // Process single token
        let token_tensor = embed_tokens(&[tokens.last().unwrap()]);
        let mut x = token_tensor;
        
        for layer_idx in 0..32 {
            x = transformer_block(&x, layer_idx, &mut cache);
        }
        
        x = final_norm(&x);
        let logits = lm_head(&x);
        let next_token = sample(logits, temperature);
        tokens.push(next_token);
        cache.current_position += 1;
    }
    
    tokens
}
```

**Key Points:**
- Prompt: Process all tokens in parallel
- Generation: Process one token at a time
- Cache: Reuse K, V from previous tokens
- Stop: When EOS token generated or max_new_tokens reached

---

## Validation Checkpoints (Updated for Llama-2)

### Standard Test Case

**Prompt:** "Hello"  
**Tokens:** `[1, 15043]` (BOS + "Hello")  
**Model:** Llama-2 7B Q8_0  
**Temperature:** 0 (greedy)  
**Max tokens:** 10

### Checkpoint Summary

| # | Checkpoint | Component | Tolerance | Critical |
|---|------------|-----------|-----------|----------|
| 1 | RMSNorm Output | First layer, first token | 1e-5 | 🔴 CRITICAL |
| 2 | QKV Projection | Attention input | 1e-4 | 🔴 CRITICAL |
| 3 | After RoPE | Q, K with position encoding | 1e-5 | 🔴 CRITICAL |
| 4 | Attention Scores | Scaled dot-product | 1e-4 | ⚠️ HIGH |
| 5 | Attention Output | After output projection | 1e-4 | ⚠️ HIGH |
| 6 | FFN Output | SwiGLU output | 1e-4 | ⚠️ HIGH |
| 7 | First Block Output | Complete block | 1e-4 | 🟢 VALIDATION |
| 8 | Full Logits | After 32 layers | 1e-3 | 🟢 VALIDATION |
| 9 | Selected Logits | Last token logits | Exact | 🔴 CRITICAL |
| 10 | Argmax Sampling | Greedy sampling | Exact | 🔴 CRITICAL |
| 11 | Softmax Probs | Temperature sampling | 1e-6 | ⚠️ MEDIUM |
| 12 | End-to-End | **FINAL VALIDATION** | Exact | 🟢 FINAL |

---

## Reference Implementations

### Primary Reference: llama.cpp

**Location:** `/home/vince/Projects/llama-orch/reference/llama.cpp`

**Key Files:**
- `llama.cpp` - Main inference implementation
- `ggml.c` - Tensor operations
- `ggml-quant.c` - Quantization support

**Checkpoint Extraction:**
Use Team 006's tool:
```bash
cd bin/llorch-cpud/tools/checkpoint-extractor
./build/llorch-checkpoint-extractor \
  /.test-models/llama2-7b/llama-2-7b.Q8_0.gguf \
  "Hello" \
  /tmp/llama2_reference_checkpoints
```

### Secondary Reference: Candle

**Location:** `/home/vince/Projects/llama-orch/reference/candle`

**Key Files:**
- `candle-transformers/src/models/llama.rs`
- `candle-core/src/quantized/gguf_file.rs`

### Tertiary Reference: Mistral.rs

**Location:** `/home/vince/Projects/llama-orch/reference/mistral.rs`

**Key Files:**
- `mistralrs-core/src/models/llama.rs`
- `mistralrs-core/src/attention/mod.rs`

---

## Implementation Roadmap

### Week 1: GGUF Loading
- [ ] Implement GGUF parser
- [ ] Load metadata and validate
- [ ] Load Q8_0 tensors
- [ ] Verify shapes

### Week 2: Core Components
- [ ] Implement RMSNorm
- [ ] Implement RoPE
- [ ] Implement SwiGLU FFN
- [ ] Unit tests for each

### Week 3: Full Inference
- [ ] Implement attention with KV cache
- [ ] Wire up transformer blocks
- [ ] Implement generation loop
- [ ] Checkpoints 1-8 validation

### Week 4: Production Ready
- [ ] Checkpoints 9-12 validation
- [ ] HTTP server integration
- [ ] Performance optimization
- [ ] Documentation

---

## Success Criteria

### Minimum (Week 3)
- ✅ Checkpoint 12 passes (end-to-end generation matches llama.cpp)

### Recommended (Week 4)
- ✅ All checkpoints 1-12 pass
- ✅ All tolerances met
- ✅ HTTP server integrated

### Production (Future)
- ✅ Multiple test cases pass
- ✅ Temperature sampling works
- ✅ Performance benchmarks met
- ✅ Quantization support (Q4, Q5, Q6, Q8)

---

## Appendix A: GGUF Q8_0 Format

**Q8_0 Quantization:**
- 8-bit integer quantization
- Block size: 32 values per block
- Each block: 32 int8 values + 1 float32 scale
- Dequantization: `value = int8_val * scale`
- Near-FP16 quality with 50% size reduction

**Block Structure:**
```rust
struct BlockQ8_0 {
    scale: f32,        // 4 bytes
    values: [i8; 32],  // 32 bytes
}
// Total: 36 bytes per 32 values = 1.125 bytes per value
```

---

## Appendix B: Differences from GPT-2

| Aspect | GPT-2 | Llama-2 |
|--------|-------|---------|
| **Position Encoding** | Learned absolute embeddings | RoPE (rotary) |
| **Normalization** | LayerNorm (mean + variance) | RMSNorm (RMS only) |
| **Activation** | GELU | SwiGLU |
| **FFN Structure** | 2 projections | 3 projections (gate, up, down) |
| **Bias Terms** | Everywhere | Nowhere |
| **Attention** | Standard MHA | Standard MHA (GQA-ready) |
| **Vocab Size** | 50257 | 32000 |
| **Tokenizer** | BPE (tiktoken) | SentencePiece BPE |
| **Weight Format** | PyTorch .pt | GGUF |
| **Quantization** | Not standard | Native GGUF support |

---

## Sign-off

**Created by:** TEAM-008 (Foundation Implementation)  
**Date:** 2025-10-08  
**Status:** Ready for implementation

This specification provides the complete behavioral foundation for Llama-2 7B inference in llorch-cpud.

**Next Steps:**
1. Update checkpoint specs (CHECKPOINT_01_*.md through CHECKPOINT_12_*.md)
2. Implement GGUF parser
3. Implement core components
4. Extract reference checkpoints
5. Begin validation

---

*"Build on the right foundation, and everything else follows."*  
— TEAM-008, Foundation Implementation Division

**END SPECIFICATION**
