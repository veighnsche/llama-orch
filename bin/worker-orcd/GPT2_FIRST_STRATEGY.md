# GPT-2 First Strategy - Simpler Model for Transformer Debugging

**Date**: 2025-10-07  
**Status**: RECOMMENDED  
**Priority**: HIGH  
**Rationale**: Debug with simpler architecture, then expand to Qwen2

---

## Executive Summary

**Problem**: Qwen2.5 transformer is complex (GQA, RoPE, RMSNorm, SwiGLU) - hard to debug

**Solution**: Start with **GPT-2 FP32** - simpler architecture, easier to verify

**Benefits**:
- ✅ Simpler attention (MHA, not GQA)
- ✅ Simpler position encoding (learned, not RoPE)
- ✅ Simpler normalization (LayerNorm, not RMSNorm)
- ✅ Simpler FFN (GELU, not SwiGLU)
- ✅ FP32 (no quantization complexity)
- ✅ Well-documented (original Transformer paper architecture)
- ✅ Multiple references available

**Strategy**: Fix GPT-2 first → Verify correctness → Expand to Qwen2

---

## Why GPT-2 is Easier to Debug

### Architecture Complexity Comparison

| Component | GPT-2 | Qwen2.5 | Complexity Reduction |
|-----------|-------|---------|---------------------|
| **Attention** | MHA (Multi-Head) | GQA (Grouped Query) | ⬇️ 50% simpler |
| **Position Encoding** | Learned embeddings | RoPE (rotary) | ⬇️ 70% simpler |
| **Normalization** | LayerNorm | RMSNorm | ⬇️ 30% simpler |
| **FFN Activation** | GELU | SwiGLU | ⬇️ 40% simpler |
| **Precision** | FP32 | Q4_K_M/MXFP4 | ⬇️ 80% simpler |
| **KV Cache** | Simple | GQA-aware | ⬇️ 50% simpler |

**Overall**: GPT-2 is **~60% simpler** to debug than Qwen2.5

---

## GPT-2 Architecture Details

### Model Structure

```
GPT-2 Transformer Block:
├── LayerNorm (pre-attention)
├── Multi-Head Attention (MHA)
│   ├── Q/K/V projections (same num_heads for all)
│   ├── Scaled dot-product attention
│   └── Output projection
├── Residual connection
├── LayerNorm (pre-FFN)
├── Feed-Forward Network
│   ├── Linear (hidden → 4*hidden)
│   ├── GELU activation
│   └── Linear (4*hidden → hidden)
└── Residual connection
```

### Key Simplifications

**1. Multi-Head Attention (MHA) vs Grouped Query Attention (GQA)**

GPT-2 (MHA):
```
num_heads = 12
num_kv_heads = 12  # Same as num_heads
Q: [batch, seq, 12 * head_dim]
K: [batch, seq, 12 * head_dim]
V: [batch, seq, 12 * head_dim]
```

Qwen2.5 (GQA):
```
num_heads = 14
num_kv_heads = 2  # Different! Need to repeat K/V
Q: [batch, seq, 14 * head_dim]
K: [batch, seq, 2 * head_dim]  # Need to repeat 7x
V: [batch, seq, 2 * head_dim]  # Need to repeat 7x
```

**2. Learned Position Embeddings vs RoPE**

GPT-2:
```cpp
// Simple lookup
pos_emb = position_embedding_table[position];
hidden = token_emb + pos_emb;
```

Qwen2.5 (RoPE):
```cpp
// Complex rotation
freq = 1.0 / pow(10000, 2*i / head_dim);
cos_val = cos(position * freq);
sin_val = sin(position * freq);
// Rotate Q and K by (cos, sin) pairs
q_rotated = rotate_half(q, cos_val, sin_val);
k_rotated = rotate_half(k, cos_val, sin_val);
```

**3. LayerNorm vs RMSNorm**

GPT-2 (LayerNorm):
```cpp
mean = sum(x) / n;
var = sum((x - mean)^2) / n;
out = gamma * (x - mean) / sqrt(var + eps) + beta;
```

Qwen2.5 (RMSNorm):
```cpp
rms = sqrt(sum(x^2) / n);
out = gamma * x / (rms + eps);
// No beta, no mean subtraction
```

**4. GELU vs SwiGLU**

GPT-2 (GELU):
```cpp
ffn_out = W2 * GELU(W1 * x);
```

Qwen2.5 (SwiGLU):
```cpp
gate = W_gate * x;
up = W_up * x;
ffn_out = W_down * (SiLU(gate) * up);
// Two projections instead of one!
```

---

## Available References for GPT-2

### C++ References (Best for Direct Comparison)

**1. llama.cpp** (Has GPT-2 support)
- File: `/reference/llama.cpp/examples/gpt-2/`
- ⚠️ BUT: You said "NOT LLAMA.CPP"
- Alternative: Use only for high-level architecture understanding

**2. ggml** (Underlying library for llama.cpp)
- File: `/reference/llama.cpp/ggml/src/ggml.c`
- Pure C implementation
- Good for understanding GEMM operations

### Rust References (Clean, Readable)

**1. candle**
- File: `/reference/candle/candle-transformers/src/models/gpt2.rs`
- ✅ Clean Rust implementation
- ✅ Well-documented
- ✅ Easy to understand

**2. mistral.rs**
- May not have GPT-2 (focused on Mistral/Qwen)
- Check: `/reference/mistral.rs/mistralrs-core/src/models/`

### Python References (For Logic Verification)

**1. vllm**
- File: `/reference/vllm/vllm/model_executor/models/gpt2.py`
- ✅ Production-grade
- ✅ Well-tested

**2. text-generation-inference**
- File: `/reference/text-generation-inference/server/text_generation_server/models/`
- ✅ HuggingFace official

**3. tinygrad**
- File: `/reference/tinygrad/`
- ✅ Minimal implementation
- ✅ Educational

---

## Implementation Strategy

### Phase 1: Add GPT-2 Support (4-6 hours)

**Goal**: Implement GPT-2 transformer alongside Qwen2

**Tasks**:

1. **Create GPT-2 model adapter** (1 hour)
   ```cpp
   // File: cuda/src/models/gpt2_model.h
   class GPT2Model {
       // Simpler than Qwen2Model
       LayerNorm ln_1;  // Pre-attention norm
       Attention attn;  // MHA (not GQA)
       LayerNorm ln_2;  // Pre-FFN norm
       FFN ffn;         // GELU (not SwiGLU)
   };
   ```

2. **Implement GPT-2 attention** (1 hour)
   ```cpp
   // File: cuda/src/models/gpt2_attention.cu
   // MHA: num_heads == num_kv_heads (simpler!)
   // No RoPE (use learned position embeddings)
   // Standard scaled dot-product attention
   ```

3. **Implement GPT-2 FFN** (30 min)
   ```cpp
   // File: cuda/src/models/gpt2_ffn.cu
   // Simple: W2 * GELU(W1 * x)
   // No gate projection, no SwiGLU
   ```

4. **Implement LayerNorm** (30 min)
   ```cpp
   // File: cuda/src/kernels/layernorm.cu
   // Standard LayerNorm with mean and variance
   ```

5. **Implement GELU** (15 min)
   ```cpp
   // File: cuda/src/kernels/gelu.cu
   // GELU(x) = x * Φ(x) where Φ is CDF of normal distribution
   // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
   ```

6. **Add GPT-2 weight loader** (1 hour)
   ```cpp
   // File: cuda/src/loaders/gpt2_weight_loader.cpp
   // Load from GGUF with "gpt2" architecture
   // Simpler weight mapping than Qwen2
   ```

7. **Test with GPT-2 model** (1 hour)
   ```bash
   # Download GPT-2 small (124M params, ~500MB FP32)
   # Test inference
   # Verify output matches reference
   ```

### Phase 2: Debug GPT-2 (2-4 hours)

**Goal**: Get GPT-2 producing correct output

**Method**: Compare with candle GPT-2 implementation

**Steps**:

1. **Compare attention implementation**
   - Reference: `/reference/candle/candle-transformers/src/models/gpt2.rs`
   - Check Q/K/V projection dimensions
   - Check attention score calculation
   - Check output projection

2. **Compare FFN implementation**
   - Check linear layer dimensions
   - Check GELU implementation
   - Check output shape

3. **Compare LayerNorm**
   - Check epsilon value
   - Check gamma/beta application
   - Check numerical stability

4. **Add logging and compare values**
   ```cpp
   // Log first 8 values at each stage
   // Compare with candle output
   // Find divergence point
   ```

5. **Fix bugs one at a time**
   - Fix bug
   - Test
   - Verify output improves
   - Repeat

### Phase 3: Expand to Qwen2 (2-3 hours)

**Goal**: Apply learnings from GPT-2 to fix Qwen2

**Method**: Use same debugging approach

**Steps**:

1. **Identify differences**
   - MHA → GQA (need to repeat K/V)
   - Learned pos → RoPE (need rotation)
   - LayerNorm → RMSNorm (simpler)
   - GELU → SwiGLU (more complex)

2. **Implement GQA correctly**
   - Use GPT-2 MHA as baseline
   - Add K/V repetition logic
   - Verify with mistral.rs reference

3. **Implement RoPE correctly**
   - Use candle Qwen3 as reference
   - Verify frequency calculation
   - Verify rotation logic

4. **Implement SwiGLU correctly**
   - Use GELU as baseline
   - Add gate projection
   - Verify element-wise multiplication

5. **Test Qwen2 with fixes**
   - Run haiku test
   - Verify output quality

---

## Comparison Methodology for GPT-2

### Step 1: Read candle GPT-2 Implementation

**File**: `/reference/candle/candle-transformers/src/models/gpt2.rs`

**Focus areas**:
```rust
// 1. Attention mechanism
impl Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Q/K/V projections
        // Attention scores
        // Softmax
        // Output projection
    }
}

// 2. MLP (FFN)
impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Linear 1
        // GELU
        // Linear 2
    }
}

// 3. Block (Layer)
impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // LayerNorm 1
        // Attention
        // Residual 1
        // LayerNorm 2
        // MLP
        // Residual 2
    }
}
```

### Step 2: Map to Our Implementation

**Our code structure**:
```cpp
// File: cuda/src/models/gpt2_model.cpp

class GPT2Block {
    void forward(float* x, int seq_len) {
        // 1. Pre-attention LayerNorm
        layernorm(x, ln_1_gamma, ln_1_beta);
        
        // 2. Attention
        attention(x, seq_len);
        
        // 3. Residual 1
        add_residual(x, residual_1);
        
        // 4. Pre-FFN LayerNorm
        layernorm(x, ln_2_gamma, ln_2_beta);
        
        // 5. FFN
        ffn(x);
        
        // 6. Residual 2
        add_residual(x, residual_2);
    }
};
```

### Step 3: Compare Tensor Shapes

**Template**:
```
Stage                  | candle shape          | Our shape          | Match?
-----------------------|-----------------------|--------------------|-------
Input                  | [B, L, H]             | [B, L, H]          | ?
After LayerNorm 1      | [B, L, H]             | [B, L, H]          | ?
After Q projection     | [B, L, num_heads*D]   | [B, L, num_heads*D]| ?
After reshape          | [B, num_heads, L, D]  | [B, num_heads, L, D]| ?
After attention        | [B, num_heads, L, D]  | [B, num_heads, L, D]| ?
After concat           | [B, L, num_heads*D]   | [B, L, num_heads*D]| ?
After O projection     | [B, L, H]             | [B, L, H]          | ?
After residual 1       | [B, L, H]             | [B, L, H]          | ?
After LayerNorm 2      | [B, L, H]             | [B, L, H]          | ?
After FFN linear 1     | [B, L, 4*H]           | [B, L, 4*H]        | ?
After GELU             | [B, L, 4*H]           | [B, L, 4*H]        | ?
After FFN linear 2     | [B, L, H]             | [B, L, H]          | ?
After residual 2       | [B, L, H]             | [B, L, H]          | ?
```

### Step 4: Compare cuBLAS Parameters

**Q Projection Example**:
```
Operation: Q = X @ W_q^T

candle (implicit):
- Input X: [batch*seq, hidden]
- Weight W_q: [num_heads*head_dim, hidden]
- Output Q: [batch*seq, num_heads*head_dim]
- Operation: matmul(X, W_q.t())

Our cuBLAS:
- M = batch * seq_len
- N = num_heads * head_dim
- K = hidden_size
- opA = CUBLAS_OP_N (X is not transposed)
- opB = CUBLAS_OP_T (W_q is transposed)
- lda = K (leading dim of X in row-major)
- ldb = N (leading dim of W_q in row-major)
- ldc = N (leading dim of Q in row-major)

Check: Do these match?
```

### Step 5: Add Logging and Compare

**Our code**:
```cpp
void GPT2Block::forward(float* x, int seq_len) {
    // Log input
    if (layer_idx == 0 && token_idx == 0) {
        float first8[8];
        cudaMemcpy(first8, x, 8*sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[INPUT] Layer 0, Token 0: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
                first8[0], first8[1], first8[2], first8[3], 
                first8[4], first8[5], first8[6], first8[7]);
    }
    
    // LayerNorm 1
    layernorm(x, ln_1_gamma, ln_1_beta);
    
    // Log after LayerNorm 1
    if (layer_idx == 0 && token_idx == 0) {
        float first8[8];
        cudaMemcpy(first8, x, 8*sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[AFTER_LN1] Layer 0, Token 0: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
                first8[0], first8[1], first8[2], first8[3], 
                first8[4], first8[5], first8[6], first8[7]);
    }
    
    // ... continue for each stage
}
```

**candle logging** (add to reference code):
```rust
// In candle GPT-2 implementation
let x = self.ln_1.forward(&x)?;
println!("[AFTER_LN1] First 8: {:?}", &x.to_vec1::<f32>()?[..8]);
```

**Compare outputs**:
```
Our code:     [0.123456, 0.234567, 0.345678, ...]
candle:       [0.123450, 0.234560, 0.345670, ...]
Difference:   [0.000006, 0.000007, 0.000008, ...]  ✅ Within FP32 tolerance
```

---

## GPT-2 Model Recommendations

### For Initial Testing

**GPT-2 Small**:
- Parameters: 124M
- Size: ~500MB (FP32)
- Layers: 12
- Hidden: 768
- Heads: 12
- Context: 1024

**Why**: Small enough to debug quickly, large enough to be meaningful

### Download

```bash
# Using HuggingFace
huggingface-cli download gpt2 --local-dir ./models/gpt2-small

# Or use GGUF format
wget https://huggingface.co/ggerganov/gpt-2/resolve/main/ggml-model-f32.gguf
```

---

## Benefits of GPT-2 First Approach

### 1. Simpler Debugging
- Fewer moving parts
- Easier to isolate bugs
- Faster iteration

### 2. Better Understanding
- Learn transformer basics first
- Understand each component
- Build confidence

### 3. Incremental Complexity
- GPT-2 (simple) → Qwen2 (complex)
- Add one feature at a time
- Verify each addition

### 4. Reusable Components
- LayerNorm works for both
- Attention core logic similar
- FFN patterns transferable

### 5. Multiple References
- More GPT-2 implementations available
- Better documented
- Easier to find help

---

## Migration Path: GPT-2 → Qwen2

### Step 1: Get GPT-2 Working
- ✅ Correct output
- ✅ Matches reference
- ✅ All tests pass

### Step 2: Add GQA Support
- Modify attention to support num_kv_heads < num_heads
- Add K/V repetition logic
- Test with Qwen2

### Step 3: Add RoPE Support
- Implement RoPE frequency calculation
- Implement rotation logic
- Replace learned position embeddings
- Test with Qwen2

### Step 4: Add RMSNorm Support
- Implement RMSNorm kernel
- Replace LayerNorm
- Test with Qwen2

### Step 5: Add SwiGLU Support
- Add gate projection
- Implement SwiGLU activation
- Replace GELU FFN
- Test with Qwen2

### Step 6: Full Qwen2 Support
- ✅ All components working
- ✅ Matches reference
- ✅ All tests pass

---

## Timeline Estimate

| Phase | Task | Duration |
|-------|------|----------|
| **Phase 1** | Implement GPT-2 | 4-6 hours |
| **Phase 2** | Debug GPT-2 | 2-4 hours |
| **Phase 3** | Expand to Qwen2 | 2-3 hours |
| **Total** | | **8-13 hours** |

**Comparison with direct Qwen2 fix**: 8-16 hours

**Benefit**: More confidence, better understanding, reusable code

---

## Decision Matrix

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Fix Qwen2 directly** | Faster if lucky | Complex, many variables | If confident |
| **GPT-2 first** | Simpler, incremental | Slightly longer | **Recommended** |

**Recommendation**: **GPT-2 first** if you want:
- Better understanding
- More confidence
- Reusable components
- Incremental verification

**Direct Qwen2 fix** if you want:
- Fastest path (if lucky)
- Single target
- No additional code

---

## Success Criteria

### GPT-2 Phase
- ✅ GPT-2 loads successfully
- ✅ GPT-2 produces coherent text
- ✅ Output matches candle reference (within tolerance)
- ✅ No crashes or errors

### Qwen2 Phase
- ✅ Qwen2 loads successfully
- ✅ Haiku test passes
- ✅ Minute word found in output
- ✅ No mojibake or repetitive tokens
- ✅ Coherent English text

---

## Files to Create/Modify

### New Files
```
cuda/src/models/gpt2_model.h
cuda/src/models/gpt2_model.cpp
cuda/src/models/gpt2_attention.cu
cuda/src/models/gpt2_ffn.cu
cuda/src/kernels/layernorm.cu
cuda/src/kernels/gelu.cu
cuda/src/loaders/gpt2_weight_loader.cpp
tests/gpt2_inference_test.rs
```

### Modified Files
```
cuda/src/model_adapter.h          # Add GPT2ModelAdapter
cuda/src/model_adapter.cpp        # Implement adapter
src/cuda_backend.rs                # Add GPT-2 model type
tests/integration_test.rs          # Add GPT-2 tests
```

---

## Next Steps

1. **Read this document** - Understand the strategy
2. **Read candle GPT-2** - Study reference implementation
3. **Implement GPT-2** - Start with simplest components
4. **Debug GPT-2** - Compare with reference
5. **Expand to Qwen2** - Add complexity incrementally

---

## References

**candle GPT-2**:
- `/reference/candle/candle-transformers/src/models/gpt2.rs`

**vllm GPT-2**:
- `/reference/vllm/vllm/model_executor/models/gpt2.py`

**Original GPT-2 Paper**:
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)

**Transformer Paper**:
- "Attention Is All You Need" (Vaswani et al., 2017)

---

**Remember**: Simple first, complex later. GPT-2 is your training ground.
