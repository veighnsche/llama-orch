# Quick Start: Transformer Bugfix by Reference Comparison

**TL;DR**: Use **mistral.rs** and **candle** as references. NOT llama.cpp (already tried, caused bugs).

---

## The Answer: mistral.rs + candle

### Why These Two?

**mistral.rs** = Production Rust implementation
- File: `/reference/mistral.rs/mistralrs-core/src/models/qwen2.rs`
- ✅ Rust (easy to read)
- ✅ Production-grade
- ✅ Full Qwen2 support
- ✅ Clean code structure

**candle** = HuggingFace Rust ML framework
- File: `/reference/candle/candle-transformers/src/models/qwen3.rs`
- ✅ Rust (easy to read)
- ✅ Educational quality
- ✅ Qwen3 (similar to Qwen2.5)
- ✅ Minimal, clear abstractions

### Why NOT llama.cpp?

**You said**: "NOT LLAMA.CPP, we started with that and everything went messed up"

**Policy**: `/reference/README.md`, `/NO_LLAMA_CPP.md`

---

## What to Compare

### Top 3 Bug Suspects (Priority Order)

**1. Attention Output Projection (`W_o`)**
- **Location**: After softmax·V, before residual add
- **Check**: Head concatenation order, transpose flags, lda/ldb/ldc
- **Reference**: 
  - mistral.rs line 104-111 (`o_proj`)
  - candle line 147-150 (`o_proj`)

**2. FFN Down Projection (`ffn_down`)**
- **Location**: After SwiGLU, before residual add
- **Check**: Weight loading, cuBLAS params, shapes
- **Reference**:
  - mistral.rs line 79 (`down_proj`)
  - candle line 79 (`down_proj`)

**3. RoPE Application**
- **Location**: After Q/K projection, before attention
- **Check**: Frequency calculation, sin/cos order, dimension split
- **Reference**:
  - mistral.rs line 179 (`rotary_emb.forward()`)
  - candle line 55-64 (`Qwen3RotaryEmbedding::apply()`)

---

## How to Compare

### Step 1: Read Reference Code (30 min each)

**mistral.rs Qwen2**:
```bash
# Open in editor
code /home/vince/Projects/llama-orch/reference/mistral.rs/mistralrs-core/src/models/qwen2.rs
```

Focus on:
- Lines 48-133: `Attention` struct
- Lines 135-250: `Attention::forward()`
- Lines 252-350: `Mlp` implementation

**candle Qwen3**:
```bash
# Open in editor
code /home/vince/Projects/llama-orch/reference/candle/candle-transformers/src/models/qwen3.rs
```

Focus on:
- Lines 30-64: `Qwen3RotaryEmbedding`
- Lines 66-91: `Qwen3MLP`
- Lines 93-200: `Qwen3Attention`

### Step 2: Compare Tensor Shapes

**Example**: Attention output projection

mistral.rs (line 104-111):
```rust
let o_proj = RowParallelLayer::new(
    num_heads * head_dim,  // Input: concatenated heads
    hidden_sz,             // Output: hidden size
    ...
);
```

Our code (qwen_transformer.cpp):
```cpp
// Check: Do we concatenate heads correctly?
// Check: Are input/output dimensions correct?
// Check: Is lda/ldb/ldc correct for row-major?
```

### Step 3: Compare cuBLAS Parameters

**Template**:
```
Operation: Q projection
mistral.rs: MatMul.qmethod_matmul(&xs, &*self.q_proj)
Our code: cublasSgemm(handle, opA, opB, M, N, K, ...)

Check:
- M = ? (should be seq_len)
- N = ? (should be num_heads * head_dim)
- K = ? (should be hidden_size)
- opA = ? (CUBLAS_OP_N or CUBLAS_OP_T?)
- opB = ? (CUBLAS_OP_N or CUBLAS_OP_T?)
- lda = ? (leading dimension of A)
- ldb = ? (leading dimension of B)
- ldc = ? (leading dimension of C)
```

### Step 4: Add Minimal Logging

**Our code** (qwen_transformer.cpp):
```cpp
// After attention output projection
if (layer_idx == 0 && token_idx == 0) {
    float first8[8];
    cudaMemcpy(first8, attn_out, 8*sizeof(float), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[ATTN_OUT] Layer 0, Token 0: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
            first8[0], first8[1], first8[2], first8[3], 
            first8[4], first8[5], first8[6], first8[7]);
}
```

**Compare with mistral.rs output** (add similar logging if needed)

---

## Quick Comparison Checklist

### Attention Mechanism

- [ ] Q/K/V projection dimensions match
- [ ] Q/K/V projection transpose flags match
- [ ] Head reshape: `[B, L, H*D] → [B, H, L, D]` correct?
- [ ] RoPE frequency calculation matches
- [ ] RoPE sin/cos application order matches
- [ ] Attention scores: `Q @ K^T / sqrt(head_dim)` correct?
- [ ] Softmax applied correctly
- [ ] V multiplication: `softmax(scores) @ V` correct?
- [ ] Head concatenation: `[B, H, L, D] → [B, L, H*D]` correct?
- [ ] Output projection dimensions match
- [ ] Output projection transpose flags match

### FFN Mechanism

- [ ] Gate projection dimensions match
- [ ] Up projection dimensions match
- [ ] SwiGLU: `gate_proj(x).silu() * up_proj(x)` correct?
- [ ] Down projection dimensions match
- [ ] Down projection weights loaded correctly

### Residual Connections

- [ ] Pre-norm vs post-norm architecture matches
- [ ] Residual add: `x = x + attn_out` correct?
- [ ] Residual add: `x = x + ffn_out` correct?

### RMSNorm

- [ ] Epsilon value matches (typically 1e-6 or 1e-5)
- [ ] Norm placement: before attention? before FFN?
- [ ] Norm weights loaded correctly

---

## Files to Check

### Our Implementation
```
/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp
/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/cuda/weight_loader.cpp
/home/vince/Projects/llama-orch/bin/worker-orcd/tests/haiku_generation_anti_cheat.rs
```

### References
```
/home/vince/Projects/llama-orch/reference/mistral.rs/mistralrs-core/src/models/qwen2.rs
/home/vince/Projects/llama-orch/reference/candle/candle-transformers/src/models/qwen3.rs
/home/vince/Projects/llama-orch/reference/vllm/vllm/model_executor/models/qwen2.py
```

---

## Test Command

```bash
# Run haiku test
cd /home/vince/Projects/llama-orch
cargo test --test haiku_generation_anti_cheat -- --ignored

# Expected output (when fixed):
# ✅ Minute word found in output
# ✅ No mojibake
# ✅ Coherent English text
```

---

## Common Bugs to Look For

### Bug Pattern 1: Transpose Flag Wrong
```cpp
// WRONG: opB = CUBLAS_OP_N (no transpose)
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, ...);

// RIGHT: opB = CUBLAS_OP_T (transpose weight matrix)
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, ...);
```

### Bug Pattern 2: Leading Dimension Wrong
```cpp
// WRONG: lda = M (assumes column-major)
cublasSgemm(..., lda=M, ldb=K, ldc=M);

// RIGHT: lda = K (row-major input)
cublasSgemm(..., lda=K, ldb=N, ldc=N);
```

### Bug Pattern 3: Head Concatenation Order Wrong
```cpp
// WRONG: Concatenate heads in wrong order
for (int h = 0; h < num_heads; h++) {
    for (int d = 0; d < head_dim; d++) {
        out[h * head_dim + d] = heads[h][d];  // Wrong stride
    }
}

// RIGHT: Flatten heads correctly
// [B, H, L, D] → [B, L, H*D]
// out[b][l][h*D + d] = heads[b][h][l][d]
```

### Bug Pattern 4: Weight Not Loaded
```cpp
// WRONG: Forgot to load weight
layer.ffn_down = nullptr;  // Oops!

// RIGHT: Load all weights
layer.ffn_down = get_ptr(prefix + "ffn_down.weight");
```

---

## Timeline

- **30 min**: Read mistral.rs Qwen2 attention
- **30 min**: Read candle Qwen3 attention
- **30 min**: Compare with our attention implementation
- **1 hour**: Identify bug #1, implement fix
- **30 min**: Test fix
- **Repeat**: For bugs #2 and #3

**Total**: 4-6 hours to fix

---

## Success Criteria

**Minimum**:
- ✅ Test passes
- ✅ Minute word found
- ✅ No crashes

**Full**:
- ✅ No mojibake
- ✅ No repetitive tokens
- ✅ Coherent English
- ✅ Contextually appropriate

---

## Emergency Contact

If stuck, check:
1. **Investigation history**: `haiku_generation_anti_cheat.rs` lines 248-309
2. **Team handoffs**: `/investigation-teams/TEAM_*_HANDOFF.md`
3. **Full plan**: `TRANSFORMER_BUGFIX_PLAN.md`

---

**Remember**: mistral.rs and candle are your friends. NOT llama.cpp.
