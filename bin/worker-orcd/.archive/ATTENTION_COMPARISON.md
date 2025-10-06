# Attention Implementation Comparison

**Date**: 2025-10-06  
**Purpose**: Compare our GQA attention with llama.cpp to find the bug

---

## Key Findings from llama.cpp Validation

✅ **llama.cpp works perfectly** with the same model file  
❌ **Our implementation produces garbage** (repetitive token 78138)  
🎯 **Conclusion**: Bug is in our attention implementation

---

## Critical Differences to Investigate

### 1. **Attention Scale Application**

**Our implementation** (`gqa_attention.cu` line 104):
```cuda
scores[pos] = score * scale;  // Applied AFTER dot product
```

**llama.cpp** (`fattn-vec.cuh` lines 214-216, 232-234):
```cuda
// Scale is applied to Q BEFORE computing dot product
Q_reg[j][k] *= scale_h2;  // Q is pre-scaled
// Then: sum = vec_dot_KQ(K, Q_reg[j], ...)
```

**🔴 CRITICAL**: We scale AFTER dot product, llama.cpp scales Q BEFORE.  
This is mathematically equivalent but may have numerical stability differences.

---

### 2. **Attention Score Computation**

**Our implementation** (lines 74-105):
```cuda
// Compute Q·K for each position
for (int pos = tid; pos <= cache_len; pos += blockDim.x) {
    float score = 0.0f;
    
    if (pos < cache_len) {
        // Cached K
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * k_val;
        }
    } else {
        // Current K
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * k_val;
        }
    }
    
    scores[pos] = score * scale;
}
```

**llama.cpp** (lines 262-263):
```cuda
float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
sum = warp_reduce_sum<nthreads_KQ>(sum);
```

**Difference**: llama.cpp uses warp-level reduction for the dot product.  
Our implementation does it sequentially in each thread.

---

### 3. **Softmax Implementation**

**Our implementation** (lines 108-160):
```cuda
// Find max
if (tid == 0) {
    float max_score = -1e9f;
    for (int i = 0; i <= cache_len; i++) {
        max_score = fmaxf(max_score, scores[i]);
    }
    max_val[0] = max_score;
}

// Compute exp and sum
for (int pos = tid; pos <= cache_len; pos += blockDim.x) {
    float exp_score = expf(scores[pos] - max_val[0]);
    scores[pos] = exp_score;
    local_sum += exp_score;
}

// Reduce sum across threads
// ... parallel reduction ...

// Normalize
for (int pos = tid; pos <= cache_len; pos += blockDim.x) {
    scores[pos] /= sum_exp[0];
}
```

**llama.cpp** (lines 287-291):
```cuda
const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
KQ_max[j] = KQ_max_new[j];

KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
```

**🔴 CRITICAL**: llama.cpp uses **online softmax** (running max/sum update).  
Our implementation uses **two-pass softmax** (find max, then normalize).

Online softmax is more numerically stable for long sequences.

---

### 4. **KV Cache Indexing**

**Our implementation** (lines 81-83):
```cuda
int k_cache_idx = batch * num_kv_heads * max_seq_len * head_dim +
                  kv_head * max_seq_len * head_dim +
                  pos * head_dim + d;
```

**Cache layout**: `[batch, kv_head, pos, d]` with `max_seq_len` stride

**llama.cpp**: Uses ggml tensor indexing with strides `nb11`, `nb12`, `nb13`

**Potential issue**: Our cache indexing might be wrong if `max_seq_len` stride is incorrect.

---

### 5. **GQA Head Mapping**

**Our implementation** (line 50):
```cuda
int kv_head = q_head / (num_q_heads / num_kv_heads);
```

**For Qwen2.5-0.5B**:
- `num_q_heads = 14`
- `num_kv_heads = 2`
- `gqa_ratio = 14 / 2 = 7`

**Mapping**:
- Q heads 0-6 → KV head 0
- Q heads 7-13 → KV head 1

**llama.cpp** (line 101):
```cuda
K += nb13*sequence + nb12*(head / gqa_ratio);
```

Same logic, so this should be correct.

---

## Most Likely Bugs (Ranked)

### 1. 🔴 **KV Cache Indexing Bug** - HIGHEST PRIORITY

**Hypothesis**: The cache read/write indices are wrong, causing:
- Model reads garbage from cache
- Model writes to wrong cache positions
- Attention always sees the same (wrong) K/V values

**Evidence**:
- Repetitive output suggests model isn't seeing its own generations
- Token 78138 is very specific, suggesting it's reading from a fixed wrong position

**How to test**:
```cpp
// Add to qwen_transformer.cpp after attention:
// Copy first 10 K cache values to CPU and print them
// Verify they match what was written
```

### 2. 🟡 **Softmax Numerical Instability** - MEDIUM PRIORITY

**Hypothesis**: Two-pass softmax causes numerical issues

**Evidence**:
- llama.cpp uses online softmax (more stable)
- We use two-pass softmax (can accumulate errors)

**How to fix**:
```cuda
// Switch to online softmax like llama.cpp
// Update running max and sum as we process each position
```

### 3. 🟡 **Scale Application Timing** - MEDIUM PRIORITY

**Hypothesis**: Scaling after dot product vs before might cause issues

**Evidence**:
- llama.cpp pre-scales Q
- We post-scale the scores
- Mathematically equivalent but numerically different

**How to fix**:
```cuda
// Pre-scale Q in registers before computing dot products
for (int d = tid; d < head_dim; d += blockDim.x) {
    q_vec[d] *= scale;  // Scale Q, not scores
}
```

### 4. 🟢 **Warp-Level Reduction** - LOW PRIORITY

**Hypothesis**: Sequential dot product is slower but should still work

**Evidence**:
- llama.cpp uses warp reduction
- We use sequential loops
- Performance issue, not correctness issue

---

## Recommended Fix Order

### Step 1: Verify KV Cache Indexing (30 min)

Add host-side debug to print K/V cache values:

```cpp
// In qwen_transformer.cpp after attention kernel
if (layer_idx == 0 && pos < 3) {
    // Copy first 10 K values from cache to CPU
    half h_k_cache[10];
    int kv_head = 0;
    int cache_idx = 0 * num_kv_heads * max_seq_len * head_dim +
                    kv_head * max_seq_len * head_dim +
                    pos * head_dim;
    cudaMemcpy(h_k_cache, kv_cache_k + cache_idx, 10 * sizeof(half), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "[KV CACHE DEBUG] Layer 0, pos=%u, kv_head=0\n", pos);
    fprintf(stderr, "  K_cache[0:10]: ");
    for (int i = 0; i < 10; i++) {
        fprintf(stderr, "%.4f ", __half2float(h_k_cache[i]));
    }
    fprintf(stderr, "\n");
}
```

### Step 2: Pre-scale Q Instead of Post-scaling Scores (15 min)

```cuda
// In gqa_attention_decode_kernel_impl, after loading Q:
for (int d = tid; d < head_dim; d += blockDim.x) {
    int q_idx = batch * num_q_heads * head_dim + q_head * head_dim + d;
    q_vec[d] = __half2float(q[q_idx]) * scale;  // Apply scale here
}

// Then remove scale from score computation:
scores[pos] = score;  // No scale here
```

### Step 3: Switch to Online Softmax (1 hour)

Implement running max/sum update like llama.cpp.

---

## Next Action

**Immediately**: Add KV cache debugging to verify indices are correct.

If cache indices are wrong, that's the smoking gun. If they're correct, move to scaling/softmax fixes.

---

**See also**:
- `LLAMA_CPP_VALIDATION.md` - Proof that model file is valid
- `DEBUG_RUN_RESULTS.md` - Current debug output
- `cuda/kernels/gqa_attention.cu` - Our implementation
- `reference/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` - llama.cpp reference
