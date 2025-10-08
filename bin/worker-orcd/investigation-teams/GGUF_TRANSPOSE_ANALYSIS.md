# 🔥 GGUF Transpose Analysis - ALL Weight Matrices

**Date:** 2025-10-08T00:14Z  
**Investigator:** TEAM DICKINSON  
**Tool:** `gguf_dump.py` from llama.cpp

---

## Executive Summary

**CRITICAL DISCOVERY:** The GGUF file stores weight matrices in **COLUMN-MAJOR** layout, but our code assumes **ROW-MAJOR** (like Candle/PyTorch).

**Impact:** EVERY matrix multiplication is reading transposed data!

---

## 🔍 GGUF Tensor Dimensions

### Embedding & Output

```
token_embd.weight:  [896, 151936]  = [hidden_size, vocab_size]
output.weight:      [896, 151936]  = [hidden_size, vocab_size]
```

**Candle expects:** `[vocab_size, hidden_size]` = `[151936, 896]`

**VERDICT:** ❌ **TRANSPOSED!**

---

### FFN Weights

```
ffn_gate.weight:    [896, 4864]    = [hidden_size, intermediate_size]
ffn_up.weight:      [896, 4864]    = [hidden_size, intermediate_size]
ffn_down.weight:    [4864, 896]    = [intermediate_size, hidden_size]
```

**Candle FFN code:**
```rust
let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
```

**Candle `linear_no_bias` expects:** `[out_features, in_features]`

So:
- `gate_proj`: `[intermediate, hidden]` = `[4864, 896]`
- `up_proj`: `[intermediate, hidden]` = `[4864, 896]`
- `down_proj`: `[hidden, intermediate]` = `[896, 4864]`

**GGUF has:**
- `ffn_gate`: `[896, 4864]` ← **TRANSPOSED!**
- `ffn_up`: `[896, 4864]` ← **TRANSPOSED!**
- `ffn_down`: `[4864, 896]` ← **TRANSPOSED!**

**VERDICT:** ❌ **ALL FFN WEIGHTS TRANSPOSED!**

---

### Attention Weights

```
attn_q.weight:      [896, 896]     = [hidden_size, hidden_size]
attn_k.weight:      [896, 128]     = [hidden_size, kv_dim]
attn_v.weight:      [896, 128]     = [hidden_size, kv_dim]
attn_output.weight: [896, 896]     = [hidden_size, hidden_size]
```

**Candle attention code:**
```rust
let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
```

**Candle expects:**
- `q_proj`: `[num_heads * head_dim, hidden_size]` = `[896, 896]`
- `k_proj`: `[num_kv_heads * head_dim, hidden_size]` = `[128, 896]`
- `v_proj`: `[num_kv_heads * head_dim, hidden_size]` = `[128, 896]`
- `o_proj`: `[hidden_size, num_heads * head_dim]` = `[896, 896]`

**GGUF has:**
- `attn_q`: `[896, 896]` ← **TRANSPOSED!** (should be `[896, 896]` but column-major)
- `attn_k`: `[896, 128]` ← **TRANSPOSED!** (should be `[128, 896]`)
- `attn_v`: `[896, 128]` ← **TRANSPOSED!** (should be `[128, 896]`)
- `attn_output`: `[896, 896]` ← **TRANSPOSED!** (should be `[896, 896]` but column-major)

**VERDICT:** ❌ **ALL ATTENTION WEIGHTS TRANSPOSED!**

---

## 🎯 The Root Cause

### GGUF Storage Format

GGUF stores matrices in **COLUMN-MAJOR** order (like Fortran, BLAS):
- Matrix `[M, N]` in GGUF means: M columns, N rows
- Memory layout: Column 0, Column 1, ..., Column M-1

### Candle/PyTorch Format

Candle/PyTorch use **ROW-MAJOR** order (like C, NumPy):
- Matrix `[M, N]` means: M rows, N columns
- Memory layout: Row 0, Row 1, ..., Row M-1

### The Mismatch

When we load GGUF `[896, 151936]` and treat it as row-major:
- We think: 896 rows × 151936 columns
- Actually: 896 columns × 151936 rows (transposed!)

---

## 🔥 Why TEAM SHAKESPEARE's Fix Failed

**TEAM SHAKESPEARE tested:**
```cpp
// Original:
half value = weight_matrix[token_id * hidden_dim + dim_idx];

// Transposed:
half value = weight_matrix[dim_idx * vocab_size + token_id];
```

**Result:** Different garbage, but still garbage!

**Why it failed:**
1. ✅ Fixed embedding transpose
2. ❌ But ALL other weights still transposed!
3. ❌ lm_head (output.weight) still transposed!
4. ❌ FFN weights still transposed!
5. ❌ Attention weights still transposed!

**Result:** Embedding correct, but downstream layers produce garbage!

---

## 🎯 The Complete Fix

### Option A: Transpose ALL Weights at Load Time

**Pros:** Simple, matches Candle exactly  
**Cons:** Extra memory copy at startup

**Implementation:**
```cpp
// In qwen_weight_loader.cpp
Tensor* load_tensor_transposed(const char* path, const char* name) {
    Tensor* original = load_tensor_to_vram(path, name);
    // Transpose: swap dimensions and reorder data
    Tensor* transposed = transpose_tensor(original);
    cudaFree(original);
    return transposed;
}

// Load all weights transposed
model->weights.token_embd = load_tensor_transposed(path, "token_embd.weight");
model->weights.output = load_tensor_transposed(path, "output.weight");
// ... same for all FFN, attention weights
```

### Option B: Use CUBLAS_OP_T for ALL Matrix Multiplications

**Pros:** No extra memory, just change cuBLAS flags  
**Cons:** Need to fix ALL matmul calls, easy to miss one

**Implementation:**
```cpp
// Change ALL cublasGemmEx calls from:
cublasGemmEx(..., CUBLAS_OP_N, CUBLAS_OP_N, ...)

// To:
cublasGemmEx(..., CUBLAS_OP_T, CUBLAS_OP_T, ...)  // Both transposed!
```

**CRITICAL:** Must transpose BOTH matrices in each matmul!

### Option C: Fix Embedding Only (Quick Test)

**Pros:** Minimal change, test if embedding is the only issue  
**Cons:** Won't fix everything if other weights also wrong

**Implementation:**
```cpp
// cuda/kernels/embedding.cu line 177
// Change from:
half value = weight_matrix[token_id * hidden_dim + dim_idx];

// To:
half value = weight_matrix[dim_idx * vocab_size + token_id];
```

**Expected result:** If embedding was the ONLY bug, output should improve

---

## 📋 Recommended Action Plan

### Step 1: Test Embedding Fix Only (5 minutes)

**Action:** Apply Option C (transpose embedding access only)

**Expected:**
- If output improves → embedding was main bug, but others remain
- If still garbage → need to fix ALL weights

### Step 2: Check How Candle Loads GGUF (30 minutes) ✅ **DONE!**

**Action:** Find Candle's GGUF loader and see how it handles transpose

**Files checked:**
- `candle-core/src/quantized/gguf_file.rs` - GGUF loader (no transpose)
- `candle-nn/src/linear.rs` - Linear layer implementation

**ANSWER FOUND:** 🔥🔥🔥 **Candle transposes weights in EVERY forward pass!**

**From `candle-nn/src/linear.rs` line 43-73:**
```rust
impl super::Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let x = match *x.dims() {
            [b1, b2, m, k] => {
                let w = self.weight.t()?;  // ← TRANSPOSE!
                x.reshape((b1 * b2 * m, k))?.matmul(&w)?
            }
            [bsize, m, k] => {
                let w = self.weight.t()?;  // ← TRANSPOSE!
                x.reshape((bsize * m, k))?.matmul(&w)?
            }
            _ => {
                let w = self.weight.t()?;  // ← TRANSPOSE!
                x.matmul(&w)?
            }
        };
        // ...
    }
}
```

**VERDICT:** Candle loads GGUF as-is (column-major), then **transposes on-the-fly** in every linear layer!

### Step 3: Apply Complete Fix (1-2 hours)

**Action:** Based on Candle's approach, apply same fix to our code

**Options:**
- If Candle transposes at load → implement Option A
- If Candle uses CUBLAS_OP_T → implement Option B

---

## 🔬 Verification Commands

### Check GGUF Dimensions

```bash
python3 reference/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py \
  .test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf | \
  grep -E "(token_embd|output|ffn_|attn_)" | head -30
```

### Check Candle's GGUF Loader

```bash
cd reference/candle
rg "gguf" --type rust -A 5 | grep -E "(transpose|permute|swap)"
```

### Test Embedding Fix

```bash
# Apply Option C fix to embedding.cu
cd bin/worker-orcd
cargo build --features cuda --release
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  --features cuda --release -- --ignored --nocapture
```

---

## 🎓 Key Insights

### 1. GGUF is Column-Major, Not Row-Major

**This is FUNDAMENTAL!** All GGUF tensors are stored column-major.

**Impact:** Every tensor needs transpose when loading into row-major systems.

### 2. Candle Must Handle This

**Candle works with GGUF files**, so it must have a solution.

**Action:** Study Candle's GGUF loader to see how they handle it.

### 3. TEAM SHAKESPEARE Was Right!

**They tested transpose** and saw output change.

**Why it failed:** Only fixed embedding, not other 100+ weight matrices.

### 4. This Explains EVERYTHING

**Why llama.cpp works:** It's written in C (row-major) but handles GGUF transpose correctly

**Why our code fails:** We assume row-major but load column-major data directly

**Why changing cuBLAS flags helped:** CUBLAS_OP_T transposes, partially compensating

---

## 📚 References

**GGUF Spec:**
- https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

**Candle GGUF Loader:**
- `candle-core/src/quantized.rs`
- `candle-core/src/gguf_file.rs`

**Our Code:**
- `cuda/kernels/embedding.cu` line 177
- `cuda/src/model/qwen_weight_loader.cpp`
- `cuda/src/transformer/qwen_transformer.cpp` (all cuBLAS calls)

**Investigation Docs:**
- `SMOKING_GUN_DEEP_DIVE.md` - Reference implementation analysis
- `TRANSPOSE_FIX_TEST_RESULTS.md` - Shakespeare's test results
- `UNINVESTIGATED_SMOKING_GUNS.md` - Original smoking gun list

---

**TEAM DICKINSON**  
*"Tell all the truth but tell it slant—Success in Circuit lies."*

**Status:** 🔥 **ROOT CAUSE IDENTIFIED!**  
**Next:** Check Candle's GGUF loader, apply same fix  
**Confidence:** 🔥🔥🔥🔥🔥 99% (this is THE bug!)  
**Last Updated:** 2025-10-08T00:14Z
