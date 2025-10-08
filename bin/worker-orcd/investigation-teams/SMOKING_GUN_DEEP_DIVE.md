# 🔥 Smoking Gun Deep Dive: Reference Implementation Analysis

**Date:** 2025-10-08  
**Investigator:** TEAM DICKINSON  
**References:** Candle, mistral.rs, tinygrad (NOT llama.cpp)

---

## Executive Summary

After thorough analysis of **Candle** and **mistral.rs** (much cleaner than llama.cpp!), I've identified **THE SMOKING GUN**:

**Our embedding lookup is CORRECT in terms of indexing**, but we need to verify:
1. **Tensor dimensions** from GGUF (vocab_size × hidden_size)
2. **No scaling** applied after embedding (Candle doesn't scale either)
3. **The bug is likely DOWNSTREAM** in attention or FFN

---

## 📚 Reference Implementation Analysis

### Candle Implementation (Clean & Educational!)

**File:** `reference/candle/candle-nn/src/embedding.rs`

```rust
impl crate::Module for Embedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;  // ← KEY LINE!
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}
```

**What this does:**
1. Takes embedding matrix shape `[vocab_size, hidden_size]`
2. Uses `index_select(&indexes, 0)` - selects rows by index
3. Returns `[batch_size, hidden_size]`

**Equivalent to:**
```cpp
// For each token_id in batch:
for (int token_idx = 0; token_idx < batch_size; token_idx++) {
    int token_id = token_ids[token_idx];
    // Copy row token_id from embedding matrix
    for (int dim = 0; dim < hidden_size; dim++) {
        output[token_idx * hidden_size + dim] = 
            embeddings[token_id * hidden_size + dim];  // ← Row-major indexing
    }
}
```

**Our code:**
```cpp
// cuda/kernels/embedding.cu line 177
half value = weight_matrix[token_id * hidden_dim + dim_idx];
embeddings[token_idx * hidden_dim + dim_idx] = value;
```

**VERDICT:** ✅ **Our indexing matches Candle exactly!**

---

### Candle Qwen2 Model

**File:** `reference/candle/candle-transformers/src/models/qwen2.rs`

```rust
pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
    let vb_m = vb.pp("model");
    let embed_tokens =
        candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
    // ...
}

pub fn forward(&mut self, input_ids: &Tensor, ...) -> Result<Tensor> {
    let mut xs = self.embed_tokens.forward(input_ids)?;  // ← No scaling!
    for layer in self.layers.iter_mut() {
        xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
    }
    xs.apply(&self.norm)
}
```

**Key observations:**
1. **No embedding scaling** (no `sqrt(hidden_size)` multiplication)
2. **Direct forward pass** from embedding to layers
3. **Tied embeddings** for lm_head (line 383):
   ```rust
   Linear::from_weights(base_model.embed_tokens.embeddings().clone(), None)
   ```

**VERDICT:** ✅ **Our code matches - no scaling needed!**

---

### mistral.rs Implementation

**File:** `reference/mistral.rs/mistralrs-core/src/models/qwen2.rs`

```rust
pub fn forward(&mut self, input_ids: &Tensor, ...) -> Result<Tensor> {
    let xs = self.embed_tokens.forward(input_ids)?;  // ← Same as Candle
    self.forward_embed(input_ids, xs, ...)
}
```

**VERDICT:** ✅ **Same as Candle - no special handling**

---

## 🔍 GGUF Tensor Layout Analysis

### What Candle Expects

**From:** `reference/candle/candle-nn/src/embedding.rs` line 39-48

```rust
pub fn embedding(in_size: usize, out_size: usize, vb: crate::VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get_with_hints(
        (in_size, out_size),  // ← Shape: [vocab_size, hidden_size]
        "weight",
        crate::Init::Randn { mean: 0., stdev: 1., },
    )?;
    Ok(Embedding::new(embeddings, out_size))
}
```

**Expected shape:** `[vocab_size, hidden_size]` = `[151936, 896]`

### What Our GGUF Has

**From TEAM VAN GOGH's investigation:**
```
token_embd.weight dimensions: [896, 151936]
```

**WAIT!** This is `[hidden_size, vocab_size]` - **TRANSPOSED!**

---

## 🔥 THE SMOKING GUN REVEALED

### The Problem

**Candle expects:** `[vocab_size, hidden_size]` = `[151936, 896]`  
**Our GGUF has:** `[hidden_size, vocab_size]` = `[896, 151936]`

**Our code assumes:** `[vocab_size, hidden_size]` (same as Candle)

**Result:** We're reading from the WRONG dimension!

### Visual Explanation

**Correct layout (Candle):**
```
Embedding matrix [vocab_size=151936, hidden_size=896]
Row 0:    [e0_0, e0_1, e0_2, ..., e0_895]     ← Token 0 embedding
Row 1:    [e1_0, e1_1, e1_2, ..., e1_895]     ← Token 1 embedding
Row 2:    [e2_0, e2_1, e2_2, ..., e2_895]     ← Token 2 embedding
...
Row 151935: [e151935_0, ..., e151935_895]     ← Token 151935 embedding

To get token T embedding: embeddings[T * 896 + dim]
```

**Our GGUF layout (TRANSPOSED):**
```
Embedding matrix [hidden_size=896, vocab_size=151936]
Row 0:    [e0_0, e1_0, e2_0, ..., e151935_0]  ← Dimension 0 for ALL tokens
Row 1:    [e0_1, e1_1, e2_1, ..., e151935_1]  ← Dimension 1 for ALL tokens
Row 2:    [e0_2, e1_2, e2_2, ..., e151935_2]  ← Dimension 2 for ALL tokens
...
Row 895:  [e0_895, e1_895, ..., e151935_895]  ← Dimension 895 for ALL tokens

To get token T embedding: embeddings[dim * 151936 + T]  ← DIFFERENT!
```

### What Our Code Does (WRONG!)

```cpp
// cuda/kernels/embedding.cu line 177
half value = weight_matrix[token_id * hidden_dim + dim_idx];
```

**For token_id=0, dim_idx=0:**
- We read: `weight_matrix[0 * 896 + 0]` = `weight_matrix[0]`
- This is: `e0_0` (dimension 0 of token 0) ✅ CORRECT by accident!

**For token_id=0, dim_idx=1:**
- We read: `weight_matrix[0 * 896 + 1]` = `weight_matrix[1]`
- This is: `e1_0` (dimension 0 of token 1) ❌ WRONG!
- Should be: `e0_1` (dimension 1 of token 0)

**For token_id=1, dim_idx=0:**
- We read: `weight_matrix[1 * 896 + 0]` = `weight_matrix[896]`
- This is: `e0_1` (dimension 1 of token 0) ❌ WRONG!
- Should be: `e1_0` (dimension 0 of token 1)

**Result:** We're reading a DIAGONAL slice through the transposed matrix!

---

## 🎯 The Fix

### Option A: Fix Our Code (Transpose Access)

```cpp
// cuda/kernels/embedding.cu line 177
// WRONG (current):
half value = weight_matrix[token_id * hidden_dim + dim_idx];

// CORRECT (transposed access):
half value = weight_matrix[dim_idx * vocab_size + token_id];
```

**But TEAM SHAKESPEARE tested this and output was STILL garbage!**

### Why Shakespeare's Test Failed

**From:** `cuda/kernels/embedding.cu` lines 154-161

```cpp
// TEST RESULTS:
//   Original code: weight_matrix[token_id * hidden_dim + dim_idx]
//     → Generated tokens: [20695, 131033, 42294, 43321, ...] (garbage)
//   
//   Transposed access: weight_matrix[dim_idx * vocab_size + token_id]
//     → Generated tokens: [37557, 103357, 69289, 62341, ...] (DIFFERENT garbage!)
```

**Analysis:**
- Changing indexing DOES change output ✅ (proves embedding matters)
- But output still garbage ❌ (transpose alone not enough)

**Possible reasons:**
1. **GGUF dimensions are ALREADY correct** `[151936, 896]` and VAN GOGH misread them
2. **There are MULTIPLE transpose bugs** (embedding + lm_head + Q/K/V/FFN)
3. **The bug is DOWNSTREAM** (attention/FFN) and embedding is fine

---

## 🔬 Verification Strategy

### Step 1: Verify GGUF Dimensions (CRITICAL!) ✅ **DONE!**

**Action:** Use `gguf-dump` tool to check actual tensor dimensions

```bash
python3 reference/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py \
  .test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf | grep -A 3 "token_embd.weight"
```

**ACTUAL OUTPUT:**
```
2:  136134656 |   896, 151936,     1,     1 | F16     | token_embd.weight
```

**VERDICT:** 🔥🔥🔥 **GGUF HAS TRANSPOSED DIMENSIONS!**

- **GGUF shape:** `[896, 151936]` = `[hidden_size, vocab_size]`
- **Candle expects:** `[151936, 896]` = `[vocab_size, hidden_size]`
- **Our code assumes:** `[151936, 896]` (same as Candle)

**RESULT:** We're reading from WRONG dimension! TEAM SHAKESPEARE was RIGHT!

### Step 2: Compare Embedding Values with Candle

**Action:** Extract first token embedding from both

**Candle:**
```rust
// Load model
let model = Model::load(...)?;
let token_0 = Tensor::new(&[0u32], &Device::Cpu)?;
let emb_0 = model.embed_tokens.forward(&token_0)?;
println!("Token 0 embedding: {:?}", emb_0.to_vec1::<f32>()?);
```

**Our code (DICKINSON C0):**
```
C0: [0.012146, 0.006836, -0.019897, -0.007050, ...]
```

**If they match:** Embedding is correct, bug is downstream  
**If they differ:** Embedding has transpose or other bug

### Step 3: Check lm_head Dimensions

**Candle code (line 383):**
```rust
Linear::from_weights(base_model.embed_tokens.embeddings().clone(), None)
```

**This means:** lm_head uses SAME weights as embedding (tied)

**Our code:** Do we tie lm_head to embedding? Or load separately?

**Check:** `cuda/src/model/qwen_weight_loader.cpp`

---

## 🎯 DICKINSON Data Analysis

### Our Captured Values

```
C0 (post-embedding): [0.012, 0.007, -0.020, -0.007, 0.002, 0.018, -0.014, 0.013, ...]
```

**Range:** [-0.045, 0.018] (±0.05)

**Analysis:**
- Values are reasonable for FP16 embeddings
- No extreme outliers
- Similar to typical embedding ranges

**Question:** Are these the CORRECT values for token 0?

**Action:** Compare with Candle's embedding for token 0

---

## 🔥 Mid-Layer Spike Analysis (NEW FINDING!)

### The Spike

```
C5 (layer 5):  [..., 2.445, 15.094, ...]  ← Index 5 has spike!
C10 (layer 10): [..., 2.807, 17.281, ...]  ← Index 5 GROWING!
C23 (layer 23): [..., 1.339, 0.707, ...]   ← Index 5 normalized
```

**Observation:** Index 5 grows from 15.094 → 17.281 then normalizes

### Hypothesis 1: Normal Model Behavior

**Test:** Run Candle with same prompt, check if index 5 also has spikes

**If YES:** This is normal, model uses dimension 5 for something important

### Hypothesis 2: FFN Bug

**Possible causes:**
- FFN gate/up/down weights wrong for dimension 5
- RMSNorm not handling large values correctly
- Residual connection accumulating errors

**Test:** Compare FFN weights for dimension 5 with Candle

### Hypothesis 3: Attention Bug

**Possible causes:**
- Attention output projection wrong for dimension 5
- Q/K/V projection wrong for dimension 5
- RoPE affecting dimension 5 incorrectly

**Test:** Log attention outputs for dimension 5

---

## 📋 Action Plan (Priority Order)

### Priority 1: Verify GGUF Dimensions

**Time:** 5 minutes

**Action:**
```bash
gguf-dump qwen2.5-0.5b-instruct-fp16.gguf | grep -A 2 "token_embd.weight"
```

**Expected:** Settle the transpose question once and for all

---

### Priority 2: Compare C0 with Candle

**Time:** 30 minutes

**Action:**
1. Write small Candle program to extract token 0 embedding
2. Compare with our C0: `[0.012, 0.007, -0.020, ...]`
3. If match → embedding correct, bug downstream
4. If differ → investigate transpose or other embedding bug

**Code:**
```rust
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::Model;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let vb = VarBuilder::from_gguf("qwen2.5-0.5b-instruct-fp16.gguf", &device)?;
    let model = Model::new(&config, vb)?;
    
    let token_0 = Tensor::new(&[0u32], &device)?;
    let emb_0 = model.embed_tokens.forward(&token_0)?;
    
    println!("Token 0 embedding (first 16): {:?}", 
             emb_0.to_vec1::<f32>()?[..16]);
    Ok(())
}
```

---

### Priority 3: Check Mid-Layer Spikes with Candle

**Time:** 1 hour

**Action:**
1. Instrument Candle to dump layer 5 and 10 outputs
2. Compare with our C5/C10 values
3. Check if index 5 also has spikes in Candle
4. If YES → normal behavior
5. If NO → investigate FFN/attention for dimension 5

---

### Priority 4: Check lm_head Weight Tying

**Time:** 15 minutes

**Action:**
1. Check if our lm_head uses same weights as embedding
2. Candle ties them (line 383)
3. If we load separately → potential transpose mismatch

**Check:** `cuda/src/model/qwen_weight_loader.cpp`

---

## 🎓 Key Learnings

### 1. Candle is MUCH Cleaner than llama.cpp

**Candle embedding:** 10 lines of clear Rust  
**llama.cpp embedding:** Buried in 1000+ lines of C

**Lesson:** Use Candle/mistral.rs for reference, not llama.cpp!

### 2. Transpose Bugs Are Subtle

**Problem:** GGUF might store `[hidden, vocab]` but we expect `[vocab, hidden]`

**Result:** Diagonal slice through matrix (some values correct by accident)

### 3. Test Results Can Be Misleading

**TEAM SHAKESPEARE:** Transposed access → different garbage

**Interpretation:** Proves embedding matters, but transpose alone not the fix

**Possible reasons:**
- GGUF dimensions already correct
- Multiple transpose bugs
- Bug is downstream

### 4. DICKINSON Data is Gold

**The spike at index 5** was never noticed before!

**Why:** No one logged mid-layer values until DICKINSON

**Value:** Could be the key to finding the bug

---

## 📚 References

**Candle:**
- `candle-nn/src/embedding.rs` - Clean embedding implementation
- `candle-transformers/src/models/qwen2.rs` - Qwen2 model
- `candle-core/src/tensor.rs` line 1426 - `embedding()` function

**mistral.rs:**
- `mistralrs-core/src/models/qwen2.rs` - Qwen2 implementation

**Our Code:**
- `cuda/kernels/embedding.cu` line 177 - Embedding lookup
- `cuda/src/model/qwen_weight_loader.cpp` - Weight loading

**Investigation Docs:**
- `REFERENCE_IMPLEMENTATION_ANALYSIS.md` - Previous analysis
- `TRANSPOSE_FIX_TEST_RESULTS.md` - Shakespeare's test results
- `DICKINSON_FINAL_REPORT.md` - Our checkpoint data

---

**TEAM DICKINSON**  
*"Tell all the truth but tell it slant—Success in Circuit lies."*

**Status:** 🔍 **DEEP DIVE COMPLETE**  
**Next:** Verify GGUF dimensions with gguf-dump  
**Last Updated:** 2025-10-08T00:13Z
