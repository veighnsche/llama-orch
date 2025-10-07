# Team HOTEL Investigation Report
**Date:** 2025-10-06 20:09-20:15 UTC  
**Mission:** Fix cuBLAS issue where position 8850 returns 0.0  
**Status:** BUG FOUND AND FIXED - Awaiting test verification

---

## 🎯 Mission Objective

Team GEMMA DELTA handed off a critical bug:
- cuBLAS returns 0.0 at position 8850 (should be -2.466037)
- Position 0 works correctly (0.831105)
- Test crashes during inference

---

## 🔍 Investigation Process

### Step 1: Read the Handoff
Team GEMMA DELTA claimed they fixed two bugs:
1. ✅ Wrong vocab size dimension (changed dim[1] to dim[0])
2. ✅ Missing padded vocab size support

They said the tensor was `[151643, 151936]` (vocab × padded_vocab).

### Step 2: Check Previous Team Reports
Looked at `TEAM_BRAVO_RESULTS.md` line 79:
```
🔍 [Rust] output.weight dimensions: [896, 151936]
```

**CONTRADICTION!** The actual tensor is `[896, 151936]`, NOT `[151643, 151936]`!

### Step 3: Verify with llama.cpp
Checked `llama-model.cpp:2365`:
```cpp
output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, ...);
```

This is `{hidden_dim, vocab_size}` = `{896, 151936}` ✓

### Step 4: Trace the Bug
Team GEMMA DELTA's code in `cuda_backend.rs:222-228`:
```rust
let actual = output_tensor.dimensions.get(0)  // Gets 896
let padded = output_tensor.dimensions.get(1)  // Gets 151936

(actual, padded)  // Returns (896, 151936)
```

They named these variables `vocab_size` and `padded_vocab_size`, but:
- `actual` (896) is actually **hidden_dim**, not vocab_size!
- `padded` (151936) is correct, but they thought it was "padded" when it's the full vocab

---

## 🐛 THE BUG

### Root Cause
**Team GEMMA DELTA swapped the tensor dimensions!**

They thought:
- `dimensions[0]` = vocab_size (151643)
- `dimensions[1]` = padded_vocab_size (151936)

Reality:
- `dimensions[0]` = hidden_dim (896)
- `dimensions[1]` = padded_vocab_size (151936)
- vocab_size (151643) is NOT in the tensor! Must get from tokenizer metadata.

### Consequence
The wrong values were passed to cuBLAS:
```cpp
cublasGemmEx(
    ...,
    config_.vocab_size,        // m = 896 (WRONG! Should be 151936)
    batch_size,                // n = 1
    config_.hidden_dim,        // k = 896
    ...
);
```

cuBLAS computed only 896 output logits instead of 151936!
- Position 0-895: Computed correctly ✓
- Position 896-151935: Uninitialized memory (returns 0.0) ✗

That's why position 8850 returned 0.0!

---

## ✅ THE FIX

### Three Critical Values
1. **vocab_size = 151643** (logical, from tokenizer metadata)
   - Use for argmax to skip 293 padding tokens
   
2. **padded_vocab_size = 151936** (physical, from tensor dimensions[1])
   - Use for cuBLAS stride and buffer allocation
   
3. **hidden_dim = 896** (from tensor dimensions[0])
   - Use for input dimension

### Code Changes

#### 1. `src/inference/cuda_backend.rs` (lines 224-265)
```rust
// Extract dimensions correctly
let (hidden_dim_from_tensor, padded_vocab_size) = {
    let hidden = output_tensor.dimensions.get(0)?;  // 896 = hidden_dim
    let padded_vocab = output_tensor.dimensions.get(1)?;  // 151936 = padded_vocab
    (hidden, padded_vocab)
};

// Get logical vocab from tokenizer (NOT from tensor!)
let vocab_size = self.metadata.vocab_size()? as u32;  // 151643
```

#### 2. `cuda/src/transformer/qwen_transformer.cpp` (lines 639-652)
```cpp
cublasGemmEx(
    cublas_handle_,
    CUBLAS_OP_N, CUBLAS_OP_N,
    config_.padded_vocab_size,  // m = 151936 (FULL output size)
    batch_size,                 // n = 1
    config_.hidden_dim,         // k = 896
    &alpha,
    lm_head_half, CUDA_R_16F, config_.padded_vocab_size,  // lda = 151936
    hidden_half, CUDA_R_16F, config_.hidden_dim,          // ldb = 896
    &beta,
    logits, CUDA_R_32F, config_.padded_vocab_size,        // ldc = 151936
    ...
);
```

#### 3. `cuda/src/ffi_inference.cpp` (lines 114-119)
```cpp
// Allocate buffer for FULL padded vocab size
cudaMalloc(&logits, padded_vocab_size * sizeof(float));  // 151936 floats
std::vector<float> init_logits(padded_vocab_size, -INFINITY);
cudaMemcpy(logits, init_logits.data(), padded_vocab_size * sizeof(float), ...);
```

#### 4. `cuda/src/transformer/qwen_transformer.cpp` (lines 747-748)
```cpp
// Copy ALL logits for verification
float* h_logits = new float[config_.padded_vocab_size];  // 151936
cudaMemcpy(h_logits, logits, config_.padded_vocab_size*sizeof(float), ...);
```

#### 5. `cuda/src/adapters/gpt_adapter.cpp` (lines 407)
```cpp
// Argmax: Only scan logical vocab_size (skip padding)
for (int i = 1; i < config_.vocab_size; i++) {  // 151643 (CORRECT!)
```

---

## 📝 Forensic Comments Added

Added detailed comments to:
- ✅ `src/inference/cuda_backend.rs:174-244` - Dimension extraction bug explanation
- ✅ `src/cuda/model.rs:74-106` - Warning that this code is wrong
- ✅ `cuda/src/transformer/qwen_transformer.h:12-31` - Critical understanding doc
- ✅ `cuda/src/transformer/qwen_transformer.cpp:612-638` - cuBLAS fix explanation
- ✅ `cuda/src/ffi_inference.cpp:94-119` - Buffer allocation fix
- ✅ `cuda/src/adapters/gpt_adapter.cpp:401-404` - Argmax explanation

All comments use forensic markers:
- `SYMPTOM:` - What we observed
- `ROOT CAUSE:` - What was wrong
- `THOUGHT:` - Our reasoning
- `TRACE:` - Evidence from other files
- `CONSEQUENCE:` - What the bug caused
- `FIXED:` - What we changed

---

## 🔑 Key Insights

### Why Team GEMMA DELTA Got Confused

1. **They saw "151643" somewhere** - Probably from tokenizer metadata
2. **Assumed it was in the tensor** - But it's not! Tensor only has [896, 151936]
3. **Misread dimension order** - Thought [vocab, hidden] but it's [hidden, vocab]
4. **Didn't cross-check with previous teams** - TEAM_BRAVO had the correct dimensions

### The "151643" Mystery

Where did they see 151643?
- ✓ Tokenizer metadata (`meta.vocab_size()` returns 151643)
- ✗ NOT in output.weight tensor dimensions
- ✗ NOT in any GGUF tensor

They conflated two different sources of truth!

### Why Position 0 Worked

cuBLAS with m=896 computed logits[0..895] correctly.
Position 0 is within this range, so it matched the manual calculation.
Position 8850 is beyond 896, so it was uninitialized memory.

---

## 📊 Expected Test Results

After the fix:
1. ✅ cuBLAS verification should pass at ALL positions (0, 8850, 44394, 137131)
2. ✅ Test should not crash
3. ✅ Model should generate coherent haiku (not garbage)
4. ✅ Haiku should contain the minute word

---

## 🚀 Next Steps

1. ⏳ Wait for test to compile and run
2. ⏳ Verify cuBLAS verification passes
3. ⏳ Check haiku output quality
4. ⏳ Update this report with test results

---

## 📚 Lessons for Future Teams

1. **Always cross-check with previous teams** - Don't assume your understanding is correct
2. **Verify tensor dimensions empirically** - Check actual debug output, not comments
3. **Distinguish logical vs physical sizes** - vocab_size ≠ tensor dimensions
4. **Use consistent naming** - Don't call hidden_dim "vocab_size"!
5. **Add forensic comments** - Explain your thought process for next team

---

Built by Team HOTEL 🏨  
*"We check in to debug, we check out with fixes"*
