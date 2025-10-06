# Bug Fix Progress - 2025-10-06

## ✅ Bug #1: Q Vector Loading (FIXED)

**Problem**: Q vector was not loading correctly into the attention kernel. Only `q_vec[0]` had a value, rest were zeros.

**Root Cause**: 
```cuda
// WRONG: Each thread writes to q_vec[d], but only thread d reads q_vec[d]
float q_vec[64];
for (int d = tid; d < head_dim; d += blockDim.x) {
    q_vec[d] = __half2float(q[q_idx]);  // Thread 0 writes q_vec[0], Thread 1 writes q_vec[1], etc.
}
// Thread 0 reads q_vec[1] but it was written by Thread 1 (different register file!)
```

**Fix**: Use shared memory instead of per-thread registers:
```cuda
// CORRECT: All threads can access shared memory
__shared__ float q_shared[64];
for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = __half2float(q[q_idx]);
}
__syncthreads();
// Now all threads can read q_shared[0:64]
```

**Status**: ✅ FIXED - Q vector now loads all 64 dimensions correctly

---

## 🔴 Bug #2: Attention Scores Too Large (ACTIVE)

**Problem**: Attention scores are extremely large (~1000 before scaling, ~125 after scaling)

**Evidence**:
```
Q[0:5]: -0.2646, -0.0967, -0.1523, 0.0200, -13.3359
Raw scores (before softmax): 125.1664 124.3007
```

**Expected**: With `scale = 1/sqrt(64) = 0.125`, scores should be in range -10 to +10

**Actual**: Scores are 125+ (unscaled would be ~1000!)

**Possible Causes**:

### Hypothesis A: Q/K values are too large
- Q[4] = -13.3359 is suspiciously large
- Most Q values are < 1.0, but some are > 10
- This could be a RoPE bug or weight loading issue

### Hypothesis B: Dot product is accumulating incorrectly
- We're summing over all 64 dimensions
- If values aren't normalized, sum could explode

### Hypothesis C: Scale is wrong
- We use `scale = 1/sqrt(head_dim) = 1/sqrt(64) = 0.125`
- llama.cpp might use a different scale

---

## 🔍 Investigation Results

### ✅ Found: Q/K Magnitudes Are Too Large

**Q magnitude**: 60.57 (should be ~8.0 for normalized 64-dim vector)  
**Q values before RoPE**: `-0.22, 0.17, -0.11, 0.11, -14.49, 0.28, 0.47, 0.08, -15.18, -34.06`  
**Q values after RoPE**: `-0.26, -0.10, -0.15, 0.02, -13.34, -5.68, 0.43, 0.21, -8.91, -36.22`

**Problem**: Values like `-34.06` and `-36.22` are WAY too large!

### Root Cause: QKV Projection Output Is Not Normalized

The issue appears BEFORE RoPE - the QKV projection is producing unnormalized vectors.

**Possible causes**:
1. Weight matrices are not normalized (but llama.cpp uses same weights and works!)
2. We're missing a normalization step after QKV projection
3. Weight loading is incorrect (dimensions, transpose, etc.)

### Next Steps

1. **Compare with llama.cpp**: Add debug to llama.cpp to see what Q/K magnitudes it produces
2. **Check weight loading**: Verify QKV weight matrices are loaded correctly
3. **Check for missing normalization**: See if there's a normalization step we're missing

---

## 📊 Current Status

- ✅ Model file is valid (llama.cpp works perfectly)
- ✅ Q vector loads correctly (all 64 dimensions)
- ✅ K/V cache indexing appears correct
- ❌ Attention scores are too large
- ❌ Model still generates repetitive token 78138

---

## 🎯 Root Cause Theory

The attention scores being too large causes softmax to saturate:
```
exp(125) / (exp(125) + exp(124)) ≈ 0.73  // Should be more balanced
```

This means the model always strongly attends to one position, ignoring context.

**Why token 78138?** 
- With broken attention, the model can't use context
- It falls back to some default behavior
- Token 78138 happens to be the highest probability token in that state

---

## 📝 Files Modified

1. `cuda/kernels/gqa_attention.cu` - Fixed Q loading, changed to shared memory
2. `cuda/src/transformer/qwen_transformer.cpp` - Added KV cache debugging

---

## 🔬 Debug Commands

```bash
# Rebuild
cargo build --features cuda --release

# Run test with debug output
cargo test --test haiku_generation_anti_cheat --features cuda --release -- --nocapture --ignored 2>&1 | tee debug_q_fixed.log

# Check Q values
grep "Q\[0:5\]" debug_q_fixed.log

# Check attention scores
grep "Raw scores" debug_q_fixed.log
```

---

**Next Action**: Investigate why Q/K values are so large, especially after RoPE.
