# Team GREEN - Partial Fix Applied

**Date:** 2025-10-06 20:49 UTC  
**Status:** ⚠️ **PARTIAL FIX - BUG STILL PRESENT**

---

## ✅ What I Fixed

### Bug Found: Missing Q/K/V Bias Addition

**Root Cause:**
The model HAS biases for Q, K, and V projections, but we were:
1. Setting them to `nullptr` in the weight loader
2. Not adding them after the projections

**Evidence:**
```
Test output showed:
   - blk.0.attn_q.bias -> 0x7dfbaa5c1200
   - blk.0.attn_k.bias -> 0x7dfbaa401000
   - blk.0.attn_v.bias -> 0x7dfbaa5c1a00
```

But our code in `qwen_weight_loader.cpp` line 352-356 was:
```cpp
layer.attn_q_bias = nullptr;  // Qwen2.5 doesn't use biases
layer.attn_k_bias = nullptr;  // Qwen2.5 doesn't use biases
layer.attn_v_bias = nullptr;  // Qwen2.5 doesn't use biases
```

This was WRONG! The model DOES have biases.

**Fix Applied:**

1. **File:** `cuda/src/model/qwen_weight_loader.cpp` (lines 352-360)
   ```cpp
   layer.attn_q_bias = get_ptr(prefix + "attn_q.bias");
   layer.attn_k_bias = get_ptr(prefix + "attn_k.bias");
   layer.attn_v_bias = get_ptr(prefix + "attn_v.bias");
   ```

2. **File:** `cuda/src/transformer/qwen_transformer.cpp` (lines 269-290)
   ```cpp
   // After Q projection
   if (layer.attn_q_bias != nullptr) {
       cuda_add_bias(q_proj_, layer.attn_q_bias, 1, batch_size, q_dim, nullptr);
   }
   
   // After K projection
   if (layer.attn_k_bias != nullptr) {
       cuda_add_bias(k_proj_, layer.attn_k_bias, 1, batch_size, kv_dim, nullptr);
   }
   
   // After V projection
   if (layer.attn_v_bias != nullptr) {
       cuda_add_bias(v_proj_, layer.attn_v_bias, 1, batch_size, kv_dim, nullptr);
   }
   ```

3. **File:** `cuda/kernels/gpt_ffn.cu` (line 41)
   ```cpp
   extern "C" void cuda_add_bias(...)  // Added extern "C"
   ```

4. **File:** `cuda/src/transformer/qwen_transformer.cpp` (lines 114-121)
   ```cpp
   void cuda_add_bias(...);  // Added forward declaration
   ```

**Verification:** llama.cpp DOES add these biases (see llama-model.cpp lines 6632-6648)

---

## ❌ What's Still Broken

### Test Result: STILL GENERATES GARBAGE

**Output (2025-10-06 20:49 UTC):**
```
çĤĬĠmilitĠmilitĠscarcityå¯¹å¤ĸå¼ĢæĶ¾Ġà¸ĺà¸±à¸Ļà¸§à¸²Ġà¸ĺà¸±à¸Ļà¸§à¸²ĠskeĠconciseçĶ·ç¥ŀ...
```

**Symptoms (UNCHANGED):**
- Mojibake (Chinese/Thai tokens)
- Repetitive tokens:
  - "Ġmilit" appears 2 times
  - "ĠÐ»ÐµÑĩ" appears 10+ times
  - "Ġconcise" appears 3 times
  - "Ġscarcity" appears 3 times
  - "èįĥ" appears 5+ times

**Conclusion:** The biases were a bug that needed fixing, but they were NOT the root cause of the garbage output.

---

## 🔍 Next Investigation Steps

The bug is STILL in the forward pass. Since biases didn't fix it, focus on:

### Priority 1: Check Bias Values
Maybe the biases themselves are corrupted or wrong?
```cpp
// Add debug logging
if (layer.attn_q_bias != nullptr && pos == 0) {
    half h_bias[10];
    cudaMemcpy(h_bias, layer.attn_q_bias, 10 * sizeof(half), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[GREEN] Q bias[0..9]: ");
    for (int i = 0; i < 10; i++) {
        fprintf(stderr, "%.4f ", __half2float(h_bias[i]));
    }
    fprintf(stderr, "\n");
}
```

### Priority 2: Compare Q/K/V Values with llama.cpp
After adding biases, are the Q/K/V values correct?
```cpp
// After bias addition
if (pos == 0) {
    half h_q[10];
    cudaMemcpy(h_q, q_proj_, 10 * sizeof(half), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[GREEN] Q after bias[0..9]: ");
    for (int i = 0; i < 10; i++) {
        fprintf(stderr, "%.4f ", __half2float(h_q[i]));
    }
    fprintf(stderr, "\n");
}
```

### Priority 3: Check Attention Mask
Maybe the causal mask is wrong?

### Priority 4: Check Final Projection
Maybe the issue is in the final logits projection?

---

## 📊 Test Command

```bash
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1
```

**Current Result:** ❌ FAIL - Still generates mojibake and repetitive tokens

---

## 📝 Files Modified

1. `cuda/src/model/qwen_weight_loader.cpp` (lines 352-360)
2. `cuda/src/transformer/qwen_transformer.cpp` (lines 114-121, 269-290)
3. `cuda/kernels/gpt_ffn.cu` (line 41)

---

## 🔑 Key Insight

**The biases were a real bug** - we were ignoring them when the model has them and llama.cpp uses them.

**But they weren't THE bug** - fixing them didn't resolve the garbage output.

**The root cause is still in the forward pass** - something else is corrupting the logits.

---

**Team GREEN 🌿**  
*"Found one bug, but the hunt continues"*
