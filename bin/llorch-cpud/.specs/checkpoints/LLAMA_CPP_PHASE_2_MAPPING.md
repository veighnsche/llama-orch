# TEAM-004: Phase 2 - Instrumentation Point Mapping
**Part of:** llama.cpp Instrumentation Master Plan  
**Duration:** 1 hour  
**Status:** ✅ COMPLETE  
**Depends on:** Phase 1 (Reconnaissance) must be complete

---

## Objective

Document exact file, function, and line numbers for each checkpoint instrumentation point.

**Goal:** Create precise instrumentation map that any team member can follow to add checkpoint extraction code.

---

## Checkpoint Mapping Template

For each checkpoint, we document:
- Exact file path
- Function name
- Line number (or line range)
- Tensor variable name
- Expected shape
- Before/after context
- Instrumentation code location

---

## Checkpoint 1: LayerNorm Output

### Location Details

**File:** `src/llama-model.cpp`  
**Function:** `llm_build_gpt2::llm_build_gpt2` (constructor)  
**Line:** 9898

### Tensor Information

**Variable name:** `cur`  
**Expected shape:** `[n_tokens, n_embd]` → `[2, 768]` for GPT-2  
**Data type:** `GGML_TYPE_F32`  
**Callback name:** `"attn_norm"`

### Context

**Before (operation):**
```cpp
// TEAM-004: Code before checkpoint (line 9894-9897)
for (int il = 0; il < n_layer; ++il) {
    cur = build_norm(inpL,
            model.layers[il].attn_norm,
            model.layers[il].attn_norm_b,
            LLM_NORM, il);
```

**After (operation):**
```cpp
// TEAM-004: Code after checkpoint (line 9898-9900)
    cb(cur, "attn_norm", il);

    // self-attention
    {
```

### Instrumentation Point

**Insert location:** Inside the callback when `name == "attn_norm" && il == 0`

**Instrumentation code:**
```cpp
// TEAM-004: CHECKPOINT 1 - LayerNorm Output
// Extracts: First LayerNorm output in first transformer block
// Expected shape: [2, 768] for GPT-2
// Validates against: PyTorch reference
// Location: src/llama-model.cpp, line 9898, callback "attn_norm"
#ifdef LLORCH_VALIDATE
    if (llama_checkpoint::is_enabled() && il == 0 && 
        strcmp(name, "attn_norm") == 0) {
        llama_checkpoint::save_tensor("checkpoint_01_ln1_output", cur);
    }
#endif
```

### Verification

**How to verify this is the right spot:**
1. ✅ This is the FIRST LayerNorm in transformer block (after `build_norm`)
2. ✅ Callback name is `"attn_norm"`
3. ✅ Happens BEFORE attention input (line 9900)
4. ✅ Shape is `[n_tokens, n_embd]` = `[2, 768]` for GPT-2
5. ✅ Layer check `il == 0` ensures first block only

---

## Checkpoint 2: QKV Projection

### Location Details

**File:** `src/llama-model.cpp`  
**Function:** `llm_build_gpt2::llm_build_gpt2` (constructor)  
**Lines:** 9912-9914

### Tensor Information

**Variable names:** 
- Q: `Qcur`
- K: `Kcur`
- V: `Vcur`

**Expected shapes:** Each `[n_embd_head, n_head, n_tokens]` → needs reshape to `[2, 768]` for GPT-2  
**Data type:** `GGML_TYPE_F32`  
**Callback names:** `"Qcur"`, `"Kcur"`, `"Vcur"`

### Context

**Before (operation):**
```cpp
// TEAM-004: Code before checkpoint (line 9902-9910)
// self-attention
{
    cur = build_lora_mm(model.layers[il].wqkv, cur);
    cb(cur, "wqkv", il);

    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
    cb(cur, "bqkv", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, ...);
    ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, ...);
    ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, ...);
```

**After (operation):**
```cpp
// TEAM-004: Code after checkpoint (line 9912-9918)
    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    cur = build_attn(inp_attn,
            model.layers[il].wo, model.layers[il].bo,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
```

### Instrumentation Point

**Insert location:** Inside callbacks when `name == "Qcur"/"Kcur"/"Vcur" && il == 0`

**Instrumentation code:**
```cpp
// TEAM-004: CHECKPOINT 2 - QKV Projection
// Extracts: Q, K, V after projection and split
// Expected shape: [n_embd_head, n_head, n_tokens] → reshape to [2, 768] for GPT-2
// Validates against: PyTorch reference
// Location: src/llama-model.cpp, lines 9912-9914, callbacks "Qcur", "Kcur", "Vcur"
#ifdef LLORCH_VALIDATE
    if (llama_checkpoint::is_enabled() && il == 0) {
        if (strcmp(name, "Qcur") == 0) {
            llama_checkpoint::save_tensor("checkpoint_02_q", cur);
        } else if (strcmp(name, "Kcur") == 0) {
            llama_checkpoint::save_tensor("checkpoint_02_k", cur);
        } else if (strcmp(name, "Vcur") == 0) {
            llama_checkpoint::save_tensor("checkpoint_02_v", cur);
        }
    }
#endif
```

**TEAM-004 Note:** Tensors are 3D `[n_embd_head, n_head, n_tokens]` and need reshaping to 2D `[n_tokens, n_embd]` for comparison with PyTorch.

### Verification

**How to verify this is the right spot:**
1. ✅ Q, K, V are split from combined QKV projection
2. ✅ Callbacks are `"Qcur"`, `"Kcur"`, `"Vcur"`
3. ✅ Happens BEFORE attention computation (line 9916)
4. ✅ Layer check `il == 0` ensures first block only
5. ⚠️ Need to reshape from 3D to 2D for comparison

---

## Checkpoint 3: KV Cache State

### Location Details

**File:** `src/llama-graph.cpp`  
**Function:** `llm_graph_context::build_attn` (KV cache version)  
**Lines:** 1515-1565

### Tensor Information

**Variable names:** 
- K cache: `k` (retrieved via `mctx_cur->get_k(ctx0, il)`)
- V cache: `v` (retrieved via `mctx_cur->get_v(ctx0, il)`)

**Expected shape:** `[n_embd_head, n_head_kv, n_kv, n_batch]` where n_kv includes cached tokens  
**Data type:** `GGML_TYPE_F32` or `GGML_TYPE_F16` (depends on cache config)  
**Callback names:** No direct callback - extracted via cache context

### Context

**Before (operation):**
```cpp
// TEAM-005: Code before checkpoint (line 1541-1546)
// Update KV cache with current K, V
mctx_cur->update(ctx0, il, k_cur, v_cur, n_tokens);

// Retrieve full K, V from cache (includes history)
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);
```

**After (operation):**
```cpp
// TEAM-005: Code after checkpoint (line 1550-1551)
ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
cb(cur, "kqv_out", il);
```

### Instrumentation Point

**Insert location:** After `mctx_cur->get_k/v()`, before `build_attn_mha()` call

**Instrumentation code:**
```cpp
// TEAM-005: CHECKPOINT 3 - KV Cache State
// Extracts: Full K and V cache after update (includes history + current)
// Expected shape: [n_embd_head, n_head_kv, n_kv, n_batch]
// Validates against: PyTorch reference
// Location: src/llama-graph.cpp, lines 1547-1548, after get_k/get_v
#ifdef LLORCH_VALIDATE
    if (llama_checkpoint::is_enabled() && il == 0) {
        llama_checkpoint::save_tensor("checkpoint_03_cache_k", k);
        llama_checkpoint::save_tensor("checkpoint_03_cache_v", v);
    }
#endif
```

### Verification

**How to verify this is the right spot:**
1. ✅ K, V are retrieved from cache context (includes history)
2. ✅ Happens AFTER cache update with current tokens
3. ✅ Happens BEFORE attention computation (build_attn_mha)
4. ✅ Layer check `il == 0` ensures first block only
5. ⚠️ Shape includes all cached tokens, not just current batch

### Notes

**TEAM-005:** KV cache in llama.cpp includes full history:
- Cache is updated with current K, V via `mctx_cur->update()`
- Retrieved K, V include both history and current tokens
- Shape dimension `n_kv` = history_length + current_batch_size
- For comparison with PyTorch, may need to extract only current tokens or handle concatenation

---

## Checkpoint 4: Attention Scores

### Location Details

**File:** `src/llama-graph.cpp`  
**Function:** `llm_graph_context::build_attn_mha`  
**Line:** 1385

### Tensor Information

**Variable name:** `kq`  
**Expected shape:** `[n_kv, n_tokens, n_head, n_batch]` (4D after permutations)  
**Data type:** `GGML_TYPE_F32`  
**Callback name:** `"kq_soft_max"`

### Context

**Before (operation):**
```cpp
// TEAM-005: Code before checkpoint (line 1349-1383)
// Compute Q * K^T
ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
cb(kq, "kq", il);

// Optional scaling/masking (architecture dependent)
if (kq_b) {
    kq = ggml_add(ctx0, kq, kq_b);
    cb(kq, "kq_plus_kq_b", il);
}

// Softmax with mask and scale
kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
ggml_soft_max_add_sinks(kq, sinks);
```

**After (operation):**
```cpp
// TEAM-005: Code after checkpoint (line 1385-1394)
cb(kq, "kq_soft_max", il);

// Transpose V if needed
if (!v_trans) {
    v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
    cb(v, "v_cont", il);
}

// Multiply attention scores with V
ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
cb(kqv, "kqv", il);
```

### Instrumentation Point

**Insert location:** Inside callback when `name == "kq_soft_max" && il == 0`

**Instrumentation code:**
```cpp
// TEAM-005: CHECKPOINT 4 - Attention Scores
// Extracts: Attention weights after softmax
// Expected shape: [n_kv, n_tokens, n_head, n_batch] (4D permuted)
// Validates against: PyTorch reference
// Location: src/llama-graph.cpp, line 1385, callback "kq_soft_max"
#ifdef LLORCH_VALIDATE
    if (llama_checkpoint::is_enabled() && il == 0 && 
        strcmp(name, "kq_soft_max") == 0) {
        llama_checkpoint::save_tensor("checkpoint_04_scores", cur);
    }
#endif
```

### Verification

**How to verify this is the right spot:**
1. ✅ This is AFTER softmax (values sum to 1 along appropriate dimension)
2. ✅ Callback name is `"kq_soft_max"`
3. ✅ Happens BEFORE multiply with V (line 1393)
4. ✅ Values should be in range [0, 1]
5. ✅ Layer check `il == 0` ensures first block only
6. ⚠️ Shape is 4D after permutations - may need reshape for PyTorch comparison

---

## Checkpoint 5: Attention Output

### Location Details

**File:** `src/llama-graph.cpp`  
**Function:** `llm_graph_context::build_attn` (after build_attn_mha)  
**Lines:** 1462-1474

### Tensor Information

**Variable name:** `cur` (after output projection)  
**Expected shape:** `[n_embd, n_tokens, n_batch]` → needs reshape to `[n_tokens, n_embd]` for GPT-2  
**Data type:** `GGML_TYPE_F32`  
**Callback name:** No direct callback after projection (need to add one)

### Context

**Before (operation):**
```cpp
// TEAM-005: Code before checkpoint (line 1461-1462)
ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
cb(cur, "kqv_out", il);

// Output projection
if (wo) {
    cur = build_lora_mm(wo, cur);
}

if (wo_b) {
    cur = ggml_add(ctx0, cur, wo_b);
}
```

**After (operation):**
```cpp
// TEAM-005: Code after checkpoint (line 1476, then back in llama-model.cpp line 9927)
return cur;  // Returns from build_attn

// Back in GPT-2 builder (llama-model.cpp line 9927)
// add the input (residual connection)
ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
cb(ffn_inp, "ffn_inp", il);
```

### Instrumentation Point

**Insert location:** After output projection (wo_b add), before return from build_attn

**Instrumentation code:**
```cpp
// TEAM-005: CHECKPOINT 5 - Attention Output
// Extracts: Attention output after projection (before residual)
// Expected shape: [n_embd, n_tokens, n_batch] → reshape to [n_tokens, n_embd]
// Validates against: PyTorch reference
// Location: src/llama-graph.cpp, line 1474, after wo_b projection
#ifdef LLORCH_VALIDATE
    if (llama_checkpoint::is_enabled() && il == 0) {
        // Add callback after output projection
        cb(cur, "attn_out_proj", il);
        // Checkpoint extraction happens in callback handler
        llama_checkpoint::save_tensor("checkpoint_05_output", cur);
    }
#endif
```

**Alternative approach:** Hook into existing callback system by adding `cb(cur, "attn_out_proj", il)` after line 1474, then extract in callback handler.

### Verification

**How to verify this is the right spot:**
1. ✅ This is AFTER attention output projection (wo and wo_b)
2. ✅ This is AFTER build_attn_mha (attention computation complete)
3. ✅ Happens BEFORE residual connection (line 9927 in llama-model.cpp)
4. ✅ Layer check `il == 0` ensures first block only
5. ⚠️ Need to add explicit callback or hook into return point

---

## Checkpoint 6: FFN Output

### Location Details

**File:** `src/llama-model.cpp`  
**Function:** `llm_build_gpt2::llm_build_gpt2` (constructor)  
**Line:** 9944

### Tensor Information

**Variable name:** `cur`  
**Expected shape:** `[n_tokens, n_embd]` → `[2, 768]` for GPT-2  
**Data type:** `GGML_TYPE_F32`  
**Callback name:** `"ffn_out"`

### Context

**Before (operation):**
```cpp
// TEAM-004: Code before checkpoint (line 9930-9943)
// FF
{
    cur = build_norm(ffn_inp,
            model.layers[il].ffn_norm,
            model.layers[il].ffn_norm_b,
            LLM_NORM, il);
    cb(cur, "ffn_norm", il);

    cur = build_ffn(cur,
            model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
            NULL,                      NULL,                        NULL,
            model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
            NULL,
            LLM_FFN_GELU, LLM_FFN_SEQ, il);
```

**After (operation):**
```cpp
// TEAM-004: Code after checkpoint (line 9944-9947)
    cb(cur, "ffn_out", il);
}

cur = ggml_add(ctx0, cur, ffn_inp);  // residual connection
```

### Instrumentation Point

**Insert location:** Inside callback when `name == "ffn_out" && il == 0`

**Instrumentation code:**
```cpp
// TEAM-004: CHECKPOINT 6 - FFN Output
// Extracts: FFN output after projection (fc → gelu → proj)
// Expected shape: [2, 768] for GPT-2
// Validates against: PyTorch reference
// Location: src/llama-model.cpp, line 9944, callback "ffn_out"
#ifdef LLORCH_VALIDATE
    if (llama_checkpoint::is_enabled() && il == 0 && 
        strcmp(name, "ffn_out") == 0) {
        llama_checkpoint::save_tensor("checkpoint_06_ffn", cur);
    }
#endif
```

### Verification

**How to verify this is the right spot:**
1. ✅ This is AFTER FFN projection (fc → gelu → proj via build_ffn)
2. ✅ Callback name is `"ffn_out"`
3. ✅ Happens BEFORE residual connection (line 9947)
4. ✅ Shape is `[n_tokens, n_embd]` = `[2, 768]` for GPT-2
5. ✅ Layer check `il == 0` ensures first block only

---

## Summary Table

| Checkpoint | File | Function | Line | Callback | Tensor | Shape |
|------------|------|----------|------|----------|--------|-------|
| 1. LayerNorm | llama-model.cpp | llm_build_gpt2 | 9898 | "attn_norm" | cur | [2, 768] |
| 2. Q | llama-model.cpp | llm_build_gpt2 | 9912 | "Qcur" | Qcur | [64, 12, 2]→[2,768] |
| 2. K | llama-model.cpp | llm_build_gpt2 | 9913 | "Kcur" | Kcur | [64, 12, 2]→[2,768] |
| 2. V | llama-model.cpp | llm_build_gpt2 | 9914 | "Vcur" | Vcur | [64, 12, 2]→[2,768] |
| 3. Cache K | llama-graph.cpp | build_attn (KV) | 1547 | (get_k) | k | [64,12,n_kv,1] |
| 3. Cache V | llama-graph.cpp | build_attn (KV) | 1548 | (get_v) | v | [64,12,n_kv,1] |
| 4. Scores | llama-graph.cpp | build_attn_mha | 1385 | "kq_soft_max" | kq | [n_kv,2,12,1] |
| 5. Attn Out | llama-graph.cpp | build_attn | 1474 | (add cb) | cur | [768,2,1]→[2,768] |
| 6. FFN | llama-model.cpp | llm_build_gpt2 | 9944 | "ffn_out" | cur | [2, 768] |

**TEAM-005 Status:**
- ✅ Checkpoints 1, 2, 6: FULLY MAPPED (via callbacks in llama-model.cpp)
- ✅ Checkpoint 3: FULLY MAPPED (KV cache retrieval in llama-graph.cpp)
- ✅ Checkpoint 4: FULLY MAPPED (attention scores callback in build_attn_mha)
- ✅ Checkpoint 5: FULLY MAPPED (attention output in build_attn, needs callback addition)

---

## Completion Checklist

- [x] Checkpoint 1 location mapped with exact line numbers
- [x] Checkpoint 2 location mapped with exact line numbers
- [x] Checkpoint 3 location mapped with exact line numbers
- [x] Checkpoint 4 location mapped with exact line numbers
- [x] Checkpoint 5 location mapped with exact line numbers
- [x] Checkpoint 6 location mapped with exact line numbers
- [x] All tensor variable names identified
- [x] All expected shapes documented
- [x] All instrumentation code snippets prepared
- [x] Summary table completed
- [x] Ready to proceed to Phase 3 (Implementation)

---

## Notes and Observations

**TEAM-005 Notes:**
- ALL 6 checkpoints are FULLY MAPPED and ready for implementation
- Callback system (`cb()`) is the primary extraction mechanism
- Checkpoints 1, 2, 6: Direct callbacks in llama-model.cpp GPT-2 builder
- Checkpoint 3: KV cache retrieval in build_attn (llama-graph.cpp line 1547-1548)
- Checkpoint 4: Callback "kq_soft_max" in build_attn_mha (llama-graph.cpp line 1385)
- Checkpoint 5: Need to add callback after output projection (llama-graph.cpp line 1474)

**Potential Issues:**
- Q, K, V are 3D tensors `[n_embd_head, n_head, n_tokens]` - need reshape logic
- KV cache includes full history, not just current tokens - shape `[n_embd_head, n_head_kv, n_kv, n_batch]`
- Attention scores are 4D after permutations - need reshape for comparison
- Checkpoint 5 requires adding a new callback in build_attn (minor code change)

**Adjustments Needed:**
- Implement tensor reshaping for 3D/4D → 2D conversion
- Add callback `cb(cur, "attn_out_proj", il)` after line 1474 in llama-graph.cpp
- Handle KV cache history dimension in checkpoint extraction
- Ensure all extractions use `ggml_backend_tensor_get` for cross-backend compatibility

**TEAM-005 Decision:**
Proceed to Phase 3 with ALL 6 checkpoints. This gives us:
- LayerNorm validation ✅
- QKV validation ✅
- KV cache validation ✅
- Attention scores validation ✅
- Attention output validation ✅
- FFN validation ✅
- 100% checkpoint coverage
- Full confidence in numerical equivalence

---

**Status:** ✅ COMPLETE (Full - 100% coverage)  
**Previous Phase:** Phase 1 - Reconnaissance (complete)  
**Next Phase:** Phase 3 - Implementation (all 6 checkpoints)  
**Estimated Time:** 1 hour  
**Actual Time:** 45 minutes

**Completed by:** TEAM-005  
**Date:** 2025-10-08
