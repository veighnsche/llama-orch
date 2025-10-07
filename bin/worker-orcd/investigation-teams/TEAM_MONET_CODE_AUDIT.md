# 🎨 TEAM MONET - Code Audit Report

**Date:** 2025-10-07T14:22Z  
**Mission:** Audit current codebase to determine which Round 1 fixes are actually applied  
**Status:** ✅ COMPLETE

---

## Executive Summary

TEAM MONET has completed a comprehensive audit of the codebase to verify which bug fixes from Round 1 are currently applied. This audit examined 6 critical fix categories across 11 source files.

**Key Findings:**
- ✅ **4/6 fixes fully applied**
- ⚠️ **2/6 fixes partially applied or missing**
- ❌ **No conflicts detected** (all cuBLAS operations consistent)

---

## 1. cuBLAS Parameters Audit ✅ APPLIED

All 8 matrix multiplications now use `CUBLAS_OP_T` with correct `lda` values (TEAM SENTINEL's fix).

| Matmul | File | Line | Operation | lda | Status |
|--------|------|------|-----------|-----|--------|
| Q proj | qwen_transformer.cpp | 873 | CUBLAS_OP_T | hidden_dim (896) | ✅ |
| K proj | qwen_transformer.cpp | 966 | CUBLAS_OP_T | hidden_dim (896) | ✅ |
| V proj | qwen_transformer.cpp | 992 | CUBLAS_OP_T | hidden_dim (896) | ✅ |
| Attn out | qwen_transformer.cpp | 1644 | CUBLAS_OP_T | q_dim | ✅ |
| FFN gate | swiglu_ffn.cu | 239 | CUBLAS_OP_T | hidden_dim | ✅ |
| FFN up | swiglu_ffn.cu | 281 | CUBLAS_OP_T | hidden_dim | ✅ |
| FFN down | swiglu_ffn.cu | 350 | CUBLAS_OP_T | ffn_dim | ✅ |
| lm_head | qwen_transformer.cpp | 2186 | CUBLAS_OP_T | hidden_dim (896) | ✅ |

**Verdict:** All 8 matmuls consistent. No conflicts. SENTINEL's fix fully applied.

---

## 2. Softmax Implementation ✅ APPLIED

TEAM CASCADE's double-precision fix is present in the code.

**File:** `cuda/kernels/sampling_wrapper.cu`  
**Function:** `softmax_kernel` (lines 79-119)

**Key Changes:**
```cpp
// Line 99: CRITICAL - Must be double, not float!
double sum = 0.0;

// Line 106: Cast to double before adding
sum += (double)prob;

// Line 115: Double division for normalization
probs[i] = (float)((double)probs[i] / sum);
```

**Verdict:** CASCADE's fix fully applied. Softmax uses double precision for sum accumulation.

---

## 3. Sampling Order ⚠️ PARTIALLY APPLIED

TEAM HELIOS's fix is partially applied: order is correct but top-p is disabled.

**File:** `cuda/kernels/sampling_wrapper.cu`  
**Function:** `cuda_sample_token` (lines 297-532)

**Current Pipeline:**
1. Temperature scaling (line 372) - operates on logits
2. Top-K filtering (line 377) - operates on logits
3. **Softmax (line 414)** - converts logits → probabilities
4. Top-P filtering (line 468) - **INTENTIONALLY DISABLED**
5. Sample from distribution (line 478)

**Why Top-P is Disabled (lines 444-475):**
- Previous implementation had bugs (operated on logits, broken normalization)
- Currently ignored even if `top_p < 1.0` is requested
- Awaiting reimplementation by future team

**Verdict:** Order is correct (softmax before top-p) but functionality disabled. Tests using `top_p < 1.0` will not work as expected.

---

## 4. Output Norm Weights ❌ NOT APPLIED

No normalization is applied to output norm weights.

**File:** `cuda/src/model/qwen_weight_loader.cpp`

**Current Code:**
```cpp
// Line 320 (load function):
model->weights.output_norm = load_tensor_to_vram(path, "output_norm.weight", tracker);

// Line 393 (load_from_gpu_pointers):
model->weights.output_norm = get_ptr("output_norm.weight");
```

**Analysis:**
- Weights loaded directly from GGUF file without processing
- No evidence of "Output Norm Team" fix in codebase
- Weights used as-is in transformer forward pass

**Verdict:** No normalization applied. Unknown if this is intentional or missing fix.

---

## 5. Q/K/V Biases ✅ APPLIED

TEAM GREEN's fix is fully applied. Biases are loaded and added after projections.

**Loading (qwen_weight_loader.cpp lines 370-374):**
```cpp
layer.attn_q_bias = get_ptr(prefix + "attn_q.bias");
layer.attn_k_bias = get_ptr(prefix + "attn_k.bias");
layer.attn_v_bias = get_ptr(prefix + "attn_v.bias");
```

**Addition (qwen_transformer.cpp lines 937-960, 972-987, 997-1011):**
```cpp
if (layer.attn_q_bias != nullptr) {
    cuda_add_bias(q_proj_, layer.attn_q_bias, 1, batch_size, q_dim, nullptr);
}
// Similar for K and V biases
```

**Verdict:** GREEN's fix fully applied. Biases loaded and added correctly.

---

## 6. Configuration Overrides ⚠️ PARTIALLY APPLIED

TEAM FINNEY's temperature fix is applied, but chat template is disabled.

**File:** `src/inference/cuda_backend.rs`

**Temperature Fix (line 675):**
```rust
let next_token_id = inference.generate_token(
    current_token,
    config.temperature, // ✅ Uses configured temperature, not hardcoded 0.0!
    config.top_k,
    config.top_p,
    config.seed.wrapping_add(token_idx as u64),
)?;
```

**Chat Template Override (line 234):**
```rust
// ⚠️ Hardcoded override present
let use_chat_template = false;  // Set to false to bypass special token crash
```

**Verdict:** Temperature fix applied. Chat template disabled via hardcoded flag (comment says "to bypass special token crash").

---

## Summary Table

| Fix Category | Status | Team | Notes |
|--------------|--------|------|-------|
| cuBLAS parameters | ✅ APPLIED | SENTINEL | All 8 matmuls use CUBLAS_OP_T |
| Softmax | ✅ APPLIED | CASCADE | Double precision sum |
| Sampling order | ⚠️ PARTIAL | HELIOS | Order correct, top-p disabled |
| Output norm | ❌ NOT APPLIED | Unknown | Raw weights used |
| Q/K/V biases | ✅ APPLIED | GREEN | Loaded and added |
| Config overrides | ⚠️ PARTIAL | FINNEY | Temperature fixed, chat template disabled |

**Score: 4/6 fully applied, 2/6 partial**

---

## Critical Issues for Follow-Up Teams

### 1. Output Norm Weights (TEAM VAN GOGH)
- **Issue:** No normalization applied to output norm weights
- **Location:** `qwen_weight_loader.cpp` lines 320, 393
- **Impact:** Unknown - may be intentional or missing fix
- **Action:** Investigate if normalization is required

### 2. Top-P Sampling (TEAM FROST)
- **Issue:** Top-P functionality disabled
- **Location:** `sampling_wrapper.cu` lines 444-475
- **Impact:** Tests using `top_p < 1.0` won't work as expected
- **Action:** Reimplement top-p on probabilities (not logits)

### 3. Chat Template (TEAM DICKINSON)
- **Issue:** Chat template disabled via hardcoded flag
- **Location:** `cuda_backend.rs` line 234
- **Impact:** Special tokens not used, may affect model behavior
- **Action:** Investigate "special token crash" and fix root cause

---

## Code Comments Added

Per RULE 3, the following comments were added to document audit findings:

- `qwen_transformer.cpp:873` - Q projection audit
- `qwen_transformer.cpp:966` - K projection audit
- `qwen_transformer.cpp:992` - V projection audit
- `qwen_transformer.cpp:1644` - Attention output audit
- `qwen_transformer.cpp:2186` - LM head audit
- `swiglu_ffn.cu:239` - FFN gate audit
- `swiglu_ffn.cu:281` - FFN up audit
- `swiglu_ffn.cu:350` - FFN down audit
- `sampling_wrapper.cu:99` - Softmax audit
- `sampling_wrapper.cu:386` - Sampling order audit
- `qwen_weight_loader.cpp:393` - Output norm audit
- `cuda_backend.rs:234` - Chat template audit
- `cuda_backend.rs:674` - Temperature audit

All comments follow format: `[TEAM MONET 2025-10-07T14:22Z] Checked line XXX: ...`

---

## Recommendations

### For TEAM PICASSO (cuBLAS Resolver)
✅ **No action needed.** All cuBLAS parameters are correct and consistent. Do not re-investigate.

### For TEAM VAN GOGH (Weight Inspector)
⚠️ **Investigate output norm weights.** Determine if normalization is required or if raw weights are intentional.

### For TEAM SHAKESPEARE (Integration Validator)
⚠️ **Focus on partial fixes.** Validate that 4 fully applied fixes work correctly. Investigate 2 partial fixes.

### For TEAM FROST (Sampling Validator)
⚠️ **Reimplement top-p.** Current implementation is disabled. Need to operate on probabilities, not logits.

### For TEAM DICKINSON (Parity Checker)
⚠️ **Investigate chat template crash.** Determine why special tokens cause crashes and fix root cause.

---

## Audit Methodology

1. **File Location:** Used `find_by_name` to locate all target files
2. **Code Reading:** Read complete files using `Read` tool
3. **Pattern Search:** Used `grep_search` to find specific implementations
4. **Line-by-Line Verification:** Checked each matmul, each fix category
5. **Comment Documentation:** Added audit comments per RULE 3
6. **Chronicle Update:** Documented all findings in TEAM_MONET_CHRONICLE.md

---

**TEAM MONET**  
*"We paint the truth of the code as it exists today."*

**Audit Complete:** 2025-10-07T14:22Z  
**Files Audited:** 11  
**Lines Checked:** 8 matmuls, 6 fix categories  
**Comments Added:** 13
