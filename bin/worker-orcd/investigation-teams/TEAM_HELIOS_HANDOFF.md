# 🛰️ TEAM HELIOS → Next Team Handoff

**Date:** 2025-10-08  
**Status:** Sampling architecture fixed, bug isolated to transformer forward pass

---

## ✅ What We Fixed

### Sampling Pipeline Order (Critical Bug)
- **File**: `cuda/kernels/sampling_wrapper.cu` lines 251-340
- **Fix**: Moved softmax BEFORE top-p (was: temp → top-k → top-p → softmax → sample)
- **Now**: temp → top-k → **softmax** → top-p → sample
- **Why**: Top-p operates on probabilities (cumulative probability mass), not logits
- **Evidence**: llama.cpp does softmax before top-p (src/llama-sampling.cpp:783)

### Top-P Temporarily Disabled
- **File**: `cuda/kernels/sampling_wrapper.cu` lines 305-337
- **Reason**: Old implementation had broken normalization (computed softmax over only 1000 tokens)
- **Impact**: Tests using top_p=1.0 unaffected; tests using top_p<1.0 will be more peaked
- **TODO**: Rewrite launch_top_p() to accept probabilities, not logits (see line 321-326)

### Debug Instrumentation Added
- **File**: `cuda/kernels/sampling_wrapper.cu` lines 347-389
- **Purpose**: Log first 20 generation tokens with temperature/seed/probabilities
- **Limitation**: Uses seed-change heuristic to detect generation phase (brittle)
- **TODO**: Replace with explicit `is_generation` parameter from caller (see line 359-362)

---

## 🔍 What We Confirmed

### Sampling Works Correctly ✅
Evidence from foreground test run (see test output):
```
[HELIOS GEN #00] token=110893, temp=0.70, top_k=0, top_p=1.00, seed=1759794426
[HELIOS GEN #00] First 5 probs: 0.000000 0.000015 0.000000 0.000000 0.000000
```

1. **Temperature**: 0.7 correctly applied (not 0.0 as in prefill)
2. **Seeds**: Increment correctly (1759794426, 1759794427, ...)
3. **Tokens**: Vary across generation (86398, 34462, 71443, 8028, ...)
4. **Probabilities**: Peaked distribution (expected for temp<1.0)

### Model Still Generates Mojibake ❌
Output: `macros-closeĳľĠminimumĳľ(libraryĳľĳľularitytees...`

**Conclusion**: Sampling is NOT the root cause. Bug is upstream.

---

## 🎯 Where the Bug Actually Is

### Evidence
- ✅ Sampling parameters correct (temp, seed, top-k)
- ✅ Probabilities have reasonable shape
- ✅ Tokens are sampled probabilistically (not deterministic)
- ❌ Tokens are semantically wrong (macros, tees, ĳľ, etc.)

### Root Cause Location
The bug is in **transformer forward pass** (attention/FFN/normalization):
- Model computes WRONG logits
- Sampling correctly picks from WRONG distribution
- Result: mojibake output

### What Previous Teams Verified
- ✅ Team AEGIS: All 8 matmuls correct
- ✅ Team AEGIS: Weight loading matches llama.cpp
- ✅ Team AEGIS: UTF-8 decoding correct
- ✅ Team HELIOS: Sampling architecture correct

### What Still Needs Investigation
1. **Attention mechanism**: Are attention scores computed correctly?
2. **RMSNorm**: Verify normalization outputs match llama.cpp
3. **SwiGLU FFN**: Check gate/up projections and activation
4. **Residual connections**: Verify add operations preserve values
5. **Weight application**: Even if weights load correctly, are they applied correctly?

---

## 🧪 Recommended Next Steps

### Step 1: Compare Logits with llama.cpp
**Goal**: Isolate divergence point in forward pass

```bash
# Run llama.cpp with same model + prompt, dump logits
llama-cli -m model.gguf -p "exact prompt" --dump-logits logits_llama.txt

# Run our code with same prompt, dump logits
# (Add logging to qwen_transformer.cpp line 183)

# Compare first divergence
diff logits_llama.txt logits_ours.txt
```

**Expected**: Logits diverge at some layer → that layer has the bug.

### Step 2: Instrument Attention Mechanism
**File**: `cuda/src/transformer/qwen_transformer.cpp`

Add logging after each attention step:
- Q/K/V projections (already has some logging)
- RoPE application
- Attention scores (Q · K^T)
- Softmax over attention scores
- Attention output (scores · V)
- Output projection

Compare each step with llama.cpp using same input.

### Step 3: Verify RMSNorm
**Suspicion**: RMSNorm might have numerical issues

```cpp
// Check if RMSNorm output matches llama.cpp
// File: cuda/src/kernels/rms_norm.cu
// Add debug logging for:
// 1. Input values (first 10)
// 2. RMS value (sqrt of mean of squares)
// 3. Output values (first 10)
// 4. Weight values (first 10)
```

### Step 4: FFN Verification
**File**: `cuda/src/kernels/swiglu.cu`

Verify SwiGLU computation:
```cpp
// SwiGLU formula: (W_gate(x) * silu(W_up(x)))
// Check:
// 1. W_gate and W_up projections separately
// 2. silu activation (x * sigmoid(x))
// 3. Element-wise multiplication
// 4. Final down projection
```

---

## ⚠️ Known Limitations

### What We Cannot Claim
We did NOT perform side-by-side parity test with llama.cpp on identical logits.

**To prove exact parity**, next team should:
1. Create unit test with known logits array
2. Feed to our sampler and llama.cpp sampler
3. Use same seed, verify same token selected
4. Test across temperature ranges (0.0, 0.5, 0.7, 1.0, 1.5)

See `TEAM_HELIOS_FINDINGS.md` lines 313-322 for test template.

### What May Break
1. **Top-p disabled**: If test expects top-p<1.0, output will be more peaked
2. **Logging heuristic**: Seed-based generation detection may fail if caller changes seed logic
3. **Static counters**: Multiple concurrent tests may interfere with each other

---

## 📝 Code Changed

### Files Modified
1. `cuda/kernels/sampling_wrapper.cu`:
   - Lines 251-277: Added comment block explaining fix
   - Lines 289-303: Moved softmax before top-p
   - Lines 305-337: Disabled top-p with detailed TODO
   - Lines 347-389: Added generation-phase logging

### Files Created
1. `investigation-teams/TEAM_HELIOS_FINDINGS.md`: Detailed bug analysis
2. `investigation-teams/TEAM_HELIOS_SUMMARY.md`: Complete investigation report
3. `investigation-teams/TEAM_HELIOS_HANDOFF.md`: This document

### Files to Review
1. `cuda/src/transformer/qwen_transformer.cpp`: Forward pass implementation
2. `cuda/src/kernels/rms_norm.cu`: Normalization
3. `cuda/src/kernels/swiglu.cu`: FFN activation
4. `cuda/src/kernels/attention.cu`: Attention mechanism

---

## 🎓 Key Insights

### Lesson 1: Architecture > Implementation
The bug wasn't in HOW we computed things, but in the ORDER. Even with correct individual functions, wrong sequence produces wrong results.

### Lesson 2: Isolate by Elimination
By proving sampling is correct, we've eliminated a major component. This narrows the search space significantly.

### Lesson 3: Use Reference Implementation
llama.cpp's architecture is battle-tested. When we deviated (top-p before softmax), we introduced bugs. Stick to reference order.

### Lesson 4: Test in Foreground
Running tests in background hides critical information. Always run foreground with full logs for investigative work.

---

## 📊 Test Results Summary

**Test**: `haiku_generation_anti_cheat::test_haiku_generation_stub_pipeline_only`  
**Runtime**: 9.07s  
**Result**: PASSED (pipeline works, output is mojibake)

**Generated Output**:
```
macros-closeĳľĠminimumĳľ(libraryĳľĳľularityteesncyĳľĳľĠoriginallyncy...
```

**Expected**: Haiku containing "forty-six"  
**Actual**: Random code-like tokens, no coherent language

**Conclusion**: Pipeline works end-to-end, but model forward pass produces wrong logits.

---

## 🔗 References

- **llama.cpp sampling**: `reference/llama.cpp/src/llama-sampling.cpp`
- **Previous investigation**: `investigation-teams/TEAM_AEGIS_FINDINGS.md`
- **Test file**: `tests/haiku_generation_anti_cheat.rs`
- **Our fix**: `cuda/kernels/sampling_wrapper.cu` (git diff available)

---

**TEAM_HELIOS**  
**Mission: Pass the baton with clarity**  
**Status: Complete**  
**Date: 2025-10-08**

---

*"Good handoffs enable great investigations."*

---

## [TEAM ORION] 2025-10-06T23:58Z

**OBJECTIVE**: Find first activation divergence between our FP16 forward pass and llama.cpp

**STATUS**: ⚠️ DIVERGENCE FOUND - Q Projection Has Extreme Values

### Parity Trace Results (Layer 0, Tokens 0 & 1)

**Token 0 (POS 0):**
- Input: min=-0.039 max=0.056 mean=0.000 ✅ NORMAL
- After attn RMSNorm: min=-0.576 max=1.038 mean=0.003 ✅ NORMAL
- **After Q proj: min=-16.047 max=14.336 mean=-0.057 ❌ EXTREME VALUES!**
- After K proj: min=-4.645 max=3.166 mean=-0.142 ⚠️ LARGE
- After V proj: min=-0.281 max=0.094 mean=-0.005 ✅ NORMAL

**Token 1 (POS 1):**
- Input: min=-0.051 max=0.042 mean=-0.000 ✅ NORMAL
- After attn RMSNorm: min=-0.542 max=0.425 mean=0.001 ✅ NORMAL
- **After Q proj: min=-3.912 max=3.695 mean=-0.011 ⚠️ LARGE**
- After K proj: min=-1.443 max=0.910 mean=-0.027 ✅ ACCEPTABLE
- After V proj: min=-0.125 max=0.058 mean=-0.002 ✅ NORMAL

### Critical Issue Identified

**Q Projection has abnormally large range (±16)**:
- Expected range for FP16 activations after projection: [-2, 2]
- Observed range for Token 0 Q: [-16.047, 14.336]
- This suggests matmul configuration error (transpose/lda/broadcast)

**Current Q matmul configuration**:
```cpp
cublasGemmEx(..., CUBLAS_OP_T, CUBLAS_OP_N, 
             q_dim, batch_size, hidden_dim,
             layer.attn_q_weight, CUDA_R_16F, hidden_dim,  // lda=896
             normed_half, CUDA_R_16F, hidden_dim,          // ldb=896
             q_half, CUDA_R_16F, q_dim, ...)               // ldc=896
```

**Hypothesis**: CUBLAS_OP_T with lda=hidden_dim may be reading weights incorrectly.

### Extreme Value Location Analysis

**Token 0 Q projection extremes:**
- **min=-16.047 at Q[95]** → head 1, dimension 31
- **max=14.336 at Q[126]** → head 1, dimension 62

🔴 **Critical Finding**: Both extreme values are in HEAD 1 (not distributed across heads)

This concentration suggests:
1. **Weight stride issue**: lda parameter doesn't match actual memory layout
2. **Possible root cause**: For Q projection, lda should be q_dim=896, NOT hidden_dim=896
3. **Why it matters**: Even though both are 896, the semantic meaning differs

### Comparison Across Projections

| Projection | Dimensions | Range | Status |
|------------|------------|-------|---------|
| Q | [896→896] | [-16.0, 14.3] | ❌ EXTREME |
| K | [896→128] | [-4.6, 3.2] | ⚠️ LARGE |
| V | [896→128] | [-0.3, 0.1] | ✅ NORMAL |

**Pattern**: Q (square matrix) has worst divergence, K (rectangular) is borderline, V is normal.

### Root Cause Hypothesis

The Q weight matrix is stored in GGUF as `[hidden_dim, q_dim]` = `[896, 896]` row-major.

**Current code**:
```cpp
cublasGemmEx(..., CUBLAS_OP_T, CUBLAS_OP_N, 
             q_dim, batch_size, hidden_dim,           // M=896, N=1, K=896
             layer.attn_q_weight, ..., hidden_dim,    // A: lda=896 (WRONG?)
             normed_half, ..., hidden_dim,            // B: ldb=896
             q_half, ..., q_dim, ...)                 // C: ldc=896
```

**Issue**: With CUBLAS_OP_T, we're saying "transpose matrix A". But what IS the leading dimension of A?
- If A is stored row-major [896, 896], lda should be **896** (number of columns)
- But we're passing lda=hidden_dim=896, which HAPPENS to be the same value
- However, cuBLAS interprets this differently based on the operation flag

**Proposed fix**: Check if lda should be q_dim (the output dimension) instead of hidden_dim (the input dimension) when using CUBLAS_OP_T.

### Next Steps for Investigation

1. **Verify against llama.cpp**: Check their exact cuBLAS parameters for Q projection
2. **Test lda=q_dim**: Change line 346 from lda=hidden_dim to lda=q_dim
3. **Verify with parity logs**: Re-run and check if Q range normalizes to [-2, 2]
4. **Check K projection**: If Q fix works, apply same logic to K (lda=kv_dim)

**FINDINGS UPDATE 2025-10-06T23:58Z:**

Attempted fix by changing lda from hidden_dim to kv_dim for K/V projections → FAILED
- Result: K and V became all zeros
- cuBLAS error: "parameter number 9 had an illegal value" (parameter 9 = lda)
- This confirms lda=hidden_dim IS correct for K/V

**Conclusion**: The extreme Q values (±16) are NOT due to wrong lda parameter.

**New Hypothesis**: Q weight matrix may be stored in a different layout than K/V in GGUF.
- Q is square (896x896) → may use different storage convention
- K/V are rectangular (896x128) → standard storage works
- Need to verify actual GGUF tensor dimensions and compare Q vs K/V storage

**Action Required**: Verify GGUF tensor dimensions and storage order:
1. Check if blk.0.attn_q.weight is [896, 896] or [896, 896] (same numerically but different semantically)
2. Check if blk.0.attn_k.weight is [896, 128] or [128, 896]
3. Compare with llama.cpp's weight loading to see if they handle Q differently

**STATUS**: Investigation paused. Reverted changes. Q projection still has extreme values.

---

## [TEAM ORION] 2025-10-07T00:06Z - Bias Investigation Complete

**FINDINGS - Bias Investigation:**

✅ **Q bias is ALL ZEROS**:
- First 16 values: 0.000000 (all)
- Stats: min=0.000000 max=0.000000 mean=0.000000
- **Conclusion**: Bias is NOT causing the ±16 extreme values

✅ **Q weight looks normal**:
- First 16 values: -0.001090 -0.002918 0.007435 0.008759... (±0.01 range)
- Hex dump: 9477 99fa 1f9d 207c 18c0 9c9d 1ab7 9297...
- **Conclusion**: No outliers in Q weight, values are normal for FP16

✅ **Manual Q[0] verification passes**:
- Token 0: manual=-0.043045, cuBLAS=-0.043060, diff=0.000015 ✅
- Token 1: manual=-0.015185, cuBLAS=-0.015182, diff=0.000003 ✅
- **Conclusion**: cuBLAS params are correct for Q[0]

❌ **Extreme values persist at specific indices**:
- Token 0: Q[95]=-16.047 (head 1, dim 31), Q[126]=+14.336 (head 1, dim 62)
- Token 1: Q[95]=-3.912 (head 1, dim 31), Q[126]=+3.695 (head 1, dim 62)
- **Pattern**: SAME INDICES across tokens, both in head 1

### Critical Insight

The fact that:
1. Q[0] manual verification passes
2. Bias is zero
3. Weights are normal
4. But Q[95] and Q[126] have ±16 values

Suggests **the matmul is reading the wrong memory locations for elements beyond position 0**. The stride/lda parameter may be correct for the first row but incorrect for subsequent rows.

**Hypothesis**: With CUBLAS_OP_T, the weight matrix is being accessed with the wrong stride for rows after the first. Q[0] works because it reads the first 896 elements correctly, but Q[95] reads from the wrong offset.

**Next Action**: Need to verify the weight matrix layout in GGUF. Is Q weight stored as [hidden_dim, q_dim] = [896, 896] or [q_dim, hidden_dim] = [896, 896]? Even though dimensions are the same, the semantic meaning affects how cuBLAS interprets lda with CUBLAS_OP_T.

---
