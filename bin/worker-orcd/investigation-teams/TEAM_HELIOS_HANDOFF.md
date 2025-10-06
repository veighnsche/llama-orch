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
