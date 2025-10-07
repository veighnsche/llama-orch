# Critical Bugs Hypothesis

**Date**: 2025-10-07  
**Status**: Analysis Complete - Ready for Implementation  
**Source**: Systematic comparison with mistral.rs and candle references  

---

## Executive Summary

After comprehensive code analysis comparing our C++/CUDA implementation with proven Rust references (mistral.rs and candle), I've identified the **most likely bug locations**.

**Key Finding**: The infrastructure (cuBLAS, RoPE, RMSNorm, KV cache) appears mathematically correct based on extensive team investigations. However, output is still garbage, suggesting a **subtle but critical** implementation detail is wrong.

---

## Top 3 Bug Hypotheses (Prioritized)

### 🔴 HYPOTHESIS #1: Tensor Reshape/Transpose Missing (CRITICAL - Priority 1)

**Likelihood**: ⭐⭐⭐⭐⭐ **VERY HIGH**

**Issue**: References show explicit reshape and transpose operations between projection and attention that may be missing in our code.

**Evidence from References**:

**mistral.rs** (lines 161-177):
```rust
// After Q projection: [B, L, num_heads*head_dim]
let q = q.reshape((b_sz, q_len, self.num_heads, self.head_dim))?  // → [B, L, H, D]
         .transpose(1, 2)?;                                         // → [B, H, L, D]

// After attention: [B, H, L, D]
let attn_output = attn_output.transpose(1, 2)?     // → [B, L, H, D]
                              .reshape((b_sz, q_len, hidden_sz))?;  // → [B, L, H*D]
```

**candle Qwen3** (lines 195-235):
```rust
// After Q/K/V projections
let q = q.reshape((b, l, self.num_heads, self.head_dim))?
         .transpose(1, 2)?;  // [B, L, H, D] → [B, H, L, D]

// After attention
ctx.transpose(1, 2)?                  // [B, H, L, D] → [B, L, H, D]
   .reshape((b, l, self.hidden_size))?  // [B, L, H, D] → [B, L, H*D]
```

**Our Code Analysis**:
- cuBLAS Q/K/V projections produce: `[batch, num_heads * head_dim]` = `[1, 896]`
- GQA kernel expects: `[batch, num_heads, head_dim]` = `[1, 14, 64]`
- **Status**: Data is already in correct memory layout (contiguous heads), but kernel indexing assumes per-head structure

**Impact if Wrong**:
- Attention would mix data from different heads
- Position information would be scrambled
- Output would be garbage (matches observed behavior)

**How to Fix**:
Option A: Add explicit reshape/transpose if kernel layout doesn't match projection output
Option B: Verify kernel indexing matches actual memory layout from cuBLAS

**Verification**:
1. Add logging to dump Q/K/V memory layout after projection
2. Compare with reference implementation output at same stage
3. Check if head boundaries are respected in attention computation

---

### 🟡 HYPOTHESIS #2: GQA K/V Repetition Edge Case (Priority 2)

**Likelihood**: ⭐⭐⭐⭐ **HIGH**

**Issue**: While GQA head mapping appears correct (`kv_head = q_head / group_size`), there may be an edge case in how K/V are accessed across cache and current token.

**Evidence from References**:

**candle repeat_kv** (line 220):
```rust
// Explicit repeat_kv function call
let k = repeat_kv(k, self.num_kv_groups)?;  // [B, 2, L, 64] → [B, 14, L, 64]
let v = repeat_kv(v, self.num_kv_groups)?;
```

**Our Code** (gqa_attention.cu lines 227-228):
```cpp
int group_size = num_q_heads / num_kv_heads;  // 14 / 2 = 7
int kv_head = q_head / group_size;            // Maps correctly
```

**Observed Behavior**:
- TEAM_SHREDDER verified mapping: q_heads 0-6 → kv_head 0, q_heads 7-13 → kv_head 1 ✅
- Mapping logic is correct

**Potential Issue**:
- When reading from cache + current K/V, both must use same `kv_head` index
- If cache read uses different indexing than current K/V, mismatch occurs

**Verification Status**:
- Static mapping: ✅ Verified correct
- Runtime cache access: ⚠️ Need to verify cache read and current K/V use same kv_head

**How to Fix**:
1. Verify both cache and current K/V access use same `kv_head` calculation
2. Add assertion: `assert(kv_head < num_kv_heads)`
3. Log actual kv_head values for first few tokens

---

### 🟢 HYPOTHESIS #3: Attention Output Memory Layout Mismatch (Priority 3)

**Likelihood**: ⭐⭐⭐ **MEDIUM-HIGH**

**Issue**: Attention kernel output layout may not match what output projection expects.

**Evidence**:

**Kernel Output** (gqa_attention.cu line 875):
```cpp
int out_idx = batch * num_q_heads * head_dim + q_head * head_dim + d;
output[out_idx] = ...;
```
This produces layout: `[batch, num_q_heads, head_dim]` = "head-major"

**Output Projection Expects** (qwen_transformer.cpp line 1652):
```cpp
cublasGemmEx(..., config_.hidden_dim, batch_size, q_dim, ...);
// Expects: [batch, q_dim] where q_dim = num_heads * head_dim
```

**Analysis**:
- Kernel produces: `[head0_d0..d63, head1_d0..d63, ..., head13_d0..d63]`
- GEMM needs: same layout as input (flat)
- **These are compatible if memory is contiguous**

**But**: If attention kernel writes with stride/gaps, GEMM would read wrong data.

**How to Verify**:
1. Dump first 128 values of attention output (2 full heads)
2. Check if data is contiguous: `[head0_all_dims, head1_all_dims]`
3. Compare with reference attention output

---

## What's Been Ruled Out (Don't Re-investigate)

Based on extensive team investigations documented in the code:

### ✅ Verified Correct (Don't Change):

1. **cuBLAS Parameters** (TEAM SENTINEL, TEAM PEAR, TEAM MONET, TEAM PICASSO)
   - All 8 matrix multiplications use CUBLAS_OP_T with correct lda values
   - Manual verification: cuBLAS matches hand calculation (diff < 0.001)
   - **Status**: MATHEMATICALLY CORRECT

2. **RoPE Implementation** (TEAM HOLE_PUNCH)
   - Frequency calculation matches references exactly
   - Angles verified: cos/sin values match closed-form math
   - Identity transformation at pos=0 works perfectly
   - **Status**: ALL 5 GATES PASSED

3. **RMSNorm** (TEAM POLARIS, TEAM CHARLIE)
   - Formula matches llama.cpp exactly
   - `output = (input / rms) * weight` where `rms = sqrt(mean(input²) + eps)`
   - **Status**: MATHEMATICALLY CORRECT

4. **SwiGLU Activation** (TEAM POLARIS)
   - Formula: `output = silu(gate) * up` where `silu(x) = x * sigmoid(x)`
   - Matches standard definition
   - **Status**: CORRECT

5. **KV Cache Infrastructure** (TEAM WATER, TEAM DRAWER)
   - cache_len increments correctly (0→1→2→3)
   - Write positions correct
   - Read indexing correct
   - **Status**: VERIFIED WORKING

6. **FFN Weight Loading** (TEAM RACE CAR, TEAM CHARLIE BETA)
   - All three weights loaded: gate, up, down
   - Pointers non-null and verified
   - **Status**: FIXED (was missing, now loaded)

7. **GQA Head Mapping** (TEAM SHREDDER)
   - group_size = 7 (14 Q heads / 2 KV heads)
   - Q heads 0-6 → KV head 0 ✅
   - Q heads 7-13 → KV head 1 ✅
   - **Status**: MAPPING CORRECT

### 🚫 Failed Hypotheses (Don't Retry):

1. **CUBLAS_OP_N instead of CUBLAS_OP_T** - Makes output WORSE
2. **Different compute types (32F vs FAST_16F)** - No effect on extremes
3. **Transpose flags varied** - Tested extensively, all combinations fail
4. **Weight corruption** - Only 2 columns checked (sparse), but no NaNs found
5. **Input spikes** - normed input is in normal range
6. **Bias values** - All biases are zeros (no effect)

---

## Recommended Investigation Order

### Step 1: Verify Tensor Layouts (Highest Priority)

**Action**: Add comprehensive logging to verify tensor shapes match references

**Code to Add** (qwen_transformer.cpp after Q projection):
```cpp
// After Q/K/V projection, log memory layout
if (layer_idx == 0 && pos == 0) {
    half h_q[896];
    cudaMemcpy(h_q, q_proj_, 896 * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Check if heads are contiguous
    fprintf(stderr, "[LAYOUT] Q projection output:\n");
    fprintf(stderr, "  Head 0, dim 0-7: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", __half2float(h_q[i]));
    fprintf(stderr, "\n  Head 0, dim 56-63: ");
    for (int i = 56; i < 64; i++) fprintf(stderr, "%.4f ", __half2float(h_q[i]));
    fprintf(stderr, "\n  Head 1, dim 0-7: ");
    for (int i = 64; i < 72; i++) fprintf(stderr, "%.4f ", __half2float(h_q[i]));
    fprintf(stderr, "\n");
}
```

**Expected**: Heads should be contiguous blocks of 64 dimensions each

**Compare with**: mistral.rs/candle output at same point

### Step 2: Verify Attention Output Layout

**Action**: Check if attention kernel output matches output projection input expectation

**Code to Add** (qwen_transformer.cpp after GQA attention):
```cpp
if (layer_idx == 0 && pos == 0) {
    half h_attn[896];
    cudaMemcpy(h_attn, attn_output_, 896 * sizeof(half), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "[LAYOUT] Attention output:\n");
    fprintf(stderr, "  Position [0-7]: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", __half2float(h_attn[i]));
    fprintf(stderr, "\n  Position [64-71] (head 1 start): ");
    for (int i = 64; i < 72; i++) fprintf(stderr, "%.4f ", __half2float(h_attn[i]));
    fprintf(stderr, "\n");
}
```

**Expected**: Should be flat array ready for GEMM

### Step 3: Cross-Reference with llama.cpp

**Action**: Run llama.cpp with verbose output and compare intermediate values

**Command**:
```bash
/home/vince/Projects/llama-orch/reference/llama.cpp/build/bin/llama-cli \
  -m /home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
  -p "Write a haiku about autumn:" \
  --verbose \
  -n 10
```

**Compare**:
- Q/K/V projection outputs (first 16 values)
- Attention outputs (first 16 values)
- FFN outputs (first 16 values)
- Logits (first 20 values)

---

## Success Criteria

**Minimum Success** (matches plan requirements):
- ✅ Haiku test passes
- ✅ Minute word found in output
- ✅ No crashes or errors
- ✅ Coherent text output

**Full Success**:
- ✅ All above
- ✅ No mojibake (`è®«æŁ¥`, `Ġłích`, `ĠKw`)
- ✅ No repetitive tokens (same token 10+ times)
- ✅ Correct language (English, not Chinese/Thai/Korean)
- ✅ Contextually appropriate output
- ✅ Reproducible with same seed

---

## Debugging Resources

**Reference Outputs**:
- llama.cpp produces perfect haiku with same model ✅
- Proves model file is correct
- Proves bug is in our implementation

**Key Files**:
- Our transformer: `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/src/transformer/qwen_transformer.cpp`
- GQA kernel: `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/kernels/gqa_attention.cu`
- FFN kernel: `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/kernels/swiglu_ffn.cu`
- Test: `/home/vince/Projects/llama-orch/bin/worker-orcd/tests/haiku_generation_anti_cheat.rs`

**Model Config** (Qwen2.5-0.5B-Instruct):
```
vocab_size: 151936
hidden_size: 896
intermediate_size: 4864
num_hidden_layers: 24
num_attention_heads: 14
num_key_value_heads: 2  ← GQA ratio 7:1
head_dim: 64
max_position_embeddings: 131072
rope_theta: 1000000.0
rms_norm_eps: 1e-6
hidden_act: "silu"
```

---

## Next Actions

1. ✅ **Phase 1 Complete**: Reference study done
2. ✅ **Phase 2 Complete**: Bug hypotheses identified
3. ⬜ **Phase 3 Next**: Implement layout verification logging
4. ⬜ **Phase 3 Next**: Compare with reference outputs
5. ⬜ **Phase 3 Next**: Fix identified bugs
6. ⬜ **Phase 4 Next**: Run full test suite

**Estimated Time to Fix**: 2-4 hours if hypothesis #1 is correct

---

*Analysis complete - ready for implementation phase*

---

# PEER REVIEW COMMENTS

**Reviewer**: Cascade (Peer Review Agent)  
**Date**: 2025-10-07T22:19Z

## Critical Issues with Hypothesis #1

### ❌ FALSE POSITIVE: "Tensor Reshape/Transpose Missing"

**Your Rating**: ⭐⭐⭐⭐⭐ VERY HIGH likelihood

**My Assessment**: ⭐ VERY LOW likelihood - **This is a false positive based on misunderstanding**

**Why This Is Wrong**:

1. **Rust `reshape()` and `transpose()` are VIEW operations, not memory operations**
   - They change tensor metadata (strides, dimensions)
   - They DO NOT move or reorganize memory
   - The underlying memory layout is still `[batch, seq, num_heads * head_dim]` contiguous

2. **Your CUDA kernel ALREADY implements the correct indexing**
   - Line 285 of gqa_attention.cu: `int q_idx = batch * num_q_heads * head_dim + q_head * head_dim + d;`
   - This reads memory as `[batch][q_head][d]` which is exactly the "reshaped" view
   - No additional reshape needed

3. **cuBLAS output is already in the correct format**
   - cuBLAS produces: `[batch, num_heads * head_dim]` in row-major, contiguous
   - Heads are already contiguous blocks of `head_dim` elements
   - Your kernel indexing matches this layout perfectly

**Evidence Against Your Hypothesis**:
- TEAM_SENTINEL verified cuBLAS parameters correct (diff < 0.001)
- TEAM_SHREDDER verified GQA head mapping correct
- Multiple teams confirmed CUBLAS_OP_T with lda=hidden_dim is mathematically correct
- If heads were mixed, you'd see wrong kv_head assignments (you don't)

**What You're Actually Seeing**:
- Comparing high-level Rust abstractions to low-level CUDA indexing
- Mistaking view operations for memory operations
- Not recognizing that your kernel already implements the "reshaped" layout

**Recommendation**: ❌ **DO NOT IMPLEMENT THIS FIX** - It will waste 2-4 hours on a non-existent bug

---

## Re-evaluation of Hypothesis #2

### 🟡 GQA K/V Repetition Edge Case

**Your Rating**: ⭐⭐⭐⭐ HIGH likelihood

**My Assessment**: ⭐⭐ LOW-MEDIUM likelihood

**Why This Might Not Be the Bug**:

1. **TEAM_SHREDDER already verified the mapping** (gqa_attention.cu lines 214-244)
   - Q heads 0-6 → KV head 0 ✅
   - Q heads 7-13 → KV head 1 ✅
   - group_size = 7 ✅

2. **The mapping formula is trivial**:
   ```cpp
   int kv_head = q_head / group_size;  // Integer division
   ```
   This is mathematically impossible to get wrong for the values involved.

3. **No evidence of edge case**:
   - Cache read and current K/V both use same `kv_head` calculation
   - No conditional logic that could cause divergence
   - No off-by-one errors visible in the code

**However**: There COULD be an issue with:
- Cache indexing: `kv_cache_k[batch][kv_head][pos][d]` vs current K indexing
- Stride calculations for cache access
- But this would show up as wrong attention patterns, not garbage

**Recommendation**: ⚠️ **LOW PRIORITY** - Only investigate if other hypotheses fail

---

## Re-evaluation of Hypothesis #3

### 🟢 Attention Output Memory Layout Mismatch

**Your Rating**: ⭐⭐⭐ MEDIUM-HIGH likelihood

**My Assessment**: ⭐⭐⭐⭐ **HIGH likelihood** - This is actually plausible

**Why This Could Be the Bug**:

1. **TEAM_PLOTTER is actively investigating** (lines 154-171 of qwen_transformer.cpp)
   - They're checking concat order, transpose flags, lda/ldb/ldc
   - This suggests they found something suspicious

2. **Output projection is complex**:
   - Kernel outputs: `[batch, num_q_heads, head_dim]` (head-major)
   - Output projection expects: `[batch, num_heads * head_dim]` (flat)
   - These SHOULD be compatible if contiguous, but...

3. **Potential mismatch**:
   - If kernel writes with gaps/strides (non-contiguous)
   - If output projection reads with wrong stride
   - If there's a transpose flag error in the output projection cuBLAS call

**Evidence Supporting This**:
- Line 1588-1590: `opA=CUBLAS_OP_T (transpose), opB=CUBLAS_OP_N (no transpose)`
- This is the output projection GEMM
- If the input to this GEMM is not in the expected format, garbage output

**Recommendation**: ✅ **HIGH PRIORITY** - Investigate this BEFORE hypothesis #1

---

## What You Should Actually Investigate

### Priority 1: Output Projection (Hypothesis #3)

**Action**:
1. Dump first 128 values of attention kernel output (2 full heads)
2. Verify memory is contiguous: `[head0_all_dims, head1_all_dims]`
3. Check if output projection cuBLAS call expects this format
4. Compare with reference implementation output at same stage

**Expected Time**: 1-2 hours

### Priority 2: FFN Down Projection

**Evidence**:
- TEAM_RACE_CAR investigating (lines 109-130)
- TEAM_PAPER_CUTTER investigating last block (lines 133-151)
- Multiple teams suspect FFN, not attention

**Action**:
1. Verify FFN down projection cuBLAS parameters
2. Check weight loading (is ffn_down actually loaded correctly?)
3. Compare FFN output with reference

**Expected Time**: 1-2 hours

### Priority 3: Numerical Stability in Attention Kernel

**Evidence from gqa_attention.cu**:
- Line 274: Static `q_shared[64]` (potential overflow)
- Line 267: No bounds check on `cache_len`
- Softmax reduction could have numerical issues

**Action**:
1. Add bounds checks
2. Check for NaN/Inf in attention scores
3. Verify softmax normalization

**Expected Time**: 1 hour

### ❌ DO NOT INVESTIGATE: Reshape/Transpose (Hypothesis #1)

**Reason**: This is a false positive based on misunderstanding Rust tensor operations.

---

## Summary of Peer Review

**Hypothesis #1 (Reshape/Transpose)**: ❌ **REJECTED** - False positive, waste of time

**Hypothesis #2 (GQA Repetition)**: ⚠️ **LOW PRIORITY** - Already verified correct

**Hypothesis #3 (Output Projection)**: ✅ **HIGH PRIORITY** - Actually plausible

**Alternative Focus**: FFN down projection, numerical stability

**Estimated Time Saved**: 2-4 hours (by not implementing hypothesis #1)

**Recommended Next Steps**:
1. Investigate output projection (TEAM_PLOTTER's work)
2. Investigate FFN down projection (TEAM_RACE_CAR's work)
3. Add value logging to compare with llama.cpp
4. Stop comparing architectures, start comparing VALUES

---

**Peer Reviewer**: Cascade  
**Date**: 2025-10-07T22:19Z
