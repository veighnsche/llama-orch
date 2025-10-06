# Debug Run Results - 2025-10-06

**UPDATE 2025-10-06 10:49**: ✅ **Matrix layout issue fixed**  
Q values now correct. Model still produces garbage due to attention mechanism issue.  
See `TEST_RESULTS_AFTER_FIX.md` for latest analysis.

---

**Status**: Debug test completed, garbage output captured  
**Test**: `haiku_generation_anti_cheat` with full debug logging

---

## Test Output Summary

### Repetitive Token Generation

The model generates a repeating pattern of two tokens:

1. **Token 78138** (`"ĠcomponentWillMount"`) - Repeated ~65 times
2. **Token 118530** (`"åįĥçĵ¦"`) - Repeated ~30 times

**Pattern**: `componentWillMount` × 33 → `åįĥçĵ¦` × 30 → `componentWillMount` × 60

---

## What's Working ✅

1. **Pipeline Execution**
   - All 24 layers process successfully
   - No crashes or CUDA errors
   - Position tracking increments correctly (0→1→2→3...)

2. **Logits Generation**
   - Logits are computed and in reasonable range (-5 to +2)
   - Max logit varies between forward passes
   - No NaN or Inf values

3. **Embeddings**
   - Token embeddings load correctly
   - Values in expected range (-0.03 to 0.04)
   - Embedding table accessible

4. **QKV Projections** (Host-side debug visible)
   - Q, K, V projections execute
   - Values after projection are non-zero
   - RoPE application completes

---

## What's NOT Working ❌

### 1. **Repetitive Output** 🔴 CRITICAL

**Symptom**: Model generates same token repeatedly, ignoring context

**Evidence**:
- Token 78138 appears 65+ times consecutively
- Different prompts likely produce same output
- Model acts as if it has no memory of previous tokens

**Root Cause Hypothesis**: **Attention mechanism is broken**
- Model is not properly attending to KV cache
- Attention weights may be uniform (not learning from context)
- RoPE may not be applying position information correctly

### 2. **CUDA Kernel Debug Output Missing** 🟡

**Issue**: Printf statements in CUDA kernels (attention, RoPE) are not appearing in logs

**Attempted Fixes**:
- Added `cudaDeviceSynchronize()` after kernel calls
- Conditional debug based on position < 5
- Still no output

**Why This Matters**: We can't see:
- Attention scores (are they all the same?)
- Attention weights after softmax (do they sum to 1.0?)
- RoPE rotation values (is position encoding working?)
- K/V cache reads (is cache being used?)

**Workaround Needed**: Add host-side debug by copying attention scores back to CPU

---

## Debug Output Analysis

### Forward Pass #0 (Position 0)
```
Token ID: 7985
Logits[0:10]: [-3.60, 0.04, -0.75, 0.25, 0.76, -2.14, -2.46, 1.80, -2.16, -0.47]
Max logit: 1.80 at index 7
Sampled: 118530
```

### Forward Pass #1 (Position 1)
```
Token ID: 264
Logits[0:10]: [-4.26, -2.83, -1.27, -0.92, 0.68, -3.13, -3.68, 0.45, -2.27, -0.77]
Max logit: 0.68 at index 4
Sampled: 118530
```

### Forward Pass #2 (Position 2)
```
Token ID: 6386
Logits[0:10]: [-4.83, -2.50, -1.80, -1.32, 2.02, -3.04, -3.71, 0.23, -2.42, -0.17]
Max logit: 2.02 at index 4
Sampled: 118530
```

**Observation**: During prefill, logits vary. But during generation, they collapse to always favor token 78138.

---

## Next Investigation Steps

### Priority 1: Verify Attention Weights 🔴

**Goal**: Determine if attention is broken

**Method**: Add host-side debug to copy attention scores from GPU to CPU

**File**: `cuda/kernels/gqa_attention.cu`

**What to add**:
```cpp
// After softmax, copy first 10 attention weights to global memory
if (tid == 0 && batch == 0 && q_head == 0 && cache_len < 5) {
    // Allocate debug buffer in host code
    // Copy attn_weights[0:10] to debug buffer
    // Print from host after kernel
}
```

**Expected if broken**:
- All attention weights are identical (e.g., all 0.333 for 3 positions)
- Attention always attends to same position
- Weights don't sum to 1.0

### Priority 2: Check RoPE Application 🟡

**Goal**: Verify position embeddings are working

**Method**: Compare Q/K values before and after RoPE from host-side debug

**Evidence from logs**:
```
Q after projection[0:10]: [values]
Q after RoPE[0:10]: [values]
```

**What to check**:
- Do Q values change after RoPE? (They should)
- Do K values change after RoPE? (They should)
- Are changes position-dependent? (Should vary with pos)

### Priority 3: Test with llama.cpp ✅ COMPLETED

**Goal**: Verify model file is not corrupted

**Result**: ✅ **Model file is VALID** - llama.cpp produces perfect output:

```
Forty-six,  
CUDA's power,  
Compute's might.
```

**Conclusion**: The bug is **definitely in our implementation**, not the model. See `LLAMA_CPP_VALIDATION.md` for details.

### Priority 4: Try Different Sampling 🟢

**Goal**: Rule out sampling as the issue

**Method**: Test with temperature > 0

**File**: `src/inference/cuda_backend.rs`

**Change**:
```rust
let next_token_id = inference.generate_token(
    current_token,
    0.7,  // ← Was 0.0 (greedy)
    50,   // ← Enable top-k
    0.9,  // ← Enable top-p
    seed,
)?;
```

**Expected**: If this produces different output, issue is in logits, not sampling

---

## Files with Debug Logging

### CUDA Kernels (printf not working)
- `cuda/kernels/gqa_attention.cu` - Attention scores, softmax
- `cuda/kernels/rope.cu` - RoPE parameters, rotations

### C++ Host Code (working)
- `cuda/src/transformer/qwen_transformer.cpp`:
  - `forward()` - Position, token ID, final hidden state
  - `forward_layer()` - QKV projections, attention output
  - `project_to_vocab()` - Logits, max values
  - `embed_tokens()` - Embedding lookup

### Rust Code (working)
- `src/inference/cuda_backend.rs` - Prefill tokens, generation loop

---

## Hypothesis: Attention is Broken

### Why This Explains Everything

1. **Repetitive output**: Model can't see previous tokens, so generates same thing
2. **Specific rare token**: Token 78138 ("componentWillMount") suggests weights are misaligned or attention is attending to wrong position
3. **Logits collapse**: During generation, logits always favor same token because attention output is constant

### How to Confirm

**Add this to `qwen_transformer.cpp` after attention kernel**:

```cpp
// Copy attention scores to CPU for debugging
if (layer_idx == 0 && pos < 5) {
    // Allocate temp buffer for attention scores
    float* h_attn_scores = new float[pos + 1];
    
    // Copy from GPU (need to modify kernel to output scores)
    cudaMemcpy(h_attn_scores, d_attn_scores, 
               (pos + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "\n[ATTENTION SCORES Layer 0, pos=%u]\n", pos);
    for (int i = 0; i <= pos; i++) {
        fprintf(stderr, "  pos[%d]: %.6f\n", i, h_attn_scores[i]);
    }
    
    delete[] h_attn_scores;
}
```

---

## Success Criteria

We'll know attention is fixed when:

1. ✅ Model generates diverse, non-repetitive tokens
2. ✅ Output changes based on prompt
3. ✅ Attention weights are diverse (not uniform)
4. ✅ Attention weights sum to ~1.0
5. ✅ Model produces coherent text

---

## Log Files

- `debug_with_sync.log` - Full test output with cudaDeviceSynchronize()
- `debug_full_rebuild.log` - Test after clean rebuild
- `debug_full.log` - Initial debug run

---

**Next Action**: Modify attention kernel to output scores to global memory, then copy to host for inspection.
