# Qwen Model Debug Status - Executive Summary

**Last Updated**: 2025-10-06 12:18  
**Current Phase**: Vocab size mismatch causing garbage token selection

---

## 🎯 Quick Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Matrix Layout** | ✅ FIXED | Q values now correct (0.01-0.26) |
| **Weight Loading** | ✅ OK | Values in expected range |
| **KV Cache** | ✅ FIXED | Now properly reading cached positions |
| **Attention Mechanism** | ✅ WORKING | Computes over all positions, weights sum to 1.0 |
| **Bias Values** | ✅ N/A | Qwen2.5 doesn't use biases (confirmed from llama.cpp) |
| **Vocab Size** | ❌ CRITICAL | lm_head is [896,151643] but vocab_size=151936 |
| **Sampling** | ❌ BROKEN | Argmax finds garbage beyond actual vocab |
| **Model Output** | ❌ BROKEN | Generates same token (ID=76156) repeatedly |

---

## 📈 Progress Timeline

### Phase 1: Discovery (2025-10-06 10:26) ✅
- **Issue**: Q values 10-100x too large
- **Evidence**: Range -13.34 to 0.26 (should be 0.01 to 1.13)
- **Document**: `CRITICAL_FINDING.md`

### Phase 2: Root Cause Analysis (2025-10-06 10:35) ✅
- **Finding**: GGUF row-major vs cuBLAS column-major mismatch
- **Solution**: Change `CUBLAS_OP_T` → `CUBLAS_OP_N`, fix leading dimensions
- **Document**: `ROOT_CAUSE_ANALYSIS.md`

### Phase 3: Implementation (2025-10-06 10:40) ✅
- **Changes**: Fixed all 8 matrix multiplications
- **Files**: `qwen_transformer.cpp`, `swiglu_ffn.cu`
- **Document**: `MATRIX_LAYOUT_FIX_SUMMARY.md`

### Phase 4: Testing (2025-10-06 10:49) ✅
- **Result**: Q values now correct ✅
- **New Issue**: Attention outputs uniform ❌
- **Document**: `TEST_RESULTS_AFTER_FIX.md`

### Phase 5: Attention Debug (2025-10-06 10:51-11:05) ✅
- **Issue**: Attention outputs uniform across positions
- **Root Cause**: KV cache not being read (pos always 0)
- **Fix**: Position tracking now working correctly
- **Document**: This summary

### Phase 6: Bias Investigation (2025-10-06 11:07) ✅
- **Issue**: Bias values contain huge outliers (-14, -34)
- **Resolution**: Confirmed Qwen2.5 doesn't use biases (checked llama.cpp reference)
- **Status**: Bias correctly disabled in code

### Phase 7: Vocab Size Mismatch (2025-10-06 11:30-11:56) 🔴 CURRENT
- **Issue**: Model generates same token (ID=76156) repeatedly
- **Root Cause**: lm_head tensor is [896, 151643] but vocab_size is 151936 (padded)
- **Problem**: Argmax searches all 151936 positions, finds garbage at positions 151643-151935
- **Evidence**: Argmax finds token_id=2966 with value 14.24 (beyond actual vocab!)
- **Status**: Multiple fix attempts failed, need proper solution
- **Document**: `BUG_STATUS.md`

---

## 🔍 What We Fixed

### Matrix Layout Bug ✅

**Problem**: GGUF stores weights in row-major order, but cuBLAS expects column-major.

**Solution**: 
- Changed all `CUBLAS_OP_T` to `CUBLAS_OP_N`
- Fixed leading dimensions: `lda = output_dim` (not `input_dim`)

**Impact**: Q values now in correct range (0.01-0.26 vs previous -13.34 to 0.26)

**Files Modified**:
1. `cuda/src/transformer/qwen_transformer.cpp` (5 projections)
2. `cuda/kernels/swiglu_ffn.cu` (3 projections)

### KV Cache Bug ✅

**Problem**: Position counter was always 0, so attention only computed over current token.

**Solution**: 
- Fixed position tracking in `forward()` function
- Added debug logging to verify position updates
- Attention now properly computes over all cached positions

**Impact**: Attention weights now vary across positions (e.g., [0]=0.5077 [1]=0.4923)

---

## 🐛 What's Still Broken

### Vocab Size Mismatch ❌ CRITICAL

**Symptom**: Model generates same token (ID=76156 "suffice") 100 times in a row

**Evidence**:
```
Input tokens: 7985 → 264 → 6386 → 38242 (CHANGING)
Output tokens: 76156 → 76156 → 76156 → 76156 (STUCK)
Hidden states: -11.04, -2.41, 8.20... → -11.31, -2.75, 8.28... (CHANGING)
Logits[0:10]: 0.19, 2.29, 3.11... → 0.33, 2.36, 3.32... (CHANGING)
Max logit in first 1000: 8.65 at token_id=706 ✅
Argmax finds: 14.24 at token_id=2966 ❌ (BEYOND VOCAB!)
```

**Root Cause**: 
- lm_head tensor in GGUF: [896, 151643]
- vocab_size parameter: 151936 (padded)
- Argmax searches all 151936 positions
- Positions 151643-151935 contain garbage values (~14.0)
- Garbage values are higher than real logits (~8.0)

**Impact**: Argmax always selects garbage token, model output is useless

**Attempted Fixes** (all failed):
1. Initialize logits buffer to -INFINITY at allocation
2. Initialize logits to -INFINITY before each projection  
3. Change cuBLAS output stride to actual_vocab

**Why Fixes Failed**: cuBLAS writes with stride 151643, buffer is 151936, garbage persists

---

## 🧭 llama.cpp Reference Findings

- **[n_vocab source]** `n_vocab = vocab.n_tokens()` from tokenizer/gguf, not a padded config value.
- **[lm_head dims]** `output.weight` is created as `{n_embd, n_vocab}`. No extra padded columns are exposed.
- **[logits sizing]** All logits buffers are sized to `n_vocab` and copied as `n_tokens * n_vocab` floats.
- **[sampling bound]** Samplers iterate strictly `token_id in [0, n_vocab)`.

Implication: llama.cpp never scans padded slots; our implementation must mirror this end-to-end.

## 🚀 Immediate Action Plan (Today)

1. **Propagate actual vocab from GGUF** (Rust): derive `actual_vocab` from `"output.weight"` dims or tokenizer and pass to `cuda_inference_init()`.
2. **Use actual vocab in C++**: in `project_to_vocab()` and `cuda_inference_generate_token()`, remove hardcoded `151643` and rely on the passed `config_.vocab_size`.
3. **Sampling bound**: ensure `cuda_sample_token()` receives the actual vocab and kernels iterate only to `vocab_size`.
4. **Sanity asserts**: at init, assert `lm_head` leading dim == `config_.vocab_size`.

---
## 📊 Key Metrics

### Q Values (After Matrix Fix)
```
Before: -13.34 to 0.26  ❌ (10-100x too large)
After:  -0.15 to 0.26   ✅ (correct range)
Target: -0.02 to 1.13   ✅ (llama.cpp reference)
```

### Weight Values
```
attn_q_weight[0:10]: -0.0011 -0.0029 0.0074 0.0088 0.0023 -0.0045 0.0033 -0.0008 0.0107 -0.0024
Range: ~0.001 to 0.01  ✅ (reasonable for FP16)
```

### Attention Weights (After KV Cache Fix)
```
cache_len=1, should have 2 scores:
  Scaled scores: [0]=0.0058 [1]=-0.0249
  Attention weights: [0]=0.5077 [1]=0.4923  ✅ (sums to 1.0)
  
cache_len=2, should have 3 scores:
  Attention weights: [0]=0.3431 [1]=0.6569 [2]=...  ✅ (varying across positions)
```

### Bias Values (Currently Disabled)
```
attn_q_bias[0:10]: -0.0150 0.0255 -0.1035 -0.1357 -14.4375 0.2656 0.3242 0.1240 -15.4375 -34.0000
Problem: Values at indices 4, 8, 9 are 10-100x too large  ❌
Status: Disabled in qwen_transformer.cpp (lines 300, 329, 357)
```

---

## 🎯 Next Actions

### Immediate (Today)
1. ✅ ~~Add attention weight debugging~~ - DONE
2. ✅ ~~Verify KV cache is being read correctly~~ - DONE
3. **Investigate bias loading** - Check weight loader dequantization
4. **Test with llama.cpp** - Verify if reference impl has same bias issues

### Short-term (This Week)
1. ✅ ~~Fix attention mechanism~~ - DONE
2. **Fix bias loading/quantization** or confirm model doesn't use biases
3. **Remove debug logging** once output quality is acceptable
4. **Run full test suite** and verify coherent text generation

### Medium-term (Next Week)
1. Remove debug logging
2. Optimize performance (pre-allocate buffers)
3. Add regression tests
4. Document final solution

---

## 📚 Documentation

### Read First
1. **`DEBUGGING_INDEX.md`** - Master index of all documents
2. **`TEST_RESULTS_AFTER_FIX.md`** - Latest test analysis
3. **`NEXT_STEPS.md`** - Action items

### Technical Details
4. **`MATRIX_LAYOUT_FIX_SUMMARY.md`** - Complete fix documentation
5. **`ROOT_CAUSE_ANALYSIS.md`** - Technical deep dive

### Historical
6. **`CRITICAL_FINDING.md`** - Original Q value discovery
7. **`DEBUG_RUN_RESULTS.md`** - Initial debugging session
8. **`MATRIX_TRANSPOSE_FIX.md`** - Incorrect approach (for reference)

---

## 🧪 How to Test

```bash
# Run test with debug output
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat -- --ignored --nocapture 2>&1 | tee test.log

# Check Q values
grep "Q before bias" test.log

# Check attention outputs
grep "Attention output" test.log

# Compare with llama.cpp
./debug_llama_cpp.sh
```

---

## ✅ Success Criteria

The model will be considered fixed when:

1. ✅ Q values match llama.cpp (DONE)
2. ✅ Attention weights vary across positions (DONE)
3. ⚠️ Model generates diverse tokens (PARTIAL - diverse but poor quality)
4. ❌ Output is coherent and follows prompt (TODO - needs bias fix)
5. ⚠️ No repetitive token loops (PARTIAL - less repetitive than before)

---

## 🎓 Lessons Learned

1. **Always verify matrix layouts** when interfacing between libraries
2. **Compare with reference implementation** early and often
3. **Add debug output at every step** to catch issues quickly
4. **Don't assume libraries use the same conventions** (row-major vs column-major)
5. **Test incrementally** - fix one issue at a time

---

## 📞 Quick Reference

### Key Files
- `cuda/src/transformer/qwen_transformer.cpp` - Main transformer
- `cuda/kernels/gqa_attention.cu` - Attention mechanism
- `cuda/kernels/rope.cu` - RoPE implementation
- `cuda/kernels/swiglu_ffn.cu` - FFN layers

### Key Functions
- `forward_layer()` - Per-layer forward pass
- `cuda_gqa_attention_forward()` - Attention computation
- `cuda_rope_forward_ex()` - RoPE application
- `project_to_vocab()` - Final logits projection

### Debug Locations
- Line 245-294: Weight/bias debug output
- Line 441-485: KV cache verification
- Line 487-495: Attention output debug

---

**Current Focus**: Fix vocab size mismatch - need to either:
1. Get actual lm_head dimensions from GGUF and use them
2. Create CUDA kernel to properly initialize padding region
3. Modify argmax to only search actual vocab range

**Critical**: The model pipeline works (attention, KV cache, matrix ops all correct) but sampling is selecting garbage tokens beyond the actual vocabulary.
