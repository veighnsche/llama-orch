# Debug Strategy - Q/K Magnitude Investigation

**Date**: 2025-10-06  
**Status**: Running llama.cpp with debug instrumentation

---

## The Problem

Our implementation generates repetitive token 78138 because:
1. ✅ Q vector loading was broken (FIXED - now uses shared memory)
2. 🔴 Q/K magnitudes are ~60 instead of expected ~8
3. 🔴 Attention scores are ~125 (should be ~1-10)
4. 🔴 Softmax saturates, model can't properly weight attention

---

## Current Investigation

### Hypothesis
We're either:
- **A)** Missing a normalization step that llama.cpp has
- **B)** Computing Q·K incorrectly (wrong weights, indexing, etc.)
- **C)** Using the wrong scale factor

### Test Method
Add debug output to llama.cpp to see:
- What Q magnitudes does llama.cpp produce?
- What attention scores does llama.cpp compute?
- What scale factor does llama.cpp use?

### Expected Outcomes

#### Scenario 1: llama.cpp has similar large magnitudes (~60)
**Conclusion**: Large magnitudes are normal for this model  
**Implication**: Our Q/K values are correct  
**Next**: Investigate why our softmax/attention fails with these values  
**Possible causes**:
- Softmax implementation difference (two-pass vs online)
- Numerical precision issues
- Weight loading for V matrix
- Attention output computation

#### Scenario 2: llama.cpp has small magnitudes (~8)
**Conclusion**: We're missing normalization  
**Implication**: Need to find where llama.cpp normalizes  
**Next**: Search llama.cpp for:
- RMSNorm after QKV projection
- LayerNorm after QKV projection
- Manual normalization in attention kernel
**Fix**: Add same normalization to our code

#### Scenario 3: Attention scores match (~125 in both)
**Conclusion**: Q·K computation is correct  
**Implication**: Problem is in softmax or V computation  
**Next**: Compare softmax implementations in detail  
**Possible fixes**:
- Switch to online softmax (like llama.cpp)
- Improve numerical stability
- Check V matrix computation

#### Scenario 4: Attention scores differ significantly
**Conclusion**: Q·K computation is wrong  
**Implication**: Weight loading or indexing bug  
**Next**: Debug weight matrices:
- Print first 10 values of attn_q_weight
- Compare with GGUF file
- Check matrix dimensions and transpose
- Verify cache indexing

---

## Files Modified for Debug

### llama.cpp
- `reference/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`
  - Lines 218-232: Q debug (FAST_FP16 path)
  - Lines 252-264: Q debug (non-FAST_FP16 path)
  - Lines 295-299: Attention score debug
  - Lines 330-333: Softmax state debug

### Scripts
- `bin/worker-orcd/debug_llama_cpp.sh` - Build and run llama.cpp
- `bin/worker-orcd/analyze_llama_debug.sh` - Parse debug output

### Documentation
- `bin/worker-orcd/LLAMA_CPP_DEBUG_COMPARISON.md` - Comparison template
- `bin/worker-orcd/DEBUG_STRATEGY.md` - This file

---

## Running the Test

```bash
# 1. Build and run llama.cpp with debug (currently running)
./debug_llama_cpp.sh

# 2. Once complete, analyze output
./analyze_llama_debug.sh

# 3. Compare with our values
# Our Q magnitude: 60.57
# Our attention scores: ~125
# Our scale: 0.125
```

---

## What Happens Next

### If we find the root cause:
1. Document the finding in `LLAMA_CPP_DEBUG_COMPARISON.md`
2. Implement the fix in our CUDA code
3. Rebuild and test
4. Verify output is no longer repetitive

### If results are inconclusive:
1. Add more detailed debug (K values, V values, output computation)
2. Compare weight matrices directly
3. Step through both implementations side-by-side

---

## Success Criteria

We'll know we've fixed it when:
1. ✅ Model generates diverse, non-repetitive tokens
2. ✅ Output changes based on prompt
3. ✅ Attention scores are in reasonable range
4. ✅ Model produces coherent haiku (like llama.cpp does)

---

**Status**: Waiting for llama.cpp build to complete (~5-10 minutes)

**Next**: Analyze `llama_cpp_debug.log` and compare with our implementation
