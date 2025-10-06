# llama.cpp Debug Comparison

**Date**: 2025-10-06  
**Purpose**: Compare Q/K values and attention scores between llama.cpp and our implementation

---

## Setup

### Debug Instrumentation Added to llama.cpp

Modified `reference/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`:

1. **Q values after scaling** (lines 218-232, 252-264)
   - Prints first 10 Q values after applying scale
   - Computes partial Q magnitude
   - Only for first sequence, first head, first position

2. **Attention scores** (lines 295-299)
   - Prints raw Q·K dot product scores
   - Before softmax, before masking
   - First 3 positions only

3. **Softmax values** (lines 330-333)
   - Prints KQ_max, KQ_sum, KQ_reg
   - Shows online softmax state

### How to Run

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
./debug_llama_cpp.sh
```

This will:
1. Rebuild llama.cpp with debug output
2. Run same prompt as our test
3. Save output to `llama_cpp_debug.log`

---

## Comparison Points

### 1. Q Vector Magnitude

**Our implementation** (from `BUG_FIX_PROGRESS.md`):
```
Q magnitude: 60.57
Q values: -0.22, 0.17, -0.11, 0.11, -14.49, 0.28, 0.47, 0.08, -15.18, -34.06
```

**llama.cpp** (to be filled after running):
```
Q magnitude: ???
Q values: ???
```

**Expected**: If llama.cpp has similar magnitudes (~60), then this is normal. If llama.cpp has much smaller magnitudes (~8), we're missing normalization.

---

### 2. Attention Scores (Q·K)

**Our implementation** (from `BUG_FIX_PROGRESS.md`):
```
Unscaled scores: ~1000
Scaled scores: ~125
```

**llama.cpp** (to be filled):
```
Raw scores: ???
```

**Expected**: If llama.cpp also has scores ~125, then our scores are correct. If llama.cpp has scores ~1-10, we have a problem.

---

### 3. Scale Factor

**Our implementation**:
```
scale = 1/sqrt(head_dim) = 1/sqrt(64) = 0.125
```

**llama.cpp** (to be filled):
```
scale = ???
```

**Expected**: Should be identical (0.125 for 64-dim heads).

---

### 4. Softmax Behavior

**Our implementation**:
- Two-pass softmax (find max, then normalize)
- Scores ~125 cause saturation

**llama.cpp**:
- Online softmax (running max/sum)
- To be observed

---

## Analysis After Running

### If Q magnitudes match (~60 in both):
✅ **Conclusion**: Large Q magnitudes are normal for this model  
🔍 **Next**: Check if attention scores also match  
❓ **Question**: Why do large magnitudes work in llama.cpp but not in ours?

### If Q magnitudes differ (llama.cpp ~8, ours ~60):
🔴 **Conclusion**: We're missing a normalization step  
🔍 **Next**: Find where llama.cpp normalizes Q/K  
✅ **Fix**: Add normalization after QKV projection

### If attention scores match (~125 in both):
✅ **Conclusion**: Our attention computation is correct  
🔍 **Next**: Look at softmax implementation differences  
❓ **Question**: Why does llama.cpp's softmax handle large scores better?

### If attention scores differ (llama.cpp ~1-10, ours ~125):
🔴 **Conclusion**: Our Q·K computation is wrong  
🔍 **Next**: Check weight loading, indexing, or scale application  
✅ **Fix**: Debug weight matrices and dot product

---

## Files Modified

- `reference/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` - Added debug printf statements
- `bin/worker-orcd/debug_llama_cpp.sh` - Build and run script
- `bin/worker-orcd/LLAMA_CPP_DEBUG_COMPARISON.md` - This file

---

## Next Steps

1. ✅ Run `./debug_llama_cpp.sh`
2. ⏳ Analyze `llama_cpp_debug.log`
3. ⏳ Fill in comparison values above
4. ⏳ Determine root cause
5. ⏳ Implement fix in our code

---

**Status**: Ready to run llama.cpp debug test
