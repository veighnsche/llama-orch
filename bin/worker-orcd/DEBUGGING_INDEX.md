# Qwen Model Debugging - Master Index

**Last Updated**: 2025-10-06 10:51  
**Current Status**: Matrix layout fixed, debugging attention mechanism

---

## 📊 Current State

### ✅ Fixed Issues
1. **Matrix Layout Bug** (2025-10-06 10:49)
   - GGUF row-major vs cuBLAS column-major mismatch
   - Q values now in correct range (0.01-0.26)
   - All 8 matrix multiplications corrected

### ❌ Active Issues
1. **Attention Mechanism Broken**
   - Attention outputs nearly identical across positions
   - Model produces repetitive garbage tokens
   - Likely causes: RoPE, KV cache, or attention score calculation

---

## 📚 Document Guide

### Current Analysis (Read These First)
1. **`TEST_RESULTS_AFTER_FIX.md`** ⭐ **START HERE**
   - Latest test results after matrix fix
   - Q value comparison (before/after)
   - Current issues and next steps
   - Attention mechanism analysis

2. **`MATRIX_LAYOUT_FIX_SUMMARY.md`** ⭐ **SOLUTION DOCS**
   - Complete fix documentation
   - Before/after comparison
   - All files modified
   - Matrix dimension reference tables

3. **`NEXT_STEPS.md`**
   - Immediate action items
   - Cleanup tasks
   - Performance optimization ideas
   - Verification checklist

### Technical Deep Dives
4. **`ROOT_CAUSE_ANALYSIS.md`**
   - Technical analysis of matrix layout bug
   - How llama.cpp handles it
   - Understanding cuBLAS leading dimensions
   - Row-major vs column-major explanation

### Historical Context (For Reference)
5. **`CRITICAL_FINDING.md`**
   - Original discovery of Q value mismatch (10-100x too large)
   - Comparison with llama.cpp reference
   - ✅ RESOLVED by matrix layout fix

6. **`DEBUG_RUN_RESULTS.md`**
   - Initial debugging session
   - Repetitive token generation analysis
   - What's working vs not working
   - ✅ Matrix issue identified and fixed

7. **`MATRIX_TRANSPOSE_FIX.md`**
   - ⚠️ INCORRECT approach (kept for reference)
   - Attempted transpose fix (didn't work)
   - Shows evolution of understanding

---

## 🔍 Timeline

### 2025-10-06 10:26 - Initial Discovery
- **Document**: `CRITICAL_FINDING.md`
- **Finding**: Q values 10-100x larger than llama.cpp
- **Evidence**: Q values ranging from -13.34 to 0.26 (should be 0.01 to 1.13)

### 2025-10-06 10:30 - First Fix Attempt
- **Document**: `MATRIX_TRANSPOSE_FIX.md`
- **Approach**: Changed `CUBLAS_OP_N` to `CUBLAS_OP_T`
- **Result**: ❌ Incorrect - misunderstood the problem

### 2025-10-06 10:35 - Root Cause Analysis
- **Document**: `ROOT_CAUSE_ANALYSIS.md`
- **Discovery**: GGUF row-major vs cuBLAS column-major mismatch
- **Insight**: Row-major `[rows, cols]` = column-major `[cols, rows]`

### 2025-10-06 10:40 - Correct Fix Applied
- **Document**: `MATRIX_LAYOUT_FIX_SUMMARY.md`
- **Changes**: 
  - Changed `CUBLAS_OP_T` → `CUBLAS_OP_N` (no transpose)
  - Fixed leading dimensions for all 8 matrix multiplications
- **Files Modified**:
  - `cuda/src/transformer/qwen_transformer.cpp`
  - `cuda/kernels/swiglu_ffn.cu`

### 2025-10-06 10:49 - Testing & New Issue
- **Document**: `TEST_RESULTS_AFTER_FIX.md`
- **Result**: ✅ Q values now correct (0.01-0.26)
- **New Issue**: ❌ Attention outputs uniform across positions
- **Symptom**: Model still produces repetitive garbage

---

## 🎯 Quick Reference

### Q Value Comparison

| Version | Range | Status |
|---------|-------|--------|
| **Before Fix** | -13.34 to 0.26 | ❌ 10-100x too large |
| **After Fix** | -0.15 to 0.26 | ✅ Correct range |
| **llama.cpp** | -0.02 to 1.13 | ✅ Reference |

### Files Modified

| File | Changes | Status |
|------|---------|--------|
| `qwen_transformer.cpp` | Q, K, V, attn_out, lm_head projections | ✅ Fixed |
| `swiglu_ffn.cu` | FFN gate, up, down projections | ✅ Fixed |
| `haiku_generation_anti_cheat.rs` | Updated test comments | ✅ Updated |

### Matrix Operations Fixed

1. ✅ Q projection (line ~232)
2. ✅ K projection (line ~316)
3. ✅ V projection (line ~345)
4. ✅ Attention output projection (line ~502)
5. ✅ LM head projection (line ~624)
6. ✅ FFN gate projection (swiglu_ffn.cu line ~96)
7. ✅ FFN up projection (swiglu_ffn.cu line ~114)
8. ✅ FFN down projection (swiglu_ffn.cu line ~144)

---

## 🐛 Current Debugging Focus

### Issue: Attention Outputs Uniform Across Positions

**Evidence**:
```
pos=0: 0.0182 -0.0056 -0.0002 -0.0258 -0.0089 -0.0017 -0.0065 0.0315 -0.0245 -0.0162
pos=1: 0.0179 -0.0060  0.0001 -0.0258 -0.0087 -0.0016 -0.0066 0.0315 -0.0247 -0.0162
```

**Hypothesis**: Attention mechanism not learning from context

**Possible Causes**:
1. Attention weights are uniform (not varying with position)
2. RoPE not applying position information correctly
3. KV cache not being read during attention
4. Softmax scaling incorrect

**Next Steps**:
1. Add debug output for attention weights after softmax
2. Compare RoPE output with llama.cpp
3. Verify KV cache is being read correctly
4. Check attention score calculation (Q·K^T)

---

## 🔧 How to Run Tests

### Run the haiku test (with debug output)
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat -- --ignored --nocapture 2>&1 | tee test_output.log
```

### Compare with llama.cpp reference
```bash
./debug_llama_cpp.sh
```

### Check specific debug output
```bash
# Q values
grep "Q before bias" test_output.log

# Attention outputs
grep "Attention output" test_output.log

# Weight values
grep "attn_q_weight" test_output.log
```

---

## 📝 Related Issues

### Known Issues (Not Yet Fixed)
1. **Bias Values Too Large** - Some bias values are 10-100x larger than expected
   - Example: -14.4375, -15.4375, -34.0000
   - Currently disabled in code (line 298)
   - Need to investigate bias loading/quantization

2. **Memory Leaks** - FFN allocates temporary buffers on every forward pass
   - Should be pre-allocated during initialization
   - Performance impact

3. **Weight Quantization** - Loading Q4_K_M weights without dequantization
   - Currently using FP16 model to avoid this
   - Need proper dequantization implementation

---

## 🎓 Key Learnings

### Matrix Layout Fundamentals
- **GGUF**: Row-major storage
- **cuBLAS**: Column-major expectation
- **Conversion**: Row-major `[rows, cols]` = Column-major `[cols, rows]` (transposed)
- **Solution**: Use `CUBLAS_OP_N` with correct leading dimensions

### Debugging Approach
1. Compare with reference implementation (llama.cpp)
2. Add debug output at every transformation step
3. Verify intermediate values match expected ranges
4. Isolate issues by testing components independently

### Common Pitfalls
- Don't assume matrix layouts match between libraries
- Always verify leading dimensions in matrix operations
- Check both values AND their distribution
- Test with known-good reference implementation

---

## 📞 Support

If you encounter issues:
1. Check `TEST_RESULTS_AFTER_FIX.md` for current status
2. Review debug output in test logs
3. Compare with llama.cpp reference values
4. Add more debug output to isolate the issue

---

**Remember**: The matrix layout issue is FIXED. Current focus is on the attention mechanism.
