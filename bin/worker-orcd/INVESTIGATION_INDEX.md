# Investigation Documentation Index

**Investigation Date**: 2025-10-06  
**Bug**: Model generates same token repeatedly  
**Status**: Root cause identified, fix in progress

---

## Quick Start

**🆕 UPDATE 2025-10-06 13:08**: Deep llama.cpp investigation complete! Solution ready.

**New to this bug?** Read these in order:

1. 📋 **INVESTIGATION_SUMMARY.md** - Complete summary with solution (10 min) ⭐ **START HERE**
2. 🔧 **ACTION_PLAN_REVISED.md** - Implementation guide (5 min)
3. 📊 **CURRENT_VS_LLAMA_CPP_COMPARISON.md** - Parameter comparison (5 min)

**For historical context:**

4. 📚 **COMPLETE_INVESTIGATION_REPORT.md** - Full investigation (30 min read)
5. 🎯 **FINAL_DIAGNOSIS.md** - Technical root cause

---

## All Documentation Files

### 🎯 Start Here (NEW - 2025-10-06 13:08)

| File | Purpose | Read Time |
|------|---------|-----------|
| **INVESTIGATION_SUMMARY.md** | 📋 **READ THIS FIRST** - Complete summary with solution | 10 min |
| **ACTION_PLAN_REVISED.md** | 🔧 **Implementation guide** - Step-by-step fix instructions | 5 min |
| **CURRENT_VS_LLAMA_CPP_COMPARISON.md** | 📊 Side-by-side parameter comparison | 5 min |

### Investigation Reports

| File | Purpose | Read Time |
|------|---------|-----------|
| **START_HERE.md** | Quick overview for new engineers | 5 min |
| **COMPLETE_INVESTIGATION_REPORT.md** | Everything we learned and tested | 30 min |
| **BUG_STATUS_UPDATED.md** | Current status, what works/broken | 10 min |
| **FINAL_DIAGNOSIS.md** | Technical details of root cause | 15 min |
| **LLAMA_CPP_MATRIX_ANALYSIS.md** | Deep analysis of llama.cpp CUDA implementation | 15 min |
| **FIX_ATTEMPT_FAILED.md** | Why changing GEMM params failed | 5 min |
| **VOCAB_SIZE_INVESTIGATION.md** | Initial findings (now outdated) | 5 min |

### Original Documents (Historical)

| File | Status | Notes |
|------|--------|-------|
| **BUG_STATUS.md** | ⚠️ OUTDATED | Original hypothesis was WRONG |
| **NEXT_STEPS.md** | ⚠️ OUTDATED | Based on incorrect hypothesis |
| **STATUS_SUMMARY.md** | ⚠️ OUTDATED | Pre-investigation status |
| **LLAMA_CPP_VALIDATION.md** | ✅ VALID | Proves model file works |

---

## Key Findings Summary

### What the Original Bug Report Claimed (WRONG ❌)

- lm_head tensor is [896, 151643]
- vocab_size is 151936
- Garbage at positions 151643-151935

### What We Actually Found (CORRECT ✅)

- lm_head tensor IS [896, 151936]
- vocab_size IS 151936
- Garbage at positions 8850, 44394, 137131 (scattered)
- Garbage values: ~14-15 (normal logits: -4 to +4)
- Garbage changes over time (not uninitialized memory!)

---

## Investigation Timeline

| Time | Phase | Key Discovery |
|------|-------|---------------|
| 12:33 | Started implementing "fix" | Assumed bug report was correct |
| 12:38 | First test run | "Fix" had no effect |
| 12:40 | Added debug output | Found garbage at unexpected positions |
| 12:45 | Checked logits values | Confirmed garbage at 44394, 137131 |
| 12:47 | Checked lm_head weights | Weights are normal! |
| 12:48 | Checked hidden state | Hidden state is normal! |
| 12:50 | Final analysis | Root cause: GEMM produces wrong values |

---

## What We Tested

### ✅ Things That Work

- Model loading (all 291 tensors)
- Matrix operations (Q values correct)
- KV cache (position tracking works)
- Attention (weights sum to 1.0)
- lm_head weights (normal values)
- Hidden state (no spikes)
- llama.cpp (same model works!)

### ❌ Things That Are Broken

- Logits at specific positions
- Argmax (selects garbage)
- Token generation (same token 100x)
- Output quality (unusable)

### 🔧 Fixes We Tried (All Failed)

1. Derive vocab from tensor dims → No effect
2. Fill high positions with -INFINITY → Failed
3. Remove hardcoded values → No effect

---

## Next Steps for Engineers

### Priority 1: Compare with llama.cpp

**Location**: `reference/llama.cpp/` directory in this repo

**What to check**:
1. How does llama.cpp load lm_head?
2. Does it transpose the tensor?
3. What cuBLAS parameters does it use?

**Commands**:
```bash
cd reference/llama.cpp
grep -r "output.weight" src/
grep -r "cublasGemmEx" ggml/src/ggml-cuda/
```

### Priority 2: Verify Memory Layout

**Theory**: Tensor not transposed correctly from GGUF row-major to cuBLAS column-major

**Test**: Manually compute dot product and compare with GEMM

### Priority 3: Try Different Approaches

1. Explicit transpose of lm_head
2. Different cuBLAS compute mode
3. FP32 instead of FP16

---

## Code Changes Made

### Modified Files

1. `src/inference/cuda_backend.rs` - Derive vocab from tensor
2. `src/cuda/model.rs` - Debug output
3. `cuda/src/transformer/qwen_transformer.cpp` - Extensive debug
4. `cuda/src/ffi_inference.cpp` - Remove hardcoded values
5. `cuda/kernels/sampling_wrapper.cu` - Extended debug

### Files to Review

1. `cuda/src/model/qwen_weight_loader.cpp` - How lm_head loaded
2. `src/cuda/weight_loader.rs` - Rust tensor loading
3. `reference/llama.cpp/src/llama-model.cpp` - Reference impl

---

## Test Commands

**Run failing test**:
```bash
cd bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1
```

**Run llama.cpp (works)**:
```bash
cd reference/llama.cpp/build
./bin/main -m /path/to/qwen2.5-0.5b-instruct-fp16.gguf \
  -p "Write a haiku" -n 50 --temp 0.0
```

---

## Reference Materials

### llama.cpp Codebase

**Location**: `reference/llama.cpp/`

**Key Files**:
- `src/llama-model.cpp` - Model loading
- `src/llama-context.cpp` - Inference
- `ggml/src/ggml-cuda/` - CUDA kernels

### Our Implementation

**Key Files**:
- `src/cuda/model.rs` - Model loading (Rust)
- `cuda/src/transformer/qwen_transformer.cpp` - Forward pass
- `cuda/kernels/sampling_wrapper.cu` - Sampling

---

## Success Criteria

You'll know it's fixed when:

✅ Argmax finds tokens with values -4 to +8 (not 14-15)  
✅ Model generates different tokens each step  
✅ Output is coherent text  
✅ Test passes with good output  

---

## Questions?

1. Read **COMPLETE_INVESTIGATION_REPORT.md** - Most comprehensive
2. Check **BUG_STATUS_UPDATED.md** - Current status
3. Look at **reference/llama.cpp/** - Working implementation

---

**Last Updated**: 2025-10-06 13:08  
**Status**: ✅ Deep investigation complete - Solution ready for implementation  
**Next Update**: After fix is tested

---

## Summary of Latest Investigation (2025-10-06 13:08)

### What We Found

After analyzing llama.cpp's CUDA implementation, we discovered:

1. **Root Cause**: Our cuBLAS call uses wrong parameters
   - ❌ Wrong transpose flag: `CUBLAS_OP_N` instead of `CUBLAS_OP_T`
   - ❌ Wrong matrix dimensions: `m = vocab_size` instead of `m = hidden_dim`

2. **Why Previous Fix Failed**: Only changed transpose flag, not dimensions
   - This created a mismatch between transpose operation and matrix shape
   - Result: Catastrophic failure

3. **The Correct Fix**: Change BOTH transpose flag AND dimensions
   - Change `CUBLAS_OP_N` → `CUBLAS_OP_T`
   - Change `m = vocab_size` → `m = hidden_dim`
   - This matches llama.cpp exactly

### Confidence Level

**High (85%)** - The fix matches llama.cpp's working implementation exactly.

### Next Steps

1. Read **INVESTIGATION_SUMMARY.md** for complete overview
2. Follow **ACTION_PLAN_REVISED.md** for implementation
3. Test and verify the fix works

**Good luck!** 🚀
