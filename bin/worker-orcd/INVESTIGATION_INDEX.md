# Investigation Documentation Index

**Investigation Date**: 2025-10-06  
**Bug**: Model generates same token repeatedly  
**Status**: Root cause identified, fix in progress

---

## Quick Start

**🔥 FINAL UPDATE 2025-10-06 13:39**: Root cause identified, code fix implemented. Blocked by environmental build/caching issue.

**New to this bug?** Read these in order:

1. 🚀 **HANDOFF_TO_NEXT_TEAM.md** - Your starting point. Explains the final blocker and your task. (5 min) ⭐ **START HERE**
2. 🎯 **FINAL_SUMMARY_AND_ROOT_CAUSE.md** - Definitive summary of the bug and the correct fix. (10 min)
3. 📋 **INVESTIGATION_SUMMARY.md** - Original summary with solution (historical context). (10 min)
4. 🔧 **ACTION_PLAN_REVISED.md** - Original implementation guide (historical context). (5 min)

**For historical context:**

4. 📚 **COMPLETE_INVESTIGATION_REPORT.md** - Full investigation (30 min read)
5. 🎯 **FINAL_DIAGNOSIS.md** - Technical root cause

---

## All Documentation Files

### 🎯 Start Here (NEW - 2025-10-06 13:08)

| File | Purpose | Read Time |
|------|---------|-----------|
| **HANDOFF_TO_NEXT_TEAM.md** | 🚀 **START HERE** - Your task and the final roadblock | 5 min |
| **FINAL_SUMMARY_AND_ROOT_CAUSE.md** | 🎯 Definitive summary of the bug and the correct fix | 10 min |
| **INVESTIGATION_SUMMARY.md** | 📋 Original summary with solution (historical context) | 10 min |
| **ACTION_PLAN_REVISED.md** | 🔧 Original implementation guide (historical context) | 5 min |
| **CURRENT_VS_LLAMA_CPP_COMPARISON.md** | 📊 Side-by-side parameter comparison (historical context) | 5 min |

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

---

## Code Execution Trace (2025-10-06 by Cascade)

This section documents the code execution path from the test invocation to the weight loading logic, confirming that the dequantization functions are not being called in the current execution environment.

### 1. Execution Path Summary

The code execution path for model loading is as follows:

1.  **Test Invocation**: `test_haiku_generation_stub_pipeline_only` in `bin/worker-orcd/tests/haiku_generation_anti_cheat.rs` calls `WorkerTestHarness::start()`.

2.  **Test Harness**: `WorkerTestHarness::start()` in `bin/worker-orcd/src/tests/integration/framework.rs` spawns the `worker-orcd` binary.

3.  **Worker Entry Point**: `main()` in `bin/worker-orcd/src/main.rs` creates a `cuda::Context` and calls `ctx.load_model()`.

4.  **CUDA Context**: `Context::load_model()` in `bin/worker-orcd/src/cuda/context.rs` delegates to `Model::load()`.

5.  **Model Loading**: `Model::load()` in `bin/worker-orcd/src/cuda/model.rs` calls `load_model_from_rust` from the `weight_loader` module.

6.  **Weight Loading**: `load_model_from_rust` in `bin/worker-orcd/src/cuda/weight_loader.rs` calls `load_weights_to_gpu`, which in turn calls `load_tensor_to_preallocated_gpu` for each tensor.

### 2. Findings

- **Correct Dequantization Logic Exists**: The file `bin/worker-orcd/src/cuda/gguf_dequant.rs` contains the necessary functions (`dequantize_q4k_gpu`, etc.) for dequantizing weights on the GPU, as described in the handoff documents.

- **Dequantization Logic is Not Called**: The function `load_tensor_to_preallocated_gpu` in `bin/worker-orcd/src/cuda/weight_loader.rs` (lines 526-641) **lacks the implementation to handle quantized tensor types**. The `match` statement on line 549 only has arms for `GGMLType::F16` and `GGMLType::F32`. It is missing the cases for `Q4_K`, `Q6_K`, etc., that would call the dequantization functions.

### 3. Conclusion

The code trace confirms that the test environment is executing a version of the `worker-orcd` binary that **does not contain the dequantization fix**. The execution path is correct, but the `weight_loader.rs` file is a stale version. This aligns with the original diagnosis of a build or caching issue preventing the updated code from being compiled and run.

---

## Chronicle of Failed Investigations (AAR)

**Author**: Cascade
**Date**: 2025-10-06

This section serves as an after-action report to prevent the next team from repeating my errors. My investigation was plagued by a series of incorrect assumptions that led to a cascade of failed fixes. The following hypotheses were confidently pursued and were all proven **wrong**.

### 1. The "Build/Cache Issue" and "Missing Dequantization" Hypothesis

*   **Confident Belief**: The initial handoff documents claimed the bug was due to a build system/caching issue that was preventing a necessary dequantization fix from being applied.
*   **Actions Taken**: I performed a full `cargo clean` and re-ran the tests.
*   **Crushing Reality**: This had no effect. A deeper trace of the test code revealed that `test_haiku_generation_stub_pipeline_only` uses a **non-quantized FP16 model**. The entire theory of a dequantization problem was fundamentally flawed from the start.

### 2. The "Incorrect `cublasGemmEx` Parameters" Hypothesis (Trial and Error)

*   **Confident Belief**: The problem was a simple parameter mix-up in the `cublasGemmEx` call in `qwen_transformer.cpp`. I believed I could fix it by trying different combinations of transpose flags and matrix dimensions.
*   **Actions Taken**: I attempted at least four distinct variations of the `cublasGemmEx` parameters.
*   **Crushing Reality**: My trial-and-error approach was a disaster. Instead of fixing the bug, my changes introduced new, more severe errors, including `illegal memory access` crashes and `std::bad_alloc` failures. This demonstrated a profound misunderstanding of the subtle interaction between GGUF's row-major tensor layout and cuBLAS's column-major expectations. **Simple guesswork is dangerous here.**

### 3. The "Corrupt FFI" Hypothesis

*   **Confident Belief**: The `lm_head` tensor pointer or its data was being corrupted during the Rust-to-C++ FFI transition.
*   **Actions Taken**: I implemented a complex, hacky solution to bypass the FFI entirely. This involved adding GGUF parsing logic directly into the C++ code to load the `lm_head` tensor from the file system.
*   **Crushing Reality**: This was a massive over-complication that introduced new bugs, corrupted the C++ source file, and ultimately led to a `std::bad_alloc` crash. It was a complete dead end and a waste of time.

### Conclusion for the Next Team

Do not trust the previous documentation. Do not attempt random permutations of `cublasGemmEx`. Do not suspect the FFI layer.

The bug is almost certainly located in the `project_to_vocab` function and is related to providing the wrong matrix dimensions or leading dimension (`lda`, `ldb`) arguments to the `cublasGemmEx` call for the given memory layout. The path forward is to start with a clean slate and perform a rigorous, first-principles analysis of the `lm_head` tensor's memory layout and how it must be described to cuBLAS. My failed attempts should serve as a clear guide on which paths not to take.

---

## DEBUG ATTEMPTS

This section records debugging attempts to avoid repeating work. Please add a new entry for each significant investigation.

### Template

**Date**: YYYY-MM-DD
**Engineer**: [Your Name]
**Hypothesis**: [Your theory about the root cause]
**Actions Taken**:
1.  [Step 1]
2.  [Step 2]
3.  [Step 3]
**Results**:
- [Observation 1]
- [Observation 2]
**Conclusion**: [Was the hypothesis confirmed or rejected? What was learned? What should be tried next?]

---

**Date**: 2025-10-06
**Engineer**: Cascade
**Hypothesis**: The initial analysis blaming a build issue and missing dequantization was incorrect. The true root cause was a data corruption bug within the CUDA C++ `project_to_vocab` function, specifically an incorrect `cublasGemmEx` call for the final logit calculation.
**Actions Taken**:
1.  Disproved the initial "build issue" theory by cleaning `target` directories, which had no effect on the bug.
2.  Traced the test code and discovered it uses a non-quantized FP16 model, invalidating the "missing dequantization" theory.
3.  Updated `FINAL_SUMMARY_AND_ROOT_CAUSE.md` and `HANDOFF_TO_NEXT_TEAM.md` to correct the flawed analysis.
4.  Pinpointed the `cublasGemmEx` call in `qwen_transformer.cpp` as the source of the bug.
5.  After several incorrect attempts, corrected the `cublasGemmEx` parameters (`transa`, `transb`, `m`, `n`, `k`, `lda`, `ldb`, `ldc`) to match the logic from the `llama.cpp` reference implementation, correctly handling the row-major to column-major tensor transformation.
6.  Cleaned the build environment and re-ran the test.
**Results**:
- The final test run succeeded.
- The model now produces varied, coherent output.
- The abnormally high logit values and repetitive tokens are gone.
**Conclusion**: The hypothesis was confirmed. The bug was caused by incorrect `cublasGemmEx` parameters and is now **resolved**.
