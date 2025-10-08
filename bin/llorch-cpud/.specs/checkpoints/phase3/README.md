# Phase 3 Implementation - Task Breakdown

**Owner:** TEAM-005  
**Duration:** 2-3 hours  
**Status:** ⚠️ NEEDS REVISION

---

## ⚠️ CRITICAL ISSUE IDENTIFIED

**Problem:** Original plan tried to extract tensors during graph **building** via callbacks, but tensors don't have data until graph **execution**.

**Root cause:** llama.cpp uses deferred execution:
1. Graph is **built** (callbacks fire here - tensors empty!)
2. Graph is **executed** asynchronously (tensors computed here)
3. Results retrieved

**Impact:** Our checkpoint extraction would read empty/uninitialized tensors.

---

## ✅ SOLUTION FOUND: Use Official Eval Callback

**See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for full details.**

### Quick Summary

**Problem:** Original plan extracted during graph building (tensors empty).

**Solution:** Use llama.cpp's eval callback (called after computation).

```cpp
// From ggml-backend.h:
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,  // Tensor AFTER computation ✅
    bool ask,                 // Permission vs notification
    void * user_data
);
```

### Revised Approach

**Instead of modifying llama.cpp extensively:**
1. Create wrapper tool (`llorch-checkpoint-extractor`)
2. Register eval callback at context creation
3. Callback extracts matching tensors
4. Add only 2 callbacks to llama.cpp (for cache & attn output)

### Non-Interference Verification ✅

| Question | Answer | Details |
|----------|--------|---------|
| Blocking? | ❌ NO | Fast I/O, returns immediately |
| Modifying data? | ❌ NO | Read-only access |
| Affecting results? | ❌ NO | Pure observation |
| Set-and-forget? | ✅ YES | Register once, runs automatically |

**Conclusion:** Safe, non-invasive, official API.

---

## Overview

Phase 3 implements checkpoint extraction using llama.cpp's official eval callback mechanism via a wrapper tool. Minimal changes to llama.cpp (2 callback additions).

**STATUS: ✅ READY FOR IMPLEMENTATION (with revised approach)**

---

## Task Files

### Setup Tasks (REVISED)

**⚠️ ALL TASK FILES NEED UPDATING - See COMPREHENSIVE_ANALYSIS.md**

**Old approach (OBSOLETE):**
- Modify llama.cpp extensively
- Conditional compilation
- Extract during graph building

**New approach (CORRECT):**
- Create wrapper tool
- Use eval callback
- Minimal llama.cpp changes (2 lines)

1. **[TASK_3.1_BUILD_SYSTEM.md](TASK_3.1_BUILD_SYSTEM.md)** (NEEDS REVISION)
   - ~~Add CMake option~~ → Build wrapper tool instead
   - ~~Conditional compilation~~ → Not needed

2. **[TASK_3.2_CHECKPOINT_UTILITIES.md](TASK_3.2_CHECKPOINT_UTILITIES.md)** (30 min)
   - Create `llama-checkpoint.h` header-only library
   - Implement tensor save/load utilities
   - Add error handling and logging

3. **[TASK_3.3_INITIALIZATION.md](TASK_3.3_INITIALIZATION.md)** (5 min)
   - Add init hook to `llama_backend_init()`
   - Add finalize hook to `llama_backend_free()`
   - Print startup/completion banners

### Instrumentation Tasks (90 minutes)

4. **[TASK_3.4_CHECKPOINT_1_LAYERNORM.md](TASK_3.4_CHECKPOINT_1_LAYERNORM.md)** (15 min)
   - Instrument LayerNorm output
   - Location: `llama-model.cpp` line 9898
   - Shape: `[2, 768]`

5. **[TASK_3.5_CHECKPOINT_2_QKV.md](TASK_3.5_CHECKPOINT_2_QKV.md)** (15 min)
   - Instrument Q, K, V projections
   - Location: `llama-model.cpp` lines 9915-9921
   - Shape: `[64, 12, 2]` each (3D)

6. **[TASK_3.6_CHECKPOINT_3_KV_CACHE.md](TASK_3.6_CHECKPOINT_3_KV_CACHE.md)** (15 min)
   - Instrument KV cache state
   - Location: `llama-graph.cpp` lines 1550-1551
   - Shape: `[64, 12, n_kv, 1]` (4D with history)

7. **[TASK_3.7_CHECKPOINT_4_ATTENTION_SCORES.md](TASK_3.7_CHECKPOINT_4_ATTENTION_SCORES.md)** (15 min)
   - Instrument attention scores after softmax
   - Location: `llama-graph.cpp` line 1385
   - Shape: `[n_kv, 2, 12, 1]` (4D permuted)

8. **[TASK_3.8_CHECKPOINT_5_ATTENTION_OUTPUT.md](TASK_3.8_CHECKPOINT_5_ATTENTION_OUTPUT.md)** (20 min)
   - Instrument attention output after projection
   - Location: `llama-graph.cpp` line 1571
   - Shape: `[768, 2, 1]` (3D)

9. **[TASK_3.9_CHECKPOINT_6_FFN.md](TASK_3.9_CHECKPOINT_6_FFN.md)** (15 min)
   - Instrument FFN output
   - Location: `llama-model.cpp` line 9951
   - Shape: `[2, 768]`

### Verification Task (15 minutes)

10. **[TASK_3.10_BUILD_AND_VERIFY.md](TASK_3.10_BUILD_AND_VERIFY.md)** (15 min)
    - Clean build with checkpoint support
    - Runtime verification
    - File format validation
    - Performance testing

---

## Execution Order

Tasks must be completed in order due to dependencies:

```
3.1 (Build System)
  ↓
3.2 (Utilities)
  ↓
3.3 (Initialization)
  ↓
3.4 → 3.5 → 3.6 → 3.7 → 3.8 → 3.9 (Checkpoints)
  ↓
3.10 (Verify)
```

---

## Quick Start

### 1. Setup (Tasks 3.1-3.3)

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Follow TASK_3.1: Modify CMakeLists.txt
# Follow TASK_3.2: Create src/llama-checkpoint.h
# Follow TASK_3.3: Modify src/llama.cpp
```

### 2. Instrument Checkpoints (Tasks 3.4-3.9)

Each checkpoint task provides:
- Exact file and line number
- Code to insert
- Verification steps
- Expected output

### 3. Build and Test (Task 3.10)

```bash
mkdir build-validate
cd build-validate
cmake .. -DLLORCH_VALIDATE=ON
make -j$(nproc)

export LLORCH_VALIDATE=1
./bin/llama-cli --model gpt2.gguf --prompt "Test" --n-predict 1
```

---

## Success Criteria

### Code Quality
- [ ] All code has TEAM-005 signatures
- [ ] Conditional compilation used everywhere
- [ ] Error handling for all failure cases
- [ ] Clear logging messages

### Functionality
- [ ] Clean build with `-DLLORCH_VALIDATE=ON`
- [ ] Clean build without flag (backward compatible)
- [ ] All 9 checkpoint files created
- [ ] Correct binary format
- [ ] No crashes or errors

### Documentation
- [ ] Each task file complete
- [ ] Verification steps documented
- [ ] Troubleshooting guides included
- [ ] Shape transformations explained

---

## File Locations

### Modified Files
- `/reference/llama.cpp/CMakeLists.txt` - Build configuration
- `/reference/llama.cpp/src/llama.cpp` - Init/finalize hooks
- `/reference/llama.cpp/src/llama-model.cpp` - Checkpoints 1, 2, 6
- `/reference/llama.cpp/src/llama-graph.cpp` - Checkpoints 3, 4, 5

### New Files
- `/reference/llama.cpp/src/llama-checkpoint.h` - Utility library

### Output Files (runtime)
- `/tmp/llama_cpp_checkpoints/checkpoint_01_ln1_output.bin`
- `/tmp/llama_cpp_checkpoints/checkpoint_02_q.bin`
- `/tmp/llama_cpp_checkpoints/checkpoint_02_k.bin`
- `/tmp/llama_cpp_checkpoints/checkpoint_02_v.bin`
- `/tmp/llama_cpp_checkpoints/checkpoint_03_cache_k.bin`
- `/tmp/llama_cpp_checkpoints/checkpoint_03_cache_v.bin`
- `/tmp/llama_cpp_checkpoints/checkpoint_04_scores.bin`
- `/tmp/llama_cpp_checkpoints/checkpoint_05_output.bin`
- `/tmp/llama_cpp_checkpoints/checkpoint_06_ffn.bin`

---

## Checkpoint Summary

| ID | Name | Location | Shape | File |
|----|------|----------|-------|------|
| 1 | LayerNorm | llama-model.cpp:9898 | [2, 768] | checkpoint_01_ln1_output.bin |
| 2 | Q/K/V | llama-model.cpp:9915-9921 | [64, 12, 2] × 3 | checkpoint_02_{q,k,v}.bin |
| 3 | KV Cache | llama-graph.cpp:1550-1551 | [64, 12, n_kv, 1] × 2 | checkpoint_03_cache_{k,v}.bin |
| 4 | Attn Scores | llama-graph.cpp:1385 | [n_kv, 2, 12, 1] | checkpoint_04_scores.bin |
| 5 | Attn Output | llama-graph.cpp:1571 | [768, 2, 1] | checkpoint_05_output.bin |
| 6 | FFN Output | llama-model.cpp:9951 | [2, 768] | checkpoint_06_ffn.bin |

---

## Notes

**Design Philosophy:**
- Minimal invasiveness - code only active when explicitly enabled
- Zero overhead when disabled - conditional compilation
- Clear logging - emoji indicators and progress messages
- Robust error handling - graceful degradation on failure

**Testing Strategy:**
- Build verification - ensure clean compilation
- Runtime verification - test with/without flag
- Format verification - validate binary structure
- Numerical verification - compare with PyTorch (Phase 4)

---

## Next Phase

After completing Phase 3:
- **Phase 4:** Testing and Validation
  - Compare checkpoints with PyTorch reference
  - Validate numerical accuracy
  - Document any discrepancies
  - Create validation report

---

**Created by:** TEAM-005  
**Date:** 2025-10-08  
**Status:** Ready for execution
