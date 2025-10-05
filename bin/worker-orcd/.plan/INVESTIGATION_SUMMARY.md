# Investigation Summary - HTTP Connection Bug

**Date**: 2025-10-05  
**Duration**: 22:15 - 22:37 (22 minutes)  
**Status**: ✅ RESOLVED  

## Investigation Trail

This document shows how the investigation evolved through multiple documents:

### 1. Initial Discovery (22:15)
**Document**: `HTTP_CONNECTION_INVESTIGATION.md`

**Symptoms Observed**:
- ✅ Worker starts, model loads
- ✅ `/health` endpoint works
- ❌ `/execute` endpoint fails with "error sending request"

**Initial Hypotheses**:
- Middleware blocking POST requests?
- State/backend issue?
- Request body malformed?
- Axum/Tower bug?

**Status**: All hypotheses were wrong - it was deeper than HTTP layer

### 2. Root Cause Analysis (22:27)
**Document**: `ROOT_CAUSE_FOUND.md`

**Discovery**: Request WAS reaching the server, but inference was crashing!

**First Bug Found**: Use-after-free in GPU pointer management
- GPU pointers stored in HashMap
- HashMap dropped when function returned
- C++ still trying to use those pointers

**Evidence**:
```
🎉 [C++] Using pre-loaded model from Rust (VRAM: 11047898204684.29 MB)  ← Garbage!
First 10 embedding values: 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00  ← All zeros!
```

**Fix Applied**: Global GPU pointer registry

**Status**: Partially correct, but not the full story

### 3. Deeper Debugging (22:30-22:35)
**Added Debug Logging**:
- Logged pointer values at each step
- Verified data was copied to GPU correctly
- Checked pointer values in C++

**Key Discovery**:
```
🔍 [Rust] Copying token_embd.weight to GPU
   GPU pointer: 0x771d4a000000  ← Correct pointer with data

🔍 [C++] Stored token_embd.weight pointer: 0x771d4a000000  ← Stored correctly

🔍 [C++] Retrieved token_embd.weight pointer: 0x771d4a000000  ← Retrieved correctly

   embedding_table = 0x60ea4276e820  ← WRONG POINTER!
```

**Realization**: Pointer was being corrupted AFTER retrieval!

### 4. The Real Bug (22:35)
**Document**: `BUG_FIXED_HTTP_CONNECTION.md`

**Location**: `cuda/src/ffi_inference.cpp:62`

**The Bug**:
```cpp
// WRONG - treats CudaModel* as if it's a QwenModel*
auto* qwen_model = reinterpret_cast<worker::model::QwenModel*>(model_ptr);
```

**Why This Failed**:
- `CudaModel*` is actually a `ModelImpl*`
- `ModelImpl` has a different memory layout than `QwenModel`
- The cast made C++ read from wrong memory offsets
- This gave garbage VRAM values and wrong pointers

**The Fix**:
```cpp
// CORRECT - properly access the QwenModel through ModelImpl
auto* model_impl = reinterpret_cast<worker::ModelImpl*>(model_ptr);
auto* qwen_model = model_impl->get_qwen_model();
```

### 5. Final Result (22:37)
**Both Bugs Fixed**:
1. ✅ GPU pointer lifetime (global registry)
2. ✅ Type cast bug (proper pointer access)

**Test Results**:
```
✅ Generated 100 tokens
✅ Got response with status: 200 OK
✅ HTTP streaming works
✅ Inference completes without crashing
```

## Key Lessons

### What Made This Hard to Debug

1. **Two bugs compounding** - Pointer lifetime + type cast both contributing
2. **Type safety illusion** - `reinterpret_cast` hid the type mismatch
3. **Symptoms far from cause** - HTTP error, but bug was in C++ pointer access
4. **Initial diagnosis incomplete** - Use-after-free was real but not the main issue

### What Broke Through

1. **Systematic pointer tracing** - Logged values at every step
2. **Verification with actual data** - Checked GPU memory contents
3. **Not stopping at first answer** - Kept digging when fix #1 didn't work
4. **Reading actual pointer values** - Revealed the mismatch

### The Smoking Gun

The moment we saw:
```
Retrieved: 0x771d4a000000  ← Correct
Used:      0x60ea4276e820  ← Wrong
```

We knew the pointer was being corrupted between retrieval and use. That led directly to the type cast bug.

## Timeline

- **22:15** - HTTP connection failure discovered
- **22:18** - Confirmed request reaches server
- **22:27** - Found use-after-free bug, implemented fix #1
- **22:30** - Fix #1 applied, still failing
- **22:32** - Added extensive debug logging
- **22:35** - Discovered type cast bug
- **22:37** - Fix #2 applied, **BOTH BUGS FIXED**

**Total Time**: 22 minutes from discovery to resolution

## Documents Created

1. `HTTP_CONNECTION_INVESTIGATION.md` - Initial investigation
2. `ROOT_CAUSE_FOUND.md` - First bug analysis
3. `BUG_FIXED_HTTP_CONNECTION.md` - Complete fix documentation
4. `INVESTIGATION_SUMMARY.md` - This document

## Supporting Evidence

All documents support the final conclusion:
- Initial investigation correctly identified symptoms
- Root cause analysis found first bug (real but incomplete)
- Final fix addressed both bugs
- Test results confirm complete resolution

The investigation documents tell a complete story of systematic debugging that uncovered two separate but related bugs.
