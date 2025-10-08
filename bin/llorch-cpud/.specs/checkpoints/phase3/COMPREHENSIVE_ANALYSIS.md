# TEAM-005: Comprehensive Phase 3 Analysis
**Date:** 2025-10-08  
**Status:** ✅ COMPLETE

---

## Critical Issues Identified and Resolved

### Issue 1: Graph Building vs Execution Timing ❌ → ✅

**Problem:**
- Original plan extracted tensors during graph **building** via `cb()` callbacks
- Tensors are empty/uninitialized during building
- Data only exists after graph **execution**

**Root Cause:**
```
llama.cpp execution flow:
1. Build graph (cb() callbacks fire here) ← Tensors EMPTY
2. Execute graph asynchronously          ← Tensors COMPUTED
3. Retrieve results
```

**Solution:**
Use llama.cpp's official **eval callback** mechanism:
```cpp
// From ggml-backend.h
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,  // Tensor AFTER computation
    bool ask,                 // Permission vs notification
    void * user_data
);
```

**Benefits:**
- ✅ Official API (documented in ggml-backend.h)
- ✅ Called AFTER tensor computation (data is valid)
- ✅ Non-invasive (no llama.cpp source changes)
- ✅ Set-and-forget (register once at context creation)

---

### Issue 2: Blocking and Interference Analysis ✅

**Question:** Will checkpoint extraction block or interfere with llama.cpp?

**Analysis:**

| Aspect | Assessment | Details |
|--------|------------|---------|
| **Blocking execution?** | ❌ NO | Eval callback is synchronous but fast (just I/O) |
| **Modifying tensors?** | ❌ NO | Read-only access, no modifications |
| **Affecting results?** | ❌ NO | Pure observation, no side effects |
| **Memory overhead?** | ✅ MINIMAL | Temporary buffer for tensor copy |
| **Performance impact?** | ✅ ACCEPTABLE | ~10-20% slowdown only when enabled |
| **Set-and-forget?** | ✅ YES | Register once, runs automatically |

**Conclusion:** Safe to use, no interference with llama.cpp operation.

---

### Issue 3: Tensor Name Matching ✅

**How it works:**

1. **During graph building**, `cb()` sets tensor names:
   ```cpp
   cb(cur, "attn_norm", il);  // Sets tensor->name = "attn_norm"
   ```

2. **During execution**, eval callback receives tensor with name:
   ```cpp
   bool eval_callback(ggml_tensor * t, bool ask, void * user_data) {
       const char * name = ggml_get_name(t);
       // name = "attn_norm" (from cb() call)
   }
   ```

3. **Tensor struct** (from ggml.h line 653):
   ```cpp
   struct ggml_tensor {
       // ...
       char name[GGML_MAX_NAME];  // Name set by cb()
       // ...
   };
   ```

**Checkpoint mapping:**
```cpp
// Phase 2 identified these callback names:
"attn_norm"      → Checkpoint 1 (LayerNorm)
"Qcur"           → Checkpoint 2 (Q)
"Kcur"           → Checkpoint 2 (K)
"Vcur"           → Checkpoint 2 (V)
// KV cache: get_k/get_v → Need special handling
"kq_soft_max"    → Checkpoint 4 (Attention scores)
// Attn output: Need to add callback
"ffn_out"        → Checkpoint 6 (FFN)
```

---

### Issue 4: KV Cache Extraction ⚠️ SPECIAL CASE

**Problem:** KV cache tensors retrieved via `mctx_cur->get_k/v()` don't go through `cb()`.

**Solution Options:**

**Option A: Add callbacks for cache tensors** (RECOMMENDED)
```cpp
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);
// TEAM-005: Add callbacks so eval callback can find them
cb(k, "cache_k", il);
cb(v, "cache_v", il);
```

**Option B: Extract via different mechanism**
- Hook into cache update logic
- More invasive, not recommended

**Decision:** Use Option A - minimal code change, consistent with other checkpoints.

---

### Issue 5: Attention Output Callback Missing ⚠️ NEEDS ADDITION

**Problem:** No callback after attention output projection.

**Solution:** Add callback in `build_attn`:
```cpp
if (wo_b) {
    cur = ggml_add(ctx0, cur, wo_b);
}
// TEAM-005: Add callback for checkpoint extraction
cb(cur, "attn_out_proj", il);
return cur;
```

**Impact:** Minimal - one line addition to llama-graph.cpp.

---

## Revised Implementation Strategy

### Approach: Eval Callback + Minimal Code Changes

**What we DON'T need to change:**
- ❌ No changes to graph building logic
- ❌ No changes to execution logic
- ❌ No changes to CMakeLists.txt (use existing build)
- ❌ No conditional compilation needed

**What we DO need:**

1. **Create wrapper tool** (`llorch-checkpoint-extractor`)
   - Links against llama.cpp
   - Registers eval callback
   - Extracts matching tensors

2. **Minimal llama.cpp changes** (2 lines):
   - Add `cb(k, "cache_k", il)` after cache retrieval
   - Add `cb(cur, "attn_out_proj", il)` after attention output

---

## Implementation Plan (Revised)

### Phase 3A: Wrapper Tool (NEW)

**Create:** `bin/llorch-cpud/tools/checkpoint-extractor/`

**Files:**
- `main.cpp` - CLI wrapper around llama.cpp
- `checkpoint_callback.cpp` - Eval callback implementation
- `checkpoint_callback.h` - Callback interface
- `CMakeLists.txt` - Build configuration

**Functionality:**
```cpp
// Pseudo-code
int main(int argc, char** argv) {
    // Parse args (model, prompt, checkpoint_dir)
    
    // Create context with eval callback
    llama_context_params params = llama_context_default_params();
    params.cb_eval = checkpoint_eval_callback;
    params.cb_eval_user_data = &checkpoint_state;
    
    llama_context * ctx = llama_new_context_with_model(model, params);
    
    // Run inference (checkpoints extracted automatically)
    llama_decode(ctx, batch);
    
    // Cleanup
    llama_free(ctx);
}

bool checkpoint_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;  // Always allow
    
    const char * name = ggml_get_name(t);
    if (is_checkpoint_tensor(name)) {
        save_tensor_to_disk(t, name);
    }
    return true;
}
```

### Phase 3B: Minimal llama.cpp Changes

**File 1:** `src/llama-graph.cpp` (2 additions)

```cpp
// Line ~1553: After cache retrieval
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);
cb(k, "cache_k", il);  // TEAM-005: For checkpoint extraction
cb(v, "cache_v", il);  // TEAM-005: For checkpoint extraction

// Line ~1574: After attention output projection
if (wo_b) {
    cur = ggml_add(ctx0, cur, wo_b);
}
cb(cur, "attn_out_proj", il);  // TEAM-005: For checkpoint extraction
return cur;
```

---

## Verification Checklist

### Non-Interference
- [ ] Eval callback doesn't modify tensors
- [ ] Callback returns `true` (allows execution to continue)
- [ ] No blocking I/O in critical path
- [ ] Memory allocated/freed properly

### Correctness
- [ ] Tensors have valid data when callback fires
- [ ] Tensor names match Phase 2 mapping
- [ ] All 6 checkpoints extractable
- [ ] Shapes match expectations

### Performance
- [ ] Acceptable slowdown (~10-20% with extraction)
- [ ] Zero overhead when not enabled
- [ ] No memory leaks
- [ ] Async I/O if needed for large tensors

---

## Task File Updates Required

All 10 task files need updating to reflect:
1. Use eval callback instead of inline extraction
2. Create wrapper tool instead of modifying llama.cpp extensively
3. Add 2 minimal callbacks to llama.cpp
4. Remove conditional compilation approach
5. Update verification steps

---

## Summary

**Original Approach:** ❌
- Extract during graph building
- Tensors empty
- Extensive llama.cpp modifications

**Revised Approach:** ✅
- Extract via eval callback
- Tensors valid
- Minimal llama.cpp changes (2 lines)
- Wrapper tool for extraction logic

**Status:** Ready to update all task files with correct approach.

---

**Completed by:** TEAM-005  
**Date:** 2025-10-08  
**Next:** Update all 10 task files
