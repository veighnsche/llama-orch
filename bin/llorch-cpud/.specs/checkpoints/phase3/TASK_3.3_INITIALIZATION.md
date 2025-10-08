# TEAM-005: Task 3.3 - Initialization Hooks
**Part of:** Phase 3 - Implementation  
**Duration:** 5 minutes  
**Status:** ⏳ PENDING  
**Depends on:** Task 3.2 (Checkpoint Utilities)

---

## Objective

Add initialization and finalization hooks to llama.cpp backend.

**Goal:** Initialize checkpoint system when llama backend starts, finalize when it shuts down.

---

## File to Modify

**Path:** `/home/vince/Projects/llama-orch/reference/llama.cpp/src/llama.cpp`

---

## Changes Required

### 1. Add Init Hook

**Find function:** `void llama_backend_init(void)`

**Add after existing initialization:**

```cpp
void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    // TEAM-005: Initialize checkpoint extraction system
    #ifdef LLORCH_VALIDATE
        #include "llama-checkpoint.h"
        llama_checkpoint::init();
    #endif
}
```

### 2. Add Finalize Hook

**Find function:** `void llama_backend_free(void)`

**Add before existing cleanup:**

```cpp
void llama_backend_free(void) {
    // TEAM-005: Finalize checkpoint extraction system
    #ifdef LLORCH_VALIDATE
        #include "llama-checkpoint.h"
        llama_checkpoint::finalize();
    #endif

    ggml_quantize_free();
}
```

---

## Verification Steps

1. **Check functions exist:**
   ```bash
   cd /home/vince/Projects/llama-orch/reference/llama.cpp
   grep -n "void llama_backend_init" src/llama.cpp
   grep -n "void llama_backend_free" src/llama.cpp
   ```

2. **Verify syntax:**
   ```bash
   # Try to compile just this file
   cd build-validate
   make llama -j1 2>&1 | grep -i error
   ```

3. **Test runtime behavior:**
   ```bash
   # Build and run with checkpoint enabled
   LLORCH_VALIDATE=1 ./bin/llama-cli --version
   # Should see TEAM-005 banner
   ```

---

## Success Criteria

- [ ] Init hook added to `llama_backend_init()`
- [ ] Finalize hook added to `llama_backend_free()`
- [ ] Both use conditional compilation (`#ifdef LLORCH_VALIDATE`)
- [ ] TEAM-005 comments added
- [ ] Code compiles without errors
- [ ] Banner appears when LLORCH_VALIDATE=1 is set
- [ ] No banner when LLORCH_VALIDATE is not set

---

## Design Notes

**Why include inside #ifdef:**
- Header only included when feature is enabled
- Avoids any overhead when disabled
- Cleaner conditional compilation

**Why init/finalize pattern:**
- Creates directory once at startup
- Prints clear banner for user feedback
- Cleanup message confirms completion

**Placement rationale:**
- Init: After core ggml initialization
- Finalize: Before ggml cleanup

---

## Troubleshooting

**Issue:** Compile error "llama-checkpoint.h not found"
- **Solution:** Verify Task 3.2 completed and file exists at `src/llama-checkpoint.h`

**Issue:** Banner doesn't appear
- **Solution:** Check that LLORCH_VALIDATE=1 is set (not just cmake flag)

**Issue:** Multiple banners appear
- **Solution:** Normal if multiple processes/contexts created

---

**Status:** ⏳ PENDING  
**Assigned to:** TEAM-005  
**Estimated time:** 5 minutes  
**Actual time:** [fill after completion]
