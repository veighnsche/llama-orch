# TEAM-007 AUDIT REPORT: Critical Issues Found
**Agent:** TEAM-007 (Bond Division)  
**Date:** 2025-10-08  
**Mission:** Verify and validate TEAM-006's Phase 3 implementation  
**Status:** 🔴 **FAILED - Multiple Critical Issues**

---

## Executive Summary

TEAM-007 conducted a thorough audit of TEAM-006's checkpoint extraction implementation. While the approach is sound and callbacks are correctly added to llama.cpp, **the implementation has critical bugs that prevent it from working as specified**.

**Verdict:** ❌ Implementation is NOT ready for Phase 4. Requires fixes before proceeding.

---

## Critical Issues Found

### 🔴 ISSUE 1: Layer Filtering Not Implemented
**Severity:** HIGH  
**File:** `tools/checkpoint-extractor/src/checkpoint_callback.cpp`

**Problem:**
- Header declares `layer_filter = 0` to extract only from layer 0
- Callback implementation NEVER checks layer number
- Will extract from ALL layers, creating duplicate/wrong checkpoints

**Evidence:**
```cpp
// checkpoint_callback.h line 22
int layer_filter = 0;  // Only extract from layer 0

// checkpoint_callback.cpp - NO layer check anywhere!
bool checkpoint_eval_callback(
    struct ggml_tensor * t,
    bool ask,
    void * user_data
) {
    // ... checks tensor name but NEVER checks layer!
}
```

**Impact:** Extracts 9 × N_LAYERS checkpoints instead of 9, wrong data.

**Fix Required:**
The llama.cpp callbacks pass layer info (`il` parameter), but the eval callback signature doesn't receive it. Need to either:
1. Parse layer from tensor name, OR
2. Use a different extraction approach

---

### 🔴 ISSUE 2: Build System Incomplete
**Severity:** HIGH  
**File:** `tools/checkpoint-extractor/CMakeLists.txt`

**Problem:**
- Uses `find_package(llama REQUIRED)` 
- llama.cpp has NO build directory (`/reference/llama.cpp/build` doesn't exist)
- llama.cpp not built, no library to link against
- CMake will fail immediately

**Evidence:**
```bash
$ ls /home/vince/Projects/llama-orch/reference/llama.cpp/build
No such file or directory
```

**Fix Required:**
1. Build llama.cpp first with CMake
2. Install it OR set CMAKE_PREFIX_PATH correctly
3. Document build prerequisites

---

### 🔴 ISSUE 3: Misleading Documentation
**Severity:** MEDIUM  
**File:** `tools/checkpoint-extractor/README.md`

**Problem:**
README claims "Non-invasive, no llama.cpp source modifications needed" but Team 006 DID modify llama.cpp by adding 3 callbacks.

**Evidence:**
```markdown
# README.md line 16
- Non-invasive, no llama.cpp source modifications needed
```

But we verified:
- `llama-graph.cpp:1556` - Added `cb(k, "cache_k", il);`
- `llama-graph.cpp:1557` - Added `cb(v, "cache_v", il);`  
- `llama-graph.cpp:1578` - Added `cb(cur, "attn_out_proj", il);`

**Fix Required:** Update documentation to accurately state modifications required.

---

### ⚠️ ISSUE 4: Missing Layer Information in Callback
**Severity:** MEDIUM  
**File:** `tools/checkpoint-extractor/src/checkpoint_callback.h`

**Problem:**
The eval callback signature is:
```cpp
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t, 
    bool ask, 
    void * user_data
);
```

But llama.cpp's `cb()` calls include layer index:
```cpp
cb(k, "cache_k", il);  // il = layer index
```

The eval callback receives the tensor AFTER the `cb()` call, but the layer info is lost. Cannot filter by layer without parsing tensor metadata.

---

### ⚠️ ISSUE 5: No Verification of Callback Existence
**Severity:** LOW  
**Files:** Task files 3.4, 3.5, 3.9

**Problem:**
Handoff says "verify existing callbacks" but provides no actual verification that they work with the wrapper tool. The callbacks exist in llama.cpp source, but:
- No test that they fire during inference
- No test that wrapper receives them
- No validation of extracted data

---

## What Actually Works ✅

1. **Callbacks added correctly** - All 3 new callbacks are in llama.cpp
2. **Callback signature correct** - Matches `ggml_backend_sched_eval_callback`
3. **Binary format correct** - Shape + data format is sound
4. **Deduplication logic** - Uses `std::unordered_set` to avoid duplicates
5. **File I/O** - Checkpoint saving logic is correct

---

## Root Cause Analysis

**Why did Team 006 miss these issues?**

1. **No compilation attempt** - Never tried to build the tool
2. **No runtime testing** - Never ran the extractor
3. **Incomplete understanding** - Didn't realize eval callback lacks layer info
4. **Documentation inconsistency** - Claimed non-invasive but modified llama.cpp

**The approach is fundamentally sound, but implementation is incomplete.**

---

## Required Fixes

### Fix 1: Implement Layer Filtering

**Option A:** Parse layer from tensor name
```cpp
// Extract layer number from tensor name like "blk.0.attn_norm"
int get_layer_from_name(const char* name) {
    if (strncmp(name, "blk.", 4) == 0) {
        return atoi(name + 4);
    }
    return -1;
}

bool checkpoint_eval_callback(...) {
    int layer = get_layer_from_name(ggml_get_name(t));
    if (layer != state->layer_filter) return true;
    // ... rest of logic
}
```

**Option B:** Extract from all layers, organize by layer
```cpp
// Save to: /output/layer_0/checkpoint_attn_norm.bin
char filename[512];
snprintf(filename, sizeof(filename), "%s/layer_%d/checkpoint_%s.bin", 
         output_dir.c_str(), layer, name);
```

### Fix 2: Build llama.cpp First

Add to handoff instructions:
```bash
# Step 0: Build llama.cpp (REQUIRED)
cd /home/vince/Projects/llama-orch/reference/llama.cpp
mkdir -p build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
```

### Fix 3: Fix Documentation

Update README.md:
```markdown
## Requirements

- llama.cpp with 3 checkpoint callbacks added (see TEAM-006 modifications)
- Callbacks added to: `src/llama-graph.cpp` lines 1556, 1557, 1578
```

### Fix 4: Add Verification Tests

Create smoke test:
```bash
# Test that callbacks fire
./llorch-checkpoint-extractor test.gguf "test" /tmp/test
ls /tmp/test/*.bin | wc -l  # Should be 9
```

---

## Handoff to Next Team

**DO NOT PROCEED TO PHASE 4** until these issues are fixed.

### Immediate Actions Required:

1. ✅ **Fix layer filtering** (Option A or B above)
2. ✅ **Build llama.cpp** with modifications
3. ✅ **Build wrapper tool** and verify it compiles
4. ✅ **Run smoke test** with actual model
5. ✅ **Verify 9 checkpoints** extracted correctly
6. ✅ **Update documentation** to match reality

### Estimated Time: 2-3 hours

Once fixed and tested, Phase 3 will be truly complete.

---

## Files Requiring Changes

### Code Fixes:
- `tools/checkpoint-extractor/src/checkpoint_callback.cpp` - Add layer filtering
- `tools/checkpoint-extractor/src/checkpoint_callback.h` - Update layer_filter usage

### Documentation Fixes:
- `tools/checkpoint-extractor/README.md` - Fix "non-invasive" claim
- `phase3/HANDOFF_TO_TEAM_007.md` - Add llama.cpp build step
- `phase3/TASK_3.10_BUILD_AND_VERIFY.md` - Add Step 0 for llama.cpp build

---

## Lessons Learned

1. **Always compile before claiming complete** ❌
2. **Test runtime behavior, not just code review** ❌  
3. **Verify API assumptions** (eval callback doesn't get layer info) ❌
4. **Documentation must match implementation** ❌

---

## TEAM-007 Signature

**Mission Status:** Intelligence gathered, vulnerabilities identified, countermeasures proposed.

The checkpoint extraction system has potential, but current implementation is **not operational**. Recommend immediate remediation before proceeding to Phase 4 validation.

**Classified Level:** UNCLASSIFIED  
**Distribution:** All subsequent teams  
**Action Required:** IMMEDIATE

---

*"The name's Bond. Build Bond. And this code needs debugging."*  
— TEAM-007, Checkpoint Extraction Audit Division

**END REPORT**
