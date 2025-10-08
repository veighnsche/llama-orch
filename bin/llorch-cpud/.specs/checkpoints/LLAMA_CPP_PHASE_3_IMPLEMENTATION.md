# TEAM-004: Phase 3 - Implementation
**Part of:** llama.cpp Instrumentation Master Plan  
**Duration:** 2-3 hours  
**Status:** ⏳ PENDING  
**Depends on:** Phase 2 (Mapping) must be complete

---

## Objective

Implement checkpoint extraction utilities and add instrumentation code to all 6 checkpoint locations.

**Goal:** Add clean, conditional checkpoint extraction that only runs when `LLORCH_VALIDATE=1` is set.

---

## Step 3.1: Add Dependencies (15 min)

### Update Build System

**File:** `CMakeLists.txt` (root directory)

**Add option:**
```cmake
# TEAM-004: Checkpoint extraction for llorch-cpud validation
option(LLORCH_VALIDATE "Enable checkpoint extraction" OFF)

if(LLORCH_VALIDATE)
    add_definitions(-DLLORCH_VALIDATE)
    message(STATUS "TEAM-004: Checkpoint extraction enabled")
endif()
```

### Verify Build Configuration

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp
mkdir -p build
cd build

# Configure with checkpoint support
cmake .. -DLLORCH_VALIDATE=ON

# Verify option is set
grep "TEAM-004" CMakeCache.txt
```

**Expected output:**
```
-- TEAM-004: Checkpoint extraction enabled
```

### Checklist

- [ ] CMakeLists.txt updated with LLORCH_VALIDATE option
- [ ] Build configuration verified
- [ ] Option appears in cmake output

---

## Step 3.2: Create Checkpoint Utilities (30 min)

### Create Header File

**File:** `src/llama-checkpoint.h` (new file)

```cpp
// TEAM-004: Checkpoint extraction utilities for llorch-cpud validation
// Created: 2025-10-08
// Purpose: Extract intermediate tensor values for multi-reference validation
//
// Usage:
//   1. Build with -DLLORCH_VALIDATE=ON
//   2. Run with LLORCH_VALIDATE=1 environment variable
//   3. Checkpoints saved to /tmp/llama_cpp_checkpoints/*.bin
//
// Format:
//   Binary format: [n_dims:int32][shape:int64[n_dims]][data:float32[n_elements]]

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace llama_checkpoint {

// TEAM-004: Check if checkpoint extraction is enabled
inline bool is_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = getenv("LLORCH_VALIDATE") != nullptr ? 1 : 0;
    }
    return enabled == 1;
}

// TEAM-004: Get checkpoint directory path
inline const char * get_checkpoint_dir() {
    const char * dir = getenv("LLORCH_CHECKPOINT_DIR");
    return dir ? dir : "/tmp/llama_cpp_checkpoints";
}

// TEAM-004: Save tensor to binary file
// Format: [shape_dims][shape_data][tensor_data]
inline void save_tensor(
    const char * checkpoint_name,
    const struct ggml_tensor * tensor
) {
    if (!is_enabled()) {
        return;
    }
    
    // TEAM-004: Build filename
    char filename[512];
    snprintf(filename, sizeof(filename), 
             "%s/%s.bin", 
             get_checkpoint_dir(),
             checkpoint_name);
    
    // TEAM-004: Open file
    FILE * f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "TEAM-004 ERROR: Failed to open %s\n", filename);
        return;
    }
    
    // TEAM-004: Write shape metadata
    int32_t n_dims = ggml_n_dims(tensor);
    fwrite(&n_dims, sizeof(int32_t), 1, f);
    
    for (int i = 0; i < n_dims; i++) {
        int64_t dim = tensor->ne[i];
        fwrite(&dim, sizeof(int64_t), 1, f);
    }
    
    // TEAM-004: Get tensor data
    size_t n_elements = ggml_nelements(tensor);
    float * data_f32 = nullptr;
    
    // TEAM-004: Handle different tensor types and backends
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        // TEAM-004: Host tensor - direct access
        if (tensor->type == GGML_TYPE_F32) {
            data_f32 = (float *) tensor->data;
        } else {
            // TEAM-004: Need conversion to F32
            data_f32 = new float[n_elements];
            // Convert based on type
            if (tensor->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, data_f32, n_elements);
            } else {
                fprintf(stderr, "TEAM-004 WARNING: Unsupported tensor type %d\n", tensor->type);
                fclose(f);
                return;
            }
        }
    } else {
        // TEAM-004: Backend tensor (GPU) - need to copy to host
        data_f32 = new float[n_elements];
        ggml_backend_tensor_get(tensor, data_f32, 0, n_elements * sizeof(float));
    }
    
    // TEAM-004: Write data
    fwrite(data_f32, sizeof(float), n_elements, f);
    
    // TEAM-004: Cleanup
    if (data_f32 != (float *)tensor->data) {
        delete[] data_f32;
    }
    
    fclose(f);
    
    // TEAM-004: Log success
    fprintf(stderr, "✅ TEAM-004: Checkpoint %s saved [", checkpoint_name);
    for (int i = 0; i < n_dims; i++) {
        fprintf(stderr, "%lld%s", (long long)tensor->ne[i], 
                i < n_dims-1 ? "x" : "");
    }
    fprintf(stderr, "]\n");
}

// TEAM-004: Initialize checkpoint directory
inline void init() {
    if (!is_enabled()) {
        return;
    }
    
    // TEAM-004: Create checkpoint directory
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", get_checkpoint_dir());
    int result = system(cmd);
    (void)result; // Suppress unused warning
    
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  TEAM-004: Checkpoint Extraction Enabled                 ║\n");
    fprintf(stderr, "║  Directory: %-44s ║\n", get_checkpoint_dir());
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");
}

// TEAM-004: Finalize checkpoint extraction
inline void finalize() {
    if (!is_enabled()) {
        return;
    }
    
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  TEAM-004: Checkpoint Extraction Complete                ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");
}

} // namespace llama_checkpoint
```

### Checklist

- [ ] `src/llama-checkpoint.h` created
- [ ] All functions implemented
- [ ] TEAM-004 signatures on all code
- [ ] Error handling included
- [ ] Logging messages added

---

## Step 3.3: Add Initialization Call (5 min)

### Update Backend Initialization

**File:** `src/llama.cpp`

**Find:** `void llama_backend_init(void)` function

**Add after existing init code:**
```cpp
void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }
    
    // TEAM-004: Initialize checkpoint extraction
    #ifdef LLORCH_VALIDATE
        #include "llama-checkpoint.h"
        llama_checkpoint::init();
    #endif
}
```

### Add Finalization Call

**File:** `src/llama.cpp`

**Find:** `void llama_backend_free(void)` function

**Add before existing cleanup:**
```cpp
void llama_backend_free(void) {
    // TEAM-004: Finalize checkpoint extraction
    #ifdef LLORCH_VALIDATE
        #include "llama-checkpoint.h"
        llama_checkpoint::finalize();
    #endif
    
    ggml_quantize_free();
}
```

### Checklist

- [ ] Init call added to `llama_backend_init()`
- [ ] Finalize call added to `llama_backend_free()`
- [ ] Conditional compilation used (#ifdef LLORCH_VALIDATE)
- [ ] TEAM-004 comments added

---

## Step 3.4: Instrument Checkpoint 1 - LayerNorm (15 min)

### Location

**File:** [from Phase 2 mapping]  
**Function:** [from Phase 2 mapping]  
**Line:** ~[from Phase 2 mapping]

### Add Instrumentation

**Insert after LayerNorm computation:**
```cpp
// TEAM-004: CHECKPOINT 1 - LayerNorm Output
// Extracts: First LayerNorm output in first transformer block
// Expected shape: [n_tokens, hidden_size] → [2, 768] for GPT-2
// Validates against: PyTorch reference
// Location: After ln_1 computation, before attention input
#ifdef LLORCH_VALIDATE
    #include "llama-checkpoint.h"
    if (llama_checkpoint::is_enabled()) {
        llama_checkpoint::save_tensor("checkpoint_01_ln1_output", [tensor_variable_name]);
    }
#endif
```

### Verification Steps

1. Build with checkpoint support
2. Run with GPT-2 model
3. Verify checkpoint file created
4. Check shape matches [2, 768]

### Checklist

- [ ] Instrumentation code added
- [ ] Correct tensor variable used
- [ ] TEAM-004 signature present
- [ ] Conditional compilation used
- [ ] Comments explain what/why/where

---

## Step 3.5: Instrument Checkpoint 2 - QKV (15 min)

### Location

**File:** [from Phase 2 mapping]  
**Function:** [from Phase 2 mapping]  
**Line:** ~[from Phase 2 mapping]

### Add Instrumentation

**Insert after QKV split:**
```cpp
// TEAM-004: CHECKPOINT 2 - QKV Projection
// Extracts: Q, K, V after projection and split
// Expected shape: [n_tokens, hidden_size] → [2, 768] each for GPT-2
// Validates against: PyTorch reference
// Location: After QKV split, before attention computation
#ifdef LLORCH_VALIDATE
    #include "llama-checkpoint.h"
    if (llama_checkpoint::is_enabled()) {
        llama_checkpoint::save_tensor("checkpoint_02_q", [q_variable_name]);
        llama_checkpoint::save_tensor("checkpoint_02_k", [k_variable_name]);
        llama_checkpoint::save_tensor("checkpoint_02_v", [v_variable_name]);
    }
#endif
```

### Checklist

- [ ] Instrumentation code added
- [ ] All three tensors (Q, K, V) extracted
- [ ] Correct tensor variables used
- [ ] TEAM-004 signature present

---

## Step 3.6: Instrument Checkpoint 3 - KV Cache (15 min)

### Location

**File:** [from Phase 2 mapping]  
**Function:** [from Phase 2 mapping]  
**Line:** ~[from Phase 2 mapping]

### Add Instrumentation

**Insert after cache update:**
```cpp
// TEAM-004: CHECKPOINT 3 - KV Cache State
// Extracts: Cached K and V values
// Expected shape: [depends on cache implementation]
// Validates against: PyTorch reference
// Location: After cache update, before attention computation
// Note: May need adjustment based on llama.cpp cache structure
#ifdef LLORCH_VALIDATE
    #include "llama-checkpoint.h"
    if (llama_checkpoint::is_enabled()) {
        // TEAM-004: Extract current K, V from cache
        // Adjust based on actual cache structure
        llama_checkpoint::save_tensor("checkpoint_03_cache_k", [k_cache_variable]);
        llama_checkpoint::save_tensor("checkpoint_03_cache_v", [v_cache_variable]);
    }
#endif
```

### Notes

**TEAM-004:** Cache implementation may differ from PyTorch. May need to:
- Extract only current K, V (not full cache history)
- Handle cache structure differences
- Adjust shape expectations

### Checklist

- [ ] Instrumentation code added
- [ ] Cache structure understood
- [ ] Correct extraction method used
- [ ] Notes added about potential differences

---

## Step 3.7: Instrument Checkpoint 4 - Attention Scores (15 min)

### Location

**File:** [from Phase 2 mapping]  
**Function:** [from Phase 2 mapping]  
**Line:** ~[from Phase 2 mapping]

### Add Instrumentation

**Insert after softmax:**
```cpp
// TEAM-004: CHECKPOINT 4 - Attention Scores
// Extracts: Attention weights after softmax
// Expected shape: [n_heads, n_tokens, n_tokens] → [12, 2, 2] for GPT-2
// Validates against: PyTorch reference
// Location: After softmax, before multiply with V
#ifdef LLORCH_VALIDATE
    #include "llama-checkpoint.h"
    if (llama_checkpoint::is_enabled()) {
        llama_checkpoint::save_tensor("checkpoint_04_scores", [scores_variable_name]);
    }
#endif
```

### Checklist

- [ ] Instrumentation code added
- [ ] After softmax, before V multiply
- [ ] Correct tensor variable used
- [ ] Shape expectations documented

---

## Step 3.8: Instrument Checkpoint 5 - Attention Output (15 min)

### Location

**File:** [from Phase 2 mapping]  
**Function:** [from Phase 2 mapping]  
**Line:** ~[from Phase 2 mapping]

### Add Instrumentation

**Insert after attention projection:**
```cpp
// TEAM-004: CHECKPOINT 5 - Attention Output
// Extracts: Attention output after projection
// Expected shape: [n_tokens, hidden_size] → [2, 768] for GPT-2
// Validates against: PyTorch reference
// Location: After attention output projection, before residual add
#ifdef LLORCH_VALIDATE
    #include "llama-checkpoint.h"
    if (llama_checkpoint::is_enabled()) {
        llama_checkpoint::save_tensor("checkpoint_05_output", [output_variable_name]);
    }
#endif
```

### Checklist

- [ ] Instrumentation code added
- [ ] After projection, before residual
- [ ] Correct tensor variable used
- [ ] TEAM-004 signature present

---

## Step 3.9: Instrument Checkpoint 6 - FFN (15 min)

### Location

**File:** [from Phase 2 mapping]  
**Function:** [from Phase 2 mapping]  
**Line:** ~[from Phase 2 mapping]

### Add Instrumentation

**Insert after FFN projection:**
```cpp
// TEAM-004: CHECKPOINT 6 - FFN Output
// Extracts: FFN output after projection
// Expected shape: [n_tokens, hidden_size] → [2, 768] for GPT-2
// Validates against: PyTorch reference
// Location: After FFN projection (fc → gelu → proj), before residual add
#ifdef LLORCH_VALIDATE
    #include "llama-checkpoint.h"
    if (llama_checkpoint::is_enabled()) {
        llama_checkpoint::save_tensor("checkpoint_06_ffn", [ffn_variable_name]);
    }
#endif
```

### Checklist

- [ ] Instrumentation code added
- [ ] After full FFN, before residual
- [ ] Correct tensor variable used
- [ ] TEAM-004 signature present

---

## Step 3.10: Build and Verify (15 min)

### Clean Build

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Clean previous build
rm -rf build
mkdir build
cd build

# Configure with checkpoint support
cmake .. -DLLORCH_VALIDATE=ON

# Build
make -j$(nproc) 2>&1 | tee build.log
```

### Check for Errors

```bash
# Check build log for errors
grep -i "error" build.log

# Check for TEAM-004 messages
grep "TEAM-004" build.log
```

**Expected:** No errors, TEAM-004 option message present

### Verify Binaries

```bash
# Check that binaries were built
ls -lh bin/llama-cli
ls -lh bin/llama-server

# Verify checkpoint code is compiled in
strings bin/llama-cli | grep "TEAM-004"
```

### Checklist

- [ ] Clean build successful
- [ ] No compilation errors
- [ ] TEAM-004 messages in build log
- [ ] Binaries created
- [ ] Checkpoint code present in binaries

---

## Completion Checklist

### Code Changes
- [ ] `CMakeLists.txt` updated with LLORCH_VALIDATE option
- [ ] `src/llama-checkpoint.h` created with utilities
- [ ] `src/llama.cpp` updated with init/finalize calls
- [ ] Checkpoint 1 instrumented (LayerNorm)
- [ ] Checkpoint 2 instrumented (QKV)
- [ ] Checkpoint 3 instrumented (KV Cache)
- [ ] Checkpoint 4 instrumented (Attention Scores)
- [ ] Checkpoint 5 instrumented (Attention Output)
- [ ] Checkpoint 6 instrumented (FFN)

### Quality Checks
- [ ] All code has TEAM-004 signatures
- [ ] All instrumentation uses conditional compilation
- [ ] All checkpoints have descriptive comments
- [ ] Error handling included
- [ ] Logging messages added

### Build Verification
- [ ] Clean build successful
- [ ] No compilation errors or warnings
- [ ] Binaries created
- [ ] Checkpoint code compiled in

### Ready for Next Phase
- [ ] All instrumentation complete
- [ ] Build verified
- [ ] Ready to proceed to Phase 4 (Testing)

---

## Notes and Issues

**TEAM-004 Notes:**
[Document any issues encountered during implementation]

**Deviations from Plan:**
[Note any changes from the original plan]

**Lessons Learned:**
[Document insights for future instrumentation work]

---

**Status:** ⏳ PENDING  
**Previous Phase:** Phase 2 - Mapping (must be complete)  
**Next Phase:** Phase 4 - Build and Test  
**Estimated Time:** 2-3 hours  
**Actual Time:** [fill in after completion]

---

## TEAM-005 Update

**Detailed task breakdown created:** See `phase3/` directory for individual task files.

Each task has its own markdown file with:
- Exact implementation details
- Verification steps
- Troubleshooting guides
- Success criteria

**Task files:**
- `phase3/TASK_3.1_BUILD_SYSTEM.md` - CMake configuration
- `phase3/TASK_3.2_CHECKPOINT_UTILITIES.md` - Utility library
- `phase3/TASK_3.3_INITIALIZATION.md` - Init/finalize hooks
- `phase3/TASK_3.4_CHECKPOINT_1_LAYERNORM.md` - LayerNorm instrumentation
- `phase3/TASK_3.5_CHECKPOINT_2_QKV.md` - QKV instrumentation
- `phase3/TASK_3.6_CHECKPOINT_3_KV_CACHE.md` - Cache instrumentation
- `phase3/TASK_3.7_CHECKPOINT_4_ATTENTION_SCORES.md` - Scores instrumentation
- `phase3/TASK_3.8_CHECKPOINT_5_ATTENTION_OUTPUT.md` - Output instrumentation
- `phase3/TASK_3.9_CHECKPOINT_6_FFN.md` - FFN instrumentation
- `phase3/TASK_3.10_BUILD_AND_VERIFY.md` - Build and verification
- `phase3/README.md` - Overview and quick start

**See phase3/README.md for execution order and quick start guide.**
