# TEAM-005: Task 3.2 - Eval Callback Implementation
**Part of:** Phase 3 - Implementation  
**Duration:** 45 minutes  
**Status:** ⏳ PENDING (REVISED)  
**Depends on:** Task 3.1 (Wrapper Structure)

---

## ⚠️ APPROACH REVISED

**Old:** Header-only library with inline extraction  
**New:** Eval callback that extracts after tensor computation

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for details.

---

## Objective

Implement eval callback that extracts checkpoints after tensor computation.

**Goal:** Use llama.cpp's official callback API to safely extract tensor data.

---

## Files to Create

**Path:** `bin/llorch-cpud/tools/checkpoint-extractor/src/`

**Files:**
- `checkpoint_callback.h` - Callback interface
- `checkpoint_callback.cpp` - Implementation

---

## Implementation

### Header Structure

```cpp
// TEAM-005: Checkpoint extraction utilities for llorch-cpud validation
// Created by: TEAM-005
// Date: 2025-10-08
//
// Purpose: Extract intermediate tensor values for multi-reference validation
//
// Usage:
//   1. Build with -DLLORCH_VALIDATE=ON
//   2. Run with LLORCH_VALIDATE=1 environment variable
//   3. Checkpoints saved to directory specified by LLORCH_CHECKPOINT_DIR
//      (defaults to /tmp/llama_cpp_checkpoints)
//
// Binary Format:
//   [n_dims:int32][shape:int64[n_dims]][data:float32[n_elements]]
//
// Example:
//   #ifdef LLORCH_VALIDATE
//       #include "llama-checkpoint.h"
//       if (llama_checkpoint::is_enabled()) {
//           llama_checkpoint::save_tensor("checkpoint_01_ln1", tensor);
//       }
//   #endif

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>

namespace llama_checkpoint {

// TEAM-005: Check if checkpoint extraction is enabled at runtime
// Returns true if LLORCH_VALIDATE environment variable is set
inline bool is_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * env = getenv("LLORCH_VALIDATE");
        enabled = (env != nullptr && env[0] != '0') ? 1 : 0;
    }
    return enabled == 1;
}

// TEAM-005: Get checkpoint directory path
// Defaults to /tmp/llama_cpp_checkpoints if LLORCH_CHECKPOINT_DIR not set
inline const char * get_checkpoint_dir() {
    const char * dir = getenv("LLORCH_CHECKPOINT_DIR");
    return dir ? dir : "/tmp/llama_cpp_checkpoints";
}

// TEAM-005: Create directory recursively
// Returns 0 on success, -1 on failure
inline int create_directory(const char * path) {
    char tmp[512];
    char * p = nullptr;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = 0;
    }

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    return mkdir(tmp, 0755);
}

// TEAM-005: Save tensor to binary file
// Format: [n_dims:int32][shape:int64[n_dims]][data:float32[n_elements]]
//
// Parameters:
//   checkpoint_name: Name for the checkpoint file (without extension)
//   tensor: GGML tensor to save
//
// Notes:
//   - Handles both host and backend (GPU) tensors
//   - Converts F16 to F32 automatically
//   - Creates directory if it doesn't exist
//   - Logs success/failure to stderr
inline void save_tensor(
    const char * checkpoint_name,
    const struct ggml_tensor * tensor
) {
    if (!is_enabled()) {
        return;
    }

    if (!tensor) {
        fprintf(stderr, "⚠️  TEAM-005 WARNING: Null tensor for %s\n", checkpoint_name);
        return;
    }

    // TEAM-005: Build filename
    char filename[512];
    snprintf(filename, sizeof(filename),
             "%s/%s.bin",
             get_checkpoint_dir(),
             checkpoint_name);

    // TEAM-005: Open file
    FILE * f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "❌ TEAM-005 ERROR: Failed to open %s\n", filename);
        return;
    }

    // TEAM-005: Write shape metadata
    int32_t n_dims = ggml_n_dims(tensor);
    fwrite(&n_dims, sizeof(int32_t), 1, f);

    for (int i = 0; i < n_dims; i++) {
        int64_t dim = tensor->ne[i];
        fwrite(&dim, sizeof(int64_t), 1, f);
    }

    // TEAM-005: Get tensor data
    size_t n_elements = ggml_nelements(tensor);
    float * data_f32 = nullptr;
    bool need_free = false;

    // TEAM-005: Handle different tensor types and backends
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        // TEAM-005: Host tensor - direct access or conversion
        if (tensor->type == GGML_TYPE_F32) {
            data_f32 = (float *) tensor->data;
        } else if (tensor->type == GGML_TYPE_F16) {
            // TEAM-005: Convert F16 to F32
            data_f32 = new float[n_elements];
            need_free = true;
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, data_f32, n_elements);
        } else {
            fprintf(stderr, "⚠️  TEAM-005 WARNING: Unsupported tensor type %d for %s\n",
                    tensor->type, checkpoint_name);
            fclose(f);
            return;
        }
    } else {
        // TEAM-005: Backend tensor (GPU) - copy to host
        data_f32 = new float[n_elements];
        need_free = true;
        ggml_backend_tensor_get(tensor, data_f32, 0, n_elements * sizeof(float));
    }

    // TEAM-005: Write data
    size_t written = fwrite(data_f32, sizeof(float), n_elements, f);
    if (written != n_elements) {
        fprintf(stderr, "⚠️  TEAM-005 WARNING: Incomplete write for %s (%zu/%zu elements)\n",
                checkpoint_name, written, n_elements);
    }

    // TEAM-005: Cleanup
    if (need_free) {
        delete[] data_f32;
    }

    fclose(f);

    // TEAM-005: Log success with shape
    fprintf(stderr, "✅ TEAM-005: %s [", checkpoint_name);
    for (int i = 0; i < n_dims; i++) {
        fprintf(stderr, "%lld%s", (long long)tensor->ne[i],
                i < n_dims-1 ? " × " : "");
    }
    fprintf(stderr, "] → %s\n", filename);
}

// TEAM-005: Initialize checkpoint system
// Creates directory and prints banner
inline void init() {
    if (!is_enabled()) {
        return;
    }

    // TEAM-005: Create checkpoint directory
    const char * dir = get_checkpoint_dir();
    if (create_directory(dir) != 0 && errno != EEXIST) {
        fprintf(stderr, "⚠️  TEAM-005 WARNING: Failed to create directory %s\n", dir);
    }

    // TEAM-005: Print banner
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  TEAM-005: Checkpoint Extraction Enabled                     ║\n");
    fprintf(stderr, "║  Directory: %-48s ║\n", dir);
    fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");
}

// TEAM-005: Finalize checkpoint system
// Prints completion banner
inline void finalize() {
    if (!is_enabled()) {
        return;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  TEAM-005: Checkpoint Extraction Complete                    ║\n");
    fprintf(stderr, "║  Files saved to: %-43s ║\n", get_checkpoint_dir());
    fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");
}

} // namespace llama_checkpoint
```

---

## Success Criteria

- [ ] File created at correct path
- [ ] All functions implemented
- [ ] TEAM-005 signatures on all code
- [ ] Error handling for all failure cases
- [ ] Logging with emoji indicators (✅ ❌ ⚠️)
- [ ] Supports both host and GPU tensors
- [ ] Handles F16 to F32 conversion
- [ ] Creates directory if missing
- [ ] Header-only (no .cpp file needed)

---

## Design Decisions

**Why header-only:**
- Easy to include where needed
- No build system changes required
- Inline functions have zero overhead when disabled

**Why binary format:**
- Compact storage
- Fast to write
- Easy to read from Python/Rust
- Shape metadata included

**Why environment variables:**
- Runtime control without recompilation
- Standard Unix pattern
- Easy to set in scripts

---

## Testing

After creating the file, verify it compiles:

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp
echo '#include "src/llama-checkpoint.h"' | g++ -x c++ -c - -I. -o /dev/null
```

Should compile without errors.

---

**Status:** ⏳ PENDING  
**Assigned to:** TEAM-005  
**Estimated time:** 30 minutes  
**Actual time:** [fill after completion]
