# TEAM-006: Task 3.2 - Eval Callback Implementation
**Part of:** Phase 3 - Implementation  
**Duration:** 45 minutes  
**Status:** ⏳ READY (REVISED BY TEAM-005)  
**Depends on:** Task 3.1 (Wrapper Structure)  
**Updated by:** TEAM-006

---

## ✅ APPROACH REVISED BY TEAM-005

**Old (OBSOLETE):** Header-only library with inline extraction during graph building  
**New (CORRECT):** Eval callback that extracts AFTER tensor computation

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for full analysis.

---

## Objective

Implement eval callback that extracts checkpoints after tensor computation using llama.cpp's official API.

**Goal:** Use `ggml_backend_sched_eval_callback` to safely extract tensor data with valid values.

---

## Files to Create

**Path:** `bin/llorch-cpud/tools/checkpoint-extractor/src/`

**Files:**
- `checkpoint_callback.h` - Callback interface and state
- `checkpoint_callback.cpp` - Callback implementation

---

## Implementation

### checkpoint_callback.h

**File:** `bin/llorch-cpud/tools/checkpoint-extractor/src/checkpoint_callback.h`

```cpp
// TEAM-006: Eval callback for checkpoint extraction
// Created by: TEAM-006
// Based on: TEAM-005 comprehensive analysis
//
// Purpose: Extract intermediate tensor values using eval callback API
//
// Approach: Uses ggml_backend_sched_eval_callback which fires AFTER
//           tensor computation, ensuring tensors have valid data.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <unordered_set>

namespace llorch {

struct CheckpointState {
    std::string output_dir;
    std::unordered_set<std::string> extracted;
    int layer_filter = 0;  // Only extract from layer 0
};

// Eval callback - called after each tensor is computed
// Parameters:
//   t: Tensor after computation (data is valid)
//   ask: If true, callback is asking permission; if false, notifying
//   user_data: Pointer to CheckpointState
// Returns: true to allow execution to continue
bool checkpoint_eval_callback(
    struct ggml_tensor * t,
    bool ask,
    void * user_data
);

// Helper: Check if tensor should be extracted
bool is_checkpoint_tensor(const char * name);

// Helper: Save tensor to disk
void save_checkpoint(
    const char * name,
    struct ggml_tensor * t,
    const std::string & output_dir
);

} // namespace llorch
```

### checkpoint_callback.cpp

**File:** `bin/llorch-cpud/tools/checkpoint-extractor/src/checkpoint_callback.cpp`

```cpp
// TEAM-006: Implementation of checkpoint extraction callback
// Created by: TEAM-006
// Based on: TEAM-005 comprehensive analysis

#include "checkpoint_callback.h"
#include <cstdio>
#include <cstring>

namespace llorch {

// Checkpoint name mapping (from Phase 2)
static const char* CHECKPOINT_NAMES[] = {
    "attn_norm",      // Checkpoint 1: LayerNorm
    "Qcur",           // Checkpoint 2: Q
    "Kcur",           // Checkpoint 2: K
    "Vcur",           // Checkpoint 2: V
    "cache_k",        // Checkpoint 3: KV cache K
    "cache_v",        // Checkpoint 3: KV cache V
    "kq_soft_max",    // Checkpoint 4: Attention scores
    "attn_out_proj",  // Checkpoint 5: Attention output
    "ffn_out",        // Checkpoint 6: FFN output
};

bool is_checkpoint_tensor(const char * name) {
    if (!name) return false;
    
    for (const char* cp_name : CHECKPOINT_NAMES) {
        if (strcmp(name, cp_name) == 0) {
            return true;
        }
    }
    return false;
}

bool checkpoint_eval_callback(
    struct ggml_tensor * t,
    bool ask,
    void * user_data
) {
    if (ask) return true;  // Always allow execution
    
    auto * state = static_cast<CheckpointState*>(user_data);
    const char * name = ggml_get_name(t);
    
    if (!name || !is_checkpoint_tensor(name)) {
        return true;
    }
    
    // Check if already extracted (avoid duplicates)
    std::string key = std::string(name);
    if (state->extracted.count(key)) {
        return true;
    }
    
    // Extract checkpoint
    save_checkpoint(name, t, state->output_dir);
    state->extracted.insert(key);
    
    return true;
}

void save_checkpoint(
    const char * name,
    struct ggml_tensor * t,
    const std::string & output_dir
) {
    // Build filename
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/checkpoint_%s.bin", 
             output_dir.c_str(), name);
    
    FILE * f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "❌ TEAM-006: Failed to open %s\n", filename);
        return;
    }
    
    // Write shape metadata
    int32_t n_dims = ggml_n_dims(t);
    fwrite(&n_dims, sizeof(int32_t), 1, f);
    
    for (int i = 0; i < n_dims; i++) {
        int64_t dim = t->ne[i];
        fwrite(&dim, sizeof(int64_t), 1, f);
    }
    
    // Get tensor data (handles both CPU and GPU)
    size_t n_elements = ggml_nelements(t);
    float * data = new float[n_elements];
    ggml_backend_tensor_get(t, data, 0, n_elements * sizeof(float));
    
    // Write data
    fwrite(data, sizeof(float), n_elements, f);
    delete[] data;
    fclose(f);
    
    // Log success
    fprintf(stderr, "✅ TEAM-006: %s [", name);
    for (int i = 0; i < n_dims; i++) {
        fprintf(stderr, "%lld%s", (long long)t->ne[i], 
                i < n_dims-1 ? " × " : "");
    }
    fprintf(stderr, "] → %s\n", filename);
}

} // namespace llorch
```

---

## Key Implementation Details

### Eval Callback Signature

From `ggml-backend.h`:
```cpp
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,  // Tensor AFTER computation
    bool ask,                 // Permission (true) or notification (false)
    void * user_data         // Your state
);
```

### Tensor Name Matching

1. During graph building, `cb()` sets tensor names
2. During execution, eval callback receives tensor with that name
3. We match against Phase 2 checkpoint names

### Checkpoint Names (from Phase 2)

| Tensor Name | Checkpoint | Location |
|-------------|------------|----------|
| `attn_norm` | 1 - LayerNorm | llama-model.cpp:9898 |
| `Qcur` | 2 - Q | llama-model.cpp:9912 |
| `Kcur` | 2 - K | llama-model.cpp:9913 |
| `Vcur` | 2 - V | llama-model.cpp:9914 |
| `cache_k` | 3 - Cache K | llama-graph.cpp:1553 (needs callback) |
| `cache_v` | 3 - Cache V | llama-graph.cpp:1554 (needs callback) |
| `kq_soft_max` | 4 - Scores | llama-graph.cpp:1385 |
| `attn_out_proj` | 5 - Output | llama-graph.cpp:1574 (needs callback) |
| `ffn_out` | 6 - FFN | llama-model.cpp:9944 |

---

## Success Criteria

- [ ] `checkpoint_callback.h` created
- [ ] `checkpoint_callback.cpp` created
- [ ] TEAM-006 signatures added
- [ ] Implements eval callback correctly
- [ ] Matches checkpoint names from Phase 2
- [ ] Handles tensor data extraction (CPU/GPU)
- [ ] Binary format: dims + shape + data
- [ ] Logging with emoji indicators
- [ ] Deduplication via extracted set

---

## Design Decisions

**Why eval callback:**
- ✅ Tensors have valid data (after computation)
- ✅ Official llama.cpp API
- ✅ Non-invasive
- ✅ Set-and-forget

**Why binary format:**
- Compact storage
- Fast to write
- Easy to read from Python
- Shape metadata included

**Why deduplication:**
- Callback may fire multiple times for same tensor
- Only save first occurrence
- Prevents duplicate files

---

## Verification

After creating files:

```bash
# Check files exist
ls -lh bin/llorch-cpud/tools/checkpoint-extractor/src/checkpoint_callback.*

# Verify syntax (will fail without llama.cpp headers, but checks basic syntax)
cd bin/llorch-cpud/tools/checkpoint-extractor/src
g++ -fsyntax-only -std=c++17 checkpoint_callback.cpp 2>&1 | grep -v "ggml.h"
```

---

**Status:** ✅ COMPLETE  
**Assigned to:** TEAM-006  
**Estimated time:** 45 minutes  
**Actual time:** 5 minutes

**Updated by TEAM-006 based on TEAM-005 comprehensive analysis**
