# HANDOFF TO TEAM-006: Phase 3 Implementation
**From:** TEAM-005  
**To:** TEAM-006  
**Date:** 2025-10-08  
**Subject:** Phase 3 Checkpoint Extraction - Revised Implementation Plan

---

## Executive Summary

TEAM-005 completed Phase 2 (mapping) and discovered **critical issues** with the original Phase 3 plan. We've revised the approach and documented everything for your implementation.

**Status:**
- ✅ Phase 1 (Reconnaissance) - Complete
- ✅ Phase 2 (Mapping) - Complete  
- ✅ Phase 3 Analysis - Complete
- ⏳ Phase 3 Implementation - **READY FOR YOU**

---

## Critical Discovery: Original Plan Was Flawed ❌

### The Problem

Original plan tried to extract tensors during graph **building**, but:
- Tensors are empty during building
- Data only exists after graph **execution**
- Would have extracted garbage/uninitialized memory

### The Solution ✅

Use llama.cpp's **official eval callback** API:
- Callback fires AFTER each tensor is computed
- Tensors have valid data
- Non-invasive, no blocking
- Official, documented API

**See:** `phase3/COMPREHENSIVE_ANALYSIS.md` for full details.

---

## What You Need to Do

### Task Overview

Instead of modifying llama.cpp extensively, you'll:

1. **Create wrapper tool** (`llorch-checkpoint-extractor`)
   - Links against llama.cpp
   - Registers eval callback
   - Extracts matching tensors

2. **Add 3 callbacks to llama.cpp** (minimal changes)
   - 2 for KV cache (line ~1553-1554)
   - 1 for attention output (line ~1574)

3. **Test and verify**
   - Run with GPT-2 model
   - Validate checkpoint files
   - Compare with PyTorch

### Detailed Tasks

#### Task 1: Create Wrapper Tool Structure (20 min)

**Directory:**
```
bin/llorch-cpud/tools/checkpoint-extractor/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── checkpoint_callback.cpp
│   └── checkpoint_callback.h
└── README.md
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.14)
project(llorch-checkpoint-extractor)

find_package(llama REQUIRED)

add_executable(llorch-checkpoint-extractor
    src/main.cpp
    src/checkpoint_callback.cpp
)

target_link_libraries(llorch-checkpoint-extractor PRIVATE llama)
```

#### Task 2: Implement Eval Callback (45 min)

**checkpoint_callback.h:**
```cpp
// TEAM-006: Eval callback for checkpoint extraction
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

**checkpoint_callback.cpp:**
```cpp
// TEAM-006: Implementation
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

#### Task 3: Implement Main CLI (30 min)

**main.cpp:**
```cpp
// TEAM-006: Checkpoint extractor CLI
#include "checkpoint_callback.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompt> [output_dir]\n", argv[0]);
        return 1;
    }
    
    const char * model_path = argv[1];
    const char * prompt = argv[2];
    const char * output_dir = argc > 3 ? argv[3] : "/tmp/llama_cpp_checkpoints";
    
    // Create output directory
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", output_dir);
    system(cmd);
    
    // Initialize llama backend
    llama_backend_init();
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "❌ Failed to load model\n");
        return 1;
    }
    
    // Create context with eval callback
    llorch::CheckpointState checkpoint_state;
    checkpoint_state.output_dir = output_dir;
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.cb_eval = llorch::checkpoint_eval_callback;
    ctx_params.cb_eval_user_data = &checkpoint_state;
    
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "❌ Failed to create context\n");
        llama_free_model(model);
        return 1;
    }
    
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  TEAM-006: Checkpoint Extraction Enabled                 ║\n");
    fprintf(stderr, "║  Output: %-47s ║\n", output_dir);
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n\n");
    
    // Tokenize prompt
    std::vector<llama_token> tokens;
    tokens.resize(llama_n_ctx(ctx));
    int n_tokens = llama_tokenize(model, prompt, strlen(prompt), 
                                   tokens.data(), tokens.size(), true, false);
    tokens.resize(n_tokens);
    
    // Run inference (checkpoints extracted via callback)
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "❌ Failed to decode\n");
    }
    
    llama_batch_free(batch);
    
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  TEAM-006: Extraction Complete                           ║\n");
    fprintf(stderr, "║  Extracted %zu checkpoints                               ║\n", 
            checkpoint_state.extracted.size());
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n\n");
    
    // Cleanup
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    
    return 0;
}
```

#### Task 4: Add Callbacks to llama.cpp (5 min)

**File:** `reference/llama.cpp/src/llama-graph.cpp`

**Change 1 (line ~1553-1554):**
```cpp
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);
// TEAM-006: Add callbacks for checkpoint extraction
cb(k, "cache_k", il);
cb(v, "cache_v", il);
```

**Change 2 (line ~1574):**
```cpp
if (wo_b) {
    cur = ggml_add(ctx0, cur, wo_b);
}
// TEAM-006: Add callback for checkpoint extraction
cb(cur, "attn_out_proj", il);
return cur;
```

#### Task 5: Build and Test (15 min)

```bash
# Build wrapper tool
cd bin/llorch-cpud/tools/checkpoint-extractor
mkdir build && cd build
cmake ..
make

# Test with GPT-2
./llorch-checkpoint-extractor \
    /path/to/gpt2.gguf \
    "Hello world" \
    /tmp/checkpoints

# Verify checkpoints
ls -lh /tmp/checkpoints/
# Should see 9 files:
# - checkpoint_attn_norm.bin
# - checkpoint_Qcur.bin, checkpoint_Kcur.bin, checkpoint_Vcur.bin
# - checkpoint_cache_k.bin, checkpoint_cache_v.bin
# - checkpoint_kq_soft_max.bin
# - checkpoint_attn_out_proj.bin
# - checkpoint_ffn_out.bin
```

---

## Important Notes

### Non-Interference Verified ✅

| Aspect | Status | Details |
|--------|--------|---------|
| Blocking? | ❌ NO | Callback is fast, returns immediately |
| Modifying data? | ❌ NO | Read-only access |
| Affecting results? | ❌ NO | Pure observation |
| Set-and-forget? | ✅ YES | Register once, runs automatically |

### Checkpoint Mapping

From Phase 2, these tensor names map to checkpoints:

| Tensor Name | Checkpoint | Location | Shape (GPT-2) |
|-------------|------------|----------|---------------|
| `attn_norm` | 1 - LayerNorm | llama-model.cpp:9898 | [2, 768] |
| `Qcur` | 2 - Q | llama-model.cpp:9912 | [64, 12, 2] |
| `Kcur` | 2 - K | llama-model.cpp:9913 | [64, 12, 2] |
| `Vcur` | 2 - V | llama-model.cpp:9914 | [64, 12, 2] |
| `cache_k` | 3 - Cache K | llama-graph.cpp:1553 | [64, 12, n_kv, 1] |
| `cache_v` | 3 - Cache V | llama-graph.cpp:1554 | [64, 12, n_kv, 1] |
| `kq_soft_max` | 4 - Scores | llama-graph.cpp:1385 | [n_kv, 2, 12, 1] |
| `attn_out_proj` | 5 - Output | llama-graph.cpp:1574 | [768, 2, 1] |
| `ffn_out` | 6 - FFN | llama-model.cpp:9944 | [2, 768] |

---

## Files and Documentation

### What TEAM-005 Delivered

1. **Phase 2 Mapping:** `LLAMA_CPP_PHASE_2_MAPPING.md`
   - All 6 checkpoints mapped
   - Exact line numbers
   - TEAM-005 markers in code

2. **Comprehensive Analysis:** `phase3/COMPREHENSIVE_ANALYSIS.md`
   - Problem identification
   - Solution details
   - Implementation strategy

3. **Updated Task Files:** `phase3/TASK_3.*.md`
   - ⚠️ Some still need updating
   - Use this handoff as primary guide

4. **Code Markers:** TEAM-005 comments in llama.cpp
   - Search for "TEAM-005" to find locations

### What You Need to Create

1. Wrapper tool in `bin/llorch-cpud/tools/checkpoint-extractor/`
2. 3 callback additions to llama.cpp
3. Test results and validation

---

## Success Criteria

- [ ] Wrapper tool builds successfully
- [ ] Links against llama.cpp
- [ ] Eval callback registered
- [ ] All 9 checkpoint files created
- [ ] Correct binary format (dims + shape + data)
- [ ] No crashes or errors
- [ ] No interference with llama.cpp operation

---

## Timeline

**Estimated:** 2-3 hours total

| Task | Duration | Priority |
|------|----------|----------|
| Wrapper structure | 20 min | P0 |
| Callback implementation | 45 min | P0 |
| Main CLI | 30 min | P0 |
| llama.cpp changes | 5 min | P0 |
| Build & test | 15 min | P0 |
| Validation | 30 min | P1 |

---

## Questions?

**Primary documentation:**
- `phase3/COMPREHENSIVE_ANALYSIS.md` - Full analysis
- `phase3/README.md` - Overview
- `LLAMA_CPP_PHASE_2_MAPPING.md` - Checkpoint locations

**Contact:**
- TEAM-005 (completed analysis)
- Check TEAM-005 markers in code

---

## Next Phase

After Phase 3 complete:
- **Phase 4:** Testing and Validation
  - Compare checkpoints with PyTorch
  - Validate numerical accuracy
  - Document discrepancies

---

**Good luck, TEAM-006!**

**Signed,**  
TEAM-005  
2025-10-08 17:54 CET

---

## Appendix: Quick Reference

### Eval Callback Signature
```cpp
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,  // Tensor after computation
    bool ask,                 // Permission (true) or notification (false)
    void * user_data         // Your state
);
```

### Registration
```cpp
llama_context_params params = llama_context_default_params();
params.cb_eval = your_callback;
params.cb_eval_user_data = &your_state;
llama_context * ctx = llama_new_context_with_model(model, params);
```

### Tensor Access
```cpp
const char * name = ggml_get_name(tensor);
int n_dims = ggml_n_dims(tensor);
int64_t shape[4] = {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]};
size_t n_elements = ggml_nelements(tensor);

// Get data (works for CPU and GPU)
float * data = new float[n_elements];
ggml_backend_tensor_get(tensor, data, 0, n_elements * sizeof(float));
```
