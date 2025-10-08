# TEAM-006: Task 3.3 - Main CLI Wrapper
**Part of:** Phase 3 - Implementation  
**Duration:** 30 minutes  
**Status:** ⏳ READY (REVISED BY TEAM-005)  
**Depends on:** Task 3.2 (Eval Callback)  
**Updated by:** TEAM-006

---

## ✅ APPROACH REVISED BY TEAM-005

**Old (OBSOLETE):** Add hooks to llama.cpp backend  
**New (CORRECT):** Create standalone CLI that registers eval callback

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for full analysis.

---

## Objective

Create main CLI wrapper that loads llama.cpp, registers eval callback, and runs inference.

**Goal:** Standalone tool that extracts checkpoints without modifying llama.cpp.

---

## File to Create

**Path:** `bin/llorch-cpud/tools/checkpoint-extractor/src/main.cpp`

---

## Implementation

**File:** `bin/llorch-cpud/tools/checkpoint-extractor/src/main.cpp`

```cpp
// TEAM-006: Checkpoint extractor CLI
// Created by: TEAM-006
// Based on: TEAM-005 comprehensive analysis

#include "checkpoint_callback.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompt> [output_dir]\n", argv[0]);
        fprintf(stderr, "\nExtracts intermediate tensor checkpoints from llama.cpp inference.\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s gpt2.gguf \"Hello world\" /tmp/checkpoints\n", argv[0]);
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
        fprintf(stderr, "❌ Failed to load model: %s\n", model_path);
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
    
    fprintf(stderr, "Tokenized prompt: %d tokens\n", n_tokens);
    
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

---

## Key Implementation Details

### Eval Callback Registration

```cpp
llama_context_params ctx_params = llama_context_default_params();
ctx_params.cb_eval = llorch::checkpoint_eval_callback;  // Register callback
ctx_params.cb_eval_user_data = &checkpoint_state;       // Pass state
```

### Workflow

1. Load model
2. Create context with eval callback registered
3. Tokenize prompt
4. Run inference → callback fires automatically after each tensor
5. Cleanup

### Banner Output

```
╔══════════════════════════════════════════════════════════╗
║  TEAM-006: Checkpoint Extraction Enabled                 ║
║  Output: /tmp/llama_cpp_checkpoints                      ║
╚══════════════════════════════════════════════════════════╝

Tokenized prompt: 2 tokens
✅ TEAM-006: attn_norm [2 × 768] → /tmp/llama_cpp_checkpoints/checkpoint_attn_norm.bin
✅ TEAM-006: Qcur [64 × 12 × 2] → /tmp/llama_cpp_checkpoints/checkpoint_Qcur.bin
...

╔══════════════════════════════════════════════════════════╗
║  TEAM-006: Extraction Complete                           ║
║  Extracted 9 checkpoints                                 ║
╚══════════════════════════════════════════════════════════╝
```

---

## Success Criteria

- [ ] `main.cpp` created
- [ ] TEAM-006 signatures added
- [ ] Registers eval callback correctly
- [ ] Loads model and creates context
- [ ] Tokenizes and runs inference
- [ ] Prints banners and status
- [ ] Proper cleanup
- [ ] Error handling for all failure cases

---

## Verification

After creating file:

```bash
# Check file exists
ls -lh bin/llorch-cpud/tools/checkpoint-extractor/src/main.cpp

# Verify syntax (will fail without llama.cpp headers, but checks basic syntax)
cd bin/llorch-cpud/tools/checkpoint-extractor/src
g++ -fsyntax-only -std=c++17 main.cpp 2>&1 | grep -v "llama.h"
```

---

## Design Notes

**Why standalone CLI:**
- ✅ No llama.cpp modifications
- ✅ Clean separation of concerns
- ✅ Easy to test and debug
- ✅ Reusable for different models

**Why eval callback:**
- Callback fires AFTER tensor computation
- Tensors have valid data
- Official llama.cpp API
- Set-and-forget

**Error handling:**
- Check model load
- Check context creation
- Check decode result
- Clean up on all paths

---

## Troubleshooting

**Issue:** Compile errors about llama.h
- **Solution:** Ensure llama.cpp is built and headers are accessible
- **Solution:** Set include paths in CMakeLists.txt

**Issue:** Callback not firing
- **Solution:** Verify cb_eval and cb_eval_user_data are set correctly
- **Solution:** Check that tensors have names (set by cb() during graph building)

**Issue:** No checkpoints extracted
- **Solution:** Verify llama.cpp has cb() calls for checkpoint tensors
- **Solution:** Check that Phase 2 mapping is correct

---

**Status:** ✅ COMPLETE  
**Assigned to:** TEAM-006  
**Estimated time:** 30 minutes  
**Actual time:** 5 minutes

**Updated by TEAM-006 based on TEAM-005 comprehensive analysis**
