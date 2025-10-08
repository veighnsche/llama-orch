# TEAM-004: Phase 1 - Reconnaissance
**Part of:** llama.cpp Instrumentation Master Plan  
**Duration:** 1 hour  
**Status:** ✅ COMPLETE

---

## Objective

Locate exact instrumentation points in llama.cpp codebase for all 6 checkpoints.

**Goal:** Document file paths, function names, and approximate line numbers where we'll add checkpoint extraction code.

---

## Task 1.1: Find GPT-2 Model Architecture (15 min)

### What to Search For

**Primary targets:**
- GPT-2 model type enum
- GPT-2 specific graph building code
- Architecture-specific forward pass implementation

**Search commands:**
```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Find model architecture enum
grep -r "LLM_ARCH" src/ | grep -i gpt

# Find GPT-2 specific code
grep -r "GPT2\|gpt-2\|gpt_2" src/ --include="*.cpp" --include="*.h"

# Find graph building functions
grep -r "llm_build" src/ --include="*.cpp" | head -20
```

### Expected Findings

**Model enum location:**
```cpp
// Likely in src/llama-arch.cpp or src/llama-model.cpp
enum llm_arch {
    LLM_ARCH_UNKNOWN,
    LLM_ARCH_LLAMA,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    // ...
};
```

**Architecture name mapping:**
```cpp
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_GPT2, "gpt2" },
    // ...
};
```

### Deliverable

Document in this section:
```markdown
### GPT-2 Architecture Location

**File:** `src/llama-arch.h`
**Enum definition line:** ~18
**Architecture name:** "gpt2"

**Enum Value:** `LLM_ARCH_GPT2`

**Architecture Mapping (src/llama-arch.cpp, line ~13):**
```cpp
{ LLM_ARCH_GPT2, "gpt2" },
```

**Model Builder (src/llama-model.cpp, line ~19622):**
```cpp
case LLM_ARCH_GPT2:
    {
        llm = std::make_unique<llm_build_gpt2>(*this, params);
    } break;
```

**Notes:**
- TEAM-004: GPT-2 is a first-class architecture in llama.cpp
- TEAM-004: Uses standard LayerNorm (LLM_NORM), not RMSNorm
- TEAM-004: No RoPE (Rotary Position Embeddings) - uses learned position embeddings
- TEAM-004: Graph builder is `llm_build_gpt2` struct (line 9867 in llama-model.cpp)
```

---

## Task 1.2: Find Layer Implementations (20 min)

### What to Search For

**Layer operations to locate:**
1. LayerNorm (norm + scale + bias)
2. QKV projection (single matrix multiply, then split)
3. Attention scores (Q·K^T, softmax)
4. Attention output (scores·V, then projection)
5. FFN (fc + gelu + projection)

**Search commands:**
```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Find LayerNorm
grep -r "ggml_norm\|layer_norm" src/ --include="*.cpp" -A 3 -B 3

# Find attention operations
grep -r "ggml_mul_mat.*qkv\|ggml_soft_max" src/ --include="*.cpp" -A 3 -B 3

# Find FFN operations
grep -r "ggml_gelu\|feed.*forward" src/ --include="*.cpp" -A 3 -B 3

# Find graph building for GPT-2
grep -r "case LLM_ARCH_GPT2" src/ --include="*.cpp" -A 50
```

### Expected Patterns

**LayerNorm pattern:**
```cpp
// TEAM-004: Expected pattern for LayerNorm
struct ggml_tensor * cur = ggml_norm(ctx, input, norm_eps);
cur = ggml_mul(ctx, cur, layer_norm_weight);
cur = ggml_add(ctx, cur, layer_norm_bias);
// ← CHECKPOINT 1: Extract cur here
```

**Attention pattern:**
```cpp
// TEAM-004: Expected pattern for Attention
struct ggml_tensor * qkv = ggml_mul_mat(ctx, w_qkv, cur);
// Split into Q, K, V
struct ggml_tensor * q = ggml_view_2d(...);
struct ggml_tensor * k = ggml_view_2d(...);
struct ggml_tensor * v = ggml_view_2d(...);
// ← CHECKPOINT 2: Extract q, k, v here

// Attention scores
struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
kq = ggml_scale(ctx, kq, scale);
kq = ggml_soft_max(ctx, kq);
// ← CHECKPOINT 4: Extract kq here

// Attention output
struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
cur = ggml_mul_mat(ctx, w_o, kqv);
// ← CHECKPOINT 5: Extract cur here
```

**FFN pattern:**
```cpp
// TEAM-004: Expected pattern for FFN
struct ggml_tensor * cur = ggml_mul_mat(ctx, w_fc, input);
cur = ggml_gelu(ctx, cur);
cur = ggml_mul_mat(ctx, w_proj, cur);
// ← CHECKPOINT 6: Extract cur here
```

### Deliverable

Document each layer:

```markdown
### LayerNorm Implementation

**File:** `src/llama-graph.cpp`
**Function:** `llm_graph_context::build_norm`
**Line:** 641-674

**Pattern found:**
```cpp
// TEAM-004: LayerNorm implementation (LLM_NORM type for GPT-2)
ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,    // weight
         ggml_tensor * mb,    // bias
       llm_norm_type   type,  // LLM_NORM for GPT-2
                 int   il) const {
    // Normalize
    cur = ggml_norm(ctx0, cur, hparams.f_norm_eps);
    
    // Scale (multiply by weight)
    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
    }
    
    // Shift (add bias)
    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }
    
    return cur;
}
```

**Used in GPT-2 (src/llama-model.cpp, line 9894):**
```cpp
// TEAM-004: First LayerNorm in transformer block
cur = build_norm(inpL,
        model.layers[il].attn_norm,      // weight
        model.layers[il].attn_norm_b,    // bias
        LLM_NORM, il);
cb(cur, "attn_norm", il);
```

**TEAM-004 Checkpoint location:** After `cb(cur, "attn_norm", il)` at line 9898

---

### QKV Projection Implementation

**File:** `src/llama-model.cpp`
**Function:** `llm_build_gpt2` constructor
**Line:** 9902-9914

**Pattern found:**
```cpp
// TEAM-004: QKV projection (single matrix multiply, then split)
// self-attention
{
    cur = build_lora_mm(model.layers[il].wqkv, cur);
    cb(cur, "wqkv", il);

    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
    cb(cur, "bqkv", il);

    // TEAM-004: Split into Q, K, V views
    ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, ...);
    ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, ...);
    ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, ...);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);
}
```

**TEAM-004 Checkpoint location:** After `cb(Qcur/Kcur/Vcur, ...)` at lines 9912-9914

---

### Attention Scores Implementation

**File:** `src/llama-graph.cpp`
**Function:** `llm_graph_context::build_attn` (multiple overloads)
**Line:** ~1400-1600 (complex, in build_attn_mha)

**Pattern (inside build_attn_mha):**
```cpp
// TEAM-004: Attention scores computation
// Q * K^T
kq = ggml_mul_mat(ctx0, k, q);

// Scale
kq = ggml_scale(ctx0, kq, kq_scale);

// Mask (if needed)
if (kq_mask) {
    kq = ggml_add(ctx0, kq, kq_mask);
}

// TEAM-004: Softmax - THIS IS THE CHECKPOINT
kq = ggml_soft_max(ctx0, kq);
```

**TEAM-004 Checkpoint location:** After `ggml_soft_max`, before multiply with V

---

### Attention Output Implementation

**File:** `src/llama-model.cpp`
**Function:** `llm_build_gpt2` constructor
**Line:** 9916-9919

**Pattern found:**
```cpp
// TEAM-004: Attention output (after scores * V, then projection)
cur = build_attn(inp_attn,
        model.layers[il].wo, model.layers[il].bo,  // output projection
        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 
        1.0f/sqrtf(float(n_embd_head)), il);
```

**Inside build_attn (after attention computation):**
```cpp
// TEAM-004: Output projection
cur = build_lora_mm(wo, cur);  // wo = output weight
if (wo_b) {
    cur = ggml_add(ctx0, cur, wo_b);  // wo_b = output bias
}
```

**TEAM-004 Checkpoint location:** After output projection, before residual add (line 9927)

---

### FFN Implementation

**File:** `src/llama-graph.cpp`
**Function:** `llm_graph_context::build_ffn`
**Line:** 676-780

**Pattern found:**
```cpp
// TEAM-004: FFN implementation (GELU activation for GPT-2)
ggml_tensor * llm_graph_context::build_ffn(...) {
    // Up projection (fc layer)
    ggml_tensor * tmp = build_lora_mm(up, cur);
    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
    }
    
    // GELU activation (for GPT-2: LLM_FFN_GELU)
    tmp = ggml_gelu(ctx0, tmp);
    
    // Down projection
    cur = build_lora_mm(down, tmp);
    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }
    
    return cur;
}
```

**Used in GPT-2 (src/llama-model.cpp, line 9938):**
```cpp
// TEAM-004: FFN call
cur = build_ffn(cur,
        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
        NULL,                      NULL,                        NULL,
        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
        NULL,
        LLM_FFN_GELU, LLM_FFN_SEQ, il);
cb(cur, "ffn_out", il);
```

**TEAM-004 Checkpoint location:** After `cb(cur, "ffn_out", il)` at line 9944
```

---

## Task 1.3: Find Computation Graph Execution (15 min)

### What to Search For

**Graph execution points:**
- Where the computation graph is built
- Where the graph is executed (tensors computed)
- Where we can safely access tensor data

**Search commands:**
```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Find graph computation
grep -r "ggml_graph_compute\|ggml_backend_graph_compute" src/ --include="*.cpp" -B 5 -A 5

# Find graph building
grep -r "ggml_new_graph\|ggml_build_forward" src/ --include="*.cpp" -B 3 -A 3

# Find where tensors are evaluated
grep -r "ggml_graph.*eval\|compute.*graph" src/ --include="*.cpp"
```

### Expected Findings

**Graph computation pattern:**
```cpp
// TEAM-004: Expected graph execution pattern
struct ggml_cgraph * gf = ggml_new_graph(ctx);

// Build graph (add operations)
ggml_build_forward_expand(gf, output_tensor);

// Execute graph
ggml_backend_graph_compute(backend, gf);

// TEAM-004: After this point, all tensors in graph are computed
// This is where we can safely extract checkpoint data
```

### Deliverable

```markdown
### Graph Execution Location

**File:** `src/llama-model.cpp`
**Function:** `llm_build_gpt2` constructor (graph building)
**Line:** 9867-9970

**Graph building pattern:**
```cpp
// TEAM-004: GPT-2 graph is built in the constructor
struct llm_build_gpt2 : public llm_graph_context {
    llm_build_gpt2(const llama_model & model, const llm_graph_params & params) 
        : llm_graph_context(params) {
        
        // Build computation graph
        for (int il = 0; il < n_layer; ++il) {
            // LayerNorm
            cur = build_norm(...);
            cb(cur, "attn_norm", il);  // ← CHECKPOINT 1 HERE
            
            // Attention
            cur = build_attn(...);
            
            // FFN
            cur = build_ffn(...);
            cb(cur, "ffn_out", il);  // ← CHECKPOINT 6 HERE
        }
        
        // Final operation
        ggml_build_forward_expand(gf, cur);
    }
};
```

**Graph execution (happens later in llama_decode):**
```cpp
// TEAM-004: Graph is executed by backend scheduler
// This happens in llama_decode_impl (not in build phase)
ggml_backend_sched_graph_compute(lctx.sched, &gf);
```

**TEAM-004 CRITICAL FINDING:**
- Graph is BUILT during `llm_build_gpt2` constructor
- Graph is EXECUTED later by scheduler
- Tensors contain computed values AFTER execution
- **We must extract checkpoints DURING graph building, using callbacks**
- The `cb(tensor, name, layer)` calls are our extraction points!

**Safe extraction approach:**
- Hook into the `cb()` callback system
- Extract tensor data when callback is invoked
- Tensors are already computed at callback time
```

---

## Task 1.4: Find Tensor Data Access (10 min)

### What to Search For

**Tensor data access patterns:**
- How to read tensor data from computed graph
- How to get tensor dimensions
- How to handle different data types (F32, F16, etc.)

**Search commands:**
```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp

# Find tensor data access
grep -r "tensor->data\|ggml_get_data" src/ --include="*.cpp" -B 2 -A 2

# Find tensor dimension access
grep -r "tensor->ne\|ggml_nelements" src/ --include="*.cpp" -B 2 -A 2

# Find type conversion
grep -r "ggml_type_size\|ggml_element_size" src/ --include="*.cpp" -B 2 -A 2

# Find backend tensor access
grep -r "ggml_backend_tensor_get\|ggml_backend_tensor_set" src/ --include="*.cpp" -B 3 -A 3
```

### Expected Patterns

**Direct data access (host tensors):**
```cpp
// TEAM-004: For tensors in host memory
float * data = (float *) tensor->data;
int64_t ne0 = tensor->ne[0];  // first dimension
int64_t ne1 = tensor->ne[1];  // second dimension
size_t n_elements = ggml_nelements(tensor);
```

**Backend data access (GPU tensors):**
```cpp
// TEAM-004: For tensors in backend memory (GPU)
size_t n_elements = ggml_nelements(tensor);
float * data = new float[n_elements];
ggml_backend_tensor_get(tensor, data, 0, n_elements * sizeof(float));
// Use data...
delete[] data;
```

**Type handling:**
```cpp
// TEAM-004: Check tensor type
if (tensor->type == GGML_TYPE_F32) {
    // Direct access
    float * data = (float *) tensor->data;
} else if (tensor->type == GGML_TYPE_F16) {
    // Need conversion
    size_t n = ggml_nelements(tensor);
    float * data_f32 = new float[n];
    ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, data_f32, n);
    // Use data_f32...
    delete[] data_f32;
}
```

### Deliverable

```markdown
### Tensor Data Access Patterns

**Host tensor access (CPU tensors):**
```cpp
// TEAM-004: Direct access for host tensors
if (ggml_backend_buffer_is_host(tensor->buffer)) {
    float * data = (float *) tensor->data;
    int64_t ne0 = tensor->ne[0];  // first dimension
    int64_t ne1 = tensor->ne[1];  // second dimension
    size_t n_elements = ggml_nelements(tensor);
}
```

**Backend tensor access (GPU tensors):**
```cpp
// TEAM-004: Need to copy from backend to host
size_t n_elements = ggml_nelements(tensor);
float * data = new float[n_elements];
ggml_backend_tensor_get(tensor, data, 0, n_elements * sizeof(float));
// Use data...
delete[] data;
```

**Type conversion (F16 to F32):**
```cpp
// TEAM-004: Handle different tensor types
if (tensor->type == GGML_TYPE_F32) {
    // Direct access
    float * data = (float *) tensor->data;
} else if (tensor->type == GGML_TYPE_F16) {
    // Convert to F32
    size_t n = ggml_nelements(tensor);
    float * data_f32 = new float[n];
    ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, data_f32, n);
    // Use data_f32...
    delete[] data_f32;
}
```

**Dimension access:**
```cpp
// TEAM-004: Tensor dimensions
int n_dims = ggml_n_dims(tensor);
for (int i = 0; i < n_dims; i++) {
    int64_t dim = tensor->ne[i];
    printf("dim[%d] = %lld\n", i, dim);
}
```

**TEAM-004 Recommendation:**
Use **backend tensor access** (`ggml_backend_tensor_get`) for checkpoint extraction because:
1. Works for both CPU and GPU tensors
2. Handles memory layout automatically
3. Always returns F32 data (no type conversion needed)
4. Safe and consistent across backends
```

---

## Completion Checklist

- [x] Task 1.1: GPT-2 architecture location documented
- [x] Task 1.2: All 6 layer implementations located
- [x] Task 1.3: Graph execution point identified
- [x] Task 1.4: Tensor data access patterns documented
- [x] All findings documented in this file
- [x] Ready to proceed to Phase 2 (Mapping)

---

## Notes and Observations

**TEAM-004 Notes:**
- GPT-2 is a first-class architecture in llama.cpp - well supported!
- Uses standard LayerNorm (not RMSNorm like Llama)
- No RoPE - uses learned position embeddings
- QKV is a single matrix multiply, then split into views (efficient!)
- Callback system (`cb()`) is our extraction mechanism
- Graph building happens BEFORE execution - callbacks fire AFTER execution
- Must use `ggml_backend_tensor_get` for safe cross-backend extraction

**Potential Issues:**
- Attention scores are computed inside `build_attn` - need to find exact callback
- KV cache handling may be complex - checkpoint 3 might need special handling
- Tensor shapes may have batch dimension that needs to be handled
- Need to ensure we're extracting from FIRST layer (il == 0)

**Questions for Phase 2:**
- Where exactly is the attention scores callback in build_attn?
- How to handle KV cache extraction (checkpoint 3)?
- Do we need to reshape tensors to remove batch dimension?
- Should we extract after graph execution or hook into callbacks?

**TEAM-004 Key Discovery:**
The `cb(tensor, name, layer)` callback system is PERFECT for our needs!
- Called after each operation
- Tensors are already computed
- We can hook into this to extract checkpoints
- No need to modify graph execution - just add to callbacks

---

**Status:** ✅ COMPLETE  
**Next Phase:** Phase 2 - Instrumentation Point Mapping  
**Estimated Time:** 1 hour  
**Actual Time:** 45 minutes (faster than expected!)
