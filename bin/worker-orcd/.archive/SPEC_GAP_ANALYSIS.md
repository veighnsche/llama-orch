# M0 Spec Gap Analysis: What's Missing vs What's Specified

**Date**: 2025-10-05  
**Investigator**: Project Management Team  
**Spec**: `bin/.specs/01_M0_worker_orcd.md`

---

## Executive Summary

**ALL of our gaps ARE specified in the M0 spec.**

The spec explicitly requires:
- ✅ GGUF weight loading (M0-W-1220, M0-W-1221, M0-W-1222)
- ✅ Tokenizer (mentioned in scope, line 105)
- ✅ Transformer execution (M0-W-1213, M0-W-1214, M0-W-1215)
- ✅ Architecture adapters (M0-W-1213, M0-W-1214, M0-W-1215)

**The problem**: These requirements were **specified but not fully implemented**.

---

## Gap 1: GGUF Weight Loading

### What the Spec Says

**[M0-W-1220] Model Weights Allocation** (line 682):
```cpp
Worker-orcd MUST allocate VRAM for model weights:
size_t total_size = calculate_total_size();  // From GGUF tensors
weights_ = std::make_unique<DeviceMemory>(total_size);
vram_bytes_ = total_size;
```

**[M0-W-1221] Memory-Mapped I/O (REQUIRED)** (line 725):
```
Worker-orcd MUST use `mmap()` for host I/O for all models to avoid 
full RAM copies and to standardize the loader across model sizes.

Implementation:
- Memory-map GGUF file for reading
- Parse headers and metadata from mapped region
- Stream tensor data directly from mmap to VRAM
```

**[M0-W-1222] Chunked H2D Transfer (REQUIRED)** (line 737):
```cpp
Worker-orcd MUST copy model tensors to VRAM in bounded chunks for all models.
Chunk Size: 1MB (configurable, but 1MB default)

for (size_t offset = 0; offset < total_size; offset += CHUNK_SIZE) {
    size_t chunk_size = std::min(CHUNK_SIZE, total_size - offset);
    cudaMemcpy(device_ptr + offset, host_ptr + offset, chunk_size, H2D);
}
```

**Complete Model Loading Flow** (line 757):
```cpp
Model::Model(const Context& ctx, const std::string& path) {
    // 1. Memory-map file for efficient reading
    auto mapped_file = mmap_file(path);
    // 2. Parse GGUF format
    parse_gguf(mapped_file);
    // 3. Allocate VRAM
    allocate_vram();
    // 4. Copy weights to VRAM in chunks
    copy_weights_chunked(mapped_file);
    // 5. Unmap file
    munmap_file(mapped_file);
}
```

### What We Have

From `cuda/src/model/gpt_weights.cpp`:
```cpp
std::unique_ptr<GPTModelWeights> GPTWeightLoader::load_from_gguf(const std::string& path) {
    // TODO: Parse GGUF tensors
    std::vector<GGUFTensorInfo> tensors;
    
    // TODO: Load embeddings
    // load_embeddings(model.get(), path, tensors);
    
    // TODO: Load transformer layers
    // TODO: Load output head
}
```

**Status**: ❌ **NOT IMPLEMENTED** - Only structure exists, all TODOs

### Gap

**Missing**:
1. Actual tensor parsing from GGUF
2. Memory-mapped file reading
3. Chunked VRAM transfer
4. Weight copying to GPU

**Corresponds to**: GT-051, GT-052 in Sprint 9

---

## Gap 2: Tokenizer

### What the Spec Says

**Scope** (line 105):
```
✅ Tokenization: Two backends (`hf-json` for GPT-OSS-20B, 
   `gguf-bpe` for Qwen/Phi-3)
```

**Model 1: Qwen2.5-0.5B** (line 786):
```
- **Tokenizer**: GGUF byte-BPE (embedded in GGUF)
```

**Model 3: GPT-OSS-20B** (line 823):
```
- **Tokenizer**: HF tokenizers (Rust) loading tokenizer.json
```

### What We Have

```bash
$ find cuda/src -name "*tokenizer*"
# No results
```

**Status**: ❌ **NOT IMPLEMENTED** - No files exist

### Gap

**Missing**:
1. BPE tokenizer implementation
2. Vocab extraction from GGUF
3. Encode (text -> token IDs)
4. Decode (token IDs -> text)

**Corresponds to**: GT-053 in Sprint 9

---

## Gap 3: Transformer Execution

### What the Spec Says

**[M0-W-1213] ModelAdapter Interface** (line 1165):
```cpp
class ModelAdapter {
public:
    virtual ~ModelAdapter() = default;
    
    // Execute forward pass (prefill or decode)
    virtual void run_forward_pass(
        const uint32_t* input_tokens,
        int seq_len,
        bool is_prefill,
        float* logits_out
    ) = 0;
};
```

**[M0-W-1214] LlamaModelAdapter Implementation** (line 1204):
```cpp
void LlamaModelAdapter::run_forward_pass(...) {
    // 1. Embedding lookup
    embedding_kernel<<<...>>>(input_tokens, embeddings);
    
    // 2. Transformer layers
    for (int layer = 0; layer < num_layers; ++layer) {
        // Pre-attention norm
        rmsnorm_kernel<<<...>>>(embeddings, normed);
        // Apply RoPE to Q and K
        rope_kernel<<<...>>>(q, k, position_ids);
        // GQA attention
        gqa_attention_kernel<<<...>>>(q, k, v, attn_out, kv_cache);
        // Residual connection
        add_kernel<<<...>>>(embeddings, attn_out);
        // Pre-FFN norm
        rmsnorm_kernel<<<...>>>(attn_out, normed);
        // SwiGLU FFN
        swiglu_ffn_kernel<<<...>>>(normed, ffn_out);
        // Residual connection
        add_kernel<<<...>>>(attn_out, ffn_out, embeddings);
    }
    
    // 3. Final norm + output projection
    rmsnorm_kernel<<<...>>>(embeddings, normed);
    output_kernel<<<...>>>(normed, logits);
}
```

**[M0-W-1215] GPTModelAdapter Implementation** (line 1249):
```cpp
void GPTModelAdapter::run_forward_pass(...) {
    // 1. Token + position embeddings
    embedding_kernel<<<...>>>(input_tokens, token_emb);
    positional_embedding_kernel<<<...>>>(position_ids, pos_emb);
    add_kernel<<<...>>>(token_emb, pos_emb, embeddings);
    
    // 2. Transformer layers
    for (int layer = 0; layer < num_layers; ++layer) {
        // Pre-attention norm
        layernorm_kernel<<<...>>>(embeddings, normed);
        // MHA attention
        mha_attention_kernel<<<...>>>(normed, attn_out, kv_cache);
        // Residual connection
        add_kernel<<<...>>>(embeddings, attn_out);
        // Pre-FFN norm
        layernorm_kernel<<<...>>>(attn_out, normed);
        // GELU FFN
        gelu_ffn_kernel<<<...>>>(normed, ffn_out);
        // Residual connection
        add_kernel<<<...>>>(attn_out, ffn_out, embeddings);
    }
    
    // 3. Final norm + output projection
    layernorm_kernel<<<...>>>(embeddings, normed);
    output_kernel<<<...>>>(normed, logits);
}
```

### What We Have

From `cuda/src/model/gpt_model.cpp`:
```cpp
void GPTModel::execute_layer(int layer_idx, const half* input, half* output, bool is_prefill) {
    // TODO: Implement actual layer execution
    // For now, just copy input to output (stub)
    
    cudaMemcpy(output, input, seq_len * cfg.hidden_dim * sizeof(half), D2D);
    
    // In real implementation:
    // 1. Pre-attention LayerNorm
    // 2. Multi-Head Attention (with KV cache for decode)
    // 3. Residual connection
    // 4. Pre-FFN LayerNorm
    // 5. Feed-Forward Network
    // 6. Residual connection
}
```

**Status**: ⚠️ **STUBBED** - Structure exists, but just copies data

### Gap

**Missing**:
1. Actual layer execution (LayerNorm, Attention, FFN, Residual)
2. Wiring to existing kernels
3. KV cache integration

**Corresponds to**: GT-054 in Sprint 9

---

## Gap 4: LM Head

### What the Spec Says

**Implied in forward pass** (lines 1233-1235, 1278-1280):
```cpp
// 3. Final norm + output projection
rmsnorm_kernel<<<...>>>(embeddings, normed);  // or layernorm
output_kernel<<<...>>>(normed, logits);
```

### What We Have

From `cuda/src/model/gpt_model.cpp`:
```cpp
void GPTModel::apply_output_head(const half* input, half* logits) {
    // Apply final LayerNorm
    cuda_layernorm(normalized, input, ...);
    
    // TODO: Apply LM head (GEMM: normalized @ lm_head_weight)
    // For now, just copy (stub)
    cudaMemcpy(logits, normalized, cfg.hidden_dim * sizeof(half), D2D);
}
```

**Status**: ⚠️ **STUBBED** - LayerNorm works, but LM head is stubbed

### Gap

**Missing**:
1. LM head GEMM (hidden_dim -> vocab_size)
2. cuBLAS call for projection

**Corresponds to**: GT-055 in Sprint 9

---

## Gap 5: Architecture Detection

### What the Spec Says

**[M0-W-1212] Architecture Detection from GGUF** (line 665):
```cpp
Architecture detect_architecture(const GGUFMetadata& metadata) {
    std::string arch = metadata.get_string("general.architecture");
    if (arch == "llama") return Architecture::Llama;  // Qwen, Phi-3
    if (arch == "gpt2" || arch == "gpt") return Architecture::GPT;  // GPT-OSS-20B
    throw std::runtime_error("Unsupported architecture: " + arch);
}
```

### What We Have

From `cuda/src/model/gpt_weights.cpp`:
```cpp
GPTConfig GPTWeightLoader::parse_config_from_gguf(const std::string& path) {
    // TODO: Parse actual GGUF file
    // For now, return GPT-OSS-20B config as stub
    GPTConfig config;
    config.vocab_size = 50257;  // Hardcoded!
    config.hidden_dim = 2048;
    // ...
    return config;
}
```

**Status**: ⚠️ **STUBBED** - Returns hardcoded config instead of parsing GGUF

### Gap

**Missing**:
1. Actual GGUF metadata parsing
2. Architecture detection
3. Config extraction from metadata

**Corresponds to**: GT-051 in Sprint 9

---

## Summary Table

| Gap | Spec Requirement | Status | Missing Work | Sprint 9 Story |
|-----|-----------------|--------|--------------|----------------|
| **GGUF Config** | M0-W-1211, M0-W-1212 | ⚠️ Stubbed | Parse real GGUF metadata | GT-051 |
| **Weight Loading** | M0-W-1220, M0-W-1221, M0-W-1222 | ❌ Not implemented | Mmap, parse tensors, copy to GPU | GT-052 |
| **Tokenizer** | Line 105, 786, 823 | ❌ Not implemented | BPE encode/decode | GT-053 |
| **Transformer Exec** | M0-W-1213, M0-W-1214, M0-W-1215 | ⚠️ Stubbed | Wire kernels, execute layers | GT-054 |
| **LM Head** | Lines 1233-1235, 1278-1280 | ⚠️ Stubbed | cuBLAS GEMM | GT-055 |
| **Wire Inference** | Implied by all above | ❌ Not implemented | Connect all pieces | GT-056 |
| **Test Cleanup** | FT-050 requirement | ⚠️ Stubbed | Remove stub warnings | GT-057 |

---

## Why This Happened

### The Spec Was Clear

The M0 spec **explicitly required** all of these components:
- M0-W-1220: "Worker-orcd MUST allocate VRAM for model weights"
- M0-W-1221: "Worker-orcd MUST use `mmap()` for host I/O" (REQUIRED)
- M0-W-1222: "Worker-orcd MUST copy model tensors to VRAM in bounded chunks" (REQUIRED)
- M0-W-1213/1214/1215: "Worker MUST implement ModelAdapter" with full forward pass

### The Implementation Was Incomplete

**What was implemented**:
- ✅ Structure created (classes, headers, function signatures)
- ✅ CUDA kernels implemented (all 25 kernels)
- ✅ KV cache implemented
- ✅ HTTP/SSE pipeline implemented

**What was NOT implemented**:
- ❌ Actual weight loading (TODOs everywhere)
- ❌ Tokenizer (no files exist)
- ❌ Transformer execution (stubbed)
- ❌ LM head projection (stubbed)

### The Gap

**Between "structure" and "working"**:
- Teams created the **architecture** (classes, interfaces, kernels)
- Teams did NOT complete the **implementation** (fill in the TODOs)
- PM did NOT verify that "complete" meant "works"

---

## Conclusion

**The M0 spec is complete and correct.** It specifies everything we need.

**The implementation is incomplete.** The TODOs were never filled in.

**The fix**: Implement GT-051 to GT-057 (Sprint 9) to complete what the spec requires.

**Estimated time**: 22-31 hours (as specified in Sprint 9)

**Deadline**: 2025-10-15 (10 days from fine)

---

## Recommendations

### For Future Milestones

1. **Define "Complete"**: Create checklist for what "complete" means
   - ✅ Structure created
   - ✅ Tests pass
   - ✅ **No TODOs in critical path**
   - ✅ **Real functionality, not stubs**

2. **Story Breakdown**: Break "implement X" into:
   - Story 1: Create structure
   - Story 2: Implement core logic
   - Story 3: Wire to dependencies
   - Story 4: Remove TODOs

3. **PM Verification**: Before marking milestone complete:
   - ✅ Run the test
   - ✅ Verify real functionality
   - ✅ Check for TODOs
   - ✅ Ask: "Is this a stub?"

4. **Testing Team Earlier**: Involve Testing Team during implementation, not after

---

**Analyzed by**: Project Management Team 📋  
**Date**: 2025-10-05  
**Conclusion**: Spec is complete, implementation is incomplete  
**Action**: Sprint 9 will complete the spec requirements

---

Documented by Project Management Team 📋
