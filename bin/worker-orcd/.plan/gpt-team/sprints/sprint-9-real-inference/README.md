# Sprint 9: Real GPU Inference Implementation

**Team**: GPT-Gamma 🤖  
**Duration**: 7-10 days  
**Priority**: P0 (M0 blocker)  
**Related Fine**: test-harness/FINES.md #001

---

## Sprint Goal

Implement **real GPU inference** to replace the stub implementation that was fined by the Testing Team.

**Success Criteria**:
- ✅ GGUF weights loaded to GPU VRAM
- ✅ Tokenizer encodes/decodes correctly
- ✅ Transformer executes on GPU
- ✅ Real haiku generated (not hardcoded)
- ✅ Test renamed back to `test_haiku_generation_anti_cheat`
- ✅ All stub warnings removed

---

## Context

### The Fine

**FINE-001-20251005**: The haiku test uses a stub that hardcodes the haiku instead of doing real GPU inference. This is a **false positive** - the test passes when the product is broken.

**Remediation deadline**: 2025-10-15 (10 days)

### What's Already Done

Based on code investigation:

#### ✅ Complete (No work needed)
- GGUF header parsing (`cuda/src/gguf/header_parser.cpp`)
- Memory-mapped I/O (`cuda/src/io/mmap_file.cpp`)
- All 25 CUDA kernels (`cuda/kernels/*.cu`)
- KV cache (`cuda/src/kv_cache.cpp`)
- GPT model structure (`cuda/src/model/gpt_model.h`)
- Transformer layer structure (`cuda/src/gpt_transformer_layer.cpp`)
- CUDA context management (`cuda/src/context.cpp`)
- VRAM tracking (`cuda/src/vram_tracker.cpp`)

#### ⚠️ Partially Done (Need to complete)
- Weight loading structure exists, but has TODOs:
  - `GPTWeightLoader::load_from_gguf()` - line 266: "TODO: Parse GGUF tensors"
  - `GPTWeightLoader::load_embeddings()` - line 370: "TODO: Load from file and copy to VRAM"
  - `GPTWeightLoader::load_layer()` - line 397: "TODO: Load all layer weights"
  - `GPTWeightLoader::load_output_head()` - line 411: "TODO: Load output norm and LM head"
  - `GPTWeightLoader::parse_config_from_gguf()` - line 336: "TODO: Parse actual GGUF file"

- Transformer execution stubbed:
  - `GPTModel::execute_layer()` - line 255: "TODO: Implement actual layer execution"
  - `GPTModel::apply_output_head()` - line 329: "TODO: Apply LM head (GEMM)"

#### ❌ Not Started
- Tokenizer (no files exist, only test stubs reference it)

---

## Stories

### GT-051: GGUF Config Parsing from Real File
**Estimate**: 2-3 hours  
**Priority**: P0  
**Dependencies**: None

**Current state**:
```cpp
GPTConfig GPTWeightLoader::parse_config_from_gguf(const std::string& path) {
    // TODO: Parse actual GGUF file
    // For now, return GPT-OSS-20B config as stub
    GPTConfig config;
    config.vocab_size = 50257;  // Hardcoded!
    // ...
}
```

**Tasks**:
1. Use existing `parse_gguf_header()` to read GGUF file
2. Extract config from metadata using `llama_metadata.cpp` helpers
3. Map GGUF metadata to `GPTConfig` struct
4. Validate config

**Files to modify**:
- `cuda/src/model/gpt_weights.cpp` (line 335-350)

**Existing code to use**:
- ✅ `cuda/src/gguf/header_parser.cpp::parse_gguf_header()`
- ✅ `cuda/src/gguf/llama_metadata.cpp::extract_llama_config()`

---

### GT-052: GGUF Weight Loading to GPU
**Estimate**: 6-8 hours  
**Priority**: P0  
**Dependencies**: GT-051

**Current state**:
```cpp
std::unique_ptr<GPTModelWeights> GPTWeightLoader::load_from_gguf(const std::string& path) {
    // TODO: Parse GGUF tensors
    std::vector<GGUFTensorInfo> tensors;
    
    // TODO: Load embeddings
    // TODO: Load transformer layers
    // TODO: Load output head
}
```

**Tasks**:
1. Parse GGUF tensors from header
2. Implement `load_embeddings()` - allocate GPU memory and copy
3. Implement `load_layer()` - load all layer weights
4. Implement `load_output_head()` - load final norm and LM head
5. Implement `allocate_and_copy()` - helper to copy tensors to GPU

**Files to modify**:
- `cuda/src/model/gpt_weights.cpp` (lines 266-412)

**Existing code to use**:
- ✅ `cuda/src/gguf/header_parser.cpp` - tensor info extraction
- ✅ `cuda/src/io/mmap_file.cpp` - file access
- ✅ `cuda/src/device_memory.cpp` - GPU allocation

**Verification**:
- Use `nvidia-smi` to verify VRAM usage
- Check `weights_->total_vram_bytes` matches actual allocation

---

### GT-053: BPE Tokenizer Implementation
**Estimate**: 5-7 hours  
**Priority**: P0  
**Dependencies**: GT-051

**Current state**: No tokenizer exists

**Tasks**:
1. Create `cuda/src/tokenizer/bpe_tokenizer.h`
2. Create `cuda/src/tokenizer/bpe_tokenizer.cpp`
3. Extract vocab and merges from GGUF metadata
4. Implement BPE encode (text -> token IDs)
5. Implement BPE decode (token IDs -> text)
6. Handle special tokens (BOS, EOS, etc.)
7. Wire to `ModelImpl`

**Files to create**:
- `cuda/src/tokenizer/bpe_tokenizer.h`
- `cuda/src/tokenizer/bpe_tokenizer.cpp`

**Files to modify**:
- `cuda/src/model_impl.h` - add tokenizer member
- `cuda/src/model_impl.cpp` - initialize tokenizer
- `cuda/CMakeLists.txt` - add tokenizer files

**Existing code to use**:
- ✅ `cuda/src/gguf/llama_metadata.cpp` - vocab extraction

**Reference**:
- GGUF spec for tokenizer metadata
- BPE algorithm (standard implementation)

---

### GT-054: Transformer Layer Execution
**Estimate**: 4-6 hours  
**Priority**: P0  
**Dependencies**: GT-052

**Current state**:
```cpp
void GPTModel::execute_layer(int layer_idx, const half* input, half* output, bool is_prefill) {
    // TODO: Implement actual layer execution
    // For now, just copy input to output (stub)
    cudaMemcpy(output, input, ...);
}
```

**Tasks**:
1. Implement pre-attention LayerNorm
2. Call MHA attention kernel (already exists)
3. Add residual connection
4. Implement pre-FFN LayerNorm
5. Call FFN kernel (already exists)
6. Add residual connection
7. Wire KV cache for decode mode

**Files to modify**:
- `cuda/src/model/gpt_model.cpp` (lines 249-272)

**Existing code to use**:
- ✅ `cuda/kernels/layernorm.cu` - LayerNorm kernel
- ✅ `cuda/kernels/mha_attention.cu` - Attention kernel
- ✅ `cuda/kernels/gpt_ffn.cu` - FFN kernel
- ✅ `cuda/kernels/residual.cu` - Residual add kernel
- ✅ `cuda/src/gpt_transformer_layer.cpp` - Layer wrapper functions

---

### GT-055: LM Head Implementation
**Estimate**: 2-3 hours  
**Priority**: P0  
**Dependencies**: GT-052

**Current state**:
```cpp
void GPTModel::apply_output_head(const half* input, half* logits) {
    // TODO: Apply LM head (GEMM: normalized @ lm_head_weight)
    // For now, just copy (stub)
    cudaMemcpy(logits, normalized, ...);
}
```

**Tasks**:
1. Call cuBLAS GEMM for LM head projection
2. Map hidden_dim -> vocab_size
3. Return logits for sampling

**Files to modify**:
- `cuda/src/model/gpt_model.cpp` (lines 307-336)

**Existing code to use**:
- ✅ `cuda/src/cublas_wrapper.cpp` - cuBLAS helpers
- ✅ `cublas_handle_` - already initialized in GPTModel

---

### GT-056: Wire Real Inference to Stub
**Estimate**: 2-3 hours  
**Priority**: P0  
**Dependencies**: GT-052, GT-053, GT-054, GT-055

**Current state**: `InferenceImpl` hardcodes haiku

**Tasks**:
1. Replace stub in `cuda/src/inference_impl.cpp`
2. Call `model_.tokenizer().encode(prompt)`
3. Call `model_.gpt_model()->prefill(tokens, ...)`
4. Loop: call `model_.gpt_model()->decode(...)`
5. Call `model_.tokenizer().decode(token_id)`
6. Return real tokens

**Files to modify**:
- `cuda/src/inference_impl.cpp` - Replace entire constructor and `next_token()`
- `cuda/src/model_impl.h` - Add `tokenizer()` and `gpt_model()` accessors
- `cuda/src/model_impl.cpp` - Wire to GPTModel

**Remove**:
- All stub haiku generation code
- All fine warnings
- Minute word extraction logic

---

### GT-057: Test Cleanup and Verification
**Estimate**: 1-2 hours  
**Priority**: P0  
**Dependencies**: GT-056

**Tasks**:
1. Remove all stub warnings from test
2. Rename test back to `test_haiku_generation_anti_cheat`
3. Remove fine references
4. Run test multiple times to verify different haikus
5. Verify minute word appears in haiku
6. Submit remediation proof to Testing Team

**Files to modify**:
- `tests/haiku_generation_anti_cheat.rs` - Remove warnings, rename
- `cuda/src/inference_impl.cpp` - Remove fine comments

**Verification**:
- Test passes
- Different haiku each run
- Minute word present
- No stub warnings
- Testing Team sign-off

---

## Timeline

### Optimistic (5 days)
- Day 1: GT-051, GT-052 (start)
- Day 2: GT-052 (finish), GT-053 (start)
- Day 3: GT-053 (finish), GT-054
- Day 4: GT-055, GT-056
- Day 5: GT-057, testing

### Realistic (7-10 days)
- Days 1-2: GT-051, GT-052
- Days 3-4: GT-053
- Days 5-6: GT-054, GT-055
- Day 7: GT-056
- Days 8-9: GT-057, debugging
- Day 10: Buffer for issues

**Deadline**: 2025-10-15

---

## Risks

### Risk 1: Tokenizer Complexity
**Mitigation**: BPE is well-documented, use GGUF spec

### Risk 2: Weight Loading Bugs
**Mitigation**: Verify with `nvidia-smi`, add debug logging

### Risk 3: Transformer Execution Issues
**Mitigation**: All kernels already tested, just need to wire

### Risk 4: Time Pressure
**Mitigation**: Focus on MVP, skip optimizations

---

## Success Metrics

- ✅ GGUF weights loaded (verify with `nvidia-smi`)
- ✅ Tokenizer works (test encode/decode)
- ✅ Transformer executes (check GPU utilization)
- ✅ Real haiku generated
- ✅ Test passes consistently
- ✅ Different haiku each run
- ✅ Minute word present
- ✅ No stub warnings
- ✅ Fine remediated

---

## Notes

### Why This is Achievable

1. **80-90% done**: Most infrastructure exists
2. **Kernels complete**: All 25 CUDA kernels work
3. **Structure exists**: Just need to fill in TODOs
4. **Clear scope**: No new features, just wire existing code

### What We're NOT Doing

- ❌ Optimizations
- ❌ Quantization support (use FP16 for now)
- ❌ Batch inference
- ❌ Advanced sampling (use basic temperature)
- ❌ Multiple models

**Focus**: Get ONE model working with real inference.

---

**Created by**: Testing Team (remediation requirement)  
**Assigned to**: GPT-Gamma 🤖  
**Sprint**: 9  
**Deadline**: 2025-10-15

---

Tracked by Testing Team 🔍
