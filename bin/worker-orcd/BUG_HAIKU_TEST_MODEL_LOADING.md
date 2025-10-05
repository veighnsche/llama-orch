# Bug Report: Haiku Test Model Loading Failure

**Date**: 2025-10-05  
**Severity**: High (Blocks M0 Success Criteria Test)  
**Status**: Identified, Plan Created

---

## Summary

The haiku anti-cheat test (M0 success criteria) fails during model loading with error:
```
Error: Model load failed: Failed to load model from GGUF file
```

This prevents the human-visible demonstration of real GPU inference.

---

## Bug Details

### What Happens

1. ✅ Test compiles successfully with CUDA enabled
2. ✅ Worker binary spawns correctly
3. ✅ CUDA context initializes on GPU 0
4. ✅ Worker attempts to load model from `.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf`
5. ❌ **FAILS**: Model loading returns error "Failed to load model from GGUF file"

### Error Log

```json
{"timestamp":"2025-10-05T15:51:23.102521Z","level":"INFO","fields":{"message":"Worker starting","worker_id":"test-worker-39941","model":".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf","gpu_device":0,"port":39941}}
{"timestamp":"2025-10-05T15:51:23.229279Z","level":"INFO","fields":{"message":"CUDA context initialized","gpu_device":0}}
{"timestamp":"2025-10-05T15:51:23.229300Z","level":"INFO","fields":{"message":"Loading model to VRAM...","model":".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf"}}
Error: Model load failed: Failed to load model from GGUF file
```

### Test Command

```bash
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_anti_cheat --features cuda --release \
  -- --ignored --nocapture --test-threads=1
```

---

## Root Cause Analysis

### Primary Issue: GGUF Loader Not Wired to CUDA

The worker has the infrastructure but the actual GGUF → CUDA model loading pipeline is incomplete:

1. **GGUF Parser**: ✅ Exists (`worker-gguf` crate)
2. **CUDA Kernels**: ✅ Compiled (attention, rope, etc.)
3. **Model Structures**: ✅ Defined (QwenConfig, GPTConfig)
4. **FFI Bridge**: ✅ Exists (`src/cuda/ffi.rs`)
5. **Integration**: ❌ **MISSING** - GGUF data → CUDA memory transfer

### Specific Gaps

#### Gap 1: GGUF Weight Loading to VRAM
**Location**: `bin/worker-crates/worker-models/src/qwen.rs`

```rust
impl QwenWeightLoader {
    pub fn load_to_vram(path: &str, config: &QwenConfig) -> Result<QwenModel, ModelError> {
        // TODO: This is stub code
        // Need to:
        // 1. Parse GGUF file
        // 2. Extract weight tensors
        // 3. Allocate CUDA memory
        // 4. Copy weights to GPU
        // 5. Return model handle
        
        Ok(QwenModel {
            config: config.clone(),
            // Stub fields
        })
    }
}
```

**Status**: Returns stub model, doesn't actually load from GGUF

#### Gap 2: CUDA Memory Allocation
**Location**: `bin/worker-orcd/cuda/src/model/model_loader.cu`

```cpp
// Missing implementation:
// - cudaMalloc for weight tensors
// - cudaMemcpy for weight data
// - Tensor layout conversion (GGUF → CUDA format)
```

**Status**: CUDA functions exist but aren't called from Rust

#### Gap 3: Tokenizer Integration
**Location**: `bin/worker-crates/worker-tokenizer/src/lib.rs`

```rust
// Tokenizer exists but not wired to inference pipeline
// Need to connect:
// - GGUF embedded tokenizer OR
// - HF tokenizer loading
```

**Status**: Tokenizer code exists but not integrated

---

## Impact

### Blocked Features

- ❌ Haiku anti-cheat test (M0 success criteria)
- ❌ Real GPU inference demonstration
- ❌ End-to-end validation with actual model
- ❌ Performance baseline measurements with real inference
- ❌ Human-visible proof that M0 works

### What Still Works

- ✅ All unit tests (144 tests passing)
- ✅ CUDA compilation and initialization
- ✅ Test infrastructure
- ✅ Worker binary spawning
- ✅ HTTP server and SSE streaming (with stub data)

---

## Fix Plan

### Phase 1: GGUF → CUDA Bridge (Critical Path)

**Goal**: Load GGUF weights into CUDA memory

#### Step 1.1: Implement GGUF Weight Extraction
**File**: `bin/worker-crates/worker-models/src/qwen.rs`

```rust
impl QwenWeightLoader {
    pub fn load_to_vram(path: &str, config: &QwenConfig) -> Result<QwenModel, ModelError> {
        // 1. Open GGUF file
        let gguf = worker_gguf::GGUFFile::open(path)?;
        
        // 2. Extract metadata
        let metadata = gguf.metadata()?;
        
        // 3. Validate architecture matches
        assert_eq!(metadata.architecture, "llama");
        
        // 4. Extract weight tensors
        let weights = extract_weights(&gguf, config)?;
        
        // 5. Allocate CUDA memory via FFI
        let cuda_model = unsafe {
            cuda_allocate_model(config, &weights)?
        };
        
        Ok(QwenModel {
            config: config.clone(),
            cuda_handle: cuda_model,
        })
    }
}
```

**Estimated Time**: 4-6 hours

#### Step 1.2: Implement CUDA Memory Allocation
**File**: `bin/worker-orcd/cuda/src/model/model_loader.cu`

```cpp
extern "C" CudaModel* cuda_allocate_model(
    const ModelConfig* config,
    const WeightTensors* weights
) {
    CudaModel* model = new CudaModel();
    
    // Allocate weight buffers
    cudaMalloc(&model->token_embeddings, weights->embed_size);
    cudaMalloc(&model->layer_weights, weights->layer_size);
    // ... allocate all weight tensors
    
    // Copy weights to GPU
    cudaMemcpy(model->token_embeddings, weights->embed_data, 
               weights->embed_size, cudaMemcpyHostToDevice);
    // ... copy all weights
    
    return model;
}
```

**Estimated Time**: 3-4 hours

#### Step 1.3: Wire FFI Bindings
**File**: `bin/worker-orcd/src/cuda/ffi.rs`

```rust
extern "C" {
    pub fn cuda_allocate_model(
        config: *const ModelConfig,
        weights: *const WeightTensors,
    ) -> *mut CudaModel;
}

pub fn load_model_to_cuda(
    path: &str,
    config: &QwenConfig,
) -> Result<*mut CudaModel, CudaError> {
    // Extract weights from GGUF
    let weights = extract_gguf_weights(path)?;
    
    // Call CUDA allocation
    unsafe {
        let model = cuda_allocate_model(&config.into(), &weights);
        if model.is_null() {
            return Err(CudaError::AllocationFailed);
        }
        Ok(model)
    }
}
```

**Estimated Time**: 2-3 hours

---

### Phase 2: Tokenizer Integration

**Goal**: Enable text → tokens → inference → tokens → text

#### Step 2.1: Load Tokenizer from GGUF
**File**: `bin/worker-crates/worker-tokenizer/src/gguf_tokenizer.rs`

```rust
pub fn load_from_gguf(path: &str) -> Result<Tokenizer, TokenizerError> {
    let gguf = GGUFFile::open(path)?;
    
    // Extract tokenizer data from GGUF metadata
    let vocab = gguf.get_vocab()?;
    let merges = gguf.get_merges()?;
    
    Ok(Tokenizer::from_vocab_and_merges(vocab, merges))
}
```

**Estimated Time**: 3-4 hours

#### Step 2.2: Wire Tokenizer to Inference
**File**: `bin/worker-orcd/src/inference_executor.rs`

```rust
pub async fn execute(&mut self, request: ExecuteRequest) -> Result<InferenceStream> {
    // 1. Tokenize input
    let tokens = self.tokenizer.encode(&request.prompt)?;
    
    // 2. Run inference
    let output_tokens = self.model.generate(tokens, request.max_tokens)?;
    
    // 3. Detokenize output
    let text = self.tokenizer.decode(&output_tokens)?;
    
    // 4. Stream via SSE
    Ok(stream_tokens(text))
}
```

**Estimated Time**: 2-3 hours

---

### Phase 3: Inference Pipeline

**Goal**: Actually generate tokens on GPU

#### Step 3.1: Implement Forward Pass
**File**: `bin/worker-orcd/cuda/src/inference/forward.cu`

```cpp
extern "C" void cuda_forward_pass(
    CudaModel* model,
    const int* input_tokens,
    int num_tokens,
    float* logits_out
) {
    // 1. Embedding lookup
    embed_tokens<<<blocks, threads>>>(
        model->token_embeddings,
        input_tokens,
        num_tokens,
        embeddings
    );
    
    // 2. Transformer layers
    for (int layer = 0; layer < model->num_layers; layer++) {
        attention_layer<<<blocks, threads>>>(/*...*/);
        ffn_layer<<<blocks, threads>>>(/*...*/);
    }
    
    // 3. Output projection
    output_projection<<<blocks, threads>>>(/*...*/);
}
```

**Estimated Time**: 6-8 hours

#### Step 3.2: Implement Sampling
**File**: `bin/worker-orcd/cuda/src/inference/sampling.cu`

```cpp
extern "C" int cuda_sample_token(
    const float* logits,
    int vocab_size,
    float temperature,
    uint64_t seed
) {
    // Temperature scaling
    // Top-p/top-k filtering
    // Random sampling
    return sampled_token;
}
```

**Estimated Time**: 2-3 hours

---

## Implementation Priority

### Critical Path (Must Have for Haiku Test)

1. **GGUF Weight Extraction** (Step 1.1) - 4-6 hours
2. **CUDA Memory Allocation** (Step 1.2) - 3-4 hours  
3. **FFI Wiring** (Step 1.3) - 2-3 hours
4. **Tokenizer Loading** (Step 2.1) - 3-4 hours
5. **Forward Pass** (Step 3.1) - 6-8 hours
6. **Sampling** (Step 3.2) - 2-3 hours

**Total Critical Path**: ~22-31 hours

### Nice to Have (Can Stub Initially)

- Advanced sampling (top-p, top-k)
- KV cache optimization
- Batch processing
- Performance optimization

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_gguf_weight_extraction() {
    let weights = extract_gguf_weights("test.gguf").unwrap();
    assert!(weights.token_embeddings.len() > 0);
}

#[test]
fn test_cuda_allocation() {
    let model = cuda_allocate_model(&config, &weights).unwrap();
    assert!(!model.is_null());
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_model_loading_e2e() {
    let model = QwenWeightLoader::load_to_vram(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        &QwenConfig::qwen2_5_0_5b()
    ).await.unwrap();
    
    assert!(model.is_loaded());
}
```

### Haiku Test

```bash
# This should PASS after fixes
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_anti_cheat --features cuda --release \
  -- --ignored --nocapture --test-threads=1
```

---

## Success Criteria

### Definition of Done

- [ ] GGUF file successfully parsed
- [ ] Weights loaded into CUDA memory
- [ ] Tokenizer encodes/decodes text
- [ ] Forward pass generates logits
- [ ] Sampling produces tokens
- [ ] Haiku test passes
- [ ] **Human sees the haiku on terminal** 🎨

### Expected Output

```
🎨 M0 Haiku Anti-Cheat Test PASSED
Minute: 52 ("fifty-two")
Nonce: xY9zW2aB
Tokens: 87
Time: 8.234s

Haiku:
Fifty-two threads dance
Silicon valleys echo
CUDA's swift embrace

Artifacts: .test-results/haiku/run-abc123/
```

---

## Risks and Mitigations

### Risk 1: GGUF Format Complexity
**Mitigation**: Use existing `worker-gguf` crate, reference GGUF spec (NOT llama.cpp - we're the competitor!)

### Risk 2: CUDA Memory Management
**Mitigation**: Start with simple allocation, add optimization later

### Risk 3: Tokenizer Compatibility
**Mitigation**: Extract from GGUF first, fall back to HF tokenizer if needed

### Risk 4: Performance
**Mitigation**: Focus on correctness first, optimize in Phase 2

---

## Alternative Approaches

### ❌ Option A: Use llama.cpp Directly
**REJECTED**: We are building a llama.cpp-FREE engine. We are their competitor.

### ❌ Option B: Stub Inference, Hardcode Haiku
**REJECTED**: Not real inference, defeats anti-cheat purpose

### ✅ Option C: Implement Our Own Inference (CHOSEN)
**Pros**: 
- Proves our engine works
- No external dependencies
- We control the code
- Enables haiku test
- **We are so close - don't give up now**

**Cons**: Requires work (but we're doing it ourselves)

**Recommendation**: Option C - BUILD IT OURSELVES

---

## Timeline

### Optimistic (1 developer, focused)
- **Phase 1**: 2 days (GGUF → CUDA)
- **Phase 2**: 1 day (Tokenizer)
- **Phase 3**: 2 days (Inference)
- **Total**: ~5 days

### Realistic (with debugging)
- **Phase 1**: 3 days
- **Phase 2**: 1.5 days
- **Phase 3**: 3 days
- **Total**: ~7-8 days

### Conservative (with unknowns)
- **Phase 1**: 4 days
- **Phase 2**: 2 days
- **Phase 3**: 4 days
- **Total**: ~10 days

---

## Next Steps

1. **Immediate**: Review this plan with team
2. **Day 1**: Start Phase 1.1 (GGUF weight extraction)
3. **Day 2**: Complete Phase 1 (CUDA allocation)
4. **Day 3**: Phase 2 (Tokenizer)
5. **Day 4-5**: Phase 3 (Inference)
6. **Day 6**: Testing and debugging
7. **Day 7**: **RUN THE HAIKU TEST AND SEE THE HAIKU!** 🎨

---

## References

- **GGUF Spec**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md (spec only, NOT implementation)
- **Qwen Model**: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
- **Test Spec**: `bin/worker-orcd/.plan/foundation-team/stories/FT-041-to-FT-050/FT-050-haiku-generation-test.md`
- **Our CUDA Kernels**: `bin/worker-orcd/cuda/` - USE THESE, they're ours!
- **NO LLAMA.CPP**: See `NO_LLAMA_CPP.md` - we are the competitor!

---

**Status**: Plan Complete, Ready for Implementation  
**Priority**: High (M0 Success Criteria)  
**Owner**: TBD

---

Built by Foundation-Alpha 🏗️  
Date: 2025-10-05
