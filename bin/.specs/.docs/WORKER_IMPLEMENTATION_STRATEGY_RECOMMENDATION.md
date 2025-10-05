# Custom Worker-orcd Implementation Strategy

**Date**: 2025-10-03  
**Decision**: Build custom worker-orcd with ModelAdapter pattern  
**Status**: ✅ Approved  
**Timeline**: 6-7 weeks (M0 foundation + architecture adapters)

---

## Executive Summary

**Decision**: **Build custom worker-orcd with ModelAdapter pattern**

**Rationale**: Analysis confirms that building a custom worker is the only path that preserves the system's core value propositions. External engines would sacrifice 8-15 critical features including narration-core logging, proof bundle emission, VRAM-only enforcement, and API contract compliance.

**Timeline**: 6-7 weeks total (4-5 weeks M0 foundation + 1-2 weeks architecture adapters)

**Risk Level**: MEDIUM (controlled complexity, well-understood requirements)

---

## 1. Strategic Context

### 1.1 System Value Propositions (from 00_llama-orch.md)

The system is designed around **five core differentiators**:

1. **Test Reproducibility**: Same seed + temp=0 → Same output (for testing validation)
2. **VRAM-Only Policy**: Model fully resident in GPU VRAM (no RAM fallback)
3. **Multi-Node Orchestration**: Distribute models across GPU clusters
4. **EU Compliance**: GDPR-native, EU-only data residency
5. **Marketplace Ready**: Enable GPU provider ecosystem

### 1.2 Critical System Requirements

From the specs, the worker MUST:

- **Narration-core logging**: Custom narrative logging for debugging/observability
- **Proof bundle emission**: Standardized test artifacts (libs/proof-bundle integration)
- **VRAM-only enforcement**: Strict VRAM residency with periodic verification
- **Architecture-specific dispatch**: Support Llama-style (Qwen/Phi) AND GPT-style models
- **Quantized execution**: Q4_K_M, MXFP4, Q4_0 with on-the-fly dequantization
- **Smart/Dumb boundary**: Worker is "dumb executor" - no policy decisions
- **HTTP API contract**: Specific endpoints (/execute, /health, /cancel) with SSE streaming
- **Ready callback**: Server-first startup, then callback to pool-manager with model_ref
- **Correlation ID propagation**: X-Correlation-Id for distributed tracing

---

## 2. Architecture Design

**Architecture**:
```
┌─────────────────────────────────────┐
│ Rust Layer (worker-orcd)            │
│ • HTTP server (Axum)                │
│ • Narration-core logging            │
│ • Proof bundle emission             │
│ • SSE streaming                     │
│ • Correlation ID propagation        │
└──────────────┬──────────────────────┘
               │ FFI (unsafe extern "C")
┌──────────────▼──────────────────────┐
│ C++/CUDA Layer                      │
│ • ModelAdapter (abstract)       │
│   ├─ LlamaAdapter (Qwen/Phi)       │
│   └─ GPTAdapter (GPT-OSS-20B)      │
│ • CUDA context management           │
│ • VRAM allocation & verification    │
│ • Architecture-specific kernels     │
└─────────────────────────────────────┘
```

### 2.2 Key Capabilities

**Full Spec Compliance**:
- ✅ **Narration-core integration**: Custom logging with narrative events
- ✅ **Proof bundle support**: Direct integration with libs/proof-bundle
- ✅ **VRAM-only enforcement**: Strict control over memory allocation
- ✅ **Architecture adapters**: Clean ModelAdapter pattern for multiple model types
- ✅ **Smart/Dumb boundary**: Worker remains "dumb executor" as designed
- ✅ **API contract compliance**: Exact HTTP endpoints and SSE streaming as specified
- ✅ **Correlation ID support**: Full distributed tracing capability
- ✅ **Test reproducibility**: Control over determinism at kernel level
- ✅ **Quantization control**: Exact MXFP4 implementation as needed
- ✅ **EU compliance ready**: Can add audit logging and data residency controls

---

## 3. Why NOT External Engines?

### 3.1 Critical Features Lost

**Integrating llama.cpp, vLLM, or TensorRT-LLM would lose**:

1. **❌ Narration-core logging**: llama.cpp has its own logging system
   - Cannot emit narrative events in our format
   - Cannot integrate with libs/narration-core
   - Debugging becomes opaque

2. **❌ Proof bundle emission**: llama.cpp doesn't know about our proof bundle standard
   - Cannot emit to libs/proof-bundle format
   - Cannot respect LLORCH_RUN_ID/LLORCH_PROOF_DIR
   - Testing reproducibility artifacts lost

3. **❌ VRAM-only enforcement**: llama.cpp supports UMA and CPU fallback
   - Cannot guarantee VRAM-only policy
   - Cannot verify residency with our checks
   - Performance becomes unpredictable

4. **❌ API contract control**: llama.cpp has its own HTTP server (llama-server)
   - Different endpoint structure
   - Different SSE event format
   - Cannot match our exact API contract

5. **❌ Smart/Dumb boundary**: llama.cpp makes its own decisions
   - Has built-in scheduling logic
   - Makes memory management decisions
   - Violates our "dumb executor" principle

6. **❌ Correlation ID propagation**: No support for X-Correlation-Id
   - Cannot trace requests across services
   - Distributed debugging impossible

7. **❌ Ready callback contract**: No server-first + callback pattern
   - Cannot integrate with pool-manager lifecycle
   - Cannot report model_ref in callback

8. **❌ Quantization control**: llama.cpp decides quantization strategy
   - May not support MXFP4 exactly as we need
   - Cannot control dequantization behavior
   - Health endpoint won't match our schema

### 3.2 Spec Compliance Comparison

| Feature | Custom Worker | External Engines |
| **Narration-core logging** | ✅ Full | ❌ No |
| **Proof bundle emission** | ✅ Full | ❌ No |
| **VRAM-only enforcement** | ✅ Full | ⚠️ Partial |
| **API contract match** | ✅ Exact | ❌ Different |
| **Smart/Dumb boundary** | ✅ Yes | ❌ No |
| **Correlation ID support** | ✅ Yes | ❌ No |
| **Ready callback contract** | ✅ Yes | ❌ No |
| **GGUF support** | ✅ Yes | ⚠️ Varies |
| **MXFP4 control** | ✅ Full | ⚠️ Partial/No |
| **Test reproducibility** | ✅ Full | ⚠️ Partial |
| **Standalone binary** | ✅ Yes | ⚠️ Varies |
| **EU compliance ready** | ✅ Yes | ❌ No |
| **Spec compliance** | 100% | 30-40% |

**Verdict**: External engines cannot meet our unique requirements.

---

## 4. Industry Research Validation

### 4.1 Industry Precedent

From the architectural gap analysis research:

**llama.cpp approach**:
- ✅ Implements architecture detection and dispatch
- ✅ Supports multiple model types via adapters
- ✅ GGUF-native implementation
- ❌ But: Designed as standalone tool, not orchestrated worker

**TensorRT-LLM approach**:
- ✅ Model Definition API with architecture-specific classes
- ✅ Separate kernel paths for MHA/MQA/GQA
- ✅ Production-proven at scale
- ❌ But: NVIDIA-only, Python dependency, no GGUF

**vLLM approach**:
- ✅ Model registry with 100+ architectures
- ✅ PagedAttention optimization
- ✅ Production-proven
- ❌ But: Python dependency, batching-focused, no GGUF

**Key Insight**: All production engines implement architecture-specific dispatch, **but none are designed to be orchestrated workers in a distributed system**.

**Conclusion**: We can reference their implementations (especially llama.cpp for GGUF/kernels) but must build our own worker to meet spec requirements.

### 4.2 Adapter Pattern Validation

Research confirms that **ModelAdapter pattern is industry standard**:

| Framework | Pattern | Our Approach |
|-----------|---------|--------------|
| llama.cpp | Architecture detection → dispatch | ✅ Same pattern |
| TensorRT-LLM | Model Definition API | ✅ Same pattern |
| vLLM | Model registry | ✅ Same pattern |
| HuggingFace | AutoModel registry | ✅ Same pattern |

**Conclusion**: Building custom worker with ModelAdapter is **architecturally correct** and follows industry best practices.

---

## 5. Risk Analysis & Mitigation

### 5.1 Technical Risks
- **Risk**: CUDA kernel bugs
  - **Mitigation**: Start with simple kernels, add unit tests, reference llama.cpp implementations
  - **Severity**: Medium (can debug and fix)

- **Risk**: Performance not competitive
  - **Mitigation**: Optimize iteratively, use cuBLAS for GEMM, reference TensorRT patterns
  - **Severity**: Low (M0 focuses on correctness, performance in M1+)

- **Risk**: Architecture adapter complexity
  - **Mitigation**: Start with 2 adapters (Llama, GPT), add more incrementally
  - **Severity**: Low (well-understood pattern)

### 5.2 Business Risks
- **Risk**: Longer time to market
  - **Impact**: +1-2 weeks
  - **Mitigation**: Accept trade-off for long-term benefits
  - **Severity**: Low (6-7 weeks still reasonable)

**Conclusion**: +1-2 weeks for architecture adapters is **essential investment** to preserve all core features and avoid technical debt.

---

## 6. Implementation Plan

### 6.1 Development Phases

**Phase 1: M0 Foundation (4-5 weeks)**
1. Rust HTTP server with Axum
2. Narration-core logging integration
3. Basic CUDA FFI (context, allocation)
4. Simple kernels (GEMM via cuBLAS, RoPE, attention, RMSNorm)
5. GGUF loader with mmap + chunked transfer
6. Tokenization (gguf-bpe + hf-json backends)
7. SSE streaming with UTF-8 safety
8. Proof bundle emission

**Phase 2: Architecture Adapters (+1-2 weeks)**
1. Define `ModelAdapter` interface
2. Implement `LlamaModelAdapter` (Qwen/Phi-3)
   - RoPE kernel
   - GQA attention
   - RMSNorm
   - SwiGLU FFN
3. Implement `GPTModelAdapter` (GPT-OSS-20B)
   - Absolute positional embedding
   - MHA attention
   - LayerNorm
   - GELU FFN
4. Architecture detection from GGUF metadata
5. Factory pattern for adapter creation

**Phase 3: M1 Enhancements (future)**
1. Performance optimization
2. Additional architectures (Falcon, Mistral, etc.)
3. Advanced kernels (FlashAttention, continuous batching)

**Total Timeline**: **6-7 weeks** (M0 + adapters)

### 6.2 Fallback Option (If Timeline Critical)

**If** +1-2 weeks is absolutely unacceptable:

**Minimal Viable Adapter** (Option B fallback):
1. Implement `ModelAdapter` interface
2. Implement `LlamaModelAdapter` only (Qwen + Phi-3)
3. **Defer** GPT adapter to M1
4. Accept: Only 2 models in M0 (lose GPT-OSS-20B)

**Timeline**: 4-5 weeks (no adapter overhead)

**Trade-off**: Lose MXFP4 validation and large model testing in M0

**Risk**: Still better than external engines (preserves all other features)

---

## 7. Architecture Decision: Single Binary vs Multiple

### 7.1 Question

"Should we make different workers for different architectures?"

### 7.2 Analysis

**Option**: Separate binaries (worker-llama, worker-gpt, worker-falcon)

**Pros**:
- ✅ Simpler per-worker codebase
- ✅ Can optimize each independently

**Cons**:
- ❌ **Deployment complexity**: Multiple binaries to manage
- ❌ **Code duplication**: Shared logic (HTTP, logging, GGUF parsing) duplicated
- ❌ **Testing overhead**: Must test each binary separately
- ❌ **Pool manager complexity**: Must know which worker to spawn
- ❌ **Maintenance burden**: Bug fixes must be applied to all workers
- ❌ **Violates DRY**: Same HTTP/logging/streaming code in each

**Recommended Approach**: **Single worker binary with ModelAdapter pattern**

**Why**:
- ✅ **Single deployment artifact**: One binary to manage
- ✅ **Shared infrastructure**: HTTP, logging, GGUF parsing, streaming
- ✅ **Architecture-specific logic isolated**: In adapter classes
- ✅ **Easy to add architectures**: New adapter = new class, no new binary
- ✅ **Follows industry pattern**: llama.cpp, vLLM, TensorRT all use single binary with dispatch

**Conclusion**: ModelAdapter pattern gives us **best of both worlds** (isolation + shared infrastructure).

---

## 8. Detailed Implementation Roadmap

### 8.1 Phase 1: M0 Foundation (Weeks 1-5)

#### Week 1-2: Core Infrastructure
**Rust Layer**:
- [ ] Set up Axum HTTP server with routes (/execute, /health, /cancel)
- [ ] Implement SSE streaming infrastructure
- [ ] Integrate narration-core logging
- [ ] Add correlation ID middleware (X-Correlation-Id)
- [ ] Implement proof bundle emission hooks

**C++/CUDA FFI Setup**:
- [ ] Define C API interface (cuda_init, cuda_load_model, etc.)
- [ ] Implement Rust FFI bindings
- [ ] Set up CUDA context management
- [ ] Implement basic VRAM allocation/deallocation

#### Week 2-3: GGUF Loader
- [ ] Implement GGUF header parser
- [ ] Add metadata extraction (general.architecture, etc.)
- [ ] Implement mmap-based file I/O
- [ ] Add chunked H2D transfer (1MB chunks)
- [ ] Implement VRAM residency verification
- [ ] Add VRAM OOM error handling

#### Week 3-4: Tokenization
**GGUF byte-BPE backend** (Qwen/Phi-3):
- [ ] Parse GGUF vocab and merges
- [ ] Implement byte-level BPE in Rust
- [ ] Add UTF-8 safe streaming decode
- [ ] Handle BOS/EOS tokens

**HF-JSON backend** (GPT-OSS-20B):
- [ ] Integrate tokenizers crate
- [ ] Load tokenizer.json
- [ ] Add conformance test vectors
- [ ] Implement runtime backend selection

#### Week 4-5: Basic Kernels
- [ ] Implement embedding lookup kernel
- [ ] Add cuBLAS GEMM wrapper
- [ ] Implement RoPE kernel (Llama-style)
- [ ] Add naive attention kernel (prefill + decode)
- [ ] Implement RMSNorm kernel
- [ ] Add temperature-based sampling
- [ ] Implement greedy sampling (temp=0)

**Testing**:
- [ ] Unit tests for each kernel
- [ ] Integration test with Qwen2.5-0.5B
- [ ] Haiku generation test
- [ ] VRAM-only verification test

### 8.2 Phase 2: Architecture Adapters (Weeks 6-7)

#### Week 6: ModelAdapter Pattern
**Interface Design**:
```cpp
class ModelAdapter {
public:
    virtual ~ModelAdapter() = default;
    
    virtual void load_weights_from_gguf(
        const GGUFFile& gguf,
        DeviceMemory& vram_allocation
    ) = 0;
    
    virtual void run_forward_pass(
        const ModelWeights& weights,
        const DeviceMemory& input_tokens,
        DeviceMemory& output_logits,
        KVCache& kv_cache,
        cudaStream_t stream
    ) = 0;
};
```

**Implementation Tasks**:
- [ ] Define ModelAdapter base class
- [ ] Implement architecture detection from GGUF metadata
- [ ] Create adapter factory pattern
- [ ] Add adapter selection logic

#### Week 6-7: LlamaModelAdapter
**Llama-style pipeline** (Qwen/Phi-3):
- [ ] Implement RoPE kernel integration
- [ ] Add GQA attention kernel
- [ ] Implement RMSNorm integration
- [ ] Add SwiGLU FFN kernel
- [ ] Wire up weight loading (Q/K/V, gate/up/down projections)
- [ ] Test with Qwen2.5-0.5B and Phi-3-Mini

#### Week 7: GPTModelAdapter
**GPT-style pipeline** (GPT-OSS-20B):
- [ ] Implement absolute positional embedding kernel
- [ ] Add MHA attention kernel
- [ ] Implement LayerNorm kernel
- [ ] Add GELU activation kernel
- [ ] Wire up weight loading (wte/wpe, Q/K/V separate, fc1/fc2)
- [ ] Add MXFP4 dequantization in kernels
- [ ] Test with GPT-OSS-20B

**MXFP4 Integration**:
- [ ] Implement MXFP4 tile dequantization
- [ ] Add FP16 accumulation paths
- [ ] Wire MXFP4 into all weight consumers (embeddings, attention, FFN, LM head)
- [ ] Add numerical correctness tests (±1% tolerance)

### 8.3 Phase 3: Integration & Testing (Week 7)

#### Integration Tests
- [ ] End-to-end test (all 3 models)
- [ ] Reproducibility test (same seed + temp=0)
- [ ] UTF-8 streaming test (multibyte characters)
- [ ] OOM recovery test
- [ ] VRAM envelope validation (±20% tolerance)

#### API Contract Tests
- [ ] POST /execute endpoint
- [ ] GET /health endpoint
- [ ] POST /cancel endpoint
- [ ] SSE event format validation
- [ ] Correlation ID propagation

#### Performance Baseline (M1 prep)
- [ ] Measure first token latency
- [ ] Measure per-token latency
- [ ] Measure model load time
- [ ] Document baseline for M1 optimization

### 8.4 Deliverables Checklist

**M0 Deliverables**:
- [ ] worker-orcd binary (Rust + C++/CUDA)
- [ ] Support for 3 models (Qwen, Phi-3, GPT-OSS-20B)
- [ ] Support for 3 quantizations (Q4_K_M, MXFP4, Q4_0)
- [ ] 2 tokenizer backends (gguf-bpe, hf-json)
- [ ] HTTP API (/execute, /health, /cancel)
- [ ] SSE streaming with UTF-8 safety
- [ ] Narration-core logging
- [ ] Proof bundle emission
- [ ] VRAM-only enforcement
- [ ] Architecture adapters (Llama, GPT)
- [ ] Test suite (unit + integration)
- [ ] Documentation (README, API docs)

**M0 Success Criteria**:
- [ ] Load Qwen2.5-0.5B into VRAM
- [ ] Execute haiku prompt with temp=0
- [ ] Produce identical token IDs across 2 runs (same device)
- [ ] Stream tokens via SSE
- [ ] All operations VRAM-only (verified)
- [ ] Pass all integration tests

### 8.5 Resource Requirements

**Team**:
- 1 Rust developer (HTTP server, FFI, tokenization)
- 1 CUDA developer (kernels, adapters, GGUF loader)
- 1 QA engineer (testing, validation)

**Infrastructure**:
- Development GPU (24GB VRAM minimum)
- CI/CD with GPU runner
- Test model artifacts (Qwen, Phi-3, GPT-OSS-20B)

**External Dependencies**:
- Axum (HTTP server)
- tokenizers crate (HF tokenizer)
- cuBLAS (GEMM operations)
- CUDA Toolkit 12.x

### 8.6 Risk Mitigation Strategies

**CUDA Kernel Bugs**:
- Start with simple reference implementations
- Add unit tests for each kernel
- Use llama.cpp as reference (not dependency)
- Validate outputs against known-good results

**Performance Issues**:
- Focus on correctness in M0, optimize in M1
- Use cuBLAS for GEMM (battle-tested)
- Profile early, identify bottlenecks
- Document baseline metrics for M1

**Architecture Adapter Complexity**:
- Start with 2 adapters (Llama, GPT)
- Keep interface simple and focused
- Add adapters incrementally
- Reference TensorRT-LLM patterns

**Timeline Slippage**:
- Fallback: Defer GPT adapter to M1 (Llama-only M0)
- Prioritize Qwen2.5-0.5B (primary smoke test)
- Defer performance optimization to M1
- Keep scope flexible (hybrid approach)

---

## 9. Key Implementation Details

### 9.1 CUDA Kernel Strategy

**Approach**: Start simple, optimize later

**M0 Kernels** (functional correctness):
1. **Embedding Lookup**: Simple table lookup
2. **GEMM**: Use cuBLAS (battle-tested)
3. **RoPE**: Reference llama.cpp implementation
4. **Attention**: Naive implementation (prefill + decode)
5. **RMSNorm**: Single reduction pass
6. **LayerNorm**: Two reduction passes (mean + variance)
7. **Sampling**: Temperature scaling + greedy/random

**M1 Optimizations** (deferred):
- FlashAttention integration
- Fused kernels (RMSNorm + attention)
- Continuous batching
- KV cache optimization

### 9.2 MXFP4 Implementation

**Dequantization Strategy**:
```cpp
// In-kernel dequantization
__global__ void mxfp4_gemm_kernel(
    const uint8_t* mxfp4_weights,  // Packed MXFP4
    const half* input,              // FP16 input
    half* output,                   // FP16 output
    int M, int N, int K
) {
    // 1. Load MXFP4 tile to shared memory
    // 2. Dequantize to FP16 in registers
    // 3. Perform GEMM with FP16 accumulation
    // 4. Write FP16 output
}
```

**Weight Consumers**:
1. Embeddings: MXFP4 → FP16 lookup
2. Attention Q/K/V: MXFP4 → FP16 GEMM
3. Attention output: MXFP4 → FP16 GEMM
4. FFN up/down: MXFP4 → FP16 GEMM
5. LM head: MXFP4 → FP16 GEMM

**Validation**:
- Compare against FP32 reference
- Tolerance: ±1% relative error
- Test on GPT-OSS-20B

### 9.3 Tokenization Architecture

**Backend Trait**:
```rust
trait TokenizerBackend {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
    fn vocab_size(&self) -> usize;
}
```

**Runtime Selection**:
```rust
let backend: Box<dyn TokenizerBackend> = match model_metadata.tokenizer_type {
    TokenizerType::HuggingFace => Box::new(HfJsonBackend::new("tokenizer.json")),
    TokenizerType::GgufBpe => Box::new(GgufBpeBackend::new(&gguf_metadata)),
};
```

**UTF-8 Safety**:
- Buffer partial multibyte sequences
- Only emit complete UTF-8 characters in SSE events
- Handle token boundaries that split UTF-8

### 9.4 Error Handling Strategy

**VRAM OOM**:
```rust
// Rust layer
if let Err(CudaError::OutOfMemory) = cuda_inference_start(...) {
    emit_sse_event("error", json!({
        "code": "VRAM_OOM",
        "message": "Insufficient VRAM for KV cache"
    }));
    // Worker stays alive, can serve next request
}
```

**Model Load Failure**:
```rust
// Fail fast at startup
if let Err(e) = cuda_load_model(...) {
    error!("Model load failed: {:?}", e);
    std::process::exit(1);
}
```

**Inference Errors**:
- Emit SSE error event
- Free partial allocations
- Mark worker unhealthy (if critical)
- Continue accepting requests (if recoverable)

---

## 10. Next Steps

### 10.1 Immediate Actions (Week 0)

1. **Finalize team assignments**
   - [ ] Assign Rust developer
   - [ ] Assign CUDA developer
   - [ ] Assign QA engineer

2. **Set up development environment**
   - [ ] Provision GPU development machine (24GB VRAM)
   - [ ] Set up CUDA Toolkit 12.x
   - [ ] Configure CI/CD with GPU runner
   - [ ] Download test models (Qwen, Phi-3, GPT-OSS-20B)

3. **Create project structure**
   - [ ] Initialize Rust workspace
   - [ ] Set up C++/CUDA build system
   - [ ] Configure FFI bindings
   - [ ] Set up test infrastructure

4. **Update specifications**
   - [ ] Add ModelAdapter requirements to M0 spec
   - [ ] Document architecture detection logic
   - [ ] Update API contract with adapter details
   - [ ] Add adapter-specific test requirements

### 10.2 Week 1 Kickoff

1. **Architecture review**
   - Review ModelAdapter pattern
   - Discuss CUDA kernel strategy
   - Align on MXFP4 implementation approach

2. **Begin Phase 1 development**
   - Start Rust HTTP server
   - Begin CUDA FFI setup
   - Implement basic GGUF parser

3. **Set up monitoring**
   - Track progress against 6-7 week timeline
   - Identify blockers early
   - Adjust scope if needed (fallback to Llama-only)

### 10.3 Success Metrics

**Week 2**: HTTP server + FFI working
**Week 3**: GGUF loader functional
**Week 4**: Tokenization backends complete
**Week 5**: Basic kernels working (Qwen test passing)
**Week 6**: ModelAdapter pattern implemented
**Week 7**: All 3 models working, tests passing

**Final Deliverable**: worker-orcd binary ready for M1 integration

---

## 11. Conclusion

**Decision**: Build custom worker-orcd with ModelAdapter pattern.

**Key Rationale**:
1. ✅ **100% spec compliance** (only option that meets all requirements)
2. ✅ **Preserves core value propositions** (reproducibility, logging, VRAM-only)
3. ✅ **Industry-validated approach** (adapter pattern is standard)
4. ✅ **Enables future features** (EU compliance, marketplace, multi-vendor)
5. ✅ **Controlled complexity** (we own the code, can debug and optimize)

**Timeline**: 6-7 weeks total
- Weeks 1-5: M0 foundation (HTTP, GGUF, tokenization, basic kernels)
- Weeks 6-7: Architecture adapters (Llama + GPT)

**Fallback**: If timeline critical, defer GPT adapter to M1 (Llama-only M0)

**Risk Assessment**: MEDIUM (controlled, well-understood)

**Next Action**: Begin Phase 1 development (Section 10)

---

**Status**: ✅ **Approved - Implementation Ready**  
**Start Date**: Immediate  
**Target Completion**: 6-7 weeks from start

