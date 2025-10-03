# M0 Worker-orcd Work Breakdown Plan

**Date**: 2025-10-03  
**Scope**: M0 Worker Implementation (6-7 weeks)  
**Teams**: 3 developer teams  
**Status**: Draft for Review

---

## Executive Summary

### Timeline & Scope
- **Total Duration**: 6-7 weeks
- **Phase 1 (Foundation)**: Weeks 1-5 (4-5 weeks)
- **Phase 2 (Architecture Adapters)**: Weeks 6-7 (+1-2 weeks)
- **Teams**: 3 parallel teams with clear boundaries

### Work Distribution Philosophy
1. **Minimize dependencies** between teams
2. **Clear ownership** of components
3. **Integration points** defined upfront
4. **Parallel execution** where possible

---

## Option A: Vertical Split (By Technology Layer)

### Team Allocation

#### **Team 1: Rust Infrastructure (HTTP/Streaming/Integration)**
- **Focus**: HTTP server, API endpoints, SSE streaming, FFI integration
- **Skills**: Rust, Axum, async/await, FFI
- **Duration**: Full 6-7 weeks

#### **Team 2: CUDA Core (Kernels/Inference)**
- **Focus**: CUDA kernels, inference execution, VRAM management
- **Skills**: CUDA, C++, GPU programming
- **Duration**: Full 6-7 weeks

#### **Team 3: Model Pipeline (GGUF/Tokenization/Adapters)**
- **Focus**: GGUF loader, tokenization, architecture adapters
- **Skills**: C++, Rust, model formats, BPE
- **Duration**: Full 6-7 weeks

### Detailed Breakdown

#### **Team 1: Rust Infrastructure**

**Week 1-2: HTTP Server & API Foundation**
- [ ] Set up Axum HTTP server with routes
  - POST /execute
  - GET /health
  - POST /cancel
  - POST /shutdown (optional)
  - GET /metrics (optional)
- [ ] Implement request validation middleware
- [ ] Add correlation ID middleware (X-Correlation-Id)
- [ ] Set up error handling and HTTP response formatting
- [ ] Basic logging integration (tracing crate)

**Week 2-3: SSE Streaming Infrastructure**
- [ ] Implement SSE event stream handler
- [ ] UTF-8 boundary-safe streaming (buffer partial multibyte sequences)
- [ ] Event types: started, token, metrics, end, error
- [ ] Event ordering enforcement (started → token* → end/error)
- [ ] Backpressure handling for slow clients

**Week 3-4: FFI Integration**
- [ ] Define Rust FFI bindings for C API
- [ ] Implement safe wrappers around unsafe FFI calls
- [ ] Error code to Rust Result conversion
- [ ] Memory management across FFI boundary
- [ ] Integration tests for FFI layer

**Week 4-5: API Implementation & Testing**
- [ ] Wire /execute endpoint to CUDA inference
- [ ] Implement /health endpoint with VRAM status
- [ ] Implement /cancel with job tracking
- [ ] Single-threaded execution queue (batch=1)
- [ ] Request timeout handling
- [ ] Integration tests (all endpoints)

**Week 6-7: Polishing & Documentation**
- [ ] Performance baseline measurements
- [ ] API contract validation tests
- [ ] Error scenario testing (OOM, cancellation, etc.)
- [ ] API documentation
- [ ] Deployment guide

**Deliverables**:
- ✅ Working HTTP server with all endpoints
- ✅ SSE streaming with UTF-8 safety
- ✅ FFI integration layer
- ✅ Integration test suite
- ✅ API documentation

---

#### **Team 2: CUDA Core**

**Week 1-2: CUDA Context & Memory Management**
- [ ] Implement Context class (CUDA context init/cleanup)
- [ ] VRAM-only enforcement (disable UMA, set cache config)
- [ ] DeviceMemory RAII wrapper (cudaMalloc/cudaFree)
- [ ] VRAM residency verification (cudaPointerGetAttributes)
- [ ] Process VRAM usage tracking (cudaMemGetInfo)
- [ ] OOM error handling

**Week 2-3: Basic Kernels (Llama-style)**
- [ ] Embedding lookup kernel
- [ ] cuBLAS GEMM wrapper
- [ ] RoPE kernel (Rotary Position Embedding)
- [ ] Naive attention kernel (prefill + decode)
- [ ] RMSNorm kernel
- [ ] Residual connection kernel
- [ ] Unit tests for each kernel

**Week 3-4: Sampling & Inference Loop**
- [ ] Temperature-based sampling kernel
- [ ] Greedy sampling (temp=0) for reproducibility
- [ ] Top-k/top-p sampling (optional)
- [ ] Seeded RNG initialization
- [ ] Inference loop (autoregressive decoding)
- [ ] KV cache management
- [ ] Unit tests for sampling

**Week 4-5: SwiGLU FFN & Integration**
- [ ] SwiGLU activation kernel (gate + value)
- [ ] FFN forward pass (gate/up/down projections)
- [ ] End-to-end forward pass (Llama-style)
- [ ] Integration with Qwen2.5-0.5B
- [ ] Reproducibility tests (same seed → same output)

**Week 6: GPT-Specific Kernels**
- [ ] LayerNorm kernel (mean + variance)
- [ ] GELU activation kernel
- [ ] Absolute positional embedding kernel
- [ ] MHA attention variant (vs GQA)
- [ ] Unit tests for GPT kernels

**Week 7: MXFP4 & Final Integration**
- [ ] MXFP4 dequantization in kernels
- [ ] FP16 accumulation paths
- [ ] Wire MXFP4 into all weight consumers
- [ ] Numerical correctness tests (±1% tolerance)
- [ ] Performance baseline documentation

**Deliverables**:
- ✅ CUDA context management
- ✅ Complete kernel suite (Llama + GPT)
- ✅ MXFP4 support
- ✅ VRAM-only enforcement
- ✅ Unit test suite
- ✅ Reproducibility validation

---

#### **Team 3: Model Pipeline**

**Week 1-2: GGUF Loader Foundation**
- [ ] GGUF header parser (magic, version, tensor count)
- [ ] Metadata extraction (architecture, context_length, etc.)
- [ ] Tensor format parsing (name, dimensions, type, offset)
- [ ] Memory-mapped I/O implementation (mmap)
- [ ] Chunked H2D transfer (1MB chunks)
- [ ] GGUF v3 support (MXFP4 tensors)
- [ ] Pre-load validation (magic bytes, version, size)

**Week 2-3: Architecture Detection & Metadata**
- [ ] Architecture detection from GGUF metadata
- [ ] Architecture enum (Llama, GPT)
- [ ] Metadata structure (ModelMetadata class)
- [ ] Hyperparameter extraction (num_layers, hidden_size, etc.)
- [ ] Unsupported architecture error handling
- [ ] Unit tests for GGUF parsing

**Week 3-4: Tokenization - GGUF-BPE Backend**
- [ ] GGUF vocab and merges parsing
- [ ] Pure-Rust byte-level BPE implementation
- [ ] Encode: UTF-8 → bytes → BPE → token IDs
- [ ] Decode: token IDs → bytes → UTF-8
- [ ] BOS/EOS token handling
- [ ] Special tokens support
- [ ] UTF-8 safety (buffer partial bytes)
- [ ] Conformance test vectors (Qwen, Phi-3)

**Week 4-5: Tokenization - HF-JSON Backend**
- [ ] Integrate tokenizers crate (Rust)
- [ ] Load tokenizer.json
- [ ] Metadata exposure (eos_id, bos_id, vocab_size)
- [ ] UTF-8 streaming decode
- [ ] Conformance test vectors (GPT-OSS-20B)
- [ ] Runtime backend selection logic
- [ ] Health endpoint integration (tokenizer_kind, vocab_size)

**Week 6: InferenceAdapter Pattern**
- [ ] Define InferenceAdapter base class (C++)
- [ ] Factory pattern for adapter creation
- [ ] Architecture detection integration
- [ ] Adapter selection at model load
- [ ] Unit tests for adapter pattern

**Week 6-7: Architecture Adapters Implementation**
- [ ] LlamaInferenceAdapter
  - Weight loading (token_embd, qkv_proj, gate/up/down)
  - Forward pass (RoPE + GQA + RMSNorm + SwiGLU)
  - Integration with Qwen2.5-0.5B and Phi-3-Mini
- [ ] GPTInferenceAdapter
  - Weight loading (wte/wpe, Q/K/V, fc1/fc2)
  - Forward pass (abs pos + MHA + LayerNorm + GELU)
  - Integration with GPT-OSS-20B
- [ ] MXFP4 architecture-aware weight mapping
- [ ] End-to-end tests (all 3 models)

**Deliverables**:
- ✅ GGUF loader (mmap + chunked transfer)
- ✅ Two tokenizer backends (gguf-bpe, hf-json)
- ✅ InferenceAdapter pattern
- ✅ Llama + GPT adapters
- ✅ Architecture detection
- ✅ Conformance test suite

---

### Integration Points (Option A)

**Week 3**: Team 1 ↔ Team 2
- FFI interface definition finalized
- Basic CUDA context callable from Rust

**Week 4**: Team 2 ↔ Team 3
- Model loading returns weights to inference layer
- Tokenizer provides token IDs to inference

**Week 5**: Team 1 ↔ Team 3
- HTTP layer receives tokenizer output
- SSE streaming uses tokenizer decode

**Week 6-7**: All Teams
- Full integration testing
- End-to-end validation (all 3 models)

---

## Option B: Horizontal Split (By Feature/Model)

### Team Allocation

#### **Team 1: Foundation + Qwen2.5-0.5B (Primary Path)**
- **Focus**: Core infrastructure + complete Qwen pipeline
- **Skills**: Rust, C++, CUDA (generalist)
- **Duration**: Full 6-7 weeks

#### **Team 2: Phi-3-Mini + Llama Adapter (Stretch Path)**
- **Focus**: Llama adapter + Phi-3 integration
- **Skills**: C++, CUDA, model architectures
- **Duration**: Weeks 3-7 (starts after foundation)

#### **Team 3: GPT-OSS-20B + GPT Adapter (Large Model Path)**
- **Focus**: GPT adapter + MXFP4 + HF tokenizer
- **Skills**: C++, CUDA, quantization
- **Duration**: Weeks 3-7 (starts after foundation)

### Detailed Breakdown

#### **Team 1: Foundation + Qwen2.5-0.5B**

**Week 1-2: Core Infrastructure**
- [ ] Axum HTTP server (all endpoints)
- [ ] SSE streaming infrastructure
- [ ] CUDA context management
- [ ] FFI layer (Rust ↔ C++)
- [ ] Basic error handling

**Week 2-3: GGUF + Tokenization (Qwen)**
- [ ] GGUF loader (mmap + chunked transfer)
- [ ] GGUF-BPE tokenizer (pure Rust)
- [ ] Architecture detection (basic)
- [ ] Qwen metadata parsing

**Week 3-4: Llama Kernels (Qwen)**
- [ ] Embedding lookup
- [ ] RoPE kernel
- [ ] GQA attention
- [ ] RMSNorm
- [ ] SwiGLU FFN
- [ ] Sampling

**Week 4-5: Qwen Integration & Testing**
- [ ] End-to-end Qwen2.5-0.5B pipeline
- [ ] Haiku generation test
- [ ] Reproducibility validation
- [ ] VRAM-only verification
- [ ] API contract tests

**Week 6-7: InferenceAdapter Pattern + Coordination**
- [ ] Define InferenceAdapter interface
- [ ] Refactor Qwen pipeline into LlamaAdapter
- [ ] Integration coordination with Teams 2 & 3
- [ ] Final testing and documentation

**Deliverables**:
- ✅ Complete Qwen2.5-0.5B pipeline (end-to-end)
- ✅ Core infrastructure (HTTP, FFI, CUDA)
- ✅ InferenceAdapter pattern
- ✅ Foundation for other teams

---

#### **Team 2: Phi-3-Mini + Llama Adapter**

**Week 3-4: Phi-3 Preparation**
- [ ] Phi-3 GGUF metadata analysis
- [ ] Tokenizer conformance tests (Phi-3 specific)
- [ ] VRAM envelope calculation (~3.5 GB)
- [ ] Test model download and validation

**Week 4-5: Phi-3 Integration**
- [ ] Adapt Llama kernels for Phi-3 (if needed)
- [ ] Sliding window attention (if Phi-3 uses it)
- [ ] Phi-3 weight mapping
- [ ] End-to-end Phi-3 pipeline

**Week 5-6: LlamaInferenceAdapter**
- [ ] Formalize LlamaInferenceAdapter class
- [ ] Refactor Qwen + Phi-3 to use adapter
- [ ] Weight loading abstraction
- [ ] Forward pass abstraction
- [ ] Unit tests for adapter

**Week 6-7: Testing & Validation**
- [ ] Qwen + Phi-3 integration tests
- [ ] Reproducibility tests (both models)
- [ ] VRAM pressure tests (Phi-3)
- [ ] Context length validation
- [ ] Documentation

**Deliverables**:
- ✅ Phi-3-Mini support
- ✅ LlamaInferenceAdapter (production-ready)
- ✅ Llama-style model validation
- ✅ Test suite (Qwen + Phi-3)

---

#### **Team 3: GPT-OSS-20B + GPT Adapter**

**Week 3-4: GPT Foundation**
- [ ] HF tokenizers crate integration
- [ ] tokenizer.json loading
- [ ] GPT metadata parsing (GGUF)
- [ ] Absolute positional embedding kernel
- [ ] LayerNorm kernel
- [ ] GELU activation kernel

**Week 4-5: GPT Inference Pipeline**
- [ ] MHA attention kernel
- [ ] GPT FFN (fc1 + GELU + fc2)
- [ ] GPT weight mapping (wte/wpe, Q/K/V, etc.)
- [ ] Basic GPT forward pass (Q4_K_M fallback)

**Week 5-6: MXFP4 Implementation**
- [ ] MXFP4 dequantization kernel
- [ ] FP16 accumulation paths
- [ ] Wire MXFP4 into all weight consumers
- [ ] Numerical correctness validation (±1%)
- [ ] GGUF v3 tensor support

**Week 6-7: GPTInferenceAdapter + Integration**
- [ ] Formalize GPTInferenceAdapter class
- [ ] Architecture-aware weight mapping
- [ ] End-to-end GPT-OSS-20B pipeline (MXFP4)
- [ ] Large model testing (24 GB boundary)
- [ ] UTF-8 streaming tests
- [ ] OOM recovery tests

**Deliverables**:
- ✅ GPT-OSS-20B support (MXFP4)
- ✅ GPTInferenceAdapter (production-ready)
- ✅ HF tokenizer integration
- ✅ MXFP4 validation
- ✅ Large model test suite

---

### Integration Points (Option B)

**Week 2-3**: Team 1 completes foundation
- FFI interface stable
- HTTP server operational
- CUDA context working
- Teams 2 & 3 can start

**Week 5**: Mid-point sync
- Team 1: Qwen working
- Team 2: Phi-3 working
- Team 3: GPT basic pipeline working
- Adapter pattern design review

**Week 6**: Adapter integration
- All teams implement adapter pattern
- Cross-team code review
- Interface alignment

**Week 7**: Final integration
- All 3 models working
- Full test suite passing
- Documentation complete

---

## Option C: Hybrid Split (Infrastructure + Model Teams)

### Team Allocation

#### **Team 1: Core Infrastructure (Rust + CUDA Foundation)**
- **Focus**: HTTP, FFI, CUDA context, basic kernels
- **Skills**: Rust, C++, CUDA
- **Duration**: Weeks 1-4 (then support role)

#### **Team 2: Llama Pipeline (Qwen + Phi-3)**
- **Focus**: GGUF-BPE, Llama kernels, Llama adapter
- **Skills**: C++, CUDA, BPE
- **Duration**: Weeks 2-7

#### **Team 3: GPT Pipeline (GPT-OSS-20B + MXFP4)**
- **Focus**: HF tokenizer, GPT kernels, GPT adapter, MXFP4
- **Skills**: C++, CUDA, quantization
- **Duration**: Weeks 2-7

### Detailed Breakdown

#### **Team 1: Core Infrastructure**

**Week 1-2: Foundation**
- [ ] Axum HTTP server (all endpoints)
- [ ] SSE streaming (UTF-8 safe)
- [ ] FFI layer (Rust ↔ C++)
- [ ] CUDA context management
- [ ] VRAM allocation/tracking
- [ ] Error handling framework

**Week 3-4: Shared Kernels & Integration**
- [ ] Embedding lookup kernel (generic)
- [ ] cuBLAS GEMM wrapper
- [ ] Sampling kernels (temperature, greedy)
- [ ] Seeded RNG
- [ ] KV cache management
- [ ] Integration test framework

**Week 5-7: Support & Polishing**
- [ ] Support Teams 2 & 3 with integration issues
- [ ] Performance baseline measurements
- [ ] API documentation
- [ ] Deployment guide
- [ ] CI/CD setup

**Deliverables**:
- ✅ HTTP server + SSE streaming
- ✅ FFI layer
- ✅ CUDA foundation
- ✅ Shared kernels
- ✅ Integration framework

---

#### **Team 2: Llama Pipeline**

**Week 2-3: GGUF + Tokenization**
- [ ] GGUF loader (mmap + chunked transfer)
- [ ] GGUF metadata parsing (Llama-style)
- [ ] GGUF-BPE tokenizer (pure Rust)
- [ ] Conformance test vectors (Qwen, Phi-3)

**Week 3-4: Llama Kernels**
- [ ] RoPE kernel
- [ ] GQA attention kernel
- [ ] RMSNorm kernel
- [ ] SwiGLU FFN kernel
- [ ] Residual connections
- [ ] Unit tests

**Week 4-5: Qwen Integration**
- [ ] Qwen2.5-0.5B weight mapping
- [ ] End-to-end Qwen pipeline
- [ ] Haiku generation test
- [ ] Reproducibility validation

**Week 5-6: Phi-3 Integration**
- [ ] Phi-3 weight mapping
- [ ] Phi-3 specific adaptations (if needed)
- [ ] End-to-end Phi-3 pipeline
- [ ] VRAM pressure tests

**Week 6-7: LlamaInferenceAdapter**
- [ ] Formalize LlamaInferenceAdapter
- [ ] Refactor Qwen + Phi-3 to use adapter
- [ ] Integration tests (both models)
- [ ] Documentation

**Deliverables**:
- ✅ GGUF loader
- ✅ GGUF-BPE tokenizer
- ✅ Llama kernels
- ✅ Qwen + Phi-3 support
- ✅ LlamaInferenceAdapter

---

#### **Team 3: GPT Pipeline**

**Week 2-3: HF Tokenizer + GPT Foundation**
- [ ] HF tokenizers crate integration
- [ ] tokenizer.json loading
- [ ] Metadata exposure (eos_id, bos_id, vocab_size)
- [ ] Conformance test vectors (GPT-OSS-20B)
- [ ] GGUF metadata parsing (GPT-style)

**Week 3-4: GPT Kernels**
- [ ] Absolute positional embedding kernel
- [ ] MHA attention kernel
- [ ] LayerNorm kernel
- [ ] GELU activation kernel
- [ ] GPT FFN (fc1 + fc2)
- [ ] Unit tests

**Week 4-5: GPT Integration (Q4_K_M Fallback)**
- [ ] GPT weight mapping (Q4_K_M)
- [ ] End-to-end GPT pipeline (basic)
- [ ] UTF-8 streaming tests
- [ ] Large model validation

**Week 5-6: MXFP4 Implementation**
- [ ] GGUF v3 tensor support
- [ ] MXFP4 dequantization kernel
- [ ] FP16 accumulation paths
- [ ] Wire MXFP4 into all weight consumers
- [ ] Numerical correctness tests (±1%)

**Week 6-7: GPTInferenceAdapter + Final Integration**
- [ ] Formalize GPTInferenceAdapter
- [ ] Architecture-aware weight mapping
- [ ] End-to-end GPT-OSS-20B (MXFP4)
- [ ] 24 GB VRAM boundary tests
- [ ] OOM recovery tests
- [ ] Documentation

**Deliverables**:
- ✅ HF tokenizer integration
- ✅ GPT kernels
- ✅ MXFP4 support
- ✅ GPT-OSS-20B support
- ✅ GPTInferenceAdapter

---

### Integration Points (Option C)

**Week 2**: Team 1 → Teams 2 & 3
- HTTP server ready
- FFI interface stable
- CUDA context available
- Teams 2 & 3 start model work

**Week 4**: Teams 2 & 3 → Team 1
- Model pipelines ready for integration
- HTTP endpoints wired to inference
- SSE streaming connected to tokenizers

**Week 6**: Adapter pattern integration
- All teams implement adapter pattern
- Cross-team code review
- Interface alignment

**Week 7**: Final integration
- All 3 models working
- Full test suite passing
- Documentation complete

---

## Comparison & Recommendation

### Option A: Vertical Split (By Technology Layer)

**Pros**:
- ✅ Clear technology boundaries (Rust vs CUDA vs Model)
- ✅ Specialists can focus on their domain
- ✅ Minimal context switching
- ✅ Easy to parallelize within each layer

**Cons**:
- ❌ High integration risk (teams must sync frequently)
- ❌ Dependencies between layers create bottlenecks
- ❌ Harder to deliver end-to-end features early
- ❌ Testing requires all layers to be ready

**Best For**: Teams with deep specialization (Rust experts, CUDA experts, Model experts)

---

### Option B: Horizontal Split (By Feature/Model)

**Pros**:
- ✅ Each team owns end-to-end feature (model pipeline)
- ✅ Early delivery of working models (Qwen in Week 5)
- ✅ Lower integration risk (foundation first, then parallel)
- ✅ Clear success criteria per team (model working)

**Cons**:
- ❌ Team 1 has heavy load (foundation + Qwen)
- ❌ Teams 2 & 3 idle until Week 3
- ❌ Potential code duplication (adapter pattern implemented 3 times)
- ❌ Requires generalist skills (each team does Rust + C++ + CUDA)

**Best For**: Full-stack teams with broad skill sets

---

### Option C: Hybrid Split (Infrastructure + Model Teams)

**Pros**:
- ✅ Balanced workload (infrastructure vs models)
- ✅ Clear separation (foundation vs domain-specific)
- ✅ Parallel model development (Teams 2 & 3)
- ✅ Shared infrastructure reduces duplication
- ✅ Specialists can focus (Team 1 = infra, Teams 2/3 = models)

**Cons**:
- ⚠️ Team 1 becomes bottleneck if foundation delayed
- ⚠️ Requires coordination between Teams 2 & 3 (adapter pattern)
- ⚠️ Team 1 transitions to support role (may lose focus)

**Best For**: Mixed team composition (infra specialists + model specialists)

---

## Recommendation: **Option C (Hybrid Split)**

### Rationale

1. **Balanced Workload**
   - Team 1 builds foundation (Weeks 1-4), then supports
   - Teams 2 & 3 work in parallel on models (Weeks 2-7)
   - No team is idle or overloaded

2. **Clear Boundaries**
   - Infrastructure vs Domain-specific work
   - Shared kernels vs Architecture-specific kernels
   - Generic vs Model-specific code

3. **Risk Mitigation**
   - Foundation ready by Week 4 (before adapter work)
   - Two model teams can help each other (Llama vs GPT)
   - Adapter pattern designed collaboratively (Week 5-6)

4. **Skill Alignment**
   - Team 1: Rust + C++ generalists (infrastructure)
   - Team 2: C++ + CUDA + BPE specialists (Llama)
   - Team 3: C++ + CUDA + quantization specialists (GPT)

5. **Delivery Milestones**
   - Week 4: Foundation complete (Team 1)
   - Week 5: Qwen working (Team 2)
   - Week 6: Phi-3 + GPT basic working (Teams 2 & 3)
   - Week 7: All models + adapters complete (All teams)

### Success Criteria

**Week 4 Gate** (Foundation Complete):
- [ ] HTTP server operational (all endpoints)
- [ ] SSE streaming working (UTF-8 safe)
- [ ] FFI layer stable (Rust ↔ C++)
- [ ] CUDA context management working
- [ ] Shared kernels implemented (embedding, GEMM, sampling)
- [ ] Integration test framework ready

**Week 5 Gate** (First Model Working):
- [ ] Qwen2.5-0.5B end-to-end pipeline working
- [ ] Haiku generation test passing
- [ ] Reproducibility validated (same seed → same output)
- [ ] VRAM-only verified

**Week 6 Gate** (All Models Basic):
- [ ] Phi-3-Mini working (Llama adapter)
- [ ] GPT-OSS-20B working (Q4_K_M fallback)
- [ ] Adapter pattern designed and agreed

**Week 7 Gate** (M0 Complete):
- [ ] All 3 models working with adapters
- [ ] MXFP4 validated (GPT-OSS-20B)
- [ ] Full test suite passing
- [ ] Documentation complete

---

## Risk Management

### High-Risk Items

1. **Foundation Delay (Team 1)**
   - **Risk**: If foundation not ready by Week 4, Teams 2 & 3 blocked
   - **Mitigation**: 
     - Team 1 focuses on FFI interface first (Week 1)
     - Teams 2 & 3 can start GGUF/tokenization work independently
     - Weekly sync meetings to track progress

2. **Adapter Pattern Misalignment (Teams 2 & 3)**
   - **Risk**: Teams implement incompatible adapter interfaces
   - **Mitigation**:
     - Week 5: Joint design session (all teams)
     - Shared header file (InferenceAdapter.hpp)
     - Code review before implementation

3. **MXFP4 Complexity (Team 3)**
   - **Risk**: MXFP4 takes longer than expected
   - **Mitigation**:
     - Start with Q4_K_M fallback (Week 4-5)
     - MXFP4 as incremental addition (Week 5-6)
     - Fallback: Defer MXFP4 to M1 if critical

4. **Integration Issues (Week 6-7)**
   - **Risk**: Components don't integrate smoothly
   - **Mitigation**:
     - Integration tests from Week 4
     - Daily standups in Week 6-7
     - Dedicated integration time (Week 7)

### Medium-Risk Items

1. **CUDA Kernel Bugs**
   - **Mitigation**: Unit tests, reference llama.cpp, early testing

2. **UTF-8 Streaming Edge Cases**
   - **Mitigation**: Comprehensive test vectors, fuzzing

3. **VRAM OOM Handling**
   - **Mitigation**: Explicit OOM tests, graceful degradation

---

## Team Composition Recommendations

### Team 1: Core Infrastructure (3 people)
- **1x Rust Lead**: Axum, async/await, FFI expert
- **1x C++ Lead**: CUDA context, memory management, FFI
- **1x DevOps/QA**: CI/CD, integration tests, deployment

### Team 2: Llama Pipeline (2-3 people)
- **1x C++/CUDA Lead**: Llama kernels (RoPE, GQA, RMSNorm, SwiGLU)
- **1x Rust/C++ Dev**: GGUF loader, GGUF-BPE tokenizer
- **1x QA (optional)**: Conformance tests, validation

### Team 3: GPT Pipeline (2-3 people)
- **1x C++/CUDA Lead**: GPT kernels (LayerNorm, GELU, MHA)
- **1x Quantization Specialist**: MXFP4 implementation, numerical validation
- **1x Rust/C++ Dev**: HF tokenizer, GGUF v3 support

**Total**: 7-9 people (can scale down to 6 if teams share QA)

---

## Alternative: Minimal Viable Team (3 people)

If only 3 developers available:

### Team 1: Full-Stack Lead (Foundation + Qwen)
- Weeks 1-5: Build foundation + Qwen pipeline
- Weeks 6-7: Adapter pattern + integration support

### Team 2: Llama Specialist (Phi-3 + Llama Adapter)
- Weeks 3-7: Phi-3 + LlamaInferenceAdapter

### Team 3: GPT Specialist (GPT-OSS-20B + GPT Adapter)
- Weeks 3-7: GPT + MXFP4 + GPTInferenceAdapter

**Trade-off**: Longer timeline (8-9 weeks) or reduced scope (defer Phi-3 or GPT to M1)

---

## Next Steps

### Immediate Actions (Week 0)

1. **Finalize Team Assignments**
   - [ ] Assign developers to teams
   - [ ] Confirm skill sets match team needs
   - [ ] Set up communication channels (Slack, etc.)

2. **Set Up Development Environment**
   - [ ] Provision GPU development machines (24GB VRAM)
   - [ ] Install CUDA Toolkit 12.x
   - [ ] Configure CI/CD with GPU runner
   - [ ] Download test models (Qwen, Phi-3, GPT-OSS-20B)

3. **Define Interfaces**
   - [ ] FFI interface (Rust ↔ C++)
   - [ ] InferenceAdapter interface (C++)
   - [ ] TokenizerBackend trait (Rust)
   - [ ] API contract (HTTP endpoints)

4. **Kick-off Meeting**
   - [ ] Review work breakdown plan
   - [ ] Align on integration points
   - [ ] Set up weekly sync meetings
   - [ ] Establish code review process

### Week 1 Kickoff

**Team 1**: Start HTTP server + FFI
**Team 2**: Start GGUF research + tokenizer design
**Team 3**: Start HF tokenizer + GPT research

**All Teams**: Daily standups, weekly sync (Fridays)

---

## Success Metrics

### Quantitative
- [ ] All 3 models load successfully
- [ ] Haiku test passes (Qwen)
- [ ] Reproducibility validated (same seed → same output)
- [ ] VRAM-only verified (all models)
- [ ] API contract tests pass (100%)
- [ ] Unit test coverage >80%
- [ ] Integration test coverage >90%

### Qualitative
- [ ] Code is maintainable and well-documented
- [ ] Teams report smooth collaboration
- [ ] No major technical debt
- [ ] Architecture is extensible (easy to add new models)

---

## Appendix: Dependency Graph

```
Week 1-2: Team 1 (Foundation)
    ↓
Week 3: Teams 2 & 3 (Model work starts)
    ↓
Week 4: Integration (HTTP ↔ CUDA ↔ Models)
    ↓
Week 5: First model working (Qwen)
    ↓
Week 6: Adapter pattern (All teams)
    ↓
Week 7: Final integration (All models)
```

**Critical Path**: Team 1 (Weeks 1-4) → Adapter Pattern (Week 6) → Final Integration (Week 7)

---

## Conclusion

**Recommended Approach**: **Option C (Hybrid Split)**

**Timeline**: 6-7 weeks (realistic with 3 teams)

**Key Success Factors**:
1. ✅ Foundation ready by Week 4 (Team 1)
2. ✅ Parallel model development (Teams 2 & 3)
3. ✅ Adapter pattern collaboration (Week 5-6)
4. ✅ Integration time allocated (Week 7)
5. ✅ Clear interfaces defined upfront

**Fallback Options**:
- If timeline critical: Defer Phi-3 or GPT to M1 (Llama-only M0)
- If MXFP4 complex: Use Q4_K_M fallback for GPT-OSS-20B
- If team size limited: Extend timeline to 8-9 weeks

**Next Action**: Finalize team assignments and begin Week 0 setup.

---

**Status**: ✅ **Ready for Review**  
**Approval Required**: Project Manager + Tech Leads  
**Target Start Date**: Immediate (upon approval)
