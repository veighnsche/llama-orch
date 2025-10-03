# Llama Team Charter

**Team Name**: 🦙 Llama Team  
**Mission**: Build Llama pipeline (GGUF loader, GGUF-BPE tokenizer, Llama kernels, Qwen + Phi-3)  
**Duration**: Weeks 2-7 (6 weeks, starts after Foundation begins)  
**Status**: Active

---

## Team Composition

### Roles

**Team Lead**: [TBD]  
- Sprint planning and coordination
- Integration with Foundation Team
- Gate tracking (Gates 1, 2, 3, 4)
- Blocker resolution

**C++/CUDA Lead**: [TBD]  
- Llama-specific kernels (RoPE, GQA, RMSNorm, SwiGLU)
- GGUF loader (C++ side)
- Weight mapping for Qwen/Phi-3
- Performance optimization

**Rust/C++ Developer**: [TBD]  
- GGUF loader (Rust side)
- GGUF-BPE tokenizer (pure Rust)
- FFI integration
- Conformance tests

**QA/Integration**: [TBD] (optional, can share with Foundation)  
- Integration tests
- Conformance test vectors
- Reproducibility validation
- Documentation

---

## Responsibilities

### Core Deliverables

1. **GGUF Loader** (Weeks 2-3)
   - Header parsing (magic, version, metadata)
   - Metadata extraction (architecture, vocab, hyperparams)
   - Memory-mapped I/O
   - Chunked H2D transfer (1MB chunks)
   - Architecture detection

2. **GGUF-BPE Tokenizer** (Weeks 3-4)
   - Pure Rust implementation
   - Vocab and merges parsing from GGUF
   - Byte-level BPE encoder
   - Byte-level BPE decoder
   - UTF-8 safe streaming
   - Conformance test vectors (20-30 pairs)

3. **Llama Kernels** (Weeks 3-4)
   - RoPE (Rotary Position Embedding)
   - GQA (Grouped Query Attention)
   - RMSNorm
   - SwiGLU FFN
   - Residual connections

4. **Qwen Integration** (Week 5)
   - Qwen2.5-0.5B weight loading
   - End-to-end Qwen pipeline
   - Haiku generation test
   - Reproducibility validation

5. **Phi-3 Integration** (Week 6)
   - Phi-3-Mini weight loading
   - End-to-end Phi-3 pipeline
   - VRAM pressure tests
   - Context length validation

6. **LlamaInferenceAdapter** (Week 6-7)
   - Formal adapter class
   - Refactor Qwen + Phi-3 to use adapter
   - Integration with Foundation's adapter pattern
   - Documentation

---

## Success Criteria

### Gate 1 (Week 4): Llama Kernels Ready
- [ ] GGUF loader can parse headers and metadata
- [ ] GGUF-BPE tokenizer can encode/decode
- [ ] RoPE kernel implemented and tested
- [ ] GQA attention kernel working
- [ ] RMSNorm kernel working
- [ ] SwiGLU FFN kernel working

### Gate 2 (Week 5): Qwen Working
- [ ] Qwen2.5-0.5B loads to VRAM
- [ ] Haiku generation test passes
- [ ] Reproducibility validated (same seed → same output, 3 runs)
- [ ] VRAM-only verified
- [ ] Tokenization round-trip works

### Gate 3 (Week 6): Phi-3 + LlamaAdapter
- [ ] Phi-3-Mini working
- [ ] LlamaInferenceAdapter implemented
- [ ] Qwen + Phi-3 refactored to use adapter
- [ ] Integration tests passing

### Gate 4 (Week 7): Final Validation
- [ ] All Llama integration tests passing
- [ ] Reproducibility tests (10 runs each model)
- [ ] VRAM pressure tests
- [ ] Documentation complete

---

## Working Agreements

### Communication
- Daily standup: 9:15 AM (15 min, after Foundation Team)
- Sprint planning: Monday 10:00 AM (2h)
- Friday demo: 2:00 PM (2h, joint with all teams)
- Slack channel: #llama-team

### Code Review
- All PRs require 1 approval
- Critical changes require 2 approvals
- Review within 24 hours
- No self-merging

### Testing
- Unit tests required (>80% coverage)
- Integration tests for end-to-end flows
- Conformance test vectors for tokenizer
- No warnings (rustfmt, clippy, clang-format)

### Documentation
- Document GGUF format learnings
- Document BPE algorithm implementation
- Keep test vectors up to date
- Write handoff notes for GPT team (GGUF learnings)

---

## Dependencies

### Upstream (Blocking Us)

**Foundation Team**:
- **Week 2**: FFI interface definition (CRITICAL - must be locked)
- **Week 3**: Shared kernels (embedding, GEMM, sampling)
- **Week 4**: Integration test framework

**Timeline**: We start Week 2, Foundation must have FFI ready

### Downstream (We Block)

**GPT Team**:
- GGUF loader learnings (can share Week 3)
- Architecture detection pattern (Week 4)

**M0 Delivery**:
- Gate 2 (Qwen working) is critical milestone
- If we slip, M0 delivery at risk

---

## Risks

### High Risk

- **FFI interface instability**: If Foundation changes FFI after Week 2, we're blocked
  - Mitigation: Participate in FFI design review, lock interface early

- **GGUF format complexity**: Parsing errors, edge cases
  - Mitigation: Reference llama.cpp implementation, comprehensive tests

- **BPE algorithm bugs**: Tokenization mismatches
  - Mitigation: Conformance test vectors, compare with upstream

### Medium Risk

- **Qwen-specific quirks**: Model-specific edge cases
  - Mitigation: Start with Qwen (simpler), test thoroughly

- **Phi-3 differences**: May have architecture variations
  - Mitigation: Research Phi-3 architecture early (Week 3)

- **GQA attention complexity**: Grouped K/V heads tricky
  - Mitigation: Reference llama.cpp, unit tests with known patterns

---

## Key Interfaces

### FFI Interface (From Foundation Team)

**Functions We Use**:
- `cuda_init()` - Initialize CUDA context
- `cuda_load_model()` - Load model to VRAM
- `cuda_inference_start()` - Start inference
- `cuda_inference_next_token()` - Get next token
- `cuda_check_vram_residency()` - Health check

**Functions We Provide** (via Foundation's C++ layer):
- Llama-specific kernels (RoPE, GQA, RMSNorm, SwiGLU)
- LlamaInferenceAdapter implementation

### GGUF Format (We Own)

**Responsibilities**:
- Parse GGUF headers and metadata
- Extract vocab and merges for tokenizer
- Detect architecture ("llama" for Qwen/Phi-3)
- Load weights to VRAM via Foundation's FFI

---

## Sprint Velocity Tracking

| Sprint | Committed | Completed | Notes |
|--------|-----------|-----------|-------|
| Week 2 | TBD | TBD | GGUF foundation |
| Week 3 | TBD | TBD | Tokenization + kernels |
| Week 4 | TBD | TBD | Gate 1 week |
| Week 5 | TBD | TBD | Gate 2 week (Qwen) |
| Week 6 | TBD | TBD | Gate 3 week (Phi-3 + adapter) |
| Week 7 | TBD | TBD | Gate 4 week |

---

## Model Specifications

### Qwen2.5-0.5B-Instruct (Primary)

- **Size**: 352 MB
- **VRAM**: ~400 MB (with KV cache)
- **Quantization**: Q4_K_M
- **Tokenizer**: GGUF byte-BPE
- **Context**: 32K (test with 2K)
- **Architecture**: Llama-style (RoPE, GQA, RMSNorm, SwiGLU)

### Phi-3-Mini-4K-Instruct (Stretch)

- **Size**: 2.3 GB
- **VRAM**: ~3.5 GB (with KV cache)
- **Quantization**: Q4_K_M
- **Tokenizer**: GGUF byte-BPE
- **Context**: 4K
- **Architecture**: Llama-style (may have variations)

---

**Status**: ✅ Charter Approved  
**Last Updated**: 2025-10-03  
**Next Review**: End of Week 3
