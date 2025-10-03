# GPT Team Charter

**Team Name**: 🤖 GPT Team  
**Mission**: Build GPT pipeline (HF tokenizer, GPT kernels, MXFP4, GPT-OSS-20B)  
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
- MXFP4 complexity management

**C++/CUDA Lead**: [TBD]  
- GPT-specific kernels (LayerNorm, GELU, MHA, Abs Pos Emb)
- MXFP4 dequantization kernel
- MXFP4 GEMM integration
- Weight mapping for GPT-OSS-20B
- Performance optimization

**Quantization Specialist**: [TBD]  
- MXFP4 format expertise
- GGUF v3 tensor support
- Numerical correctness validation (±1% tolerance)
- FP16 accumulation paths
- Dequantization testing

**Rust/C++ Developer**: [TBD]  
- HF tokenizers crate integration
- tokenizer.json loading
- Metadata exposure (eos_id, bos_id, vocab_size)
- UTF-8 streaming safety
- Conformance tests

---

## Responsibilities

### Core Deliverables

1. **HF Tokenizer Integration** (Week 2)
   - HF tokenizers crate (Rust)
   - tokenizer.json loading
   - Metadata exposure
   - Conformance test vectors (20-30 pairs)

2. **GPT GGUF Metadata** (Week 3)
   - GPT-style metadata parsing
   - GGUF v3 tensor support (MXFP4)
   - Architecture detection ("gpt2"/"gpt")

3. **GPT Kernels** (Weeks 3-4)
   - Absolute positional embedding
   - LayerNorm (mean + variance, learnable scale/bias)
   - GELU activation
   - MHA (Multi-Head Attention)
   - GPT FFN (fc1 + GELU + fc2)

4. **GPT Basic Pipeline** (Week 5)
   - GPT weight loading (Q4_K_M fallback)
   - End-to-end forward pass
   - UTF-8 streaming tests
   - Large model validation

5. **MXFP4 Implementation** (Weeks 5-6)
   - GGUF v3 tensor parsing
   - MXFP4 dequantization kernel
   - FP16 accumulation paths
   - Wire MXFP4 into all weight consumers
   - Numerical correctness validation (±1%)

6. **GPTInferenceAdapter** (Week 6-7)
   - Formal adapter class
   - Architecture-aware weight mapping
   - Integration with Foundation's adapter pattern
   - Documentation

---

## Success Criteria

### Gate 1 (Week 4): GPT Kernels Ready
- [ ] HF tokenizer can encode/decode
- [ ] GPT metadata parsing working
- [ ] LayerNorm kernel implemented
- [ ] GELU kernel implemented
- [ ] MHA attention kernel working
- [ ] Absolute positional embedding working

### Gate 2 (Week 5): GPT Basic Working
- [ ] GPT-OSS-20B loads (Q4_K_M fallback)
- [ ] Basic generation working
- [ ] UTF-8 streaming safe
- [ ] Large model validation (~16 GB VRAM)

### Gate 3 (Week 6): MXFP4 + GPTAdapter
- [ ] MXFP4 dequantization working
- [ ] GPT-OSS-20B loads with MXFP4 (~16 GB VRAM)
- [ ] Numerical correctness validated (±1%)
- [ ] GPTInferenceAdapter implemented

### Gate 4 (Week 7): Final Validation
- [ ] All GPT integration tests passing
- [ ] MXFP4 end-to-end working
- [ ] 24 GB VRAM boundary tests
- [ ] OOM recovery tests
- [ ] Documentation complete

---

## Working Agreements

### Communication
- Daily standup: 9:30 AM (15 min, after Foundation & Llama)
- Sprint planning: Monday 10:00 AM (2h)
- Friday demo: 2:00 PM (2h, joint with all teams)
- Slack channel: #gpt-team

### Code Review
- All PRs require 1 approval
- MXFP4 changes require 2 approvals (critical)
- Review within 24 hours
- No self-merging

### Testing
- Unit tests required (>80% coverage)
- Numerical correctness tests for MXFP4 (±1% tolerance)
- Integration tests for end-to-end flows
- Conformance test vectors for tokenizer
- No warnings (rustfmt, clippy, clang-format)

### Documentation
- Document MXFP4 format and implementation
- Document GPT architecture specifics
- Keep numerical validation results
- Write handoff notes for M1 (MXFP4 learnings)

---

## Dependencies

### Upstream (Blocking Us)

**Foundation Team**:
- **Week 2**: FFI interface definition (CRITICAL - must be locked)
- **Week 3**: Shared kernels (embedding, GEMM, sampling)
- **Week 4**: Integration test framework

**Llama Team**:
- **Week 3**: GGUF loader learnings (can share)
- **Week 4**: Architecture detection pattern

**Timeline**: We start Week 2, Foundation must have FFI ready

### Downstream (We Block)

**M0 Delivery**:
- MXFP4 is unique to GPT team (no other team has this)
- If MXFP4 slips, GPT-OSS-20B validation at risk
- Gate 3 (MXFP4 working) is critical for M0

---

## Risks

### High Risk

- **MXFP4 complexity**: Novel quantization format, no reference implementation
  - Mitigation: Start early (Week 5), Q4_K_M fallback, numerical tests

- **GPT-OSS-20B size**: 12 GB model, ~16 GB VRAM total, close to 24 GB limit
  - Mitigation: Memory profiling, OOM tests, chunked loading

- **HF tokenizer integration**: Rust crate may have quirks
  - Mitigation: Conformance tests, compare with upstream

### Medium Risk

- **LayerNorm complexity**: Two reduction passes, numerical stability
  - Mitigation: Reference implementations, unit tests

- **MHA vs GQA**: Different from Llama's GQA, all heads unique K/V
  - Mitigation: Reference llama.cpp, unit tests

- **UTF-8 streaming**: GPT-OSS-20B generates long text, edge cases
  - Mitigation: Comprehensive test vectors, fuzzing

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
- GPT-specific kernels (LayerNorm, GELU, MHA, Abs Pos Emb)
- MXFP4 dequantization kernel
- GPTInferenceAdapter implementation

### HF Tokenizer (We Own)

**Responsibilities**:
- Load tokenizer.json from model directory
- Expose metadata (eos_id, bos_id, vocab_size)
- UTF-8 safe encode/decode
- Conformance test vectors

---

## Sprint Velocity Tracking

| Sprint | Committed | Completed | Notes |
|--------|-----------|-----------|-------|
| Week 2 | TBD | TBD | HF tokenizer + GPT metadata |
| Week 3 | TBD | TBD | GPT kernels start |
| Week 4 | TBD | TBD | Gate 1 week |
| Week 5 | TBD | TBD | GPT basic + MXFP4 start |
| Week 6 | TBD | TBD | Gate 3 week (MXFP4 + adapter) |
| Week 7 | TBD | TBD | Gate 4 week |

---

## Model Specifications

### GPT-OSS-20B (MXFP4)

- **Size**: ~12 GB (MXFP4)
- **VRAM**: ~16 GB (with KV cache)
- **Quantization**: MXFP4 (primary), Q4_K_M (fallback)
- **Tokenizer**: HF tokenizers (tokenizer.json)
- **Context**: 8K (test with 2K)
- **Architecture**: GPT-style (Abs Pos, MHA, LayerNorm, GELU)

**MXFP4 Details**:
- Microscaling FP4 format
- Block-based quantization
- Scale factors per block
- FP16 accumulation required
- ±1% numerical tolerance

---

## Unique Challenges

### MXFP4 Quantization

**What It Is**:
- Microscaling FP4: 4-bit floating point with shared exponent per block
- Block size: Typically 32-128 elements
- Scale factor: FP16 or FP8 per block
- Dequantization: In-kernel to registers/shared memory

**Why It's Hard**:
- No reference implementation in llama.cpp (new format)
- Numerical correctness critical (±1% tolerance)
- Must wire into ALL weight consumers (embeddings, attention, FFN, LM head)
- FP16 accumulation paths required

**Our Approach**:
1. Week 5: Implement dequantization kernel, unit tests
2. Week 5: Wire into GEMM (single path)
3. Week 6: Wire into all weight consumers
4. Week 6: Numerical validation (compare with Q4_K_M baseline)
5. Week 6: End-to-end GPT-OSS-20B with MXFP4

### Large Model (GPT-OSS-20B)

**Challenge**: 12 GB model + 4 GB KV cache = 16 GB VRAM (close to 24 GB limit)

**Risks**:
- OOM during inference
- Memory fragmentation
- KV cache overflow

**Mitigations**:
- Memory profiling from day 1
- OOM recovery tests
- Chunked loading (1MB chunks)
- KV cache size validation

---

**Status**: ✅ Charter Approved  
**Last Updated**: 2025-10-03  
**Next Review**: End of Week 3
