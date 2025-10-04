# Llama Team Sprint Roadmap

**Team**: Llama-Beta  
**Updated**: 2025-10-05  
**Status**: Active Planning

---

## Sprint Overview

| Sprint | Focus | Duration | Status | Stories |
|--------|-------|----------|--------|---------|
| Sprint 1 | GGUF Foundation | 12 days | 📋 Planned | GGUF parsing, memory-mapped I/O |
| Sprint 2 | GGUF-BPE Tokenizer | 9 days | 📋 Planned | Vocab, merges, BPE encoder/decoder |
| Sprint 3 | UTF-8 + Llama Kernels | 6 days | 📋 Planned | UTF-8 safety, RoPE, RMSNorm, residual |
| Sprint 4 | GQA Attention + Gate 1 | 13 days | 📋 Planned | GQA attention, SwiGLU, tests, Gate 1 |
| Sprint 5 | Qwen Integration | 13 days | 📋 Planned | First complete model pipeline |
| Sprint 6 | Phi-3 + Adapter | 11 days | 📋 Planned | Second model, adapter pattern |
| Sprint 7 | Final Integration | 9 days | 📋 Planned | Testing, reproducibility, docs |

**Total Duration**: 73 agent-days (Days 15-88)

---

## Sprint 1: GGUF Foundation 📋 PLANNED

**Duration**: 12 days (Days 15-26)  
**Status**: 📋 Planned  
**Priority**: P0 (Blocks all downstream work)

### Stories
- LT-001: GGUF Header Parser (3 days, includes +1 security)
- LT-002: GGUF Metadata Extraction (2 days)
- LT-003: Memory-Mapped I/O (2 days)
- LT-004: Chunked H2D Transfer (2 days)
- LT-005: Pre-Load Validation (2 days)
- LT-006: Architecture Detection (1 day)

### Deliverables
- Complete GGUF loader with security validation
- Memory-mapped I/O for efficient file access
- Chunked host-to-device transfer
- Pre-load validation framework
- Architecture detection for Llama models

### Success Criteria
- ✅ Parse Qwen2.5-0.5B GGUF file
- ✅ Security validation prevents heap overflow
- ✅ Memory-mapped I/O working
- ✅ Chunked transfer to VRAM
- ✅ All unit tests passing

### Dependencies
- **Upstream**: FT-006, FT-007 (FFI lock Day 15)
- **Downstream**: Sprint 2 (tokenizer needs GGUF loader)

---

## Sprint 2: GGUF-BPE Tokenizer 📋 PLANNED

**Duration**: 9 days (Days 27-35)  
**Status**: 📋 Planned  
**Priority**: P0 (Blocks Qwen integration)

### Stories
- LT-007: GGUF Vocab Parsing (2 days)
- LT-008: GGUF Merges Parsing (2 days)
- LT-009: Byte-Level BPE Encoder (3 days)
- LT-010: Byte-Level BPE Decoder (2 days)

### Deliverables
- Pure Rust byte-level BPE tokenizer
- Vocab and merges extraction from GGUF
- Encoder and decoder implementations
- Conformance test framework

### Success Criteria
- ✅ Extract vocab/merges from Qwen GGUF
- ✅ BPE encoder working
- ✅ BPE decoder working
- ✅ All unit tests passing

### Dependencies
- **Upstream**: Sprint 1 (needs GGUF loader)
- **Downstream**: Sprint 3 (UTF-8 safety needs tokenizer)

---

## Sprint 3: UTF-8 Safety + Llama Kernels 📋 PLANNED

**Duration**: 6 days (Days 36-41)  
**Status**: 📋 Planned  
**Priority**: P0 (Core kernels)

### Stories
- LT-011: UTF-8 Safe Streaming Decode (2 days)
- LT-012: RoPE Kernel (2 days)
- LT-013: RMSNorm Kernel (1 day)
- LT-014: Residual Connection Kernel (1 day)

### Deliverables
- UTF-8 safe streaming decoder
- RoPE (Rotary Position Embedding) kernel
- RMSNorm kernel
- Residual connection kernel

### Success Criteria
- ✅ UTF-8 streaming handles partial sequences
- ✅ RoPE kernel working
- ✅ RMSNorm kernel working
- ✅ Residual connection kernel working
- ✅ All unit tests passing

### Dependencies
- **Upstream**: Sprint 2 (tokenizer complete)
- **Downstream**: Sprint 4 (GQA needs these kernels)

---

## Sprint 4: GQA Attention + Gate 1 📋 PLANNED

**Duration**: 13 days (Days 42-54)  
**Status**: 📋 Planned  
**Priority**: P0 (Critical milestone)

### Stories
- LT-015: GQA Attention Kernel (Prefill) (4 days)
- LT-016: GQA Attention Kernel (Decode) (2 days)
- LT-017: SwiGLU FFN Kernel (2 days)
- LT-018: Tokenizer Conformance Tests (Qwen) (2 days)
- LT-019: Kernel Unit Tests (2 days)
- LT-020: Gate 1 Participation (1 day)

### Deliverables
- GQA attention kernels (prefill + decode)
- SwiGLU FFN kernel
- Comprehensive tokenizer conformance tests
- Kernel unit tests
- Gate 1 validation

### Success Criteria
- ✅ GQA attention working (prefill + decode)
- ✅ SwiGLU FFN working
- ✅ Tokenizer conformance tests passing
- ✅ All kernel unit tests passing
- ✅ Gate 1 checkpoint passed

### Dependencies
- **Upstream**: Sprint 3 (needs base kernels)
- **Downstream**: Sprint 5 (Qwen integration)

---

## Sprint 5: Qwen Integration 🔴 CRITICAL

**Duration**: 13 days (Days 55-67)  
**Status**: 📋 Planned  
**Priority**: P0 (First complete model)

### Stories
- LT-022: Qwen Weight Mapping (3 days)
- LT-023: Qwen Weight Loading to VRAM (2 days)
- LT-024: Qwen Forward Pass Implementation (4 days)
- LT-025: Qwen Haiku Generation Test (2 days)
- LT-026: Qwen Reproducibility Validation (2 days)
- LT-027: Gate 2 Checkpoint (integrated)

### Deliverables
- Complete Qwen2.5-0.5B pipeline
- Weight mapping and loading
- Forward pass implementation
- Haiku generation test
- Reproducibility validation
- Gate 2 validation

### Success Criteria
- ✅ Qwen2.5-0.5B generates haiku
- ✅ Reproducible outputs (10 runs)
- ✅ All integration tests passing
- ✅ Gate 2 checkpoint passed

### Dependencies
- **Upstream**: Sprint 4 (needs all kernels)
- **Downstream**: Sprint 6 (Phi-3 integration)

---

## Sprint 6: Phi-3 + Adapter 📋 PLANNED

**Duration**: 11 days (Days 68-78)  
**Status**: 📋 Planned  
**Priority**: P0 (Second model + adapter)

### Stories
- LT-029: Phi-3 Metadata Analysis (1 day)
- LT-030: Phi-3 Weight Loading (2 days)
- LT-031: Phi-3 Forward Pass (2 days)
- LT-032: Tokenizer Conformance Tests (Phi-3) (2 days)
- LT-033: LlamaInferenceAdapter Implementation (3 days)
- LT-034: Gate 3 Participation (1 day)

### Deliverables
- Phi-3 model support
- LlamaInferenceAdapter implementation
- Tokenizer conformance tests
- Gate 3 validation

### Success Criteria
- ✅ Phi-3 model working
- ✅ Adapter pattern implemented
- ✅ Tokenizer conformance tests passing
- ✅ Gate 3 checkpoint passed

### Dependencies
- **Upstream**: Sprint 5 (needs Qwen working)
- **Downstream**: Sprint 7 (final integration)

---

## Sprint 7: Final Integration 📋 PLANNED

**Duration**: 9 days (Days 79-87)  
**Status**: 📋 Planned  
**Priority**: P0 (M0 completion)

### Stories
- LT-035: Llama Integration Test Suite (3 days)
- LT-036: Reproducibility Tests (10 runs × 2 models) (2 days)
- LT-037: VRAM Pressure Tests (Phi-3) (2 days)
- LT-038: Documentation (GGUF, BPE, Llama) (2 days)

### Deliverables
- Complete integration test suite
- Reproducibility validation (20 runs total)
- VRAM pressure tests
- Comprehensive documentation

### Success Criteria
- ✅ All integration tests passing
- ✅ Reproducibility validated
- ✅ VRAM pressure tests passing
- ✅ Documentation complete
- ✅ Llama-Beta work complete

### Dependencies
- **Upstream**: Sprint 6 (needs adapter)
- **Downstream**: M0 validation

---

## Timeline Projection

### Sequential Execution
**Day 15**: FFI Lock → Start Sprint 1  
**Day 26**: Sprint 1 Complete → Start Sprint 2  
**Day 35**: Sprint 2 Complete → Start Sprint 3  
**Day 41**: Sprint 3 Complete → Start Sprint 4  
**Day 54**: Sprint 4 Complete (Gate 1) → Start Sprint 5  
**Day 67**: Sprint 5 Complete (Gate 2) → Start Sprint 6  
**Day 78**: Sprint 6 Complete (Gate 3) → Start Sprint 7  
**Day 87**: Sprint 7 Complete → Llama-Beta Done

**Total Duration**: 73 agent-days (Days 15-87)

---

## Critical Milestones

### Day 15: FFI Lock (Foundation-Alpha)
**Unblocks**: Llama-Beta Sprint 1

### Day 26: GGUF Loader Complete
**Enables**: Tokenizer implementation

### Day 35: Tokenizer Complete
**Enables**: Qwen integration

### Day 54: Gate 1 - Llama Kernels Complete
**Validation**: All Llama kernels working

### Day 67: Gate 2 - First Model Working (Qwen)
**Validation**: Complete Llama pipeline proven

### Day 78: Gate 3 - Adapter Pattern Complete
**Validation**: LlamaInferenceAdapter implemented

### Day 87: Llama-Beta Complete
**Note**: Before GPT-Gamma (Day 102)

---

## Risk Management

### Sprint 1 Risks
**High**: GGUF format edge cases  
- **Mitigation**: Reference llama.cpp, comprehensive tests

**Medium**: Security validation complexity  
- **Mitigation**: +1 day allocated, auth-min review

### Sprint 2 Risks
**High**: BPE algorithm bugs  
- **Mitigation**: Conformance test vectors, study reference

### Sprint 4 Risks
**High**: GQA attention complexity  
- **Mitigation**: Reference llama.cpp, incremental testing

### Sprint 5 Risks
**Critical**: Qwen forward pass bugs  
- **Mitigation**: Incremental integration, unit tests, Gate 2 validation

---

## Success Metrics

### Sprint Completion
- ✅ All stories in sprint complete
- ✅ All unit tests passing
- ✅ All integration tests passing
- ✅ Documentation updated

### Gate Validation
- ✅ Gate 1: Llama kernels validated
- ✅ Gate 2: First model working
- ✅ Gate 3: Adapter pattern complete

### M0 Readiness
- ✅ 2 Llama models working (Qwen, Phi-3)
- ✅ Reproducibility validated
- ✅ VRAM enforcement working
- ✅ Documentation complete

---

## References

- **Story List**: `docs/complete-story-list.md`
- **Team Charter**: `docs/team-charter.md`
- **Foundation Roadmap**: `../foundation-team/sprints/SPRINT_ROADMAP.md`
- **Coordination**: `../coordination/`

---

Built by Llama-Beta 🦙  
Coordinated by Project Management Team 📋
