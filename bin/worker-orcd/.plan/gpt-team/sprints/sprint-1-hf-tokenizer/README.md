# Sprint 1: HF Tokenizer

**Team**: GPT-Gamma  
**Days**: 15-26 (12 agent-days)  
**Goal**: Integrate HuggingFace tokenizers crate and implement GPT GGUF metadata parsing

---

## Sprint Overview

Sprint 1 begins after the FFI lock (Day 15) and focuses on tokenization infrastructure for GPT-OSS-20B. Unlike Llama models which use GGUF byte-BPE, GPT-OSS-20B requires HuggingFace tokenizer.json format.

This sprint establishes the tokenization foundation and GPT-specific GGUF parsing that all subsequent sprints depend on.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| GT-001 | HF Tokenizers Crate Integration | S | 1 | 15 |
| GT-002 | tokenizer.json Loading | M | 2 | 16-17 |
| GT-003 | Tokenizer Metadata Exposure | S | 1 | 18 |
| GT-004 | HF Tokenizer Conformance Tests | S | 1 | 19 |
| GT-005 | GPT GGUF Metadata Parsing | M | 2 | 20-21 |
| GT-006 | GGUF v3 Tensor Support (MXFP4) | M | 2 | 22-23 |
| GT-007 | Architecture Detection (GPT) | S | 1 | 24 |

**Total**: 7 stories, 12 agent-days (Days 15-26)

---

## Story Execution Order

### Day 15: GT-001 - HF Tokenizers Crate Integration
**Goal**: Add HuggingFace tokenizers crate to project  
**Key Deliverable**: Pure Rust tokenizer backend (no Python dependencies)  
**Blocks**: GT-002 (tokenizer.json loading)

### Days 16-17: GT-002 - tokenizer.json Loading
**Goal**: Load and initialize tokenizer from tokenizer.json file  
**Key Deliverable**: Working tokenizer instance  
**Blocks**: GT-003 (metadata exposure)

### Day 18: GT-003 - Tokenizer Metadata Exposure
**Goal**: Expose vocab size, special tokens, model type  
**Key Deliverable**: Tokenizer metadata API  
**Blocks**: GT-004 (conformance tests)

### Day 19: GT-004 - HF Tokenizer Conformance Tests
**Goal**: Validate encode/decode correctness  
**Key Deliverable**: Comprehensive test suite with 10+ test cases  
**Blocks**: GT-005 (GPT metadata parsing)

### Days 20-21: GT-005 - GPT GGUF Metadata Parsing
**Goal**: Parse GPT-specific GGUF metadata  
**Key Deliverable**: GPTConfig struct with all model parameters  
**Blocks**: GT-006 (GGUF v3 tensor support)

### Days 22-23: GT-006 - GGUF v3 Tensor Support (MXFP4)
**Goal**: Parse GGUF v3 tensors including MXFP4 format  
**Key Deliverable**: Tensor parser supporting MXFP4 type  
**Blocks**: GT-007 (architecture detection)

### Day 24: GT-007 - Architecture Detection (GPT)
**Goal**: Detect GPT architecture from GGUF metadata  
**Key Deliverable**: Architecture detection returning Architecture::GPT  
**Blocks**: Sprint 2 (GPT kernels)

---

## Critical Milestones

### FFI Lock (Day 15)
**What**: Foundation team publishes FFI interface  
**Why Critical**: GPT team cannot start until FFI is stable  
**Deliverable**: `FFI_INTERFACE_LOCKED.md` published by Foundation-Alpha  
**Blocks**: This entire sprint

---

## Dependencies

### Upstream (Blocks This Sprint)
- FT-006: FFI Interface Definition (FFI lock on Day 15)
- FT-007: Rust FFI Bindings (FFI lock on Day 15)

### Downstream (This Sprint Blocks)
- Sprint 2: GPT Kernels (needs tokenizer and metadata parsing)
- GT-008: Absolute Positional Embedding (needs architecture detection)

---

## Success Criteria

Sprint is complete when:
- [ ] All 7 stories marked complete
- [ ] HF tokenizer integrated and tested
- [ ] tokenizer.json loading works
- [ ] Conformance tests passing (10+ test cases)
- [ ] GPT GGUF metadata parsing works
- [ ] GGUF v3 tensor support implemented
- [ ] Architecture detection working
- [ ] Ready for Sprint 2 (GPT kernels)

---

## Next Sprint

**Sprint 2**: GPT Kernels  
**Starts**: Day 27  
**Focus**: Implement GPT-specific CUDA kernels (LayerNorm, GELU, FFN)

---

**Status**: 📋 Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Coordinated by Project Management Team 📋
