# Sprint 2: GGUF-BPE Tokenizer

**Team**: Llama-Beta  
**Days**: 27-35 (9 agent-days)  
**Goal**: Implement pure Rust byte-level BPE tokenizer

---

## Sprint Overview

Sprint 2 implements a pure Rust byte-level BPE tokenizer that extracts vocabulary and merges from GGUF files. This tokenizer is critical for Qwen integration and must achieve conformance with reference implementations.

This sprint establishes the tokenization foundation required for all Llama model inference.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| LT-007 | GGUF Vocab Parsing | M | 2 | 27-28 |
| LT-008 | GGUF Merges Parsing | M | 2 | 29-30 |
| LT-009 | Byte-Level BPE Encoder | M | 3 | 31-33 |
| LT-010 | Byte-Level BPE Decoder | M | 2 | 34-35 |

**Total**: 4 stories, 9 agent-days (Days 27-35)

---

## Story Execution Order

### Days 27-28: LT-007 - GGUF Vocab Parsing
**Goal**: Parse vocabulary from GGUF metadata  
**Key Deliverable**: Vocab extraction from GGUF  
**Blocks**: LT-009 (BPE encoder)

### Days 29-30: LT-008 - GGUF Merges Parsing
**Goal**: Parse BPE merges from GGUF metadata  
**Key Deliverable**: Merges extraction from GGUF  
**Blocks**: LT-009 (BPE encoder)

### Days 31-33: LT-009 - Byte-Level BPE Encoder
**Goal**: Implement byte-level BPE encoder  
**Key Deliverable**: Working BPE encoder  
**Blocks**: LT-010 (BPE decoder)

### Days 34-35: LT-010 - Byte-Level BPE Decoder
**Goal**: Implement byte-level BPE decoder  
**Key Deliverable**: Working BPE decoder  
**Blocks**: Sprint 3 (UTF-8 safety)

---

## Dependencies

### Upstream (Blocks This Sprint)
- LT-001: GGUF Header Parser (provides header structure)
- LT-002: GGUF Metadata Extraction (provides metadata access)

### Downstream (This Sprint Blocks)
- Sprint 3: UTF-8 Safety + Llama Kernels (needs tokenizer)
- LT-011: UTF-8 Safe Streaming Decode (needs decoder)

---

## Success Criteria

Sprint is complete when:
- [ ] All 4 stories marked complete
- [ ] Vocab parsing working for Qwen GGUF
- [ ] Merges parsing working for Qwen GGUF
- [ ] BPE encoder working
- [ ] BPE decoder working
- [ ] All unit tests passing
- [ ] Ready for Sprint 3 (UTF-8 safety)

---

## Next Sprint

**Sprint 3**: UTF-8 Safety + Llama Kernels  
**Starts**: Day 36  
**Focus**: UTF-8 streaming, RoPE, RMSNorm, residual

---

**Status**: 📋 Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-05

---
Coordinated by Project Management Team 📋
