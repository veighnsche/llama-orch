# Sprint 1: GGUF Foundation

**Team**: Llama-Beta  
**Days**: 15-26 (12 agent-days)  
**Goal**: Parse GGUF format and establish loader infrastructure

---

## Sprint Overview

Sprint 1 is the foundational sprint for Llama-Beta that establishes GGUF file format parsing with comprehensive security validation. This sprint begins immediately after Foundation-Alpha locks the FFI interface (Day 15) and creates the loader infrastructure that all downstream work depends on.

This sprint includes critical security enhancements (+1 day) to prevent heap overflow vulnerabilities discovered during research phase.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| LT-001 | GGUF Header Parser | M | 3 | 15-17 |
| LT-002 | GGUF Metadata Extraction (Llama) | M | 2 | 18-19 |
| LT-003 | Memory-Mapped I/O Implementation | M | 2 | 20-21 |
| LT-004 | Chunked H2D Transfer | M | 2 | 22-23 |
| LT-005 | Pre-Load Validation | M | 2 | 24-25 |
| LT-006 | Architecture Detection (Llama) | S | 1 | 26 |

**Total**: 6 stories, 12 agent-days (Days 15-26)

---

## Story Execution Order

### Days 15-17: LT-001 - GGUF Header Parser
**Goal**: Parse GGUF header with security validation  
**Key Deliverable**: Header parser with heap overflow prevention  
**Blocks**: LT-002 (metadata extraction)  
**Security**: +1 day for bounds validation (M0-W-1211a)

### Days 18-19: LT-002 - GGUF Metadata Extraction
**Goal**: Extract Llama-specific metadata from GGUF  
**Key Deliverable**: Metadata extraction for Llama architecture  
**Blocks**: LT-003 (memory-mapped I/O)

### Days 20-21: LT-003 - Memory-Mapped I/O
**Goal**: Implement memory-mapped file I/O for efficient access  
**Key Deliverable**: Memory-mapped GGUF file access  
**Blocks**: LT-004 (chunked transfer)

### Days 22-23: LT-004 - Chunked H2D Transfer
**Goal**: Implement chunked host-to-device transfer  
**Key Deliverable**: Efficient chunked VRAM transfer  
**Blocks**: LT-005 (pre-load validation)

### Days 24-25: LT-005 - Pre-Load Validation
**Goal**: Validate model before loading to VRAM  
**Key Deliverable**: Pre-load validation framework  
**Blocks**: LT-006 (architecture detection)

### Day 26: LT-006 - Architecture Detection
**Goal**: Detect Llama architecture from metadata  
**Key Deliverable**: Architecture detection for Llama models  
**Blocks**: Sprint 2 (tokenizer)

---

## Dependencies

### Upstream (Blocks This Sprint)
- FT-006: FFI Interface Definition (FFI lock Day 15)
- FT-007: Rust FFI Bindings (FFI lock Day 15)

### Downstream (This Sprint Blocks)
- Sprint 2: GGUF-BPE Tokenizer (needs GGUF loader)
- LT-007: GGUF Vocab Parsing (needs metadata extraction)

---

## Success Criteria

Sprint is complete when:
- [ ] All 6 stories marked complete
- [ ] GGUF header parser working with security validation
- [ ] Metadata extraction working for Llama models
- [ ] Memory-mapped I/O operational
- [ ] Chunked H2D transfer working
- [ ] Pre-load validation framework complete
- [ ] Architecture detection working
- [ ] All unit tests passing
- [ ] Security fuzzing tests passing (100+ malformed files)
- [ ] Ready for Sprint 2 (tokenizer)

---

## Next Sprint

**Sprint 2**: GGUF-BPE Tokenizer  
**Starts**: Day 27  
**Focus**: Vocab/merges parsing, BPE encoder/decoder

---

**Status**: 📋 Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-05

---
Coordinated by Project Management Team 📋
