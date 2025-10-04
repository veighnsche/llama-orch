# Sprint 6: Phi-3 + Adapter

**Team**: Llama-Beta  
**Days**: 68-78 (11 agent-days)  
**Goal**: Second Llama model + LlamaInferenceAdapter implementation

---

## Sprint Overview

Sprint 6 adds support for a second Llama model (Phi-3) and implements the LlamaInferenceAdapter pattern. This sprint proves the Llama implementation is generalizable across different Llama-family models and establishes the adapter pattern for clean integration.

Success in this sprint enables Gate 3 checkpoint.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| LT-029 | Phi-3 Metadata Analysis | S | 1 | 68 |
| LT-030 | Phi-3 Weight Loading | M | 2 | 69-70 |
| LT-031 | Phi-3 Forward Pass (Adapt from Qwen) | M | 2 | 71-72 |
| LT-032 | Tokenizer Conformance Tests (Phi-3) | M | 2 | 73-74 |
| LT-033 | LlamaInferenceAdapter Implementation | M | 3 | 75-77 |
| LT-034 | Gate 3 Participation | S | 1 | 78 |

**Total**: 6 stories, 11 agent-days (Days 68-78)

---

## Story Execution Order

### Day 68: LT-029 - Phi-3 Metadata Analysis
**Goal**: Analyze Phi-3 GGUF metadata  
**Key Deliverable**: Phi-3 metadata analysis  
**Blocks**: LT-030 (weight loading)

### Days 69-70: LT-030 - Phi-3 Weight Loading
**Goal**: Load Phi-3 weights to VRAM  
**Key Deliverable**: Phi-3 weight loading  
**Blocks**: LT-031 (forward pass)

### Days 71-72: LT-031 - Phi-3 Forward Pass
**Goal**: Adapt Qwen forward pass for Phi-3  
**Key Deliverable**: Working Phi-3 inference  
**Blocks**: LT-032 (conformance tests)

### Days 73-74: LT-032 - Tokenizer Conformance Tests (Phi-3)
**Goal**: Create conformance tests for Phi-3 tokenizer  
**Key Deliverable**: 20-30 test pairs for Phi-3  
**Blocks**: LT-033 (adapter)

### Days 75-77: LT-033 - LlamaInferenceAdapter Implementation
**Goal**: Implement LlamaInferenceAdapter pattern  
**Key Deliverable**: Adapter pattern for Llama models  
**Blocks**: LT-034 (Gate 3)

### Day 78: LT-034 - Gate 3 Participation
**Goal**: Participate in Gate 3 validation  
**Key Deliverable**: Gate 3 checkpoint passed  
**Blocks**: Sprint 7 (final integration)

---

## Dependencies

### Upstream (Blocks This Sprint)
- LT-027: Gate 2 Checkpoint (validates Qwen)
- FT-071: Adapter Pattern Definition (Day 71, provides adapter spec)

### Downstream (This Sprint Blocks)
- Sprint 7: Final Integration (needs adapter)
- LT-035: Integration Test Suite (needs both models)

---

## Critical Milestone: Gate 3 (Day 78)

**What**: Adapter pattern complete, second model working  
**Why Critical**: Proves generalization and clean integration  
**Deliverable**: Gate 3 validation report

**Checklist**:
- [ ] Phi-3 model working
- [ ] LlamaInferenceAdapter implemented
- [ ] Both models use adapter pattern
- [ ] Tokenizer conformance tests passing
- [ ] All integration tests passing
- [ ] Gate 3 report published

---

## Success Criteria

Sprint is complete when:
- [ ] All 6 stories marked complete
- [ ] Phi-3 metadata analyzed
- [ ] Phi-3 weights loaded to VRAM
- [ ] Phi-3 forward pass working
- [ ] Tokenizer conformance tests passing (Phi-3)
- [ ] LlamaInferenceAdapter implemented
- [ ] Gate 3 checkpoint passed
- [ ] All integration tests passing
- [ ] Ready for Sprint 7 (final integration)

---

## Next Sprint

**Sprint 7**: Final Integration  
**Starts**: Day 79  
**Focus**: Testing, reproducibility, documentation

---

**Status**: 📋 Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-05

---
Coordinated by Project Management Team 📋
