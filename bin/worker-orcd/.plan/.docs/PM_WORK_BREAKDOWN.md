# PM Work Breakdown - Artifact Generation

**Date**: 2025-10-04  
**Purpose**: Break down artifact generation into executable units of work  
**Total Units**: 42 work units  
**Estimated Time**: 7 days (6 units per day)  
**Status**: 📋 **READY TO EXECUTE**

---

## Work Unit Structure

Each unit of work:
- Takes ~1-2 hours to complete
- Produces 3-10 related documents
- Has clear inputs and outputs
- Can be verified independently

---

## Phase 1: Foundation Team (Days 1-2)

### Day 1: Foundation Stories (Part 1)

#### Unit 1.1: HTTP Foundation Stories (FT-001 to FT-005)
**Time**: 1.5 hours  
**Output**: 5 story cards

**Stories**:
- FT-001: HTTP Server Setup
- FT-002: POST /execute Endpoint
- FT-003: SSE Streaming
- FT-004: Correlation ID Middleware
- FT-005: Request Validation

**Verification**: All 5 cards in `foundation-team/stories/FT-001-to-FT-010/`

---

#### Unit 1.2: FFI Layer Stories (FT-006 to FT-010)
**Time**: 1.5 hours  
**Output**: 5 story cards

**Stories**:
- FT-006: FFI Interface Definition ← CRITICAL (FFI lock)
- FT-007: Rust FFI Bindings ← CRITICAL (FFI lock)
- FT-008: Error Code System C++
- FT-009: Error Code to Result Rust
- FT-010: CUDA Context Init

**Verification**: All 5 cards in `foundation-team/stories/FT-001-to-FT-010/`

---

#### Unit 1.3: CUDA Foundation Stories (FT-011 to FT-015)
**Time**: 1.5 hours  
**Output**: 5 story cards

**Stories**:
- FT-011: VRAM-Only Enforcement
- FT-012: FFI Integration Tests
- FT-013: Device Memory RAII
- FT-014: VRAM Residency Verification
- FT-015: Embedding Lookup Kernel

**Verification**: 5 cards in `foundation-team/stories/FT-011-to-FT-020/`

---

#### Unit 1.4: Shared Kernels Stories (FT-016 to FT-020)
**Time**: 1.5 hours  
**Output**: 5 story cards

**Stories**:
- FT-016: cuBLAS GEMM Wrapper
- FT-017: Temperature Scaling Kernel
- FT-018: Greedy Sampling
- FT-019: Stochastic Sampling
- FT-020: Seeded RNG

**Verification**: 5 cards in `foundation-team/stories/FT-011-to-FT-020/`

---

#### Unit 1.5: KV Cache + Integration Stories (FT-021 to FT-025)
**Time**: 1.5 hours  
**Output**: 5 story cards

**Stories**:
- FT-021: KV Cache Allocation
- FT-022: KV Cache Management
- FT-023: Integration Test Framework ← CRITICAL (blocks Gate 1)
- FT-024: HTTP-FFI-CUDA Integration Test
- FT-025: Gate 1 Validation Tests

**Verification**: 5 cards in `foundation-team/stories/FT-021-to-FT-030/`

---

#### Unit 1.6: Gate 1 + Support Stories (FT-026 to FT-030)
**Time**: 1.5 hours  
**Output**: 5 story cards

**Stories**:
- FT-026: Error Handling Integration
- FT-027: Gate 1 Checkpoint
- FT-028: Support Llama Integration
- FT-029: Support GPT Integration
- FT-030: Bug Fixes Integration

**Verification**: 5 cards in `foundation-team/stories/FT-021-to-FT-030/`

---

### Day 2: Foundation Stories (Part 2) + Sprints + Gates

#### Unit 2.1: Performance + Gate 2 Stories (FT-031 to FT-035)
**Time**: 1.5 hours  
**Output**: 5 story cards

**Stories**:
- FT-031: Performance Baseline Prep
- FT-032: Gate 2 Checkpoint
- FT-033: InferenceAdapter Interface ← CRITICAL (adapter pattern)
- FT-034: Adapter Factory Pattern
- FT-035: Architecture Detection Integration

**Verification**: 5 cards in `foundation-team/stories/FT-031-to-FT-040/`

---

#### Unit 2.2: Adapter + Gate 3 Stories (FT-036 to FT-040)
**Time**: 1.5 hours  
**Output**: 5 story cards

**Stories**:
- FT-036: Update Integration Tests Adapters
- FT-037: API Documentation
- FT-038: Gate 3 Checkpoint
- FT-039: CI/CD Pipeline
- FT-040: Performance Baseline Measurements

**Verification**: 3 cards in `FT-031-to-FT-040/`, 2 cards in `FT-041-to-FT-050/`

---

#### Unit 2.3: Final Integration Stories (FT-041 to FT-047)
**Time**: 1.5 hours  
**Output**: 7 story cards

**Stories**:
- FT-041: All Models Integration Test
- FT-042: OOM Recovery Test
- FT-043: UTF-8 Streaming Edge Cases
- FT-044: Cancellation Integration Test
- FT-045: Documentation Complete
- FT-046: Final Validation
- FT-047: Gate 4 Checkpoint

**Verification**: 7 cards in `foundation-team/stories/FT-041-to-FT-050/`

---

#### Unit 2.4: New Stories + Sprint READMEs (FT-049, FT-050 + Sprints 1-4)
**Time**: 2 hours  
**Output**: 2 story cards + 4 sprint READMEs

**Stories**:
- FT-049: Model Load Progress Events (NEW)
- FT-050: Narration-Core Logging (NEW)

**Sprint READMEs**:
- Sprint 1: HTTP Foundation (Days 1-9, 5 stories)
- Sprint 2: FFI Layer (Days 10-22, 7 stories) ← FFI LOCK
- Sprint 3: Shared Kernels (Days 23-38, 10 stories)
- Sprint 4: Integration + Gate 1 (Days 39-52, 7 stories)

**Verification**: 2 cards in `FT-041-to-FT-050/`, 4 READMEs in `sprints/`

---

#### Unit 2.5: Sprint READMEs + Gate Checklists (Sprints 5-7 + Gates)
**Time**: 2 hours  
**Output**: 3 sprint READMEs + 3 gate checklists

**Sprint READMEs**:
- Sprint 5: Support + Prep (Days 53-60, 5 stories)
- Sprint 6: Adapter + Gate 3 (Days 61-71, 6 stories)
- Sprint 7: Final Integration (Days 72-89, 9 stories)

**Gate Checklists**:
- Gate 1: Foundation Complete (Day 52)
- Gate 3: Adapter Complete (Day 71)
- Gate 4: M0 Complete (Day 89)

**Verification**: 3 READMEs in `sprints/`, 3 checklists in `integration-gates/`

---

#### Unit 2.6: Foundation Execution Templates
**Time**: 1 hour  
**Output**: 4 execution templates

**Templates**:
- `execution/day-tracker.md` (with instructions)
- `execution/dependencies.md` (upstream/downstream tracking)
- `execution/milestones.md` (FFI lock, gates, completion)
- `execution/FFI_INTERFACE_LOCKED.md.template`

**Verification**: 4 files in `foundation-team/execution/`

---

## Phase 2: Llama Team (Day 3)

### Day 3: Llama Stories + Sprints + Gates

#### Unit 3.1: Prep + GGUF Foundation Stories (LT-000 to LT-006)
**Time**: 1.5 hours  
**Output**: 7 story cards

**Stories**:
- LT-000: GGUF Format Research (NEW - prep work)
- LT-001: GGUF Header Parser
- LT-002: GGUF Metadata Extraction
- LT-003: Memory-Mapped I/O
- LT-004: Chunked H2D Transfer
- LT-005: Pre-Load Validation
- LT-006: Architecture Detection (Llama)

**Verification**: 1 card in `LT-000-prep/`, 6 cards in `LT-001-to-LT-010/`

---

#### Unit 3.2: GGUF-BPE Tokenizer Stories (LT-007 to LT-011)
**Time**: 1.5 hours  
**Output**: 5 story cards

**Stories**:
- LT-007: GGUF Vocab Parsing
- LT-008: GGUF Merges Parsing
- LT-009: Byte-Level BPE Encoder
- LT-010: Byte-Level BPE Decoder
- LT-011: UTF-8 Safe Streaming Decode

**Verification**: 4 cards in `LT-001-to-LT-010/`, 1 card in `LT-011-to-LT-020/`

---

#### Unit 3.3: Llama Kernels Stories (LT-012 to LT-017)
**Time**: 1.5 hours  
**Output**: 6 story cards

**Stories**:
- LT-012: RoPE Kernel
- LT-013: RMSNorm Kernel
- LT-014: Residual Connection Kernel
- LT-015: GQA Attention (Prefill)
- LT-016: GQA Attention (Decode)
- LT-017: SwiGLU FFN Kernel

**Verification**: 6 cards in `llama-team/stories/LT-011-to-LT-020/`

---

#### Unit 3.4: Gate 1 + Qwen Stories (LT-018 to LT-024)
**Time**: 1.5 hours  
**Output**: 7 story cards

**Stories**:
- LT-018: Tokenizer Conformance Tests (Qwen)
- LT-019: Kernel Unit Tests
- LT-020: Gate 1 Participation
- LT-022: Qwen Weight Mapping
- LT-023: Qwen Weight Loading
- LT-024: Qwen Forward Pass

**Verification**: 3 cards in `LT-011-to-LT-020/`, 4 cards in `LT-021-to-LT-030/`

---

#### Unit 3.5: Qwen Validation + Phi-3 Stories (LT-025 to LT-031)
**Time**: 1.5 hours  
**Output**: 7 story cards

**Stories**:
- LT-025: Qwen Haiku Generation Test
- LT-026: Qwen Reproducibility Validation
- LT-027: Gate 2 Checkpoint
- LT-029: Phi-3 Metadata Analysis
- LT-030: Phi-3 Weight Loading
- LT-031: Phi-3 Forward Pass

**Verification**: 3 cards in `LT-021-to-LT-030/`, 4 cards in `LT-031-to-LT-038/`

---

#### Unit 3.6: Phi-3 + Final Stories + Sprints + Gates
**Time**: 2 hours  
**Output**: 5 story cards + 8 sprint READMEs + 4 gate checklists

**Stories**:
- LT-032: Tokenizer Conformance Tests (Phi-3)
- LT-033: LlamaInferenceAdapter
- LT-034: Gate 3 Participation
- LT-035: Llama Integration Test Suite
- LT-036: Reproducibility Tests (10 runs)
- LT-037: VRAM Pressure Tests (Phi-3)
- LT-038: Documentation (GGUF, BPE, Llama)

**Sprint READMEs**:
- Sprint 0: Prep Work (Days 1-3, 1 story)
- Sprint 1: GGUF Foundation (Days 15-25, 6 stories)
- Sprint 2: GGUF-BPE Tokenizer (Days 26-34, 4 stories)
- Sprint 3: UTF-8 + Llama Kernels (Days 35-40, 4 stories)
- Sprint 4: GQA + Gate 1 (Days 41-53, 7 stories)
- Sprint 5: Qwen Integration (Days 54-66, 6 stories) ← CRITICAL
- Sprint 6: Phi-3 + Adapter (Days 67-77, 6 stories)
- Sprint 7: Final Integration (Days 78-90, 4 stories)

**Gate Checklists**:
- Gate 1: Llama Kernels (Day 53)
- Gate 2: Qwen Working (Day 66) ← CRITICAL
- Gate 3: Adapter Complete (Day 77)
- Gate 4: Participation (Day 90)

**Verification**: 5 cards in `LT-031-to-LT-038/`, 8 READMEs in `sprints/`, 4 checklists in `integration-gates/`

---

#### Unit 3.7: Llama Execution Templates
**Time**: 1 hour  
**Output**: 3 execution templates

**Templates**:
- `execution/day-tracker.md`
- `execution/dependencies.md`
- `execution/milestones.md`

**Verification**: 3 files in `llama-team/execution/`

---

## Phase 3: GPT Team (Days 4-5)

### Day 4: GPT Stories (Part 1)

#### Unit 4.1: Prep + HF Tokenizer Stories (GT-000 to GT-005)
**Time**: 1.5 hours  
**Output**: 6 story cards

**Stories**:
- GT-000: MXFP4 Spec Study (NEW - prep work)
- GT-001: HF Tokenizers Crate Integration
- GT-002: tokenizer.json Loading
- GT-003: Tokenizer Metadata Exposure
- GT-004: HF Tokenizer Conformance Tests
- GT-005: GPT GGUF Metadata Parsing

**Verification**: 1 card in `GT-000-prep/`, 5 cards in `GT-001-to-GT-010/`

---

#### Unit 4.2: GPT Metadata + Kernels Stories (GT-006 to GT-011)
**Time**: 1.5 hours  
**Output**: 6 story cards

**Stories**:
- GT-006: GGUF v3 Tensor Support (MXFP4 Parsing)
- GT-007: Architecture Detection (GPT)
- GT-008: Absolute Positional Embedding
- GT-009: LayerNorm (Mean Reduction)
- GT-010: LayerNorm (Variance + Normalize)
- GT-011: LayerNorm Unit Tests

**Verification**: 5 cards in `GT-001-to-GT-010/`, 1 card in `GT-011-to-GT-020/`

---

#### Unit 4.3: GPT Kernels Stories (GT-012 to GT-017)
**Time**: 1.5 hours  
**Output**: 6 story cards

**Stories**:
- GT-012: GELU Activation Kernel
- GT-013: GELU Unit Tests
- GT-014: GPT FFN Kernel
- GT-015: Residual Connection Kernel
- GT-016: Kernel Integration Tests
- GT-017: MHA Attention (Prefill)

**Verification**: 6 cards in `gpt-team/stories/GT-011-to-GT-020/`

---

#### Unit 4.4: MHA + Gate 1 Stories (GT-018 to GT-023)
**Time**: 1.5 hours  
**Output**: 6 story cards

**Stories**:
- GT-018: MHA Attention (Decode)
- GT-019: MHA vs GQA Differences Validation
- GT-020: MHA Unit Tests
- GT-021: GPT Kernel Suite Integration
- GT-022: Gate 1 Participation
- GT-023: FFI Integration Tests (GPT)

**Verification**: 4 cards in `GT-011-to-GT-020/`, 2 cards in `GT-021-to-GT-030/`

---

#### Unit 4.5: GPT Basic Pipeline Stories (GT-024 to GT-029)
**Time**: 1.5 hours  
**Output**: 6 story cards

**Stories**:
- GT-024: GPT Weight Mapping (Q4_K_M)
- GT-025: GPT Weight Loading
- GT-026: GPT Forward Pass (Q4_K_M)
- GT-027: GPT Basic Generation Test
- GT-028: Gate 2 Checkpoint
- GT-029: MXFP4 Dequantization Kernel ← NOVEL FORMAT

**Verification**: 6 cards in `gpt-team/stories/GT-021-to-GT-030/`

---

#### Unit 4.6: MXFP4 Dequant + Integration Stories (GT-030 to GT-035)
**Time**: 1.5 hours  
**Output**: 6 story cards

**Stories**:
- GT-030: MXFP4 Unit Tests
- GT-031: UTF-8 Streaming Safety Tests
- GT-033: MXFP4 GEMM Integration
- GT-034: MXFP4 Embedding Lookup
- GT-035: MXFP4 Attention (Q/K/V)

**Verification**: 2 cards in `GT-021-to-GT-030/`, 4 cards in `GT-031-to-GT-040/`

---

### Day 5: GPT Stories (Part 2) + Sprints + Gates

#### Unit 5.1: MXFP4 Integration Stories (GT-036 to GT-040)
**Time**: 1.5 hours  
**Output**: 5 story cards

**Stories**:
- GT-036: MXFP4 FFN (Up/Down Projections)
- GT-037: MXFP4 LM Head
- GT-038: MXFP4 Numerical Validation (±1%) ← CRITICAL
- GT-039: GPTInferenceAdapter
- GT-040: GPT-OSS-20B MXFP4 E2E

**Verification**: 5 cards in `gpt-team/stories/GT-031-to-GT-040/`

---

#### Unit 5.2: Final Integration Stories (GT-041 to GT-048)
**Time**: 1.5 hours  
**Output**: 8 story cards

**Stories**:
- GT-041: Gate 3 Participation
- GT-042: GPT Integration Test Suite
- GT-043: MXFP4 Regression Tests
- GT-044: 24 GB VRAM Boundary Tests
- GT-045: OOM Recovery Tests (GPT)
- GT-046: UTF-8 Multibyte Edge Cases
- GT-047: Documentation (GPT, MXFP4, HF)
- GT-048: Performance Baseline (GPT)

**Verification**: 8 cards in `gpt-team/stories/GT-041-to-GT-048/`

---

#### Unit 5.3: GPT Sprint READMEs (Sprints 0-4)
**Time**: 1.5 hours  
**Output**: 5 sprint READMEs

**Sprint READMEs**:
- Sprint 0: Prep Work (Days 1-3, 1 story)
- Sprint 1: HF Tokenizer (Days 15-26, 7 stories)
- Sprint 2: GPT Kernels (Days 27-41, 9 stories)
- Sprint 3: MHA + Gate 1 (Days 42-55, 7 stories)
- Sprint 4: GPT Basic (Days 56-66, 5 stories)

**Verification**: 5 READMEs in `gpt-team/sprints/`

---

#### Unit 5.4: GPT Sprint READMEs (Sprints 5-8)
**Time**: 1.5 hours  
**Output**: 4 sprint READMEs

**Sprint READMEs**:
- Sprint 5: MXFP4 Dequant (Days 67-74, 3 stories) ← NOVEL
- Sprint 6: MXFP4 Integration (Days 75-89, 6 stories) ← CRITICAL
- Sprint 7: Adapter + E2E (Days 90-96, 2 stories)
- Sprint 8: Final Integration (Days 97-110, 8 stories) ← M0 DELIVERY

**Verification**: 4 READMEs in `gpt-team/sprints/`

---

#### Unit 5.5: GPT Gate Checklists
**Time**: 1.5 hours  
**Output**: 5 gate checklists

**Gate Checklists**:
- Gate 1: GPT Kernels (Day 55)
- Gate 2: GPT Basic (Day 66)
- Gate 3: MXFP4 + Adapter (Day 96) ← CRITICAL
- Gate 4: M0 Delivery (Day 110) ← PROJECT COMPLETE
- MXFP4 Validation Framework (special)

**Verification**: 5 checklists in `gpt-team/integration-gates/`

---

#### Unit 5.6: GPT Execution Templates
**Time**: 1 hour  
**Output**: 4 execution templates

**Templates**:
- `execution/day-tracker.md`
- `execution/dependencies.md`
- `execution/milestones.md`
- `execution/mxfp4-validation-framework.md`

**Verification**: 4 files in `gpt-team/execution/`

---

## Phase 4: Coordination Documents (Day 6)

### Day 6: Coordination + Review

#### Unit 6.1: Coordination Documents
**Time**: 2 hours  
**Output**: 5 coordination documents

**Documents**:
- `coordination/master-timeline.md` (agent status tracking)
- `coordination/FFI_INTERFACE_LOCKED.md.template` (to be filled by Foundation-Alpha)
- `coordination/adapter-pattern-locked.md.template` (to be filled by Foundation-Alpha)
- `coordination/daily-standup-template.md` (optional)
- `coordination/gate-coordination-checklist.md`

**Verification**: 5 files in `coordination/`

---

#### Unit 6.2: Team Index Documents
**Time**: 1.5 hours  
**Output**: 3 team index documents

**Documents**:
- `foundation-team/README.md` (team overview, navigation)
- `llama-team/README.md` (team overview, navigation)
- `gpt-team/README.md` (team overview, navigation)

**Verification**: 3 README files in team folders

---

#### Unit 6.3: Cross-Reference Validation
**Time**: 2 hours  
**Output**: Validation report

**Tasks**:
- Verify all story dependencies are correct
- Verify all day ranges are accurate
- Verify all spec references exist
- Verify all gate checklists align with stories
- Create validation report

**Verification**: `PM_VALIDATION_REPORT.md` created

---

#### Unit 6.4: Final Documentation Polish
**Time**: 1.5 hours  
**Output**: Updated documents

**Tasks**:
- Fix any issues found in validation
- Ensure consistent formatting
- Add missing cross-references
- Update master artifact list

**Verification**: All 189 documents finalized

---

## Phase 5: Review & Handoff (Day 7)

### Day 7: Final Review

#### Unit 7.1: Foundation Team Review
**Time**: 1.5 hours  
**Output**: Review checklist

**Tasks**:
- Review all 49 story cards
- Review all 7 sprint READMEs
- Review all 3 gate checklists
- Review all 4 execution templates
- Mark Foundation Team as ✅ Ready

**Verification**: Foundation Team review checklist complete

---

#### Unit 7.2: Llama Team Review
**Time**: 1.5 hours  
**Output**: Review checklist

**Tasks**:
- Review all 39 story cards
- Review all 8 sprint READMEs
- Review all 4 gate checklists
- Review all 3 execution templates
- Mark Llama Team as ✅ Ready

**Verification**: Llama Team review checklist complete

---

#### Unit 7.3: GPT Team Review
**Time**: 1.5 hours  
**Output**: Review checklist

**Tasks**:
- Review all 49 story cards
- Review all 9 sprint READMEs
- Review all 5 gate checklists
- Review all 4 execution templates
- Mark GPT Team as ✅ Ready

**Verification**: GPT Team review checklist complete

---

#### Unit 7.4: Coordination Review + Handoff
**Time**: 1.5 hours  
**Output**: Handoff document

**Tasks**:
- Review all 5 coordination documents
- Create execution handoff document
- Create PM handoff checklist
- Mark entire M0 planning as ✅ Ready for Execution

**Verification**: `PM_HANDOFF_DOCUMENT.md` created

---

## Summary

### Total Work Units: 42

| Phase | Day | Units | Documents | Time |
|-------|-----|-------|-----------|------|
| Phase 1 | Day 1 | 6 | 30 stories | 9 hours |
| Phase 1 | Day 2 | 6 | 19 stories + 7 sprints + 3 gates + 4 exec | 10 hours |
| Phase 2 | Day 3 | 7 | 39 stories + 8 sprints + 4 gates + 3 exec | 11 hours |
| Phase 3 | Day 4 | 6 | 36 stories | 9 hours |
| Phase 3 | Day 5 | 6 | 13 stories + 9 sprints + 5 gates + 4 exec | 9 hours |
| Phase 4 | Day 6 | 4 | 5 coord + 3 index + validation | 7 hours |
| Phase 5 | Day 7 | 4 | Reviews + handoff | 6 hours |
| **TOTAL** | **7 days** | **42 units** | **189 documents** | **61 hours** |

---

## Execution Order

### Week 1: Artifact Generation

**Monday (Day 1)**: Units 1.1 → 1.6 (Foundation stories part 1)  
**Tuesday (Day 2)**: Units 2.1 → 2.6 (Foundation stories part 2 + sprints + gates)  
**Wednesday (Day 3)**: Units 3.1 → 3.7 (Llama complete)  
**Thursday (Day 4)**: Units 4.1 → 4.6 (GPT stories part 1)  
**Friday (Day 5)**: Units 5.1 → 5.6 (GPT stories part 2 + sprints + gates)  
**Saturday (Day 6)**: Units 6.1 → 6.4 (Coordination + validation)  
**Sunday (Day 7)**: Units 7.1 → 7.4 (Final review + handoff)

---

## Next Steps

1. ✅ Approve work breakdown
2. 🔄 Begin Unit 1.1 (FT-001 to FT-005)
3. 🔄 Execute units sequentially
4. ✅ Verify each unit's output
5. 🚀 Complete all 42 units in 7 days

---

**Status**: 📋 **WORK BREAKDOWN COMPLETE - READY TO EXECUTE**  
**Total Units**: 42  
**Total Documents**: 189  
**Timeline**: 7 days  
**Our Duty**: Execute every unit perfectly

---

*Created by Perfect Project Managers 📋*
