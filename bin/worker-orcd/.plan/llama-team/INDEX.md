# Llama Team - Planning Index

**Team**: 🦙 Llama Team  
**Mission**: Llama pipeline (GGUF, GGUF-BPE, Llama kernels, Qwen + Phi-3)  
**Timeline**: 6 weeks (with 3 people) OR 9 weeks (with 2 people)  
**Status**: 🔴 **PLANNING COMPLETE - TEAM SIZE DECISION REQUIRED**

---

## 📋 Quick Navigation

### Essential Documents

1. **[Executive Summary](EXECUTIVE_SUMMARY.md)** - 🔴 **START HERE** - TL;DR for decision makers
2. **[Team Charter](docs/team-charter.md)** - Team composition, responsibilities, success criteria
3. **[Complete Story List](docs/complete-story-list.md)** - All 38 stories across 6 weeks
4. **[Planning Gap Analysis](docs/PLANNING_GAP_ANALYSIS.md)** - 🔴 **CRITICAL: Team size analysis**

### Sprint Plans

- Week 2: GGUF Foundation (6 stories, 11 days) - TBD
- Week 3: Tokenization + Kernels (8 stories, 15 days) - TBD ⚠️ **BOTTLENECK**
- Week 4: Kernels Complete (7 stories, 13 days) - TBD
- Week 5: Qwen Integration (7 stories, 13 days) - TBD
- Week 6: Phi-3 + Adapter (6 stories, 11 days) - TBD
- Week 7: Final Integration (4 stories, 9 days) - TBD

### Integration Gates

- Gate 1: Llama Kernels Ready (Week 4) - TBD
- Gate 2: Qwen Working (Week 5) - TBD ← **CRITICAL MILESTONE**
- Gate 3: Phi-3 + LlamaAdapter (Week 6) - TBD
- Gate 4: M0 Complete (Week 7) - TBD

---

## 🔴 Critical Findings

### **TEAM SIZE DETERMINES FEASIBILITY**

**With 2 People**: 
- Week 3: **150% utilization** 🔴 OVERCOMMITTED
- Week 4: **130% utilization** 🔴 OVERCOMMITTED
- Week 5: **130% utilization** 🔴 OVERCOMMITTED
- **Overall**: **120% utilization** 🔴 NOT FEASIBLE
- **Need**: 9 weeks to be realistic

**With 3 People**:
- Week 3: **100% utilization** ✅ FEASIBLE
- Week 4: **87% utilization** ✅ FEASIBLE
- Week 5: **87% utilization** ✅ FEASIBLE
- **Overall**: **80% utilization** ✅ HEALTHY
- **Timeline**: 6 weeks as planned

**Decision Required**: 3 people (6 weeks) OR 2 people (9 weeks)?

**See**: [Planning Gap Analysis](docs/PLANNING_GAP_ANALYSIS.md) for full details

---

## 📊 Summary Statistics

### Work Breakdown

| Metric | Value |
|--------|-------|
| **Total Stories** | 38 |
| **Total Estimated Days** | 72 days |
| **Timeline (3 people)** | 6 weeks (Weeks 2-7) |
| **Timeline (2 people)** | 9 weeks (Weeks 2-10) |
| **Available Capacity (3 people)** | 90 days |
| **Available Capacity (2 people, 6 weeks)** | 60 days |
| **Utilization (3 people)** | 80% (healthy) |
| **Utilization (2 people, 6 weeks)** | 120% (overcommitted) |

### Story Size Distribution

| Size | Count | Days | % of Total |
|------|-------|------|------------|
| Small (S) | 6 | 6 | 8% |
| Medium (M) | 27 | 54 | 75% |
| Large (L) | 5 | 20 | 28% |

### Weekly Utilization (3-Person Plan)

| Week | Stories | Days | Capacity | Util % | Status |
|------|---------|------|----------|--------|--------|
| 2 | 6 + 2 | 15 | 15 | 100% | ✅ Good |
| 3 | 6 | 11 | 15 | 73% | ✅ Good |
| 4 | 7 | 13 | 15 | 87% | ✅ Good |
| 5 | 7 | 13 | 15 | 87% | ✅ Good |
| 6 | 6 | 11 | 15 | 73% | ✅ Good |
| 7 | 4 | 9 | 15 | 60% | ✅ Good |

### Weekly Utilization (2-Person Plan, 6 Weeks)

| Week | Stories | Days | Capacity | Util % | Status |
|------|---------|------|----------|--------|--------|
| 2 | 6 | 11 | 10 | 110% | 🟡 Tight |
| 3 | 8 | 15 | 10 | **150%** | 🔴 **OVERCOMMITTED** |
| 4 | 7 | 13 | 10 | **130%** | 🔴 **OVERCOMMITTED** |
| 5 | 7 | 13 | 10 | **130%** | 🔴 **OVERCOMMITTED** |
| 6 | 6 | 11 | 10 | 110% | 🟡 Tight |
| 7 | 4 | 9 | 10 | 90% | ✅ Good |

---

## 🎯 Success Criteria

### Gate 1 (Week 4): Llama Kernels Ready
- GGUF loader parsing headers/metadata
- GGUF-BPE tokenizer encode/decode
- RoPE, GQA, RMSNorm, SwiGLU kernels implemented

### Gate 2 (Week 5): Qwen Working ← **CRITICAL**
- Qwen2.5-0.5B loads to VRAM
- Haiku generation test passes
- Reproducibility validated (same seed → same output)
- VRAM-only verified

### Gate 3 (Week 6): Phi-3 + LlamaAdapter
- Phi-3-Mini working
- LlamaInferenceAdapter implemented
- Qwen + Phi-3 use adapter

### Gate 4 (Week 7): Final Validation
- All integration tests passing
- Reproducibility tests (10 runs each)
- Documentation complete

---

## 🚀 Getting Started

### Week 0 (Before Week 2)

1. **Read Planning Documents**:
   - [ ] Executive Summary ⚠️ **CRITICAL**
   - [ ] Team Charter
   - [ ] Complete Story List
   - [ ] Planning Gap Analysis ⚠️ **CRITICAL**

2. **Decision Required**:
   - [ ] 3 people (6 weeks) or 2 people (9 weeks)?
   - [ ] If 3 people: Allocate 3rd person
   - [ ] If 2 people: Extend timeline, update M0 date

3. **Preparation**:
   - [ ] Research GGUF format (llama.cpp reference)
   - [ ] Research BPE algorithm
   - [ ] Review Foundation Team's FFI interface

### Week 2 (Llama Team Starts)

- **Monday**: Sprint planning (10:00-12:00)
- **Tue-Thu**: Daily standups (9:15 AM, 15 min)
- **Friday**: Demo + Retro (14:00-16:00, joint with all teams)

---

## 📞 Communication

### Daily Standup
- **Time**: 9:15 AM (15 min, after Foundation Team)
- **Format**: What shipped? What shipping today? Blockers?

### Sprint Planning
- **Time**: Monday 10:00 AM (2 hours)
- **Attendees**: All team members

### Friday Demo
- **Time**: Friday 14:00 PM (2 hours)
- **Attendees**: All teams (Foundation, Llama, GPT)
- **Format**: Demo working features, integration tests

### Slack Channel
- **Channel**: #llama-team
- **Use**: Async updates, questions, blockers

---

## 📚 Key Deliverables

### GGUF Loader (Weeks 2-3)
- Header parsing (magic, version, metadata)
- Metadata extraction (architecture, vocab, hyperparams)
- Memory-mapped I/O
- Chunked H2D transfer (1MB chunks)
- Architecture detection

### GGUF-BPE Tokenizer (Weeks 3-4)
- Pure Rust implementation
- Vocab and merges parsing from GGUF
- Byte-level BPE encoder/decoder
- UTF-8 safe streaming
- Conformance test vectors (20-30 pairs)

### Llama Kernels (Weeks 3-4)
- RoPE (Rotary Position Embedding)
- GQA (Grouped Query Attention)
- RMSNorm
- SwiGLU FFN
- Residual connections

### Qwen Integration (Week 5)
- Weight loading to VRAM
- End-to-end forward pass
- Haiku generation test
- Reproducibility validation

### Phi-3 Integration (Week 6)
- Weight loading to VRAM
- End-to-end forward pass
- VRAM pressure tests

### LlamaInferenceAdapter (Week 6-7)
- Formal adapter class
- Refactor Qwen + Phi-3 to use adapter
- Integration with Foundation's pattern

---

## ⚠️ Risks

### High Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Week 3 overcommitment (2 people) | Gate 1 delay | **Add 3rd person OR extend to 9 weeks** |
| BPE algorithm bugs | Tokenization broken | Conformance test vectors, reference impl |
| GQA attention complexity | Gate 1 delay | Reference llama.cpp, unit tests |
| Qwen forward pass bugs | Gate 2 failure | Incremental integration, unit tests |

### Medium Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| GGUF format edge cases | Parsing failures | Comprehensive tests |
| Phi-3 architecture differences | Week 6 delays | Research early (Week 3-4) |
| FFI interface changes | Blocked | Participate in Week 2 FFI lock |

---

## 📈 Progress Tracking

### Story Workflow

```
Backlog → In Progress → Review → Done
```

**Backlog**: Not yet started  
**In Progress**: Developer actively working  
**Review**: Code review or integration testing  
**Done**: Merged + tests passing

### Velocity Tracking

| Sprint | Committed | Completed | Velocity |
|--------|-----------|-----------|----------|
| Week 2 | TBD | TBD | TBD |
| Week 3 | TBD | TBD | TBD |
| Week 4 | TBD | TBD | TBD |
| Week 5 | TBD | TBD | TBD |
| Week 6 | TBD | TBD | TBD |
| Week 7 | TBD | TBD | TBD |

**Target Velocity**: ~12 days/week (for 3 people) or ~10 days/week (for 2 people)

---

## 🔗 Related Documents

### Spec References

- [M0 Worker Spec](../../../.specs/01_M0_worker_orcd.md) - Complete M0 requirements
- [System Spec](../../../.specs/00_llama-orch.md) - System-level requirements
- [Architectural Gap Analysis](../../../.specs/.docs/M0_ARCHITECTURAL_GAP_ANALYSIS.md) - Adapter pattern rationale

### Cross-Team Coordination

- [Foundation Team Plan](../../foundation-team/INDEX.md) - FFI interface, shared kernels
- [GPT Team Plan](../../gpt-team/INDEX.md) - TBD
- [Integration Gates (All Teams)](../../README.md) - Master gate tracking

---

## ✅ Next Actions

### Immediate (Week 0)

1. **🔴 CRITICAL**: Decide on 3 people vs 2 people
2. Review Planning Gap Analysis with stakeholders
3. If 3 people: Allocate 3rd person (C++/CUDA or Rust)
4. If 2 people: Extend timeline to 9 weeks, update M0 date
5. Create remaining story cards (LT-003 through LT-038)

### Week 1 (Foundation Team Only)

1. Llama Team prepares for Week 2 start
2. Review Foundation Team's FFI interface design
3. Research GGUF format (llama.cpp)
4. Research BPE algorithm (prepare for Week 3)

### Week 2 Kickoff (Llama Team Starts)

1. Sprint planning Monday morning
2. Assign stories LT-001 through LT-006
3. Begin GGUF loader development
4. Participate in FFI interface lock (critical)

---

**Status**: 📋 **PLANNING COMPLETE - AWAITING DECISION**  
**Next Action**: **Team size decision (3 vs 2 people)**  
**Blocker**: Cannot start Week 2 until team size decided  
**Owner**: [Project Manager]

---

**Last Updated**: 2025-10-03  
**Document Version**: 1.0  
**Maintained By**: Llama Team Lead
