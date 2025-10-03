# Foundation Team - Planning Index

**Team**: 🏗️ Foundation Team  
**Mission**: Core infrastructure (HTTP, FFI, CUDA context, shared kernels)  
**Timeline**: 7-8 weeks (decision pending)  
**Status**: 🔴 **PLANNING COMPLETE - DECISION REQUIRED**

---

## 📋 Quick Navigation

### Essential Documents

1. **[Team Charter](docs/team-charter.md)** - Team composition, responsibilities, success criteria
2. **[Complete Story List](docs/complete-story-list.md)** - All 47 stories across 7 weeks
3. **[Planning Gap Analysis](docs/PLANNING_GAP_ANALYSIS.md)** - 🔴 **CRITICAL: Week 7 overcommitted**

### Sprint Plans

- [Week 1: HTTP Foundation](sprints/week-1/sprint-plan.md) - 5 stories, 9 days
- [Week 2: FFI Layer](sprints/week-2/sprint-plan.md) - 7 stories, 13 days (TBD)
- [Week 3: Shared Kernels](sprints/week-3/sprint-plan.md) - 8 stories, 14 days (TBD)
- [Week 4: Integration & Gate 1](sprints/week-4/sprint-plan.md) - 7 stories, 14 days (TBD)
- [Week 5: Support Role](sprints/week-5/sprint-plan.md) - 5 stories, 8 days (TBD)
- [Week 6: Adapter Coordination](sprints/week-6/sprint-plan.md) - 6 stories, 11 days (TBD)
- [Week 7: Final Integration](sprints/week-7/sprint-plan.md) - 9 stories, 16 days ⚠️ **OVERCOMMITTED**

### Integration Gates

- [Gate 1: Foundation Complete (Week 4)](integration-gates/gate-1-week-4.md) - ✅ Critical milestone
- [Gate 2: Support Role (Week 5)](integration-gates/gate-2-week-5.md) - TBD
- [Gate 3: Adapter Pattern (Week 6)](integration-gates/gate-3-week-6.md) - TBD
- [Gate 4: M0 Complete (Week 7)](integration-gates/gate-4-week-7.md) - TBD

### Story Cards

All 47 story cards are in `stories/backlog/`:
- FT-001 through FT-005: Week 1 (HTTP Foundation)
- FT-006 through FT-012: Week 2 (FFI Layer)
- FT-013 through FT-020: Week 3 (Shared Kernels)
- FT-021 through FT-027: Week 4 (Integration & Gate 1)
- FT-028 through FT-032: Week 5 (Support Role)
- FT-033 through FT-038: Week 6 (Adapter Coordination)
- FT-039 through FT-047: Week 7 (Final Integration)

---

## 🔴 Critical Findings

### **PLANNING GAP IDENTIFIED**

**Problem**: Week 7 is overcommitted (16 days of work, 15 days available = 107% utilization)

**Impact**: 
- High risk of Gate 4 failure
- M0 delivery at risk
- Team burnout

**Options**:
1. **Extend to 8 weeks** ⭐ RECOMMENDED
2. Reduce scope (no CI/CD, no perf baseline)
3. Add 4th developer for Week 7
4. Work overtime (NOT RECOMMENDED)

**Decision Required**: Before Week 1 sprint planning

**See**: [Planning Gap Analysis](docs/PLANNING_GAP_ANALYSIS.md) for full details

---

## 📊 Summary Statistics

### Work Breakdown

| Metric | Value |
|--------|-------|
| **Total Stories** | 47 |
| **Total Estimated Days** | 85 days |
| **Team Size** | 3 people |
| **Timeline** | 7 weeks (risky) or 8 weeks (recommended) |
| **Available Capacity (7 weeks)** | 105 days |
| **Available Capacity (8 weeks)** | 120 days |
| **Utilization (7 weeks)** | 81% average (but Week 7 = 107%) |
| **Utilization (8 weeks)** | 71% average (healthy) |

### Story Size Distribution

| Size | Count | Days | % of Total |
|------|-------|------|------------|
| Small (S) | 12 | 12 | 14% |
| Medium (M) | 32 | 64 | 75% |
| Large (L) | 3 | 12 | 14% |

### Weekly Utilization (7-Week Plan)

| Week | Stories | Days | Capacity | Util % | Status |
|------|---------|------|----------|--------|--------|
| 1 | 5 | 9 | 15 | 60% | ✅ Healthy |
| 2 | 7 | 13 | 15 | 87% | ✅ Good |
| 3 | 8 | 14 | 15 | 93% | ✅ Good |
| 4 | 7 | 14 | 15 | 93% | ✅ Good |
| 5 | 5 | 8 | 15 | 53% | 🟡 Underutilized |
| 6 | 6 | 11 | 15 | 73% | ✅ Good |
| 7 | 9 | 16 | 15 | **107%** | 🔴 **OVERCOMMITTED** |

---

## 🎯 Success Criteria

### Gate 1 (Week 4): Foundation Complete
- HTTP server operational
- SSE streaming working
- FFI layer stable
- CUDA context working
- Shared kernels implemented
- Integration test framework ready

### Gate 2 (Week 5): Support Role
- Teams 2 & 3 unblocked
- No critical infrastructure bugs

### Gate 3 (Week 6): Adapter Coordination
- InferenceAdapter interface designed
- All teams aligned

### Gate 4 (Week 7/8): M0 Complete
- All integration tests passing
- CI/CD green
- Documentation complete
- Performance baseline documented

---

## 🚀 Getting Started

### Week 0 (Before Sprint 1)

1. **Read Planning Documents**:
   - [ ] Team Charter
   - [ ] Complete Story List
   - [ ] Planning Gap Analysis ⚠️ **CRITICAL**

2. **Decision Required**:
   - [ ] 7 weeks (risky) or 8 weeks (recommended)?
   - [ ] If 7 weeks: Which scope to cut?

3. **Story Writing Workshop**:
   - [ ] Review all 47 story cards
   - [ ] Validate estimates (planning poker)
   - [ ] Adjust based on team input

4. **Sprint 1 Planning**:
   - [ ] Commit to Week 1 stories (FT-001 through FT-005)
   - [ ] Assign owners
   - [ ] Set up development environment

### Week 1 (Sprint 1)

- **Monday**: Sprint planning (10:00-12:00)
- **Tue-Thu**: Daily standups (9:00 AM, 15 min)
- **Friday**: Demo + Retro (14:00-16:00)

---

## 📞 Communication

### Daily Standup
- **Time**: 9:00 AM (15 min)
- **Format**: What shipped? What shipping today? Blockers?

### Sprint Planning
- **Time**: Monday 10:00 AM (2 hours)
- **Attendees**: All team members

### Friday Demo
- **Time**: Friday 14:00 PM (2 hours)
- **Attendees**: All teams (Foundation, Llama, GPT)
- **Format**: Demo working features, integration tests

### Slack Channel
- **Channel**: #foundation-team
- **Use**: Async updates, questions, blockers

---

## 📚 Key Interfaces

### FFI Interface (Locked by Week 2)

**Core Functions**:
- `cuda_init()` - Initialize CUDA context
- `cuda_load_model()` - Load model to VRAM
- `cuda_inference_start()` - Start inference
- `cuda_inference_next_token()` - Get next token
- `cuda_check_vram_residency()` - Health check

**See**: `docs/interfaces.md` (to be created in Week 2)

### HTTP API (Stable by Week 3)

**Endpoints**:
- `POST /execute` - Start inference
- `GET /health` - Worker health
- `POST /cancel` - Cancel job
- `POST /shutdown` - Graceful shutdown (optional)
- `GET /metrics` - Prometheus metrics (optional)

---

## ⚠️ Risks

### High Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Week 7 overcommitment | M0 delivery delayed | **Extend to 8 weeks** |
| FFI interface changes after Week 2 | Blocks Teams 2 & 3 | Lock interface by Week 2 end |
| Gate 1 failure | Entire project delayed | Weekly integration tests |

### Medium Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| CUDA context bugs | Integration delays | Valgrind tests, VRAM tracking |
| SSE UTF-8 edge cases | Client-facing bugs | Comprehensive test vectors |
| KV cache complexity | Week 4 delays | Start early if possible |

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
| Week 1 | 9 days | TBD | TBD |
| Week 2 | 13 days | TBD | TBD |
| Week 3 | 14 days | TBD | TBD |
| Week 4 | 14 days | TBD | TBD |
| Week 5 | 8 days | TBD | TBD |
| Week 6 | 11 days | TBD | TBD |
| Week 7 | 16 days | TBD | TBD |

**Target Velocity**: ~13 days/week (for 3 people)

---

## 🔗 Related Documents

### Spec References

- [M0 Worker Spec](../../../.specs/01_M0_worker_orcd.md) - Complete M0 requirements
- [System Spec](../../../.specs/00_llama-orch.md) - System-level requirements
- [Execution Plan](../../../.specs/.docs/M0_EXECUTION_PLAN.md) - Overall M0 execution strategy
- [Work Breakdown Plan](../../../.specs/.docs/M0_WORK_BREAKDOWN_PLAN.md) - 3-team breakdown

### Cross-Team Coordination

- [Llama Team Plan](../../llama-team/INDEX.md) - TBD
- [GPT Team Plan](../../gpt-team/INDEX.md) - TBD
- [Integration Gates (All Teams)](../../README.md) - Master gate tracking

---

## ✅ Next Actions

### Immediate (Week 0)

1. **🔴 CRITICAL**: Decide on 7 weeks vs 8 weeks
2. Review Planning Gap Analysis with stakeholders
3. Create remaining story cards (FT-003 through FT-047)
4. Story sizing workshop (all team members)
5. Set up development environment

### Week 1 Kickoff

1. Sprint planning Monday morning
2. Assign FT-001 through FT-005 to owners
3. Begin HTTP server development
4. Daily standups start Tuesday

---

**Status**: 📋 **PLANNING COMPLETE - AWAITING DECISION**  
**Next Action**: **Stakeholder decision on 7 vs 8 week timeline**  
**Blocker**: Cannot start Week 1 until timeline decided  
**Owner**: [Project Manager]

---

**Last Updated**: 2025-10-03  
**Document Version**: 1.0  
**Maintained By**: Foundation Team Lead
