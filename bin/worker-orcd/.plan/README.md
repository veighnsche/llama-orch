# M0 Worker-orcd Team Planning Structure

**Created**: 2025-10-03  
**Duration**: 6-7 weeks  
**Methodology**: Weekly sprints + Story cards + Integration gates

---

## Team Names & Responsibilities

### 🏗️ **Foundation Team** (Team 1)
**Focus**: Core infrastructure (HTTP, FFI, CUDA context, shared kernels)  
**Duration**: Weeks 1-7 (full project)  
**Skills**: Rust, C++, CUDA, DevOps

### 🦙 **Llama Team** (Team 2)
**Focus**: GGUF loader, GGUF-BPE tokenizer, Llama kernels, Qwen + Phi-3  
**Duration**: Weeks 2-7 (starts after foundation begins)  
**Skills**: C++, CUDA, BPE, model formats

### 🤖 **GPT Team** (Team 3)
**Focus**: HF tokenizer, GPT kernels, MXFP4, GPT-OSS-20B  
**Duration**: Weeks 2-7 (starts after foundation begins)  
**Skills**: C++, CUDA, quantization, HF ecosystem

---

## Folder Structure

```
.plan/
├── README.md                          # This file
├── 00_TEAM_OVERVIEW.md               # Team composition, skills, contacts
├── 01_INTEGRATION_GATES.md           # Week 4, 5, 6, 7 gate criteria
├── 02_SPRINT_CALENDAR.md             # Sprint dates, demos, retros
│
├── foundation-team/                   # Team 1: Core Infrastructure
│   ├── sprints/                       # Weekly sprint folders
│   │   ├── week-1/
│   │   │   ├── sprint-plan.md        # Sprint commitment
│   │   │   ├── stories/              # Story cards for this sprint
│   │   │   │   ├── T1-001.md
│   │   │   │   ├── T1-002.md
│   │   │   │   └── ...
│   │   │   ├── demo-notes.md         # Friday demo outcomes
│   │   │   └── retrospective.md      # What went well/poorly
│   │   ├── week-2/
│   │   ├── week-3/
│   │   ├── week-4/                   # Gate 1 week
│   │   ├── week-5/
│   │   ├── week-6/
│   │   └── week-7/                   # Gate 4 week
│   ├── stories/                       # All stories (backlog)
│   │   ├── backlog/                  # Not yet scheduled
│   │   ├── in-progress/              # Currently being worked
│   │   ├── review/                   # Code review or testing
│   │   └── done/                     # Completed and merged
│   ├── integration-gates/             # Gate-specific tracking
│   │   ├── gate-1-week-4.md          # Foundation complete
│   │   ├── gate-2-week-5.md          # Support role
│   │   ├── gate-3-week-6.md          # Adapter coordination
│   │   └── gate-4-week-7.md          # Final integration
│   └── docs/
│       ├── team-charter.md           # Team mission, roles
│       ├── interfaces.md             # FFI interface definitions
│       ├── decisions.md              # ADRs (Architecture Decision Records)
│       └── handoff-notes.md          # For Teams 2 & 3
│
├── llama-team/                        # Team 2: Llama Pipeline
│   ├── sprints/
│   │   ├── week-2/                   # Starts Week 2
│   │   │   ├── sprint-plan.md
│   │   │   ├── stories/
│   │   │   ├── demo-notes.md
│   │   │   └── retrospective.md
│   │   ├── week-3/
│   │   ├── week-4/                   # Gate 1 participation
│   │   ├── week-5/                   # Gate 2 week (Qwen working)
│   │   ├── week-6/                   # Gate 3 week (Phi-3 + adapter)
│   │   └── week-7/                   # Gate 4 week
│   ├── stories/
│   │   ├── backlog/
│   │   ├── in-progress/
│   │   ├── review/
│   │   └── done/
│   ├── integration-gates/
│   │   ├── gate-1-week-4.md          # Llama kernels ready
│   │   ├── gate-2-week-5.md          # Qwen haiku test
│   │   ├── gate-3-week-6.md          # Phi-3 + LlamaAdapter
│   │   └── gate-4-week-7.md          # Final Llama validation
│   └── docs/
│       ├── team-charter.md
│       ├── gguf-format.md            # GGUF parsing notes
│       ├── bpe-algorithm.md          # BPE implementation notes
│       ├── llama-architecture.md     # Llama-style model details
│       └── test-vectors.md           # Tokenizer conformance tests
│
└── gpt-team/                          # Team 3: GPT Pipeline
    ├── sprints/
    │   ├── week-2/                   # Starts Week 2
    │   │   ├── sprint-plan.md
    │   │   ├── stories/
    │   │   ├── demo-notes.md
    │   │   └── retrospective.md
    │   ├── week-3/
    │   ├── week-4/                   # Gate 1 participation
    │   ├── week-5/                   # GPT basic working
    │   ├── week-6/                   # Gate 3 week (MXFP4 + adapter)
    │   └── week-7/                   # Gate 4 week
    ├── stories/
    │   ├── backlog/
    │   ├── in-progress/
    │   ├── review/
    │   └── done/
    ├── integration-gates/
    │   ├── gate-1-week-4.md          # GPT kernels ready
    │   ├── gate-2-week-5.md          # GPT basic (Q4_K_M)
    │   ├── gate-3-week-6.md          # MXFP4 + GPTAdapter
    │   └── gate-4-week-7.md          # Final GPT validation
    └── docs/
        ├── team-charter.md
        ├── hf-tokenizer.md           # HF tokenizers crate usage
        ├── gpt-architecture.md       # GPT-style model details
        ├── mxfp4-spec.md             # MXFP4 quantization format
        └── numerical-validation.md   # MXFP4 correctness tests
```

---

## How to Use This Structure

### For Team Leads

**Sprint Planning (Monday)**:
1. Review previous sprint's `demo-notes.md` and `retrospective.md`
2. Create new `sprint-plan.md` in current week folder
3. Move stories from `backlog/` to `sprints/week-N/stories/`
4. Commit to sprint scope (typically 3-4 stories per team)

**Daily Standups**:
- Update story status (move between `backlog/`, `in-progress/`, `review/`, `done/`)
- Add blockers to story cards
- Update integration gate checklists

**Friday Demo**:
1. Record outcomes in `demo-notes.md`
2. Update gate progress in `integration-gates/gate-N-week-N.md`
3. Run retrospective, document in `retrospective.md`

### For Developers

**Starting a Story**:
1. Read story card in `sprints/week-N/stories/T#-XXX.md`
2. Move story to `in-progress/` folder
3. Update story with daily progress notes

**Completing a Story**:
1. Ensure all acceptance criteria checked off
2. Move story to `review/` for code review
3. After merge, move to `done/`
4. Update sprint plan with completion

### For Project Manager

**Weekly Tracking**:
- Check each team's `sprint-plan.md` for commitment vs completion
- Review `integration-gates/` for gate progress
- Identify blockers across teams (stories stuck in `in-progress/`)
- Update master timeline based on gate status

**Gate Weeks** (4, 5, 6, 7):
- All teams update their `integration-gates/gate-N-week-N.md`
- Run gate tests (automated + manual)
- Go/No-Go decision for next sprint
- Document gate outcomes

---

## Story Card Template

Each story card is a markdown file: `T#-XXX.md`

```markdown
# [T#-XXX] Story Title

**Team**: Foundation / Llama / GPT  
**Sprint**: Week N  
**Size**: S / M / L (1 day / 2-3 days / 4-5 days)  
**Owner**: [Developer Name]  
**Status**: Backlog / In Progress / Review / Done

## User Story

As a [role], I want [capability], so that [benefit]

## Acceptance Criteria

- [ ] Criterion 1 (testable)
- [ ] Criterion 2 (testable)
- [ ] Criterion 3 (testable)

## Definition of Done

- [ ] Code written and reviewed
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration test passes (if applicable)
- [ ] No compiler warnings (rustfmt, clippy)
- [ ] Documentation updated
- [ ] Demoed in Friday demo

## Dependencies

- Depends on: [Story IDs]
- Blocks: [Story IDs]

## Technical Notes

[Implementation details, design decisions, etc.]

## Progress Log

**YYYY-MM-DD**: Started work, set up environment  
**YYYY-MM-DD**: Implemented X, blocked on Y  
**YYYY-MM-DD**: Unblocked, completed acceptance criteria 1-2  
**YYYY-MM-DD**: Code review feedback addressed, merged
```

---

## Integration Gate Template

Each gate has a checklist: `gate-N-week-N.md`

```markdown
# Gate N: [Gate Name] - Week N

**Date**: YYYY-MM-DD  
**Status**: 🟡 In Progress / 🟢 Passed / 🔴 Failed

## Criteria

### Foundation Team
- [ ] Criterion 1
- [ ] Criterion 2

### Llama Team
- [ ] Criterion 1
- [ ] Criterion 2

### GPT Team
- [ ] Criterion 1
- [ ] Criterion 2

## Gate Test

**Command**:
```bash
[Test command to run]
```

**Expected Output**:
```
[What success looks like]
```

## Results

**Test Run**: YYYY-MM-DD HH:MM  
**Outcome**: Pass / Fail  
**Notes**: [Any issues or observations]

## Go/No-Go Decision

**Decision**: Go / No-Go  
**Rationale**: [Why we passed or what needs fixing]  
**Action Items** (if No-Go):
- [ ] Fix X (Owner: [Name])
- [ ] Fix Y (Owner: [Name])
```

---

## Sprint Plan Template

Each sprint has a plan: `sprint-plan.md`

```markdown
# Sprint Plan: Week N

**Team**: Foundation / Llama / GPT  
**Sprint Dates**: YYYY-MM-DD to YYYY-MM-DD  
**Sprint Goal**: [One-sentence goal]

## Committed Stories

| Story ID | Title | Size | Owner | Status |
|----------|-------|------|-------|--------|
| T#-XXX | Story title | M | Alice | ✅ Done |
| T#-YYY | Story title | L | Bob | 🟡 In Progress |
| T#-ZZZ | Story title | S | Carol | ⏸️ Blocked |

## Sprint Capacity

- **Team Size**: N developers
- **Available Days**: N days (accounting for holidays, etc.)
- **Estimated Capacity**: N story points (S=1, M=2, L=4)
- **Committed**: N story points

## Dependencies

- **Waiting On**: [Stories from other teams]
- **Blocking**: [Stories in other teams waiting on us]

## Risks

- 🔴 High Risk: [Description]
- 🟡 Medium Risk: [Description]

## Daily Progress

**Monday**: Sprint planning complete, stories assigned  
**Tuesday**: [Progress update]  
**Wednesday**: [Progress update]  
**Thursday**: [Progress update]  
**Friday**: Demo + retro complete
```

---

## Cross-Team Coordination

### Shared Documents (Root Level)

**00_TEAM_OVERVIEW.md**: Team rosters, skills, contact info  
**01_INTEGRATION_GATES.md**: Master gate criteria and schedule  
**02_SPRINT_CALENDAR.md**: All sprint dates, demo times, holidays

### Integration Points

**Week 2-3**: Foundation Team defines FFI interface  
→ Documented in `foundation-team/docs/interfaces.md`  
→ Llama & GPT teams review and approve

**Week 5-6**: All teams design InferenceAdapter interface  
→ Documented in `foundation-team/docs/interfaces.md` (adapter section)  
→ Joint design session, all teams sign off

**Week 6-7**: Final integration  
→ Cross-team pair programming sessions  
→ Documented in each team's `sprints/week-7/integration-notes.md`

---

## Best Practices

### Story Management

✅ **DO**:
- Keep stories small (max 5 days)
- Clear acceptance criteria (testable)
- Update progress daily
- Move stories through workflow (backlog → in-progress → review → done)

❌ **DON'T**:
- Leave stories in "in-progress" >3 days without update
- Change acceptance criteria mid-sprint
- Skip Definition of Done checklist
- Work on stories not in current sprint

### Documentation

✅ **DO**:
- Document decisions in `docs/decisions.md`
- Keep technical notes in story cards
- Update interface docs when APIs change
- Write handoff notes for other teams

❌ **DON'T**:
- Duplicate docs across teams (link instead)
- Let docs go stale (update with code)
- Skip retrospectives (learn from mistakes)

### Integration

✅ **DO**:
- Test integration points weekly
- Update gate checklists daily
- Communicate blockers immediately
- Pair program across teams when needed

❌ **DON'T**:
- Wait until gate week to test integration
- Hide blockers or risks
- Work in isolation (sync frequently)
- Skip Friday demos (mandatory)

---

## Getting Started

### Week 0 (Sprint 0)

1. **All Teams**: Read this README
2. **Team Leads**: Create `docs/team-charter.md` for your team
3. **All Teams**: Story writing workshop (4 hours)
4. **All Teams**: Create stories in `stories/backlog/`
5. **All Teams**: Size stories (planning poker)
6. **All Teams**: Create `sprints/week-1/sprint-plan.md` (or week-2 for Llama/GPT)

### Week 1 (Sprint 1)

**Foundation Team**:
- Sprint planning Monday
- Daily standups Tue-Thu
- Friday demo + retro

**Llama & GPT Teams**:
- Prepare for Week 2 start
- Review Foundation Team's FFI interface design
- Finalize Week 2 sprint plans

---

**Status**: ✅ Structure Ready  
**Next Action**: Teams create their `team-charter.md` and begin story writing
