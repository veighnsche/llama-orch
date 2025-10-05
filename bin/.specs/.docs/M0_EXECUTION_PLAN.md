# M0 Worker-orcd Execution Plan

**Date**: 2025-10-03  
**Work Breakdown**: Hybrid Split (Option C) - 3 Teams  
**Duration**: 6-7 weeks  
**Methodology**: Weekly sprints + Story-based tracking + Integration gates

---

## Recommended Approach: **Weekly Sprints + Story Cards + Hard Gates**

### Why This Approach?

**❌ NOT Velocity-Based**:
- No baseline velocity (new teams, new codebase)
- Fixed-scope project, not continuous delivery
- Predictability > throughput optimization

**❌ NOT Pure Epic-Based**:
- Epics too coarse for 6-7 week timeline
- Hard to track progress
- Integration risks hidden

**✅ RECOMMENDED: 1-Week Sprints + Stories + Gates**:
- **Weekly sprints** aligned with milestones
- **Story cards** (1-3 days each, clear acceptance criteria)
- **Integration gates** (Weeks 4, 5, 6, 7) - hard pass/fail
- **Daily standups** for coordination
- **Friday demos** to validate integration

---

## Sprint Structure

### Sprint Cadence (1 Week)

**Monday (2h)**: Sprint Planning
- Review previous sprint demos
- Commit to stories for this week
- Identify blockers and dependencies
- Update integration gate status

**Tue-Thu (15min daily)**: Standup
- What shipped yesterday?
- What shipping today?
- Blockers?

**Friday (2h)**: Demo + Retro
- **Mandatory**: Demo working feature with integration test
- Team retrospective
- Integration checkpoint review

### Why 1-Week Sprints?

- ✅ Catch integration issues within days, not weeks
- ✅ Forces frequent integration (aligns with gates)
- ✅ Short enough for course-correction
- ✅ Natural alignment with Week 4/5/6/7 gates

---

## Story Card Format

### Template

```markdown
### [TEAM-XXX] Story Title

**User Story**:
As a [role], I want [capability], so that [benefit]

**Acceptance Criteria** (testable):
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

**Definition of Done**:
- [ ] Code reviewed and merged
- [ ] Unit tests pass (coverage >80%)
- [ ] Integration test passes (if applicable)
- [ ] No warnings (rustfmt, clippy, clang-format)
- [ ] Documentation updated

**Size**: S/M/L (1 day / 2-3 days / 4-5 days)
**Dependencies**: [Story IDs]
**Sprint**: Week X
**Owner**: [Name]
```

### Sizing Guidelines

| Size | Days | Description | Example |
|------|------|-------------|---------|
| **S** | 1 day | Single function/module | "Implement RMSNorm kernel" |
| **M** | 2-3 days | Multiple functions, some integration | "GGUF header parser" |
| **L** | 4-5 days | Complex feature, significant integration | "End-to-end Qwen pipeline" |

**Rule**: No story >5 days. If L, consider splitting into 2-3 M stories.

---

## Integration Gates (Hard Pass/Fail)

### Gate 1: Week 4 - Foundation Complete

**Criteria** (all must pass):
- [ ] HTTP server operational (all endpoints respond)
- [ ] SSE streaming works (UTF-8 safe, event ordering correct)
- [ ] FFI layer stable (Rust can call CUDA context, allocate VRAM)
- [ ] Shared kernels implemented (embedding, GEMM, sampling)
- [ ] Integration test: HTTP → FFI → CUDA → response
- [ ] GGUF loader can parse headers (Team 2)
- [ ] HF tokenizer can encode/decode (Team 3)

**Gate Test**: HTTP request → CUDA context init → embed lookup → sample token → SSE response

**If Fail**: Sprint 5 delayed until pass. Teams 2 & 3 blocked.

---

### Gate 2: Week 5 - First Model Working

**Criteria**:
- [ ] Qwen2.5-0.5B loads to VRAM (352 MB)
- [ ] Haiku generation test passes (prompt → tokens)
- [ ] Reproducibility validated (same seed/temp=0 → identical output, 3 runs)
- [ ] VRAM-only verified (cudaPointerGetAttributes confirms device memory)
- [ ] Tokenization round-trip works (encode → decode → original text)

**Gate Test**: 
```bash
curl -X POST http://localhost:8080/execute \
  -d '{"job_id":"test","prompt":"Write a haiku","temperature":0.0,"seed":42,"max_tokens":50}' \
  | tee output1.json

# Run again
curl ... | tee output2.json

# Must be identical
diff output1.json output2.json  # No diff allowed
```

**If Fail**: Sprint 6 adapter work paused. Fix Qwen first.

---

### Gate 3: Week 6 - All Models Basic + Adapter Design

**Criteria**:
- [ ] Phi-3-Mini working (Llama adapter)
- [ ] GPT-OSS-20B working (Q4_K_M fallback, no MXFP4 yet)
- [ ] ModelAdapter interface agreed by all teams (header file signed off)
- [ ] Architecture detection working (GGUF → Llama or GPT)
- [ ] Factory pattern creates correct adapter

**Gate Test**: Load all 3 models sequentially, generate text with each

**If Fail**: Week 7 at risk. Emergency debug session.

---

### Gate 4: Week 7 - M0 Complete

**Criteria**:
- [ ] All 3 models use adapter pattern (LlamaAdapter, GPTAdapter)
- [ ] MXFP4 working for GPT-OSS-20B (~16 GB VRAM)
- [ ] All acceptance criteria from spec pass:
  - [ ] Qwen haiku test (reproducible)
  - [ ] Phi-3 generation test
  - [ ] GPT MXFP4 generation test
  - [ ] VRAM-only verified (all models)
  - [ ] UTF-8 streaming safe (multibyte test)
  - [ ] OOM recovery test
- [ ] Documentation complete
- [ ] CI/CD green

**Gate Test**: Automated test suite runs all models end-to-end

**If Fail**: M0 delivery delayed. Root cause analysis required.

---

## Weekly Sprint Breakdown

### Sprint 1 (Week 1): Foundation Starts

**Team 1 Focus**: HTTP server + FFI interface definition
**Team 2 Focus**: GGUF research + tokenizer design
**Team 3 Focus**: HF tokenizer research + GPT metadata study

**Stories** (~8-10 stories total, ~2-3 per team):
- T1-001: HTTP server setup (M)
- T1-002: POST /execute skeleton (M)
- T1-003: SSE streaming infrastructure (M)
- T1-004: FFI interface definition (S) ← **coordinate with Team 2**
- T2-001: GGUF header parser (M)
- T2-002: GGUF metadata extraction (M)
- T3-001: HF tokenizers crate integration (S)
- T3-002: HF tokenizer metadata (M)

**Demo**: HTTP server responds, GGUF headers parsed, HF tokenizer encodes text

---

### Sprint 2 (Week 2): Foundation Deepens

**Team 1 Focus**: FFI bindings + shared kernels start
**Team 2 Focus**: Memory-mapped I/O + tokenization
**Team 3 Focus**: GPT GGUF metadata + conformance tests

**Stories** (~10-12 stories):
- T1-005: Rust FFI bindings (M)
- T1-006: FFI integration tests (M)
- T1-007: Embedding kernel (S)
- T1-008: cuBLAS GEMM wrapper (M)
- T2-003: Memory-mapped I/O (M)
- T2-004: GGUF vocab parsing (M)
- T2-005: BPE encoder (M)
- T3-003: HF conformance tests (M)
- T3-004: GPT metadata parsing (M)

**Demo**: FFI working (Rust calls CUDA), GGUF mmap works, tokenizers encode/decode

---

### Sprint 3 (Week 3): Kernels + Tokenization Complete

**Team 1 Focus**: Sampling kernels + KV cache
**Team 2 Focus**: BPE decoder + Llama kernels start
**Team 3 Focus**: GGUF v3 (MXFP4) + GPT kernels start

**Stories** (~12-14 stories):
- T1-009: Sampling kernels (M)
- T1-010: KV cache management (M)
- T2-006: BPE decoder (M)
- T2-007: Tokenizer conformance tests (M)
- T2-008: RoPE kernel (M)
- T2-010: RMSNorm kernel (S)
- T3-005: GGUF v3 MXFP4 tensors (M)
- T3-006: Positional embedding kernel (S)
- T3-007: LayerNorm kernel (M)
- T3-008: GELU kernel (S)

**Demo**: Sampling works, tokenization complete (both backends), basic kernels running

---

### Sprint 4 (Week 4): Integration → Gate 1

**Team 1 Focus**: Integration test framework + Gate 1 validation
**Team 2 Focus**: Complete Llama kernels (GQA, SwiGLU)
**Team 3 Focus**: Complete GPT kernels (MHA)

**Stories** (~10-12 stories):
- T1-011: Integration test framework (M)
- T1-012: Gate 1 validation tests (M)
- T2-009: GQA attention kernel (L)
- T2-011: SwiGLU FFN kernel (M)
- T2-012: Qwen weight loading (M)
- T3-009: MHA attention kernel (L)
- T3-010: GPT weight loading Q4_K_M (M)

**Demo**: **Gate 1 Test** - Full HTTP → CUDA → SSE flow works

**Checkpoint**: If Gate 1 fails, Sprint 5 paused for fixes

---

### Sprint 5 (Week 5): First Model → Gate 2

**Team 1 Focus**: Support Teams 2 & 3 + performance baseline prep
**Team 2 Focus**: Qwen end-to-end + haiku test
**Team 3 Focus**: GPT basic pipeline (Q4_K_M)

**Stories** (~8-10 stories):
- T1-012: Performance baseline measurements (M)
- T2-013: Qwen forward pass (L)
- T2-014: Qwen haiku test (M)
- T3-011: GPT forward pass Q4_K_M (L)
- T3-011b: GPT basic generation test (M)

**Demo**: **Gate 2 Test** - Qwen haiku reproducible

**Checkpoint**: If Gate 2 fails, Sprint 6 adapter work paused

---

### Sprint 6 (Week 6): All Models + Adapter Design → Gate 3

**Team 1 Focus**: API docs + adapter coordination
**Team 2 Focus**: Phi-3 + LlamaModelAdapter
**Team 3 Focus**: MXFP4 implementation starts

**Stories** (~10-12 stories):
- T1-013: API documentation (M)
- T1-017: Adapter pattern coordination (M) ← **all teams**
- T2-015: Phi-3 weight loading (M)
- T2-016: Phi-3 integration (M)
- T2-017: LlamaModelAdapter (M)
- T3-012: MXFP4 dequant kernel (L)
- T3-013: MXFP4 GEMM integration (M)

**Demo**: **Gate 3 Test** - All 3 models load and generate (Phi-3, GPT basic)

**Checkpoint**: Adapter interface locked, no changes after this sprint

---

### Sprint 7 (Week 7): Final Integration → Gate 4 (M0 Complete)

**All Teams Focus**: MXFP4 final, adapter refactoring, testing, docs

**Stories** (~12-15 stories):
- T1-014: CI/CD pipeline (L)
- T1-015: Final integration tests (M)
- T2-018: Llama integration test suite (M)
- T2-019: Documentation (M)
- T3-014: GPT MXFP4 end-to-end (L)
- T3-015: GPTModelAdapter (M)
- T3-016: GPT integration tests (M)
- T3-017: MXFP4 numerical validation (M)
- ALL: Emergency buffer for Gate 4 blockers

**Demo**: **Gate 4 Test** - Full test suite runs, all models + adapters working

**Deliverable**: M0 worker-orcd ready for deployment

---

## Story Tracking Mechanism

### Recommended Tool: **GitHub Projects** (or Equivalent)

**Board Columns**:
1. **Backlog** - All stories, prioritized
2. **Sprint N** - Committed for current sprint
3. **In Progress** - Developer actively working
4. **Review** - Code review or integration testing
5. **Done** - Merged + tests passing

**Story Card Metadata**:
- **Labels**: team-1, team-2, team-3, size-S/M/L, gate-N
- **Milestone**: Sprint N (Week N)
- **Assignee**: Developer name
- **Dependencies**: Linked to blocking stories

### Daily Standup Update

Each developer updates their story card:
- Move to "In Progress" when starting
- Add comment with blockers
- Move to "Review" when ready
- Move to "Done" when merged

**Team lead** reviews board before standup:
- Identifies stuck stories (>2 days in same column)
- Unblocks dependencies
- Adjusts sprint scope if needed

---

## Story Estimation Session (Sprint 0)

### Before Sprint 1 Starts

**Team Workshop (4 hours)**:

1. **Story Writing** (1h):
   - Each team writes their Sprint 1-2 stories
   - Use template format
   - Clear acceptance criteria

2. **Story Review** (1h):
   - Cross-team review
   - Identify dependencies
   - Validate acceptance criteria testable

3. **Sizing** (1h):
   - Estimate each story (S/M/L)
   - Use planning poker or dot voting
   - No sandbagging (realistic estimates)

4. **Sprint 1 Commitment** (1h):
   - Each team commits to ~3-4 stories
   - Total: ~10 stories across all teams
   - Buffer for unknowns (first sprint)

**Output**: 
- Sprint 1 backlog ready
- Sprint 2-3 stories drafted
- Dependencies mapped

---

## Progress Tracking Metrics

### Weekly Metrics (Tracked by PM)

**Velocity** (stories completed per sprint):
- Track: # stories moved to Done
- Goal: Consistent velocity (not increasing)
- Flag: Velocity drops >30% sprint-over-sprint

**Gate Progress** (% toward gate criteria):
- Week 4 gate: X/7 criteria passing
- Week 5 gate: X/5 criteria passing
- Week 6 gate: X/5 criteria passing
- Week 7 gate: X/12 criteria passing

**Blocker Count** (unresolved blockers):
- Track: Stories stuck >2 days
- Goal: <3 blockers at any time
- Action: Daily unblocking session if >5

**Integration Test Pass Rate**:
- Track: % tests passing in CI
- Goal: 100% by Gate 1 (Week 4)
- Flag: <90% before gate week

### Do NOT Track

- ❌ Lines of code
- ❌ Commits per day
- ❌ Individual velocity
- ❌ Burn-down charts (fixed scope project)

**Why**: These metrics optimize wrong things. Focus on **working software** and **gate criteria**.

---

## Risk Mitigation: Story-Level

### Story Risk Indicators

**🔴 Red Flag** (act immediately):
- Story >5 days in "In Progress"
- Story blocking >3 other stories
- Acceptance criteria unclear or changing
- "Works on my machine" (no integration test)

**🟡 Yellow Flag** (watch closely):
- Story size estimated as L (4-5 days)
- Story has >2 dependencies
- Story spans multiple teams
- First time implementing pattern (e.g., first kernel)

**🟢 Green** (nominal):
- Story <2 days in flight
- Clear acceptance criteria
- Integration test exists
- No external dependencies

### Mitigation Actions

**For Red Flags**:
1. Immediate team huddle (30 min)
2. Break story into smaller stories
3. Pair programming session
4. Escalate to tech lead if blocked >1 day

**For Yellow Flags**:
1. Daily check-in with story owner
2. Pre-integration test before "Review"
3. Code review by senior dev
4. Buffer time in sprint plan

---

## Definition of Done (DoD) - Enforced

### Story-Level DoD

**All stories MUST**:
- [ ] Code merged to main branch
- [ ] All unit tests pass (coverage >80% for new code)
- [ ] Integration test passes (if story touches multiple modules)
- [ ] No compiler warnings (rustfmt, clippy, clang-format)
- [ ] Code review approved (1 approver minimum, 2 for critical)
- [ ] Documentation updated (README, API docs, code comments)
- [ ] Story owner demos feature in Friday demo

**Gate-Level DoD**:
- [ ] All gate criteria passing (100%, no exceptions)
- [ ] Integration test suite green
- [ ] Manual smoke test passed by PM
- [ ] Team lead sign-off
- [ ] Ready for next sprint to start

---

## Example: Sprint 1 Story Breakdown

### Team 1: Core Infrastructure (Sprint 1)

#### Story T1-001: HTTP Server Setup
**Size**: M (2 days)

**Acceptance Criteria**:
- [ ] Axum server binds to specified port (configurable via CLI)
- [ ] GET /health returns 200 with `{"status":"starting"}`
- [ ] Server logs startup message with port number
- [ ] Server handles SIGTERM gracefully (logs shutdown, exits 0)
- [ ] Integration test: `curl http://localhost:8080/health` succeeds

**DoD Additions**:
- [ ] README updated with "How to Run" section
- [ ] CLI help text documents --port flag

**Owner**: Alice (Rust lead)  
**Dependencies**: None

---

#### Story T1-002: POST /execute Endpoint (Skeleton)
**Size**: M (2 days)

**Acceptance Criteria**:
- [ ] Endpoint accepts POST /execute with JSON body
- [ ] Request validation: job_id required, prompt required, temperature 0.0-2.0
- [ ] Returns 400 with error message if validation fails
- [ ] Returns 202 Accepted for valid requests (no actual inference yet)
- [ ] X-Correlation-Id middleware attaches correlation ID to logs

**DoD Additions**:
- [ ] API documentation: request/response schemas
- [ ] Unit test: validation edge cases (empty prompt, temp=3.0, etc.)

**Owner**: Alice  
**Dependencies**: T1-001 (needs server running)

---

### Team 2: Llama Pipeline (Sprint 1)

#### Story T2-001: GGUF Header Parser
**Size**: M (2 days)

**Acceptance Criteria**:
- [ ] Parse magic bytes from GGUF file (must be 0x47475546)
- [ ] Parse version (must be 3)
- [ ] Parse tensor_count, metadata_kv_count
- [ ] Throw error if magic invalid or version ≠3
- [ ] Unit test with sample GGUF file (Qwen)

**DoD Additions**:
- [ ] Error messages include: file path, expected vs actual values
- [ ] Code comments explain GGUF format structure

**Owner**: Bob (GGUF specialist)  
**Dependencies**: None

---

## Summary: Why This Works

### Key Success Factors

1. **Weekly Cadence** → Catch issues early (days, not weeks)
2. **Story Cards** → Clear ownership and acceptance criteria
3. **Hard Gates** → Force integration, prevent "works in isolation" syndrome
4. **Friday Demos** → Tangible progress visible to all teams
5. **No Velocity Pressure** → Focus on quality and integration, not story count

### What This Prevents

- ❌ "Big bang" integration at the end (gates force continuous integration)
- ❌ Unclear progress (story cards + demos show exactly what's working)
- ❌ Scope creep (fixed gates, stories aligned to gates)
- ❌ Hidden blockers (daily standups + story board visibility)
- ❌ Last-minute surprises (weekly gates catch issues early)

### Expected Outcomes

**Week 4**: Foundation solid, teams 2 & 3 can proceed confidently  
**Week 5**: First model working, proof of concept validated  
**Week 6**: Adapter pattern proven, all models loading  
**Week 7**: M0 complete, ready for M1 planning

---

## Next Actions

1. **Sprint 0 Prep** (This Week):
   - [ ] Set up story tracking tool (GitHub Projects)
   - [ ] All teams write Sprint 1-2 stories
   - [ ] Story estimation workshop (4h session)
   - [ ] Commit to Sprint 1 scope

2. **Sprint 1 Kickoff** (Monday Week 1):
   - [ ] Sprint planning (2h)
   - [ ] Team stand-up setup (15min daily, 9am)
   - [ ] Friday demo scheduled (2pm)

3. **Ongoing**:
   - [ ] Daily standups (every team member updates)
   - [ ] Story board grooming (PM checks blockers daily)
   - [ ] Gate criteria tracking (PM updates gate checklist)
   - [ ] Friday demos (mandatory attendance, all teams)

**Status**: ✅ **Ready to Execute**  
**First Action**: Schedule Sprint 0 story writing workshop
