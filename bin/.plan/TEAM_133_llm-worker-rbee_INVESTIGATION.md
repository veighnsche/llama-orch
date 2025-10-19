# TEAM-133: llm-worker-rbee INVESTIGATION

**Binary:** `bin/llm-worker-rbee`  
**Team:** TEAM-133  
**Phase:** Investigation (Week 1)  
**Status:** 🔍 IN PROGRESS

---

## 🎯 MISSION

**Investigate `llm-worker-rbee` binary and propose decomposition into 6 focused crates under `worker-rbee-crates/`.**

**CRITICAL:** These crates will be SHARED by ALL future worker types!

**NO CODE CHANGES! INVESTIGATION ONLY!**

---

## 📊 CURRENT STATE

### Binary Stats
- **Total LOC:** ~2,550 (verified via cloc)
- **Files:** ~20 (verify exact count!)
- **Purpose:** LLM inference worker
- **Test coverage:** TBD (investigate!)
- **Dependencies:** candle, axum, tokio, etc.

### File Structure (Known)
```
bin/llm-worker-rbee/src/
├── main.rs
├── lib.rs
├── common/
│   ├── error.rs (336 LOC) ✅ TEAM-130 complete
│   ├── startup.rs (316 LOC) ✅ TEAM-130 fixed
│   └── ... (discover rest)
├── inference/
│   └── ... (core inference logic)
├── http/
│   ├── sse.rs (SSE streaming)
│   └── ... (HTTP routes)
├── heartbeat.rs (182 LOC)
└── ... (discover all files!)
```

---

## 🔍 INVESTIGATION TASKS

### Day 1-2: Deep Code Analysis

#### Task 1.1: Complete File Discovery
- [ ] List ALL files with LOC counts
- [ ] Map complete directory structure
- [ ] Identify largest files
- [ ] Document module organization
- [ ] Verify total LOC

**Command:**
```bash
cloc bin/llm-worker-rbee/src --by-file
```

#### Task 1.2: Read Every File
- [ ] Read inference engine code
- [ ] Read model loading logic
- [ ] Read SSE streaming implementation
- [ ] Read HTTP server setup
- [ ] Read heartbeat mechanism
- [ ] Read startup logic
- [ ] Read error handling (already done by TEAM-130!)
- [ ] Read all other files

**Questions to Answer:**
1. How does inference work? (Candle integration)
2. How are models loaded? (From disk? Download?)
3. How is SSE streaming implemented?
4. How does heartbeat work?
5. How does worker register with rbee-hive?

#### Task 1.3: Document Inference Flow
- [ ] Map end-to-end inference flow
- [ ] Document model loading
- [ ] Document tokenization
- [ ] Document generation
- [ ] Document SSE streaming
- [ ] Document error handling

**Example Flow:**
```
HTTP Request → Validate → Load Model → Tokenize → Generate → Stream Tokens → Complete
```

#### Task 1.4: Analyze Dependencies
- [ ] Read `Cargo.toml` completely
- [ ] List all dependencies
- [ ] Map Candle usage
- [ ] Check shared crate usage
- [ ] Identify missing shared crates

**Known Dependencies:**
- `candle-core`, `candle-nn`, `candle-transformers`
- `axum`, `tokio`, `tower`
- `async-stream` (for SSE)
- `reqwest` (for callbacks)
- Shared crates: `input-validation`, `narration-core`, etc.

---

### Day 2-3: Crate Boundary Analysis

#### Task 2.1: Propose 6 Crates for `worker-rbee-crates/`

**CRITICAL:** These crates must be REUSABLE by future worker types!

**Initial Proposal (VERIFY & REFINE!):**

1. **`worker-rbee-inference`** (~800 LOC estimate)
   - Purpose: Core inference engine (Candle integration)
   - Responsibilities:
     - Model inference
     - Token generation
     - Batch processing
     - GPU management
   - Public API: `InferenceEngine`, `generate()`
   - Dependencies: `candle-*`
   - **REUSABLE:** ✅ All workers need inference!

2. **`worker-rbee-model-loader`** (~600 LOC estimate)
   - Purpose: Model loading and management
   - Responsibilities:
     - Load models from disk
     - Model caching
     - Model validation
     - Format detection (GGUF, safetensors)
   - Public API: `ModelLoader`, `load_model()`
   - Dependencies: `candle-*`, `model-catalog`
   - **REUSABLE:** ✅ All workers load models!

3. **`worker-rbee-sse-streaming`** (~400 LOC estimate)
   - Purpose: SSE streaming implementation
   - Responsibilities:
     - Stream tokens via SSE
     - Handle backpressure
     - Error streaming
     - Connection management
   - Public API: `SseStream`, `stream_tokens()`
   - Dependencies: `axum`, `async-stream`
   - **REUSABLE:** ✅ All workers stream responses!

4. **`worker-rbee-health`** (~180 LOC estimate)
   - Purpose: Health checks and heartbeat
   - Responsibilities:
     - Send heartbeats to rbee-hive
     - Health status reporting
     - Readiness checks
     - Liveness checks
   - Public API: `HealthMonitor`, `send_heartbeat()`
   - Dependencies: `reqwest`, `tokio`
   - **REUSABLE:** ✅ All workers need health checks!

5. **`worker-rbee-startup`** (~316 LOC estimate)
   - Purpose: Worker startup and registration
   - Responsibilities:
     - Register with rbee-hive
     - Callback on ready
     - Configuration loading
     - Initialization
   - Public API: `startup()`, `callback_ready()`
   - Dependencies: `reqwest`
   - **REUSABLE:** ✅ All workers need startup!

6. **`worker-rbee-error`** (~336 LOC estimate)
   - Purpose: Error types and handling
   - Responsibilities:
     - Worker error types
     - Error code mapping
     - HTTP error responses
     - Retriability logic
   - Public API: `WorkerError`, error types
   - Dependencies: `thiserror`, `axum`
   - **REUSABLE:** ✅ All workers need error handling!

#### Task 2.2: Verify Reusability

**For EACH crate, answer:**
1. **Can embedding-worker-rbee use this?** Yes/No + Why
2. **Can vision-worker-rbee use this?** Yes/No + Why
3. **Can audio-worker-rbee use this?** Yes/No + Why
4. **What would need to be generic?**
5. **What would be LLM-specific?**

**Example Analysis:**
```
worker-rbee-inference:
  ✅ Embedding worker: YES (same inference engine)
  ✅ Vision worker: MAYBE (different model types)
  ✅ Audio worker: MAYBE (different model types)
  
  Generic: Inference loop, GPU management
  Specific: Tokenization (LLM-specific)
  
  Solution: Make tokenization pluggable!
```

#### Task 2.3: Identify LLM-Specific Code
- [ ] What code is LLM-specific?
- [ ] What code is generic to all workers?
- [ ] Should LLM-specific code be separate?
- [ ] Should there be `llm-worker-rbee-crates/` too?

**Possible Structure:**
```
worker-rbee-crates/        # Shared by ALL workers
    inference/
    model-loader/
    sse-streaming/
    health/
    startup/
    error/

llm-worker-rbee-crates/    # LLM-specific
    tokenization/
    llm-inference/
    text-generation/
```

#### Task 2.4: Map Dependencies Between Crates
```
llm-worker-rbee (binary)
    ├─> worker-rbee-startup
    ├─> worker-rbee-health
    ├─> worker-rbee-inference
    │       └─> worker-rbee-model-loader
    ├─> worker-rbee-sse-streaming
    └─> worker-rbee-error
```

- [ ] Verify no circular dependencies
- [ ] Document data flow
- [ ] Identify shared types
- [ ] Map public APIs

---

### Day 3-4: Shared Crate Analysis

#### Task 3.1: Audit Shared Crate Usage

**For each shared crate:**

1. **`input-validation`**
   - [ ] Where is it used?
   - [ ] Are all inputs validated?
   - [ ] Should more code use it?

2. **`narration-core`**
   - [ ] Is observability implemented?
   - [ ] Are all operations traced?
   - [ ] Should more operations be narrated?

3. **`deadline-propagation`**
   - [ ] Are deadlines propagated?
   - [ ] Is timeout handling correct?
   - [ ] Should inference use it?

4. **`model-catalog`**
   - [ ] Is it used for model info?
   - [ ] Should model-loader use it?
   - [ ] Is model metadata duplicated?

5. **`gpu-info`**
   - [ ] Is GPU detection used?
   - [ ] Should inference use it?
   - [ ] Is GPU info reported correctly?

#### Task 3.2: Identify Missing Opportunities
- [ ] Is there duplicate HTTP code?
- [ ] Should there be shared HTTP client?
- [ ] Is error handling consistent?
- [ ] Should there be shared error types?
- [ ] Is configuration management shared?

#### Task 3.3: Check Integration Points
- [ ] How does worker talk to rbee-hive? (HTTP callbacks)
- [ ] How does worker receive requests? (HTTP from queen-rbee)
- [ ] Are there shared types with rbee-hive?
- [ ] Are there shared types with queen-rbee?
- [ ] Should types be in shared crates?

**Example Shared Types:**
- Worker registration request
- Health check response
- Inference request format
- Inference response format

---

### Day 4-5: Risk Assessment & Migration Strategy

#### Task 4.1: Identify Breaking Changes
- [ ] Will inference API change?
- [ ] Will model loading change?
- [ ] Will SSE format change?
- [ ] Will heartbeat format change?
- [ ] Impact on rbee-hive integration?
- [ ] Impact on queen-rbee integration?

#### Task 4.2: Assess Migration Complexity

**For each crate:**

1. **worker-rbee-error** (PILOT CANDIDATE?)
   - Complexity: Low (already well-defined by TEAM-130!)
   - Dependencies: Standalone
   - Test coverage: 100% (TEAM-130 added tests!)
   - Risk: Low
   - Effort: 4 hours (already done!)

2. **worker-rbee-startup**
   - Complexity: Medium (callback logic)
   - Dependencies: worker-rbee-error
   - Test coverage: Good (TEAM-130 fixed tests!)
   - Risk: Medium (integration with rbee-hive)
   - Effort: 8 hours

3. **worker-rbee-health**
   - Complexity: Low (simple heartbeat)
   - Dependencies: worker-rbee-error
   - Test coverage: TBD
   - Risk: Low
   - Effort: 6 hours

4. **worker-rbee-model-loader**
   - Complexity: High (Candle integration)
   - Dependencies: model-catalog, candle
   - Test coverage: TBD
   - Risk: High (model loading is critical!)
   - Effort: 16 hours

5. **worker-rbee-inference**
   - Complexity: High (core inference)
   - Dependencies: worker-rbee-model-loader, candle
   - Test coverage: TBD
   - Risk: High (inference is critical!)
   - Effort: 20 hours

6. **worker-rbee-sse-streaming**
   - Complexity: Medium (async streaming)
   - Dependencies: worker-rbee-error
   - Test coverage: TBD
   - Risk: Medium (streaming is tricky!)
   - Effort: 12 hours

**Total Effort:** ~66 hours (2 weeks for 1 person, 1 week for 2 people)

#### Task 4.3: Document Test Strategy
- [ ] Current test coverage per module
- [ ] Unit tests per crate
- [ ] Integration tests
- [ ] BDD tests per crate
- [ ] Inference accuracy tests
- [ ] Performance tests

**Critical Tests:**
- Model loading (various formats)
- Inference correctness
- SSE streaming (backpressure)
- Heartbeat reliability
- Error handling
- Startup/registration

#### Task 4.4: Create Migration Plan

**Recommended Order:**
1. **worker-rbee-error** (PILOT - already done by TEAM-130!)
2. **worker-rbee-health** (simple, low risk)
3. **worker-rbee-startup** (depends on error, health)
4. **worker-rbee-sse-streaming** (depends on error)
5. **worker-rbee-model-loader** (complex, high risk)
6. **worker-rbee-inference** (most complex, depends on model-loader)

**Why this order?**
- Start with easiest (error - done!)
- Build confidence with simple crates
- Save complex crates for last
- Verify integration at each step

#### Task 4.5: Document Rollback Plan
- [ ] Checkpoint after each crate
- [ ] Verification steps
- [ ] Rollback procedure
- [ ] Go/No-Go criteria per crate

---

### Day 5: Report Writing

#### Task 5.1: Complete Investigation Report

**Required Sections:**
1. Executive Summary
2. Current Architecture (with actual LOC!)
3. Proposed Crate Structure (6 crates under `worker-rbee-crates/`)
4. Reusability Analysis (can future workers use these?)
5. LLM-Specific vs Generic Code
6. Shared Crate Analysis
7. Migration Strategy (recommended order)
8. Risk Assessment
9. Recommendations

#### Task 5.2: Get Peer Review
- [ ] Share with TEAM-131 (rbee-hive)
- [ ] Share with TEAM-132 (queen-rbee)
- [ ] Share with TEAM-134 (rbee-keeper)
- [ ] Incorporate feedback
- [ ] Finalize report

---

## 🚨 CRITICAL QUESTIONS TO ANSWER

### Reusability Questions (MOST IMPORTANT!):
1. **Can embedding-worker-rbee reuse these crates?** Which ones?
2. **Can vision-worker-rbee reuse these crates?** Which ones?
3. **Can audio-worker-rbee reuse these crates?** Which ones?
4. **What needs to be generic?** Inference? Model loading?
5. **What's LLM-specific?** Tokenization? Text generation?

### Architecture Questions:
1. **Is inference engine truly generic?** Or LLM-specific?
2. **Can model-loader handle all model types?** Or just LLMs?
3. **Is SSE streaming generic?** Or specific to text?
4. **Should there be `llm-worker-rbee-crates/` too?**
5. **Are there circular dependencies?**

### Integration Questions:
1. **How does worker register with rbee-hive?** HTTP callback?
2. **How does worker receive requests?** From queen-rbee?
3. **Are there shared types?** With rbee-hive? queen-rbee?
4. **Should types be in shared crates?**
5. **Is there duplicate code?**

### Migration Questions:
1. **What's the safest crate to start with?** (error - done!)
2. **What's the riskiest crate?** (inference!)
3. **Will this break integration?**
4. **How to verify each step?**
5. **How long will migration take?** (Realistic: 2 weeks)

---

## 📋 DELIVERABLES

### Required Outputs:
- [ ] **Investigation Report** (TEAM_133_llm-worker-rbee_INVESTIGATION.md)
- [ ] **Actual LOC counts** (not estimates!)
- [ ] **Complete file structure**
- [ ] **Dependency graph**
- [ ] **Inference flow diagram**
- [ ] **6 crate proposals** (under `worker-rbee-crates/`)
- [ ] **Reusability analysis** (can future workers use these?)
- [ ] **LLM-specific vs generic analysis**
- [ ] **Shared crate audit**
- [ ] **Migration plan** (recommended order)
- [ ] **Risk assessment**
- [ ] **Peer review**

### Quality Criteria:
- ✅ Every file analyzed
- ✅ Actual LOC counted
- ✅ Reusability verified
- ✅ All dependencies mapped
- ✅ All shared crates audited
- ✅ Clear recommendations
- ✅ Peer-reviewed

---

## 🎯 SUCCESS CRITERIA

### Investigation Complete When:
- [ ] All ~2,550 LOC analyzed
- [ ] 6 crates proposed under `worker-rbee-crates/`
- [ ] Reusability verified for future workers
- [ ] LLM-specific code identified
- [ ] All shared crates audited
- [ ] Migration plan complete (with order!)
- [ ] Risks assessed
- [ ] Report peer-reviewed
- [ ] Go/No-Go decision made

---

## 🔥 SPECIAL NOTE: FUTURE-PROOFING

**This is THE MOST IMPORTANT investigation!**

These crates will be used by:
- `llm-worker-rbee` (current)
- `embedding-worker-rbee` (future)
- `vision-worker-rbee` (future)
- `audio-worker-rbee` (future)
- `multimodal-worker-rbee` (future)

**Get this right, and we enable the future!**

**Get this wrong, and we duplicate code forever!**

---

## 📞 NEED HELP?

- **Slack:** `#team-133-llm-worker-rbee`
- **Daily Standup:** 9:00 AM
- **Team Lead:** [Name]
- **Peer Review:** TEAM-131, TEAM-132, TEAM-134

---

## ✅ READY TO START!

**First: Verify LOC and map complete file structure!**

**Focus: Reusability for future worker types!**

**Remember: INVESTIGATION ONLY - NO CODE CHANGES!**

**TEAM-133: Let's decompose llm-worker-rbee and enable the future! 🚀**
