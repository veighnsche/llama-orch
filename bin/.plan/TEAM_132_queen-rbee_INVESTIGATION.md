# TEAM-132: queen-rbee INVESTIGATION

**Binary:** `bin/queen-rbee`  
**Team:** TEAM-132  
**Phase:** Investigation (Week 1)  
**Status:** 🔍 IN PROGRESS

---

## 🎯 MISSION

**Investigate `queen-rbee` binary and propose decomposition into 4 focused crates.**

**NO CODE CHANGES! INVESTIGATION ONLY!**

---

## 📊 CURRENT STATE

### Binary Stats
- **Total LOC:** ~3,100 (estimate - verify!)
- **Files:** TBD (count them!)
- **Purpose:** HTTP orchestrator daemon (NOT the CLI!)
- **Test coverage:** TBD (investigate!)
- **Dependencies:** TBD (audit Cargo.toml!)

### What is queen-rbee?
- **NOT** the `rbee` CLI (that's `rbee-keeper`)
- **IS** the orchestrator HTTP daemon
- Receives inference requests
- Selects best worker via load balancing
- Forwards requests to workers
- Streams responses back to clients

### File Structure (VERIFY!)
```
bin/queen-rbee/src/
├── main.rs
├── lib.rs (if exists)
├── orchestrator/ (request orchestration)
├── registry/ (worker registry)
├── load_balancer/ (worker selection)
├── http/ (HTTP server)
└── ... (discover the rest!)
```

---

## 🔍 INVESTIGATION TASKS

### Day 1-2: Deep Code Analysis

#### Task 1.1: Discover File Structure
- [ ] List ALL files in `bin/queen-rbee/src/`
- [ ] Count total LOC (use `cloc`)
- [ ] Identify largest files
- [ ] Map directory structure
- [ ] Document module organization

**Command to run:**
```bash
cloc bin/queen-rbee/src --by-file
```

#### Task 1.2: Read Every File
- [ ] Read orchestrator logic
- [ ] Read worker registry
- [ ] Read load balancer
- [ ] Read HTTP server
- [ ] Read all other files

**Questions to Answer:**
1. How does request orchestration work?
2. How does worker selection work?
3. How is the worker registry managed?
4. How are requests forwarded?
5. How are responses streamed?

#### Task 1.3: Document Request Flow
- [ ] Map end-to-end request flow
- [ ] Document worker selection algorithm
- [ ] Identify all HTTP endpoints
- [ ] Map error handling
- [ ] Document timeout handling

**Example Flow:**
```
Client → queen-rbee HTTP → Orchestrator → Load Balancer → Select Worker → Forward Request → Stream Response
```

#### Task 1.4: Analyze Dependencies
- [ ] Read `Cargo.toml` completely
- [ ] List all dependencies
- [ ] Map which modules use which deps
- [ ] Check for shared crate usage
- [ ] Identify missing shared crates

**Expected Shared Crates:**
- `model-catalog` (for model info?)
- `auth-min` (for authentication?)
- `deadline-propagation` (for timeouts?)
- `input-validation` (for request validation?)
- `audit-logging` (for audit logs?)

---

### Day 2-3: Crate Boundary Analysis

#### Task 2.1: Propose 4 Crates

**Initial Proposal (VERIFY & REFINE!):**

1. **`queen-rbee-orchestrator`** (~1,200 LOC estimate)
   - Purpose: Request orchestration logic
   - Responsibilities:
     - Receive inference requests
     - Validate requests
     - Select worker via load balancer
     - Forward request to worker
     - Stream response back
   - Public API: `Orchestrator`, `handle_request()`
   - Dependencies: `queen-rbee-load-balancer`, `queen-rbee-registry`, `reqwest`

2. **`queen-rbee-load-balancer`** (~600 LOC estimate)
   - Purpose: Worker selection algorithm
   - Responsibilities:
     - Track worker load
     - Select best worker
     - Handle worker unavailability
     - Implement selection strategies
   - Public API: `LoadBalancer`, `select_worker()`
   - Dependencies: `queen-rbee-registry`

3. **`queen-rbee-registry`** (~800 LOC estimate)
   - Purpose: Worker registry management
   - Responsibilities:
     - Register workers
     - Deregister workers
     - Track worker health
     - Query available workers
   - Public API: `WorkerRegistry`, `register()`, `query()`
   - Dependencies: `serde`, `chrono`

4. **`queen-rbee-http-server`** (~500 LOC estimate)
   - Purpose: HTTP server and routes
   - Responsibilities:
     - HTTP server setup
     - Route definitions
     - Middleware (auth, CORS)
     - Health endpoints
   - Public API: `start_server()`, `InferenceRoutes`
   - Dependencies: `axum`, `tower`, `queen-rbee-orchestrator`

#### Task 2.2: Justify Each Crate
For each crate:
- [ ] Why separate crate?
- [ ] Single responsibility?
- [ ] Can be tested in isolation?
- [ ] Clear public API?
- [ ] Reasonable size (200-1,200 LOC)?

#### Task 2.3: Map Dependencies Between Crates
```
queen-rbee (binary)
    ├─> queen-rbee-http-server
    │       └─> queen-rbee-orchestrator
    │               ├─> queen-rbee-load-balancer
    │               │       └─> queen-rbee-registry
    │               └─> queen-rbee-registry
    └─> (all crates for CLI if needed)
```

- [ ] Verify no circular dependencies
- [ ] Document data flow
- [ ] Identify shared types
- [ ] Map public APIs

---

### Day 3-4: Shared Crate Analysis

#### Task 3.1: Audit Shared Crate Usage

**Critical Questions:**

1. **`model-catalog`**
   - [ ] Does queen-rbee use it?
   - [ ] Should it use it for model info?
   - [ ] Is model info duplicated?
   - [ ] Should orchestrator query catalog?

2. **`auth-min`**
   - [ ] Is authentication implemented?
   - [ ] Are all endpoints protected?
   - [ ] Is bearer token validation used?
   - [ ] Are there auth gaps?

3. **`deadline-propagation`**
   - [ ] Are deadlines propagated to workers?
   - [ ] Is timeout handling correct?
   - [ ] Are deadlines in HTTP headers?
   - [ ] Should more operations use it?

4. **`input-validation`**
   - [ ] Are inference requests validated?
   - [ ] Are model names validated?
   - [ ] Are parameters validated?
   - [ ] Are there validation gaps?

5. **`audit-logging`**
   - [ ] Are inference requests audited?
   - [ ] Are worker selections logged?
   - [ ] Are failures audited?
   - [ ] Should more operations be logged?

6. **`narration-core`**
   - [ ] Is observability implemented?
   - [ ] Are all operations narrated?
   - [ ] Is tracing consistent?
   - [ ] Should more operations be traced?

#### Task 3.2: Identify Missing Opportunities
- [ ] Is there duplicate HTTP client code?
- [ ] Should there be a shared `rbee-http-client`?
- [ ] Is error handling consistent?
- [ ] Should there be shared error types?
- [ ] Is configuration management shared?

#### Task 3.3: Check rbee-hive Integration
- [ ] Does queen-rbee depend on rbee-hive code?
- [ ] Are there shared types?
- [ ] Should types be in shared crates?
- [ ] Is there duplicate code?

**Example:**
- Worker registration types
- Health check types
- Model info types
- Error types

---

### Day 4-5: Risk Assessment & Migration Strategy

#### Task 4.1: Identify Breaking Changes
- [ ] Will HTTP API change?
- [ ] Will worker registration change?
- [ ] Will request format change?
- [ ] Will response format change?
- [ ] Impact on clients?

#### Task 4.2: Assess Migration Complexity

**For each crate:**

1. **queen-rbee-registry** (PILOT CANDIDATE?)
   - Complexity: Low/Medium/High?
   - Dependencies: Standalone?
   - Test coverage: TBD
   - Risk: Low/Medium/High?
   - Effort: X hours

2. **queen-rbee-load-balancer**
   - Complexity: ?
   - Dependencies: registry only?
   - Test coverage: ?
   - Risk: ?
   - Effort: ?

3. **queen-rbee-orchestrator**
   - Complexity: ?
   - Dependencies: load-balancer, registry?
   - Test coverage: ?
   - Risk: ?
   - Effort: ?

4. **queen-rbee-http-server**
   - Complexity: ?
   - Dependencies: orchestrator?
   - Test coverage: ?
   - Risk: ?
   - Effort: ?

#### Task 4.3: Document Test Strategy
- [ ] Current test coverage
- [ ] Unit tests per crate
- [ ] Integration tests
- [ ] BDD tests per crate
- [ ] E2E tests

**Test Scenarios:**
- Worker selection algorithm
- Request forwarding
- Response streaming
- Error handling
- Timeout handling
- Worker unavailability

#### Task 4.4: Create Migration Plan
1. [ ] Which crate to extract first? (Pilot)
2. [ ] What's the extraction order?
3. [ ] How to verify each step?
4. [ ] What are the checkpoints?
5. [ ] How to rollback if needed?

#### Task 4.5: Document Rollback Plan
- [ ] Checkpoint strategy
- [ ] Verification steps
- [ ] Rollback procedure
- [ ] Go/No-Go criteria

---

### Day 5: Report Writing

#### Task 5.1: Complete Investigation Report

**Required Sections:**
1. Executive Summary
2. Current Architecture (with actual LOC counts!)
3. Proposed Crate Structure (4 crates justified)
4. Shared Crate Analysis (audit findings)
5. Migration Strategy (step-by-step)
6. Risk Assessment (risks and mitigation)
7. Recommendations (Go/No-Go)

#### Task 5.2: Get Peer Review
- [ ] Share with TEAM-131 (rbee-hive)
- [ ] Share with TEAM-133 (llm-worker-rbee)
- [ ] Share with TEAM-134 (rbee-keeper)
- [ ] Incorporate feedback
- [ ] Finalize report

---

## 🚨 CRITICAL QUESTIONS TO ANSWER

### Architecture Questions:
1. **Is orchestrator truly the core?** Or is it just glue code?
2. **Can load balancer be isolated?** Or does it need registry access?
3. **Is registry independent?** Or does it depend on other modules?
4. **Can HTTP layer be separated?** Or is it tightly coupled?
5. **Are there circular dependencies?**

### Integration Questions:
1. **How does queen-rbee talk to rbee-hive?** HTTP? Shared types?
2. **How does queen-rbee talk to llm-worker-rbee?** HTTP? SSE?
3. **Are there shared types that should be in shared crates?**
4. **Is there duplicate code across binaries?**
5. **Should there be a shared `rbee-types` crate?**

### Shared Crate Questions:
1. **Should queen-rbee use `model-catalog`?**
2. **Is authentication using `auth-min`?**
3. **Are deadlines using `deadline-propagation`?**
4. **Is validation using `input-validation`?**
5. **Is auditing using `audit-logging`?**

### Migration Questions:
1. **What's the safest crate to start with?** (Pilot)
2. **What's the riskiest crate?**
3. **Will this break integration with rbee-hive?**
4. **Will this break integration with llm-worker-rbee?**
5. **How long will migration take?**

---

## 📋 DELIVERABLES

### Required Outputs:
- [ ] **Investigation Report** (TEAM_132_queen-rbee_INVESTIGATION.md)
- [ ] **Actual LOC counts** (not estimates!)
- [ ] **File structure map** (complete)
- [ ] **Dependency graph** (visual)
- [ ] **Request flow diagram** (end-to-end)
- [ ] **Crate proposal** (4 crates justified)
- [ ] **Shared crate audit** (findings)
- [ ] **Migration plan** (step-by-step)
- [ ] **Risk assessment** (risks and mitigation)
- [ ] **Peer review** (feedback incorporated)

### Quality Criteria:
- ✅ Every file analyzed
- ✅ Actual LOC counts (not estimates!)
- ✅ All dependencies mapped
- ✅ All shared crates audited
- ✅ Request flow documented
- ✅ Clear recommendations
- ✅ Peer-reviewed

---

## 🎯 SUCCESS CRITERIA

### Investigation Complete When:
- [ ] All files analyzed
- [ ] Actual LOC counted
- [ ] 4 crates proposed and justified
- [ ] All shared crates audited
- [ ] Integration points documented
- [ ] Migration plan complete
- [ ] Risks assessed
- [ ] Report peer-reviewed
- [ ] Go/No-Go decision made

---

## 📞 NEED HELP?

- **Slack:** `#team-132-queen-rbee`
- **Daily Standup:** 9:00 AM
- **Team Lead:** [Name]
- **Peer Review:** TEAM-131, TEAM-133, TEAM-134

---

## ✅ READY TO START!

**First: Count actual LOC and map file structure!**

**Remember: INVESTIGATION ONLY - NO CODE CHANGES!**

**TEAM-132: Let's decompose queen-rbee! 🚀**
