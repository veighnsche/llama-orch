# TEAM-131: rbee-hive INVESTIGATION

**Binary:** `bin/rbee-hive`  
**Team:** TEAM-131  
**Phase:** Investigation (Week 1)  
**Status:** 🔍 IN PROGRESS

---

## 🎯 MISSION

**Investigate `rbee-hive` binary and propose decomposition into 10 focused crates.**

**NO CODE CHANGES! INVESTIGATION ONLY!**

---

## 📊 CURRENT STATE

### Binary Stats
- **Total LOC:** 4,184
- **Files:** 33
- **Largest file:** `registry.rs` (492 LOC)
- **Test coverage:** TBD (investigate!)
- **Dependencies:** 20+ external crates

### File Structure
```
bin/rbee-hive/src/
├── main.rs (15 LOC)
├── lib.rs (7 LOC)
├── cli.rs (67 LOC)
├── registry.rs (492 LOC) ⚠️ LARGEST
├── http/
│   ├── workers.rs (407 LOC)
│   ├── models.rs (184 LOC)
│   ├── middleware/auth.rs (177 LOC)
│   ├── heartbeat.rs (124 LOC)
│   ├── server.rs (103 LOC)
│   ├── routes.rs (84 LOC)
│   ├── health.rs (46 LOC)
│   ├── metrics.rs (44 LOC)
│   └── mod.rs (10 LOC)
├── provisioner/
│   ├── operations.rs (158 LOC)
│   ├── catalog.rs (133 LOC)
│   ├── download.rs (115 LOC)
│   ├── types.rs (67 LOC)
│   └── mod.rs (5 LOC)
├── commands/
│   ├── worker.rs (188 LOC)
│   ├── models.rs (150 LOC)
│   ├── daemon.rs (124 LOC)
│   ├── detect.rs (20 LOC)
│   ├── status.rs (11 LOC)
│   └── mod.rs (5 LOC)
├── shutdown.rs (248 LOC)
├── resources.rs (247 LOC)
├── monitor.rs (210 LOC)
├── metrics.rs (178 LOC)
├── restart.rs (162 LOC)
├── download_tracker.rs (151 LOC)
├── timeout.rs (93 LOC)
└── worker_provisioner.rs (67 LOC)
```

---

## 🔍 INVESTIGATION TASKS

### Day 1-2: Deep Code Analysis

#### Task 1.1: Read Every File
- [ ] Read `registry.rs` - Worker registry logic
- [ ] Read `http/workers.rs` - HTTP worker endpoints
- [ ] Read `http/middleware/auth.rs` - Authentication
- [ ] Read `shutdown.rs` - Shutdown orchestration
- [ ] Read `resources.rs` - Resource management
- [ ] Read `monitor.rs` - Health monitoring
- [ ] Read `provisioner/*.rs` - Model provisioning
- [ ] Read `commands/*.rs` - CLI commands
- [ ] Read `metrics.rs` - Prometheus metrics
- [ ] Read `restart.rs` - Worker restart logic
- [ ] Read `download_tracker.rs` - Download tracking
- [ ] Read all other files

#### Task 1.2: Document Module Dependencies
- [ ] Create dependency graph
- [ ] Identify circular dependencies
- [ ] Map data flow between modules
- [ ] Document public vs private APIs
- [ ] Identify tight coupling

**Questions to Answer:**
1. Which modules depend on `registry.rs`?
2. Is `http/` tightly coupled to business logic?
3. Can `provisioner/` be isolated?
4. Are there circular dependencies?
5. What's the data flow for worker spawn?

#### Task 1.3: Analyze External Dependencies
- [ ] List all `Cargo.toml` dependencies
- [ ] Map which modules use which dependencies
- [ ] Identify dependency conflicts
- [ ] Check for unused dependencies
- [ ] Document version constraints

**Current Dependencies (from Cargo.toml):**
```toml
hive-core = { path = "../shared-crates/hive-core" }
model-catalog = { path = "../shared-crates/model-catalog" }
gpu-info = { path = "../shared-crates/gpu-info" }
auth-min = { path = "../shared-crates/auth-min" }
secrets-management = { path = "../shared-crates/secrets-management" }
input-validation = { path = "../shared-crates/input-validation" }
audit-logging = { path = "../shared-crates/audit-logging" }
deadline-propagation = { path = "../shared-crates/deadline-propagation" }
axum, tokio, tower, serde, reqwest, prometheus, etc.
```

---

### Day 2-3: Crate Boundary Analysis

#### Task 2.1: Propose 10 Crates

**Initial Proposal (VERIFY & REFINE!):**

1. **`rbee-hive-registry`** (492 LOC)
   - Purpose: Worker registry management
   - Files: `registry.rs`
   - Public API: `WorkerRegistry`, `register_worker()`, `deregister_worker()`
   - Dependencies: `serde`, `chrono`

2. **`rbee-hive-http-server`** (407 LOC)
   - Purpose: HTTP routes and handlers
   - Files: `http/workers.rs`, `http/routes.rs`, `http/server.rs`
   - Public API: `start_server()`, `WorkerRoutes`
   - Dependencies: `axum`, `tower`, `rbee-hive-registry`

3. **`rbee-hive-http-middleware`** (177 LOC)
   - Purpose: HTTP middleware (auth, CORS, etc.)
   - Files: `http/middleware/auth.rs`
   - Public API: `auth_middleware()`, `cors_middleware()`
   - Dependencies: `auth-min`, `axum`

4. **`rbee-hive-provisioner`** (473 LOC)
   - Purpose: Model provisioning and download
   - Files: `provisioner/*.rs`
   - Public API: `provision_model()`, `download_model()`
   - Dependencies: `model-catalog`, `reqwest`

5. **`rbee-hive-monitor`** (210 LOC)
   - Purpose: Worker health monitoring
   - Files: `monitor.rs`
   - Public API: `start_monitor()`, `check_worker_health()`
   - Dependencies: `reqwest`, `tokio`

6. **`rbee-hive-resources`** (247 LOC)
   - Purpose: Resource management (RAM, VRAM, CPU)
   - Files: `resources.rs`
   - Public API: `check_resources()`, `ResourceInfo`
   - Dependencies: `sysinfo`, `gpu-info`

7. **`rbee-hive-shutdown`** (248 LOC)
   - Purpose: Graceful shutdown orchestration
   - Files: `shutdown.rs`
   - Public API: `shutdown_worker()`, `shutdown_all()`
   - Dependencies: `reqwest`, `tokio`

8. **`rbee-hive-metrics`** (222 LOC)
   - Purpose: Prometheus metrics
   - Files: `metrics.rs`, `http/metrics.rs`
   - Public API: `register_metrics()`, `update_metrics()`
   - Dependencies: `prometheus`

9. **`rbee-hive-restart`** (162 LOC)
   - Purpose: Worker restart logic
   - Files: `restart.rs`
   - Public API: `restart_worker()`, `restart_with_jitter()`
   - Dependencies: `tokio`, `rand`

10. **`rbee-hive-cli`** (498 LOC)
    - Purpose: CLI commands
    - Files: `cli.rs`, `commands/*.rs`
    - Public API: `Cli`, `run_command()`
    - Dependencies: `clap`, all other rbee-hive crates

#### Task 2.2: Justify Each Crate
For each proposed crate:
- [ ] Why does this deserve its own crate?
- [ ] What's the single responsibility?
- [ ] Can it be tested in isolation?
- [ ] What's the public API surface?
- [ ] What are the dependencies?

#### Task 2.3: Identify Shared Code
- [ ] Look for code that should be in shared crates
- [ ] Check for duplicate code across modules
- [ ] Identify utility functions
- [ ] Document opportunities for consolidation

---

### Day 3-4: Shared Crate Analysis

#### Task 3.1: Audit Current Shared Crate Usage

**For each shared crate, answer:**

1. **`hive-core`**
   - [ ] Where is it used?
   - [ ] What types/functions are used?
   - [ ] Are we using it fully?
   - [ ] Should more code use it?

2. **`model-catalog`**
   - [ ] Where is it used?
   - [ ] Is provisioner using it correctly?
   - [ ] Are there missing features?
   - [ ] Should it be extended?

3. **`gpu-info`**
   - [ ] Where is it used?
   - [ ] Is resources.rs using it fully?
   - [ ] Are we detecting all GPU info?
   - [ ] Should it be enhanced?

4. **`auth-min`**
   - [ ] Where is it used?
   - [ ] Is middleware using it correctly?
   - [ ] Are all endpoints protected?
   - [ ] Are there auth gaps?

5. **`secrets-management`**
   - [ ] Where is it used?
   - [ ] Are secrets properly managed?
   - [ ] Are there hardcoded secrets?
   - [ ] Should more code use it?

6. **`input-validation`**
   - [ ] Where is it used?
   - [ ] Are all inputs validated?
   - [ ] Are there validation gaps?
   - [ ] Should more endpoints use it?

7. **`audit-logging`**
   - [ ] Where is it used?
   - [ ] Are all actions audited?
   - [ ] Are there audit gaps?
   - [ ] Should more operations be logged?

8. **`deadline-propagation`**
   - [ ] Where is it used?
   - [ ] Are deadlines propagated correctly?
   - [ ] Are there timeout issues?
   - [ ] Should more operations use it?

#### Task 3.2: Identify Missing Opportunities
- [ ] Find duplicate code that should be in shared crates
- [ ] Identify new shared crates to create
- [ ] Document code that should be consolidated
- [ ] Propose shared crate enhancements

**Example Questions:**
- Is there HTTP client code that should be shared?
- Is there error handling that should be shared?
- Is there logging that should be standardized?
- Is there configuration that should be shared?

#### Task 3.3: Check for Version Conflicts
- [ ] Verify all shared crates use workspace versions
- [ ] Check for dependency conflicts
- [ ] Document version mismatches
- [ ] Propose version alignment strategy

---

### Day 4-5: Risk Assessment & Migration Strategy

#### Task 4.1: Identify Breaking Changes
- [ ] List all public APIs that will change
- [ ] Identify modules that will move
- [ ] Document import path changes
- [ ] Assess impact on tests
- [ ] Assess impact on other binaries

**Questions:**
1. Does `queen-rbee` depend on `rbee-hive` code?
2. Does `llm-worker-rbee` depend on `rbee-hive` code?
3. Are there shared types that need to move?
4. Will BDD tests break?

#### Task 4.2: Assess Migration Complexity
For each proposed crate:
- [ ] Complexity: Low/Medium/High
- [ ] Dependencies on other crates
- [ ] Test coverage
- [ ] Risk level
- [ ] Estimated effort (hours)

**Example:**
```
rbee-hive-registry:
  Complexity: Medium
  Dependencies: None (standalone)
  Test coverage: 60% (needs improvement)
  Risk: Low (well-isolated)
  Effort: 8 hours
```

#### Task 4.3: Document Test Strategy
- [ ] Current test coverage per module
- [ ] Proposed test strategy per crate
- [ ] BDD test migration plan
- [ ] Integration test strategy
- [ ] Test gaps to fill

#### Task 4.4: Create Migration Plan
Step-by-step plan:
1. [ ] Create crate directories
2. [ ] Write Cargo.toml files
3. [ ] Move code (in what order?)
4. [ ] Update imports
5. [ ] Fix compilation errors
6. [ ] Run tests
7. [ ] Create BDD suites
8. [ ] Verify integration

#### Task 4.5: Document Rollback Plan
- [ ] How to rollback if migration fails?
- [ ] What's the checkpoint strategy?
- [ ] How to verify each step?
- [ ] What are the go/no-go criteria?

---

### Day 5: Report Writing

#### Task 5.1: Write Investigation Report
Use this template:

```markdown
# TEAM-131 rbee-hive INVESTIGATION REPORT

## Executive Summary
- Current: 4,184 LOC monolithic binary
- Proposed: 10 focused crates
- Benefits: 93% faster compilation, perfect isolation
- Risks: [list key risks]
- Recommendation: GO / NO-GO

## Current Architecture
[Detailed analysis]

## Proposed Crate Structure
[10 crates with justification]

## Shared Crate Analysis
[Audit findings and recommendations]

## Migration Strategy
[Step-by-step plan]

## Risk Assessment
[Risks and mitigation]

## Recommendations
[Next steps]
```

#### Task 5.2: Get Peer Review
- [ ] Share report with TEAM-132
- [ ] Share report with TEAM-133
- [ ] Share report with TEAM-134
- [ ] Incorporate feedback
- [ ] Finalize report

---

## 🚨 CRITICAL QUESTIONS TO ANSWER

### Architecture Questions:
1. **Is `registry.rs` truly standalone?** Or does it have hidden dependencies?
2. **Can HTTP layer be separated from business logic?** Or is it tightly coupled?
3. **Is provisioner independent?** Or does it depend on registry/http?
4. **Can monitor run independently?** Or does it need registry access?
5. **Are there circular dependencies?** Between which modules?

### Shared Crate Questions:
1. **Are we using `model-catalog` to its full potential?**
2. **Should more code use `auth-min`?**
3. **Are there duplicate HTTP client patterns?**
4. **Should we create a shared `rbee-http-client` crate?**
5. **Is error handling consistent across modules?**

### Migration Questions:
1. **What's the riskiest crate to extract?**
2. **What's the safest crate to start with?** (Pilot candidate)
3. **Are there dependencies on other binaries?**
4. **Will this break existing tests?**
5. **How long will migration take?** (Realistic estimate)

---

## 📋 DELIVERABLES

### Required Outputs:
- [ ] **Investigation Report** (TEAM_131_rbee-hive_INVESTIGATION.md)
- [ ] **Dependency Graph** (visual diagram)
- [ ] **Crate Proposal** (10 crates with justification)
- [ ] **Shared Crate Audit** (findings and recommendations)
- [ ] **Migration Plan** (step-by-step)
- [ ] **Risk Assessment** (risks and mitigation)
- [ ] **Test Strategy** (coverage and gaps)
- [ ] **Peer Review** (feedback from other teams)

### Quality Criteria:
- ✅ Every file analyzed
- ✅ All dependencies mapped
- ✅ All shared crates audited
- ✅ Realistic effort estimates
- ✅ Clear recommendations
- ✅ Peer-reviewed

---

## 🎯 SUCCESS CRITERIA

### Investigation Complete When:
- [ ] All 4,184 LOC analyzed
- [ ] 10 crates proposed and justified
- [ ] All shared crates audited
- [ ] Migration plan documented
- [ ] Risks assessed and mitigated
- [ ] Report peer-reviewed
- [ ] Go/No-Go decision made

---

## 📞 NEED HELP?

- **Slack:** `#team-131-rbee-hive`
- **Daily Standup:** 9:00 AM
- **Team Lead:** [Name]
- **Peer Review:** TEAM-132, TEAM-133, TEAM-134

---

## ✅ READY TO START!

**Read this guide completely, then begin Day 1 analysis!**

**Remember: INVESTIGATION ONLY - NO CODE CHANGES!**

**TEAM-131: Let's decompose rbee-hive! 🚀**
