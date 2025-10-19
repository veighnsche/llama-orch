# TEAM-134: rbee-keeper INVESTIGATION

**Binary:** `bin/rbee-keeper`  
**Team:** TEAM-134  
**Phase:** Investigation (Week 1)  
**Status:** 🔍 IN PROGRESS

---

## 🎯 MISSION

**Investigate `rbee-keeper` binary and propose decomposition into 5 focused crates.**

**NO CODE CHANGES! INVESTIGATION ONLY!**

---

## 📊 CURRENT STATE

### Binary Stats
- **Total LOC:** 1,252 (verified via cloc)
- **Files:** 13
- **Purpose:** Operator CLI tool (`rbee` command)
- **Test coverage:** TBD (investigate!)
- **Dependencies:** ssh2, reqwest, clap, etc.

### What is rbee-keeper?
- **IS** the `rbee` CLI command (for operators/humans)
- **NOT** the orchestrator daemon (that's `queen-rbee`)
- **NOT** the pool manager (that's `rbee-hive`)
- SSH-based remote control
- Manages pools, workers, models
- Installs and configures systems

### File Structure (Known)
```
bin/rbee-keeper/src/
├── main.rs (12 LOC)
├── cli.rs (175 LOC)
├── commands/
│   ├── setup.rs (222 LOC) ⚠️ LARGEST
│   ├── workers.rs (197 LOC)
│   ├── infer.rs (186 LOC)
│   ├── install.rs (98 LOC)
│   ├── hive.rs (84 LOC)
│   ├── logs.rs (24 LOC)
│   └── mod.rs (6 LOC)
├── pool_client.rs (115 LOC)
├── queen_lifecycle.rs (75 LOC)
├── config.rs (44 LOC)
└── ssh.rs (14 LOC)
```

---

## 🔍 INVESTIGATION TASKS

### Day 1-2: Deep Code Analysis

#### Task 1.1: Read Every File
- [ ] Read `commands/setup.rs` - Setup command (largest!)
- [ ] Read `commands/workers.rs` - Worker management
- [ ] Read `commands/infer.rs` - Inference command
- [ ] Read `commands/install.rs` - Installation
- [ ] Read `commands/hive.rs` - Hive management
- [ ] Read `commands/logs.rs` - Log viewing
- [ ] Read `cli.rs` - CLI definition
- [ ] Read `pool_client.rs` - Pool communication
- [ ] Read `queen_lifecycle.rs` - Queen management
- [ ] Read `config.rs` - Configuration
- [ ] Read `ssh.rs` - SSH operations
- [ ] Read `main.rs` - Entry point

#### Task 1.2: Document Command Flow
For each command:
- [ ] `rbee setup` - What does it do?
- [ ] `rbee workers spawn` - How does it work?
- [ ] `rbee workers list` - What info does it show?
- [ ] `rbee infer` - How does inference work?
- [ ] `rbee install` - What gets installed?
- [ ] `rbee hive start/stop` - How does it manage hive?
- [ ] `rbee logs` - How are logs retrieved?

**Example Flow:**
```
rbee workers spawn <model> 
  → Parse CLI args
  → Validate input
  → SSH to pool machine
  → Call rbee-hive HTTP API
  → Stream response
  → Display status
```

#### Task 1.3: Analyze Dependencies
- [ ] Read `Cargo.toml` completely
- [ ] List all dependencies
- [ ] Map SSH usage (ssh2 crate)
- [ ] Map HTTP client usage (reqwest)
- [ ] Check shared crate usage
- [ ] Identify missing shared crates

**Current Dependencies:**
```toml
clap = "4.5"
ssh2 = { workspace = true }
reqwest = { workspace = true }
input-validation = { path = "../shared-crates/input-validation" }
```

**Questions:**
1. Is `input-validation` used everywhere?
2. Should more shared crates be used?
3. Is error handling consistent?
4. Is configuration management shared?

---

### Day 2-3: Crate Boundary Analysis

#### Task 2.1: Propose 5 Crates

**Initial Proposal (VERIFY & REFINE!):**

1. **`rbee-keeper-commands`** (703 LOC)
   - Purpose: All CLI commands
   - Files: `commands/*.rs` (all command implementations)
   - Responsibilities:
     - `setup` command
     - `workers` commands
     - `infer` command
     - `install` command
     - `hive` commands
     - `logs` command
   - Public API: Command implementations
   - Dependencies: `clap`, `rbee-keeper-pool-client`, `rbee-keeper-ssh-client`

2. **`rbee-keeper-pool-client`** (115 LOC)
   - Purpose: HTTP client for rbee-hive communication
   - Files: `pool_client.rs`
   - Responsibilities:
     - HTTP requests to rbee-hive
     - Worker spawn requests
     - Worker list queries
     - Model management
   - Public API: `PoolClient`, `spawn_worker()`, `list_workers()`
   - Dependencies: `reqwest`, `serde`

3. **`rbee-keeper-ssh-client`** (14 LOC)
   - Purpose: SSH operations
   - Files: `ssh.rs`
   - Responsibilities:
     - SSH connection management
     - Remote command execution
     - File transfer (if needed)
   - Public API: `SshClient`, `execute_remote()`
   - Dependencies: `ssh2`

4. **`rbee-keeper-queen-lifecycle`** (75 LOC)
   - Purpose: Queen-rbee lifecycle management
   - Files: `queen_lifecycle.rs`
   - Responsibilities:
     - Start queen-rbee
     - Stop queen-rbee
     - Check queen status
     - Queen configuration
   - Public API: `start_queen()`, `stop_queen()`, `queen_status()`
   - Dependencies: `reqwest`

5. **`rbee-keeper-config`** (44 LOC)
   - Purpose: Configuration management
   - Files: `config.rs`
   - Responsibilities:
     - Load configuration
     - Save configuration
     - Validate configuration
     - Default values
   - Public API: `Config`, `load()`, `save()`
   - Dependencies: `serde`, `toml`

#### Task 2.2: Alternative Structure

**Should commands be split further?**

Option A: One `commands` crate (703 LOC)
- Pros: Simple, all commands together
- Cons: Large, hard to test individually

Option B: Split by command category
- `rbee-keeper-commands-setup` (222 LOC)
- `rbee-keeper-commands-workers` (197 LOC)
- `rbee-keeper-commands-infer` (186 LOC)
- `rbee-keeper-commands-install` (98 LOC)
- `rbee-keeper-commands-hive` (84 LOC)
- `rbee-keeper-commands-logs` (24 LOC)

**Questions:**
1. Is 703 LOC too large for one crate?
2. Would splitting help testing?
3. Would splitting help reusability?
4. What's the right balance?

#### Task 2.3: Justify Each Crate
For each proposed crate:
- [ ] Why separate crate?
- [ ] Single responsibility?
- [ ] Can be tested in isolation?
- [ ] Clear public API?
- [ ] Reasonable size?

#### Task 2.4: Map Dependencies Between Crates
```
rbee-keeper (binary)
    ├─> rbee-keeper-cli (175 LOC)
    │       └─> rbee-keeper-commands
    │               ├─> rbee-keeper-pool-client
    │               ├─> rbee-keeper-ssh-client
    │               ├─> rbee-keeper-queen-lifecycle
    │               └─> rbee-keeper-config
    └─> rbee-keeper-config
```

- [ ] Verify no circular dependencies
- [ ] Document data flow
- [ ] Identify shared types
- [ ] Map public APIs

---

### Day 3-4: Shared Crate Analysis

#### Task 3.1: Audit Shared Crate Usage

**Current Usage:**

1. **`input-validation`**
   - [ ] Where is it used?
   - [ ] Are all CLI args validated?
   - [ ] Are model names validated?
   - [ ] Are pool names validated?
   - [ ] Should more commands use it?

**Missing Shared Crates?**

2. **`auth-min`** (NOT USED)
   - [ ] Should CLI authenticate?
   - [ ] Should API keys be managed?
   - [ ] Is authentication needed?

3. **`secrets-management`** (NOT USED)
   - [ ] Should SSH keys be managed?
   - [ ] Should API keys be stored?
   - [ ] Is secrets management needed?

4. **`audit-logging`** (NOT USED)
   - [ ] Should CLI actions be audited?
   - [ ] Should commands be logged?
   - [ ] Is audit trail needed?

5. **`narration-core`** (NOT USED)
   - [ ] Should CLI be observable?
   - [ ] Should commands be traced?
   - [ ] Is observability needed?

#### Task 3.2: Identify Missing Opportunities
- [ ] Is there duplicate HTTP client code?
- [ ] Should there be shared `rbee-http-client`?
- [ ] Is error handling consistent?
- [ ] Should there be shared error types?
- [ ] Is SSH code reusable?
- [ ] Should there be shared `rbee-ssh-client`?

**Questions:**
1. Does rbee-hive also use SSH?
2. Could SSH client be shared?
3. Does queen-rbee also use HTTP client?
4. Could HTTP client be shared?

#### Task 3.3: Check Integration Points
- [ ] How does CLI talk to rbee-hive? (HTTP)
- [ ] How does CLI talk to queen-rbee? (HTTP)
- [ ] How does CLI SSH to pool machines?
- [ ] Are there shared types?
- [ ] Should types be in shared crates?

**Example Shared Types:**
- Worker spawn request
- Worker list response
- Model info
- Pool configuration

---

### Day 4-5: Risk Assessment & Migration Strategy

#### Task 4.1: Identify Breaking Changes
- [ ] Will CLI commands change?
- [ ] Will command arguments change?
- [ ] Will output format change?
- [ ] Will configuration format change?
- [ ] Impact on users?

#### Task 4.2: Assess Migration Complexity

**For each crate:**

1. **rbee-keeper-config** (PILOT CANDIDATE?)
   - Complexity: Low (simple config)
   - Dependencies: Standalone
   - Test coverage: TBD
   - Risk: Low
   - Effort: 4 hours

2. **rbee-keeper-ssh-client**
   - Complexity: Low (simple wrapper)
   - Dependencies: ssh2
   - Test coverage: TBD (hard to test SSH!)
   - Risk: Low
   - Effort: 4 hours

3. **rbee-keeper-pool-client**
   - Complexity: Medium (HTTP client)
   - Dependencies: reqwest
   - Test coverage: TBD
   - Risk: Medium (integration with rbee-hive)
   - Effort: 8 hours

4. **rbee-keeper-queen-lifecycle**
   - Complexity: Medium (lifecycle management)
   - Dependencies: reqwest
   - Test coverage: TBD
   - Risk: Medium (integration with queen-rbee)
   - Effort: 8 hours

5. **rbee-keeper-commands**
   - Complexity: High (all commands)
   - Dependencies: All other crates
   - Test coverage: TBD
   - Risk: High (user-facing!)
   - Effort: 20 hours

**Total Effort:** ~44 hours (1 week for 1 person)

#### Task 4.3: Document Test Strategy
- [ ] Current test coverage
- [ ] Unit tests per crate
- [ ] Integration tests (with mock rbee-hive?)
- [ ] BDD tests per crate
- [ ] CLI acceptance tests

**Critical Tests:**
- All CLI commands work
- SSH connection works
- HTTP client works
- Configuration loading works
- Error handling works
- Help text is correct

#### Task 4.4: Create Migration Plan

**Recommended Order:**
1. **rbee-keeper-config** (PILOT - simple, low risk)
2. **rbee-keeper-ssh-client** (simple, standalone)
3. **rbee-keeper-pool-client** (medium, important)
4. **rbee-keeper-queen-lifecycle** (medium, important)
5. **rbee-keeper-commands** (complex, depends on all others)

**Why this order?**
- Start with simplest (config)
- Build confidence with standalone crates
- Save complex commands for last
- Verify integration at each step

#### Task 4.5: Document Rollback Plan
- [ ] Checkpoint after each crate
- [ ] Verification steps (run CLI commands!)
- [ ] Rollback procedure
- [ ] Go/No-Go criteria per crate

---

### Day 5: Report Writing

#### Task 5.1: Complete Investigation Report

**Required Sections:**
1. Executive Summary
2. Current Architecture (1,252 LOC)
3. Proposed Crate Structure (5 crates)
4. Alternative Structures (split commands?)
5. Shared Crate Analysis (missing opportunities?)
6. Integration Points (rbee-hive, queen-rbee)
7. Migration Strategy (recommended order)
8. Risk Assessment
9. Recommendations

#### Task 5.2: Get Peer Review
- [ ] Share with TEAM-131 (rbee-hive)
- [ ] Share with TEAM-132 (queen-rbee)
- [ ] Share with TEAM-133 (llm-worker-rbee)
- [ ] Incorporate feedback
- [ ] Finalize report

---

## 🚨 CRITICAL QUESTIONS TO ANSWER

### Architecture Questions:
1. **Should commands be one crate or split?** 703 LOC in one crate?
2. **Is SSH client reusable?** Could rbee-hive use it?
3. **Is HTTP client reusable?** Could be shared across binaries?
4. **Should there be shared `rbee-http-client`?**
5. **Are there circular dependencies?**

### Shared Crate Questions:
1. **Should CLI use `auth-min`?** For API keys?
2. **Should CLI use `secrets-management`?** For SSH keys?
3. **Should CLI use `audit-logging`?** For command audit?
4. **Should CLI use `narration-core`?** For observability?
5. **Is `input-validation` used everywhere?**

### Integration Questions:
1. **How does CLI talk to rbee-hive?** HTTP API?
2. **How does CLI talk to queen-rbee?** HTTP API?
3. **Are there shared types?** With rbee-hive? queen-rbee?
4. **Should types be in shared crates?**
5. **Is there duplicate code?**

### User Experience Questions:
1. **Will CLI commands change?** Breaking changes?
2. **Will output format change?** JSON? Text?
3. **Will configuration change?** File format?
4. **Impact on users?** Migration guide needed?
5. **Backward compatibility?** Support old commands?

---

## 📋 DELIVERABLES

### Required Outputs:
- [ ] **Investigation Report** (TEAM_134_rbee-keeper_INVESTIGATION.md)
- [ ] **Complete file analysis** (all 1,252 LOC)
- [ ] **Command flow diagrams** (for each command)
- [ ] **Dependency graph**
- [ ] **5 crate proposals** (or alternative structure)
- [ ] **Shared crate audit** (missing opportunities)
- [ ] **Integration analysis** (with rbee-hive, queen-rbee)
- [ ] **Migration plan** (recommended order)
- [ ] **Risk assessment**
- [ ] **User impact analysis**
- [ ] **Peer review**

### Quality Criteria:
- ✅ Every file analyzed
- ✅ All 1,252 LOC accounted for
- ✅ All commands documented
- ✅ All dependencies mapped
- ✅ All shared crates audited
- ✅ User impact assessed
- ✅ Clear recommendations
- ✅ Peer-reviewed

---

## 🎯 SUCCESS CRITERIA

### Investigation Complete When:
- [ ] All 1,252 LOC analyzed
- [ ] 5 crates proposed (or alternative)
- [ ] All shared crates audited
- [ ] Integration points documented
- [ ] User impact assessed
- [ ] Migration plan complete
- [ ] Risks assessed
- [ ] Report peer-reviewed
- [ ] Go/No-Go decision made

---

## 📞 NEED HELP?

- **Slack:** `#team-134-rbee-keeper`
- **Daily Standup:** 9:00 AM
- **Team Lead:** [Name]
- **Peer Review:** TEAM-131, TEAM-132, TEAM-133

---

## ✅ READY TO START!

**First: Read all commands and document what they do!**

**Focus: User experience and integration points!**

**Remember: INVESTIGATION ONLY - NO CODE CHANGES!**

**TEAM-134: Let's decompose rbee-keeper! 🚀**
