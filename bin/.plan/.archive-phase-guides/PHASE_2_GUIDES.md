# PHASE 2: Queen Crate Behavior Discovery

**Teams:** TEAM-220, TEAM-221, TEAM-222  
**Duration:** 1 day (all teams work concurrently)  
**Output:** 3 behavior inventory documents

---

## TEAM-220: hive-lifecycle

**Component:** `bin/15_queen_rbee_crates/hive-lifecycle`  
**Complexity:** High  
**Output:** `.plan/TEAM_220_HIVE_LIFECYCLE_BEHAVIORS.md`

### Investigation Areas

#### 1. Public API (src/lib.rs)
- Document ALL exported functions
- Document request/response types
- Document error types

#### 2. Hive Operations
**Files:** `src/list.rs`, `src/get.rs`, `src/status.rs`, `src/start.rs`, `src/stop.rs`, `src/install.rs`, `src/uninstall.rs`, `src/capabilities.rs`, `src/ssh_test.rs`

**Tasks:**
- Document each operation's behavior
- Document error handling per operation
- Document narration emission points
- Document job_id propagation
- Document SSH integration points

#### 3. Validation (src/validation.rs)
- Document `validate_hive_exists()` behavior
- Document localhost special case
- Document hives.conf auto-generation
- Document error messages

#### 4. Types (src/types.rs)
- Document all request types
- Document all response types
- Document field validation

#### 5. Hive Client (src/hive_client.rs)
- Document HTTP client usage
- Document capabilities discovery
- Document timeout handling

#### 6. Critical Behaviors
- Binary resolution logic (config → debug → release)
- Health polling (exponential backoff)
- Graceful shutdown (SIGTERM → SIGKILL)
- Capabilities caching with TimeoutEnforcer
- SSH vs localhost detection

#### 7. Existing Tests
```bash
find bin/15_queen_rbee_crates/hive-lifecycle -name "*.rs" | grep test
find bin/15_queen_rbee_crates/hive-lifecycle/bdd -name "*.feature"
```

### Expected LOC
~1,629 LOC (from TEAM-210 through TEAM-215)

### Key Files
- `src/lib.rs` - Module structure
- `src/types.rs` - Request/response types
- `src/validation.rs` - Validation helpers
- `src/start.rs` - HiveStart (most complex)
- `src/stop.rs` - HiveStop
- `src/install.rs` - HiveInstall
- `src/uninstall.rs` - HiveUninstall
- `src/list.rs` - HiveList
- `src/get.rs` - HiveGet
- `src/status.rs` - HiveStatus
- `src/capabilities.rs` - HiveRefreshCapabilities
- `src/hive_client.rs` - HTTP client
- `src/ssh_test.rs` - SSH testing

---

## TEAM-221: hive-registry

**Component:** `bin/15_queen_rbee_crates/hive-registry`  
**Complexity:** Medium  
**Output:** `.plan/TEAM_221_HIVE_REGISTRY_BEHAVIORS.md`

### Investigation Areas

#### 1. Registry Interface
- Document public API
- Document hive registration
- Document hive lookup
- Document state tracking
- Document persistence (if any)

#### 2. State Management
- Document hive states
- Document state transitions
- Document invariants

#### 3. Heartbeat Integration
- Document heartbeat handling
- Document staleness detection
- Document cleanup logic

#### 4. Data Structures
- Document hive metadata
- Document indexing strategy
- Document concurrent access

#### 5. Error Handling
- Document error types
- Document validation errors
- Document conflict resolution

#### 6. Existing Tests
```bash
find bin/15_queen_rbee_crates/hive-registry -name "*.rs" | grep test
find bin/15_queen_rbee_crates/hive-registry/bdd -name "*.feature"
```

### Key Files
- `src/lib.rs` - Public API
- `Cargo.toml` - Dependencies

---

## TEAM-222: ssh-client

**Component:** `bin/15_queen_rbee_crates/ssh-client`  
**Complexity:** Medium  
**Output:** `.plan/TEAM_222_SSH_CLIENT_BEHAVIORS.md`

### Investigation Areas

#### 1. SSH Connection
- Document connection establishment
- Document authentication methods
- Document connection pooling (if any)
- Document timeout handling

#### 2. Command Execution
- Document remote command execution
- Document stdout/stderr capture
- Document exit code handling
- Document error detection

#### 3. File Transfer
- Document file upload (if any)
- Document file download (if any)
- Document transfer verification

#### 4. Error Handling
- Document connection errors
- Document authentication errors
- Document timeout errors
- Document network errors

#### 5. Configuration
- Document SSH config sources
- Document key management
- Document known_hosts handling

#### 6. Existing Tests
```bash
find bin/15_queen_rbee_crates/ssh-client -name "*.rs" | grep test
find bin/15_queen_rbee_crates/ssh-client/bdd -name "*.feature"
```

### Key Files
- `src/lib.rs` - Public API
- `Cargo.toml` - Dependencies

---

## Deliverables Template

Each team must produce:

```markdown
# [COMPONENT] BEHAVIOR INVENTORY

**Team:** TEAM-XXX  
**Component:** [crate name]  
**Date:** [date]

## 1. Public API Surface
[All exported functions, types, traits]

## 2. State Machine Behaviors
[States, transitions, lifecycle]

## 3. Data Flows
[Inputs, outputs, transformations]

## 4. Error Handling
[Error types, propagation, recovery]

## 5. Integration Points
[Dependencies, dependents, contracts]

## 6. Critical Invariants
[What must always be true]

## 7. Existing Test Coverage
[Unit tests, BDD tests, gaps]

## 8. Behavior Checklist
- [ ] All public APIs documented
- [ ] All state transitions documented
- [ ] All error paths documented
- [ ] All integration points documented
- [ ] All edge cases documented
- [ ] Existing test coverage assessed
- [ ] Coverage gaps identified
```

---

## Success Criteria

### Per-Team
- [ ] Behavior inventory document complete
- [ ] Follows template structure
- [ ] Max 3 pages
- [ ] All behaviors documented
- [ ] All edge cases identified
- [ ] Test coverage gaps identified (for IMPLEMENTED features only)
- [ ] Code signatures added (`// TEAM-XXX: Investigated`)

### CRITICAL: Test Gaps vs Future Features

**Test gaps = IMPLEMENTED code with NO tests**
**Future features = Code with TODO markers (intentional, not a gap)**

**Focus on testing what EXISTS today, not what we plan to build.**

### Phase 2
- [ ] All 3 teams completed
- [ ] All inventories delivered
- [ ] Ready for Phase 3

---

## Coordination

### Concurrent Work
- All 3 teams work independently
- No dependencies between teams
- Can start as soon as Phase 1 completes

### Documentation
- All docs in `.plan/TEAM_XXX_[component]_BEHAVIORS.md`
- Follow template exactly
- Include code examples with line numbers

---

**Status:** READY (after Phase 1)  
**Next:** Phase 3 (Hive crates)
