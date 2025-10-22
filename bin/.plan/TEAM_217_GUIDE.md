# TEAM-217: queen-rbee Behavior Investigation

**Phase:** 1 (Main Binaries)  
**Component:** `10_queen_rbee` - Queen daemon (orchestrator)  
**Duration:** 1 day  
**Output:** `TEAM_217_QUEEN_RBEE_BEHAVIORS.md`

---

## Mission

Inventory ALL behaviors in `queen-rbee` daemon to enable comprehensive test coverage.

---

## Investigation Areas

### 1. HTTP API Surface

**File:** `bin/10_queen_rbee/src/main.rs` + router files

**Tasks:**
- Document ALL HTTP endpoints
- Document request/response schemas
- Document status codes
- Document error responses
- Document authentication/authorization

**Endpoints to Document:**
- `/v1/jobs` - Job creation
- `/v1/jobs/{job_id}/stream` - SSE stream
- `/v1/hives/*` - Hive management
- `/v1/workers/*` - Worker management (if any)
- `/v1/models/*` - Model management (if any)
- `/health` - Health check
- Any other endpoints

### 2. Job Router Logic

**File:** `bin/10_queen_rbee/src/job_router.rs`

**Tasks:**
- Document ALL Operation variants
- Document operation execution flow
- Document error handling per operation
- Document narration emission points
- Document SSE routing logic

**Critical Behaviors:**
- How jobs are created
- How job_id is generated
- How correlation_id flows
- How operations are routed
- How errors are propagated

### 3. SSE Stream Management

**Files:**
- Look for SSE sink implementation
- Look for narration → SSE bridge
- Look for job_id routing

**Tasks:**
- Document SSE channel lifecycle
- Document narration event routing
- Document job_id-based filtering
- Document stream completion logic
- Document stream error handling

**Edge Cases:**
- Multiple clients on same job_id
- Stream closes before job completes
- Narration without job_id
- Client disconnects

### 4. Hive Management Integration

**Files:**
- `bin/10_queen_rbee/src/job_router.rs` (hive operations)
- Integration with `hive-lifecycle` crate

**Tasks:**
- Document how hive operations are delegated
- Document hive-lifecycle crate usage
- Document SSH client integration
- Document capabilities caching
- Document hive state tracking

**Operations to Document:**
- HiveList
- HiveGet
- HiveStart
- HiveStop
- HiveInstall
- HiveUninstall
- HiveStatus
- HiveRefreshCapabilities

### 5. Configuration Management

**Files:**
- Look for RbeeConfig usage
- Look for hives.conf loading
- Look for environment variables

**Tasks:**
- Document ALL configuration sources
- Document config validation
- Document default values
- Document localhost special case
- Document remote hive configuration

**Critical Questions:**
- What happens if hives.conf missing?
- What happens with invalid config?
- How are config errors reported?

### 6. Daemon Lifecycle

**Files:**
- `bin/10_queen_rbee/src/main.rs` (startup/shutdown)

**Tasks:**
- Document startup sequence
- Document initialization steps
- Document graceful shutdown
- Document crash recovery
- Document signal handling

**Behaviors:**
- Port binding
- Config loading
- HTTP server startup
- SSE setup
- Shutdown cleanup

### 7. Narration Integration

**Files:**
- Look for NARRATE usage
- Look for NarrationFactory
- Look for job_id propagation

**Tasks:**
- Document ALL narration emission points
- Document narration metadata (job_id, correlation_id)
- Document narration routing logic
- Document narration to SSE flow

**Critical:**
- Which operations emit narration?
- How is job_id propagated?
- What narration goes to SSE vs stdout?

### 8. Error Handling

**Tasks:**
- Document ALL error types
- Document error propagation paths
- Document error responses
- Document error narration

**Scenarios:**
- Hive unreachable
- SSH failures
- Config errors
- Invalid requests
- Internal errors

### 9. State Management

**Tasks:**
- Document job registry usage
- Document capabilities cache
- Document hive state tracking
- Document any other state

**Questions:**
- What state is persisted?
- What state is in-memory?
- What happens on restart?

---

## Investigation Methodology

### Step 1: Read Main Entry Point
```bash
# Read main.rs
cat bin/10_queen_rbee/src/main.rs
```

### Step 2: Read Job Router
```bash
# Read job_router.rs (critical file)
cat bin/10_queen_rbee/src/job_router.rs
```

### Step 3: Identify All Modules
```bash
# List all source files
find bin/10_queen_rbee/src -name "*.rs"
```

### Step 4: Check Dependencies
```bash
# Check what crates are used
cat bin/10_queen_rbee/Cargo.toml
```

### Step 5: Check Existing Tests
```bash
# Look for tests
find bin/10_queen_rbee -name "*test*.rs"
find bin/10_queen_rbee/tests -type f
```

---

## Key Files to Investigate

1. `bin/10_queen_rbee/src/main.rs` - Entry point, server setup
2. `bin/10_queen_rbee/src/job_router.rs` - Core operation routing
3. `bin/10_queen_rbee/Cargo.toml` - Dependencies
4. Any HTTP route handlers
5. Any SSE-related modules
6. Any config modules

---

## Expected Behaviors to Document

### HTTP API Behaviors
- [ ] Request parsing
- [ ] Request validation
- [ ] Response serialization
- [ ] Error responses
- [ ] Authentication/authorization

### Job Management Behaviors
- [ ] Job creation
- [ ] Job ID generation
- [ ] Correlation ID handling
- [ ] Operation routing
- [ ] Job completion

### SSE Behaviors
- [ ] Stream creation
- [ ] Event emission
- [ ] Job ID filtering
- [ ] Stream completion
- [ ] Stream errors

### Hive Management Behaviors
- [ ] Hive operations delegation
- [ ] SSH integration
- [ ] Capabilities caching
- [ ] State tracking
- [ ] Error handling

### Configuration Behaviors
- [ ] Config loading
- [ ] Config validation
- [ ] Default values
- [ ] Error handling

### Daemon Behaviors
- [ ] Startup sequence
- [ ] Graceful shutdown
- [ ] Signal handling
- [ ] Crash recovery

### Narration Behaviors
- [ ] Narration emission
- [ ] Job ID propagation
- [ ] SSE routing
- [ ] Stdout output

---

## Deliverables Checklist

- [ ] All HTTP endpoints documented
- [ ] All operation types documented
- [ ] All SSE behaviors documented
- [ ] All hive operations documented
- [ ] All error paths documented
- [ ] All configuration documented
- [ ] Daemon lifecycle documented
- [ ] Narration flow documented
- [ ] Existing test coverage assessed
- [ ] Coverage gaps identified
- [ ] Code signatures added (`// TEAM-217: Investigated`)
- [ ] Document follows template
- [ ] Document ≤3 pages
- [ ] Examples include line numbers

---

## Success Criteria

1. ✅ Complete behavior inventory document
2. ✅ All HTTP APIs documented
3. ✅ All operation flows documented
4. ✅ All SSE behaviors documented
5. ✅ All error paths documented
6. ✅ Test coverage gaps identified
7. ✅ Code signatures added
8. ✅ No TODO markers in document

---

## Critical Focus Areas

### 1. SSE + Narration Flow
This is complex and critical - document thoroughly:
- How job_id enables SSE routing
- How narration flows to clients
- How multiple clients are handled
- How streams close

### 2. Hive-Lifecycle Integration
Document how queen delegates to hive-lifecycle:
- Function call patterns
- Error propagation
- Narration emission
- State management

### 3. Error Handling
Document all error paths:
- HTTP errors
- SSH errors
- Hive errors
- Config errors
- Internal errors

---

## Next Steps After Completion

1. Hand off to TEAM-242 for test plan creation
2. Document will be used to create:
   - Unit test plan
   - BDD test plan
   - Integration test plan
   - E2E test plan

---

**Status:** READY  
**Blocked By:** None (can start immediately)  
**Blocks:** TEAM-242 (test planning)
