# PHASE 4: Shared Crate Behavior Discovery

**Teams:** TEAM-230 to TEAM-238  
**Duration:** 1 day (all teams work concurrently)  
**Output:** 9 behavior inventory documents

---

## TEAM-230: narration

**Components:** `bin/99_shared_crates/narration-core` + `bin/99_shared_crates/narration-macros`  
**Complexity:** High  
**Output:** `.plan/TEAM_230_NARRATION_BEHAVIORS.md`

### Investigation Areas

#### 1. Narration Core
- Document NarrationFactory pattern
- Document event emission
- Document job_id propagation (CRITICAL)
- Document correlation_id propagation
- Document SSE routing logic
- Document stdout vs SSE behavior

#### 2. Narration Macros
- Document NARRATE! macro
- Document action types
- Document metadata handling
- Document builder pattern

#### 3. Integration with SSE
- Document how narration reaches SSE
- Document job_id filtering
- Document event serialization

#### 4. Critical Behaviors
- job_id REQUIRED for SSE routing
- correlation_id OPTIONAL for tracing
- Events without job_id → stdout only
- Events with job_id → SSE + stdout

---

## TEAM-231: daemon-lifecycle

**Component:** `bin/99_shared_crates/daemon-lifecycle`  
**Complexity:** High  
**Output:** `.plan/TEAM_231_DAEMON_LIFECYCLE_BEHAVIORS.md`

### Investigation Areas

#### 1. Process Spawning
- Document spawn patterns
- Document binary resolution
- Document environment handling
- Document working directory

#### 2. Health Polling
- Document health check patterns
- Document exponential backoff
- Document timeout handling

#### 3. Graceful Shutdown
- Document SIGTERM → SIGKILL pattern
- Document wait periods
- Document cleanup logic

#### 4. State Tracking
- Document process state management
- Document PID tracking
- Document exit code handling

#### 5. Usage by Binaries
- Used by rbee-keeper (queen lifecycle)
- Used by queen-rbee (hive lifecycle)
- Used by rbee-hive (worker lifecycle)

---

## TEAM-232: rbee-http-client

**Component:** `bin/99_shared_crates/rbee-http-client`  
**Complexity:** Medium  
**Output:** `.plan/TEAM_232_HTTP_CLIENT_BEHAVIORS.md`

### Investigation Areas

#### 1. HTTP Client Setup
- Document client configuration
- Document timeout handling
- Document retry logic
- Document connection pooling

#### 2. Request Building
- Document request builders
- Document header handling
- Document body serialization

#### 3. Response Handling
- Document response parsing
- Document error handling
- Document status code handling

#### 4. SSE Consumption
- Document SSE stream handling
- Document event parsing
- Document stream lifecycle

---

## TEAM-233: config-operations

**Components:** `bin/99_shared_crates/rbee-config` + `bin/99_shared_crates/rbee-operations`  
**Complexity:** Medium  
**Output:** `.plan/TEAM_233_CONFIG_OPERATIONS_BEHAVIORS.md`

### Investigation Areas

#### 1. RbeeConfig
- Document config loading
- Document config validation
- Document default values
- Document hives.conf handling
- Document localhost special case

#### 2. RbeeOperations
- Document Operation enum
- Document operation variants
- Document operation serialization

#### 3. Configuration Sources
- Document config file paths
- Document environment variables
- Document precedence rules

---

## TEAM-234: job-deadline

**Components:** `bin/99_shared_crates/job-registry` + `bin/99_shared_crates/deadline-propagation`  
**Complexity:** Medium  
**Output:** `.plan/TEAM_234_JOB_DEADLINE_BEHAVIORS.md`

### Investigation Areas

#### 1. Job Registry
- Document job_id generation
- Document job storage
- Document job lookup
- Document job cleanup

#### 2. Deadline Propagation
- Document deadline setting
- Document deadline checking
- Document deadline expiration
- Document timeout handling

#### 3. Integration
- How job_id flows through system
- How deadlines are propagated
- How timeouts are enforced

---

## TEAM-235: auth-jwt

**Components:** `bin/99_shared_crates/auth-min` + `bin/99_shared_crates/jwt-guardian`  
**Complexity:** Low  
**Output:** `.plan/TEAM_235_AUTH_JWT_BEHAVIORS.md`

### Investigation Areas

#### 1. Minimal Auth
- Document auth patterns
- Document token handling
- Document validation

#### 2. JWT Guardian
- Document JWT generation
- Document JWT validation
- Document JWT expiration
- Document claims handling

#### 3. Security
- Document secret management
- Document token refresh
- Document revocation

---

## TEAM-236: audit-validation

**Components:** `bin/99_shared_crates/audit-logging` + `bin/99_shared_crates/input-validation`  
**Complexity:** Low  
**Output:** `.plan/TEAM_236_AUDIT_VALIDATION_BEHAVIORS.md`

### Investigation Areas

#### 1. Audit Logging
- Document audit event types
- Document audit storage
- Document audit queries

#### 2. Input Validation
- Document validation patterns
- Document validation rules
- Document error messages

#### 3. Security
- Document sanitization
- Document injection prevention
- Document validation bypass checks

---

## TEAM-237: heartbeat-update-timeout

**Components:** `bin/99_shared_crates/heartbeat` + `bin/99_shared_crates/auto-update` + `bin/99_shared_crates/timeout-enforcer`  
**Complexity:** Medium  
**Output:** `.plan/TEAM_237_HEARTBEAT_UPDATE_BEHAVIORS.md`

### Investigation Areas

#### 1. Heartbeat
- Document heartbeat patterns
- Document heartbeat frequency
- Document heartbeat payload
- Document staleness detection

#### 2. Auto-Update
- Document update checking
- Document update download
- Document update application

#### 3. Timeout Enforcer
- Document timeout enforcement patterns
- Document job_id integration for SSE
- Document narration during timeouts
- Document visual countdown behavior

---

## TEAM-238: secrets-sse-model

**Components:** `bin/99_shared_crates/secrets-management` + `bin/99_shared_crates/sse-relay` + `bin/99_shared_crates/model-catalog`  
**Complexity:** Medium  
**Output:** `.plan/TEAM_238_SECRETS_SSE_MODEL_BEHAVIORS.md`

### Investigation Areas

#### 1. Secrets Management
- Document secret storage patterns
- Document secret zeroization
- Document key derivation (HKDF)
- Document constant-time comparison
- Document security guarantees

#### 2. SSE Relay
- Document SSE infrastructure
- Document event routing
- Document client management
- Document stream lifecycle

#### 3. Model Catalog (Shared)
- Document model metadata
- Document catalog structure
- Document relationship to hive model-catalog
- Document consolidation opportunities

---

## Investigation Methodology

### Step 1: List Crates
```bash
ls bin/99_shared_crates/
```

### Step 2: Read Cargo.toml
```bash
cat bin/99_shared_crates/[crate-name]/Cargo.toml
```

### Step 3: Read Source
```bash
find bin/99_shared_crates/[crate-name]/src -name "*.rs"
```

### Step 4: Check Tests
```bash
find bin/99_shared_crates/[crate-name] -name "*test*.rs"
find bin/99_shared_crates/[crate-name]/bdd -name "*.feature"
```

---

## Deliverables Checklist

Each team must deliver:
- [ ] Behavior inventory document
- [ ] Follows template structure
- [ ] Max 3 pages
- [ ] All public APIs documented
- [ ] All behaviors documented
- [ ] All error paths documented
- [ ] All integration points documented
- [ ] Test coverage gaps identified
- [ ] Code signatures added (`// TEAM-XXX: Investigated`)

---

## Success Criteria

### Per-Team
- ✅ Complete behavior inventory
- ✅ All shared contracts documented
- ✅ All usage patterns documented
- ✅ Test gaps identified

### Phase 4
- ✅ All 9 teams completed
- ✅ All inventories delivered
- ✅ Ready for Phase 5

---

## Coordination

### Concurrent Work
- All 9 teams work independently
- No dependencies between teams
- Can start as soon as Phase 3 completes

### Special Notes
- Some teams investigate multiple related crates
- Document integration between related crates
- Single document per team

---

**Status:** READY (after Phase 3)  
**Next:** Phase 5 (Integration flows)
