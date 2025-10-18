# Extended BDD Research - Deep Dive

**Generated:** 2025-10-18  
**By:** TEAM-112  
**Purpose:** Extended analysis of BDD test architecture, patterns, and insights

---

## Executive Summary

**Files Analyzed:** 29 feature files + step implementations  
**Total Scenarios:** ~300  
**Key Finding:** Tests reveal sophisticated architecture with clear separation of concerns

---

## 🏗️ Architecture Insights from BDD Tests

### 1. Registry Architecture (Multi-Layer)

**Discovery:** Tests reveal THREE distinct registry layers, not one!

| Registry | Scope | Storage | Purpose | Feature File |
|----------|-------|---------|---------|--------------|
| **queen-rbee Global Registry** | Cross-node | In-memory (Arc<RwLock>) | Global worker discovery | 050-queen-rbee-worker-registry.feature |
| **rbee-hive Local Registry** | Single node | In-memory (ephemeral) | Local worker tracking | 060-rbee-hive-worker-registry.feature |
| **Beehive Registry** | Global | SQLite (~/.rbee/beehives.db) | Node SSH details | 010-ssh-registry-management.feature |

**Evidence:**
- 200-concurrency-scenarios.feature line 7: "test race conditions in queen-rbee GLOBAL registry"
- 210-failure-recovery.feature line 73: "rbee-hive's local registry is in-memory (will be lost on restart)"
- 010-ssh-registry-management.feature line 27: "rbee-hive registry is SQLite"

**Implication:** Tests expect complex multi-layer state management

---

### 2. Deleted Scenarios (Architectural Decisions)

**Discovery:** Tests document WHY certain scenarios were removed

#### Gap-C3: Concurrent Model Catalog Inserts (DELETED)
**Location:** 200-concurrency-scenarios.feature line 45-48  
**Reason:** "Each rbee-hive has separate SQLite catalog (~/.rbee/models.db)"  
**Implication:** No shared database = no concurrent INSERT conflicts possible  
**Future:** "If shared catalog is implemented (PostgreSQL), add this scenario back"

#### Gap-F3: Registry Split-Brain (DELETED)
**Location:** 210-failure-recovery.feature line 45-49  
**Reason:** "v1.0 supports only SINGLE queen-rbee instance (no HA)"  
**Implication:** Split-brain requires multi-master setup with Raft/Paxos consensus  
**Future:** "If HA is implemented in v2.0, add this scenario back with @future @v2.0 @requires-ha"

**Pattern:** Tests use deletion comments to document architectural constraints

---

### 3. Moved Scenarios (Architectural Clarity)

**Discovery:** Scenarios moved between files to reflect correct component responsibility

#### Gap-C4: Slot Allocation (MOVED)
**From:** 200-concurrency-scenarios.feature  
**To:** 130-inference-execution.feature  
**Reason:** "Slot allocation happens AT THE WORKER, not in queen-rbee registry"  
**Implication:** queen-rbee only caches slot availability (eventually consistent)

#### Gap-C5: Download Coordination (MOVED)
**From:** 200-concurrency-scenarios.feature  
**To:** 030-model-provisioner.feature  
**Reason:** "Download coordination happens at rbee-hive level, not queen-rbee"  
**Implication:** Current architecture allows duplicate downloads across nodes (by design)

**Pattern:** Tests document component boundaries through scenario placement

---

## 🔐 Security Architecture Revealed

### 1. Secrets Management (17 Scenarios)

**File:** 310-secrets-management.feature

**Key Requirements:**
- File permissions MUST be 0600 (owner read/write only)
- Rejects 0644 (world-readable) and 0640 (group-readable)
- Supports systemd credentials (/run/credentials/)
- Memory zeroization on drop (not in memory dumps)
- HKDF-SHA256 for key derivation
- Secrets NEVER appear in logs (fingerprints only)
- Hot-reload via SIGHUP (no restart required)
- Multi-line files rejected
- Empty/whitespace-only files rejected

**Evidence:** SEC-001 through SEC-017 scenarios

**Implication:** Production-grade secrets management is REQUIRED, not optional

---

### 2. Audit Logging (Tamper-Evident)

**File:** 330-audit-logging.feature

**Key Requirements:**
- Hash chain for tamper detection (SHA-256)
- Each entry includes previous_hash
- First entry has previous_hash = "0000000000000000"
- Detects log tampering (identifies exact entry)
- JSON structured format (ISO 8601 timestamps)
- Log rotation preserves hash chain across files
- Disk space monitoring (warnings at 95%, rotation at 99%)
- Correlation IDs for request tracing

**Evidence:** AUDIT-001 through AUDIT-008 scenarios

**Implication:** Compliance-grade audit logging is expected

---

### 3. Authentication Model

**File:** 300-authentication.feature

**Key Requirements:**
- Bearer token authentication (RFC 6750)
- /health endpoint is PUBLIC (no auth)
- All other endpoints PROTECTED
- Timing-safe token comparison (variance < 10%)
- Token fingerprinting (6-char SHA-256 prefix)
- Dev mode: loopback bind without token works
- Production: public bind requires token or fails to start
- Special characters in tokens handled correctly
- Concurrent auth requests (no race conditions)

**Evidence:** AUTH-001 through AUTH-017 scenarios

**Implication:** Security-first design with timing attack protection

---

## 📊 Validation Responsibility Matrix

### Clear Separation of Concerns

| Component | Validates | Feature File | Line |
|-----------|-----------|--------------|------|
| **rbee-keeper (CLI)** | Model reference format | 140-input-validation.feature | 30 |
| | Backend name (cpu/cuda/metal) | 140-input-validation.feature | 55 |
| | SSH key path exists | 010-ssh-registry-management.feature | 213 |
| **rbee-hive (Server)** | Device number in range | 140-input-validation.feature | 80 |
| | Worker ID format | 140-input-validation.feature | 251 |
| | Worker ID length (max 64) | 140-input-validation.feature | 260 |
| | Path traversal (../../) | 140-input-validation.feature | 189 |
| | Absolute paths | 140-input-validation.feature | 197 |
| | Symlinks | 140-input-validation.feature | 204 |
| | Command injection (shell metacharacters) | 140-input-validation.feature | 213 |
| | Command injection (backticks) | 140-input-validation.feature | 221 |
| | Command injection (pipe characters) | 140-input-validation.feature | 228 |
| **queen-rbee (Orchestrator)** | JWT tokens | 910-full-stack-integration.feature | 35 |
| | Bearer tokens | 300-authentication.feature | 23 |

**Pattern:** Client-side validation (rbee-keeper) for UX, server-side validation (rbee-hive) for security

---

## 🔄 Lifecycle Rules (From Tests)

### Component Lifecycles

| Component | Lifecycle | Dies When | Evidence |
|-----------|-----------|-----------|----------|
| **rbee-keeper** | Ephemeral | After inference completes | 120-queen-rbee-lifecycle.feature line 21 |
| **queen-rbee** | Persistent OR Ephemeral | SIGTERM (if spawned by rbee-keeper) | 120-queen-rbee-lifecycle.feature line 33 |
| **rbee-hive** | Persistent daemon | SIGTERM from queen-rbee | 110-rbee-hive-lifecycle.feature line 22 |
| **llm-worker-rbee** | Persistent daemon | Idle timeout (5min) OR shutdown | 100-worker-rbee-lifecycle.feature line 21 |

### Cascading Shutdown Chain

**From:** 120-queen-rbee-lifecycle.feature line 41-42

```
rbee-keeper sends SIGTERM to queen-rbee
  → queen-rbee cascades shutdown to all rbee-hive instances via SSH
    → rbee-hive cascades shutdown to workers
      → workers complete in-flight requests (max 30s)
        → workers exit with code 0
```

**Implication:** Graceful shutdown is a first-class requirement

---

## 🚀 End-to-End Flow (Happy Path)

**File:** 160-end-to-end-flows.feature

**Complete Flow (27 steps):**

1. rbee-keeper sends request to queen-rbee
2. queen-rbee queries beehive registry for node SSH details
3. queen-rbee establishes SSH connection
4. queen-rbee starts rbee-hive via SSH
5. queen-rbee updates registry with last_connected_unix
6. queen-rbee queries rbee-hive worker registry
7. Worker registry returns empty list (cold start)
8. queen-rbee performs pool preflight check
9. rbee-hive checks model catalog
10. Model not found in catalog
11. rbee-hive downloads model from Hugging Face
12. Download progress SSE stream available
13. rbee-keeper displays progress bar
14. Model download completes
15. rbee-hive registers model in SQLite catalog
16. rbee-hive performs worker preflight checks (RAM, CUDA)
17. rbee-hive spawns worker process
18. Worker HTTP server starts
19. Worker sends ready callback to rbee-hive
20. rbee-hive registers worker in in-memory registry
21. rbee-hive returns worker details to queen-rbee
22. queen-rbee returns worker URL to rbee-keeper
23. rbee-keeper polls worker readiness
24. Worker streams loading progress
25. Worker completes loading
26. rbee-keeper sends inference request
27. Worker streams tokens via SSE

**Implication:** Tests expect sophisticated orchestration with multiple async operations

---

## 📈 Concurrency Patterns

### RwLock Usage (From Tests)

**File:** 200-concurrency-scenarios.feature

**Pattern:** Arc<RwLock> for queen-rbee global registry

**Guarantees:**
- Sequential writes (no race conditions)
- Concurrent reads (performance)
- No deadlocks (RwLock prevents)
- Last-write-wins for state cache

**Evidence:**
- Line 31: "no database locks occur (in-memory registry uses Arc<RwLock>)"
- Line 40: "queen-rbee processes updates sequentially (RwLock guarantees)"
- Line 68: "no deadlocks occur (RwLock prevents deadlock)"

**Implication:** Tests expect thread-safe in-memory registry

---

## 🔍 Error Propagation Chain

**File:** 910-full-stack-integration.feature line 114-120

```
Worker validates request
  → Worker returns validation error
    → rbee-hive propagates error
      → queen-rbee propagates error
        → rbee-keeper displays error to user
```

**Pattern:** Errors flow upstream through all layers

**Implication:** Every component must preserve error context

---

## 📋 Test Patterns Discovered

### 1. Background Consistency

**Pattern:** All 29 feature files use identical Background section

```gherkin
Background:
  Given the following topology:
    | node        | hostname              | components                 | capabilities        |
    | blep        | blep.home.arpa        | rbee-keeper, queen-rbee    | cpu                 |
    | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee | cuda:0, cuda:1, cpu |
  And I am on node "blep"
  And queen-rbee is running at "http://localhost:8080"
```

**Implication:** Tests assume consistent 2-node topology

---

### 2. Port Allocation Strategy

**Discovered Pattern:**
- queen-rbee: 8080 (orchestrator)
- rbee-hive: 8081 or 9200 (pool manager)
- workers: 8001+ (incremental)

**Inconsistency Found:**
- Some tests use 8081 for rbee-hive
- Others use 9200 for rbee-hive
- 160-end-to-end-flows.feature line 50: uses 9200
- 340-deadline-propagation.feature line 19: uses 8081

**Implication:** Port allocation needs clarification

---

### 3. Priority Tags

**Discovered Tags:**
- @p0 - Critical for production (authentication, secrets, validation)
- @p1 - Important but not blocking (audit logging, metrics)
- @p2 - Nice to have (catalog backup)
- @critical - Must pass before release
- @future @v2.0 - Planned for future version

**Implication:** Tests have clear prioritization

---

## 🎯 Key Insights

### 1. Architecture is MORE Complex Than Expected

**Discovery:** 3 registry layers, not 1
- Global registry (queen-rbee)
- Local registry (rbee-hive)
- Beehive registry (SQLite)

**Implication:** Step implementations must handle all 3 layers correctly

---

### 2. Security is First-Class

**Discovery:** 
- 17 secrets management scenarios
- 8 audit logging scenarios
- 17 authentication scenarios
- Timing attack protection
- Tamper-evident logging

**Implication:** This is NOT a toy project - production security expected

---

### 3. Tests Document Architectural Decisions

**Discovery:** Deleted/moved scenarios explain WHY features don't exist

**Examples:**
- No shared catalog → No concurrent INSERT conflicts
- Single queen-rbee → No split-brain scenarios
- Worker-level slots → Not in queen-rbee registry

**Implication:** Tests are living architecture documentation

---

### 4. Validation is Layered

**Discovery:** 
- Client validates for UX (rbee-keeper)
- Server validates for security (rbee-hive)
- Orchestrator validates auth (queen-rbee)

**Implication:** Defense in depth - multiple validation layers

---

### 5. Graceful Degradation Expected

**Discovery:**
- Workers continue after rbee-hive restart
- In-flight requests complete during shutdown
- Partial downloads resume
- Catalog corruption recoverable

**Implication:** Resilience is a core requirement

---

## 📊 Statistics

| Metric | Count |
|--------|-------|
| Feature files analyzed | 29 |
| Total scenarios | ~300 |
| Security scenarios | 42 (14%) |
| Concurrency scenarios | 7 |
| Lifecycle scenarios | 15 |
| Validation scenarios | 25 |
| Deleted scenarios (documented) | 3 |
| Moved scenarios (documented) | 2 |
| Priority P0 scenarios | ~100 (33%) |
| Priority P1 scenarios | ~50 (17%) |

---

## 🔮 Future Features (From Tests)

### Marked with @future @v2.0

1. **High Availability** - Multi-master queen-rbee with consensus
2. **Automatic Failover** - Request retry with state machine
3. **Shared Catalog** - PostgreSQL instead of per-node SQLite
4. **Multi-Hive Load Balancing** - Cross-node request distribution

**Evidence:** Comments in deleted scenarios

---

## ⚠️ Gaps Between Tests and Product Code

### From PRODUCT_CODE_REALITY_CHECK.md

| Feature | Tests Expect | Product Has | Gap |
|---------|--------------|-------------|-----|
| Input Validation (rbee-keeper) | ✅ Required | ❌ Not implemented | HIGH |
| Input Validation (rbee-hive) | ✅ Required | ✅ Implemented | NONE |
| Authentication | ✅ Required | ✅ Implemented | NONE |
| Secrets Management | ✅ Required | ✅ Implemented | NONE |
| Audit Logging | ✅ Required | ❓ Unknown | MEDIUM |
| Tamper-Evident Logs | ✅ Required | ❓ Unknown | MEDIUM |
| Hot-Reload (SIGHUP) | ✅ Required | ❓ Unknown | LOW |

---

## 📝 Recommendations

### For TEAM-113

1. **Implement rbee-keeper validation** - Tests expect it, product doesn't have it
2. **Verify audit logging exists** - 8 scenarios expect tamper-evident logs
3. **Check SIGHUP handling** - Tests expect hot-reload capability
4. **Clarify rbee-hive port** - Tests use both 8081 and 9200

### For Product Team

1. **Add audit-logging shared crate** - Tests expect it
2. **Implement hash chain logging** - Required for SEC compliance
3. **Add SIGHUP handler** - Required for secret rotation
4. **Document registry layers** - Tests reveal 3 layers, docs should explain

### For Documentation

1. **Document deleted scenarios** - Explain architectural constraints
2. **Document validation layers** - Client vs server validation
3. **Document lifecycle rules** - Cascading shutdown chain
4. **Document port allocation** - Clarify 8081 vs 9200

---

## ✅ Conclusion

**Extended research reveals:**

1. **Architecture is sophisticated** - 3 registry layers, cascading shutdown, multi-component orchestration
2. **Security is paramount** - Secrets management, audit logging, timing attack protection
3. **Tests are documentation** - Deleted scenarios explain architectural decisions
4. **Validation is layered** - Client (UX) + Server (security) + Orchestrator (auth)
5. **Gaps exist** - rbee-keeper validation missing, audit logging unclear

**Key Takeaway:** BDD tests reveal a production-grade architecture with security-first design. The gap between tests and product code is smaller than initially thought - most infrastructure exists, just needs wiring.

---

**TEAM-112 Extended Research Complete**
