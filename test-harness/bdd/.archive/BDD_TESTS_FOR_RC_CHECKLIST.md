# BDD Tests Required for Release Candidate

**Created by:** TEAM-096 | 2025-10-18  
**Purpose:** Map RC checklist items to required BDD tests  
**Status:** Gap analysis complete

---

## Executive Summary

**Existing BDD Tests:** 20 feature files covering basic functionality  
**RC Checklist Items:** 18 P0-P2 items  
**BDD Coverage:** ~40% (8/18 items have tests)  
**Missing Tests:** 10 critical feature files needed

---

## Coverage Matrix

| RC Item | Existing Test | Status | Missing Tests |
|---------|--------------|--------|---------------|
| **P0-1: Worker PID Tracking** | 110-rbee-hive-lifecycle.feature (partial) | 🟡 PARTIAL | Force-kill scenarios |
| **P0-2: Authentication** | ❌ None | 🔴 MISSING | 300-authentication.feature |
| **P0-3: Input Validation** | 140-input-validation.feature | ✅ EXISTS | Expand coverage |
| **P0-4: Secrets Management** | ❌ None | 🔴 MISSING | 310-secrets-management.feature |
| **P0-5: Error Handling** | 140-input-validation.feature (partial) | 🟡 PARTIAL | 320-error-handling.feature |
| **P1-6: Worker Restart Policy** | ❌ None | 🔴 MISSING | Add to 210-failure-recovery.feature |
| **P1-7: Heartbeat Mechanism** | 110-rbee-hive-lifecycle.feature (basic) | 🟡 PARTIAL | Expand heartbeat scenarios |
| **P1-8: Audit Logging** | ❌ None | 🔴 MISSING | 330-audit-logging.feature |
| **P1-9: Deadline Propagation** | ❌ None | 🔴 MISSING | 340-deadline-propagation.feature |
| **P1-10: Resource Limits** | 230-resource-management.feature (basic) | 🟡 PARTIAL | Expand with cgroups |
| **P2-11: Metrics** | ❌ None | 🔴 MISSING | 350-metrics-observability.feature |
| **P2-12: Configuration** | ❌ None | 🔴 MISSING | 360-configuration-management.feature |
| **P2-13: Health Checks** | 110-rbee-hive-lifecycle.feature (basic) | 🟡 PARTIAL | Kubernetes-style checks |
| **P2-14: Cascading Shutdown** | 110-rbee-hive-lifecycle.feature | ✅ EXISTS | Expand with parallel shutdown |
| **P2-15: Testing** | All existing tests | ✅ EXISTS | More chaos/load tests |
| **P3-16: JWT Auth** | ❌ None | 🔴 MISSING | 370-jwt-authentication.feature |
| **P3-17: Model Verification** | ❌ None | 🔴 MISSING | Add to 030-model-provisioner.feature |
| **P3-18: Backup/Recovery** | ❌ None | 🔴 MISSING | 380-backup-recovery.feature |

**Summary:**
- ✅ **Exists:** 3 items (17%)
- 🟡 **Partial:** 5 items (28%)
- 🔴 **Missing:** 10 items (55%)

---

## Required New Feature Files

### P0 - Critical (Must Have)

#### 1. 300-authentication.feature
**Priority:** P0  
**Effort:** 2-3 days  
**Scenarios:** 15-20

```gherkin
Feature: API Authentication
  As a system securing APIs
  I want to require Bearer tokens for all endpoints
  So that unauthorized access is prevented

  @auth @p0
  Scenario: AUTH-001 - Reject request without token
    Given queen-rbee is running with auth enabled
    When I send POST /v1/workers/spawn without Authorization header
    Then response status is 401 Unauthorized
    And response body contains "Missing Authorization header"

  @auth @p0
  Scenario: AUTH-002 - Reject request with invalid token
    Given queen-rbee is running with auth enabled
    When I send POST /v1/workers/spawn with Authorization: "Bearer invalid-token"
    Then response status is 401 Unauthorized
    And response body contains "Invalid token"
    And log contains token fingerprint (not raw token)

  @auth @p0
  Scenario: AUTH-003 - Accept request with valid token
    Given queen-rbee is running with auth enabled
    And valid API token is loaded from /etc/llorch/secrets/api-token
    When I send POST /v1/workers/spawn with valid Bearer token
    Then response status is 200 OK
    And request is processed

  @auth @p0 @timing-attack
  Scenario: AUTH-004 - Timing-safe token comparison
    Given queen-rbee is running with auth enabled
    When I send 1000 requests with invalid tokens of varying lengths
    Then response times have variance < 10%
    And timing attack is prevented

  @auth @p0
  Scenario: AUTH-005 - Loopback bind without token (dev mode)
    Given queen-rbee binds to 127.0.0.1:8080
    And no API token is configured
    Then queen-rbee starts successfully
    And requests to localhost work without auth

  @auth @p0
  Scenario: AUTH-006 - Public bind requires token
    Given queen-rbee attempts to bind to 0.0.0.0:8080
    And no API token is configured
    Then queen-rbee fails to start
    And displays error: "LLORCH_API_TOKEN required for public bind"
```

---

#### 2. 310-secrets-management.feature
**Priority:** P0  
**Effort:** 1-2 days  
**Scenarios:** 10-15

```gherkin
Feature: Secrets Management
  As a system handling credentials
  I want to load secrets from files (not env vars)
  So that secrets are not exposed in process listings

  @secrets @p0
  Scenario: SEC-001 - Load API token from file
    Given API token file exists at /etc/llorch/secrets/api-token
    And file permissions are 0600
    When queen-rbee starts
    Then API token is loaded from file
    And token is stored in memory with zeroization
    And token is never logged

  @secrets @p0
  Scenario: SEC-002 - Reject world-readable secret file
    Given API token file exists at /etc/llorch/secrets/api-token
    And file permissions are 0644 (world-readable)
    When queen-rbee starts
    Then queen-rbee fails to start
    And displays error: "Secret file must be 0600 (owner read/write only)"

  @secrets @p0
  Scenario: SEC-003 - Load from systemd credentials
    Given systemd service has LoadCredential=api_token:/etc/llorch/secrets/api-token
    When queen-rbee starts
    Then API token is loaded from /run/credentials/queen-rbee/api_token
    And token is not visible in /proc/<pid>/environ

  @secrets @p0
  Scenario: SEC-004 - Memory zeroization on drop
    Given API token is loaded into memory
    When queen-rbee shuts down
    Then token memory is zeroized
    And token is not recoverable from memory dump

  @secrets @p0
  Scenario: SEC-005 - Derive encryption key from token
    Given API token is loaded
    When system needs encryption key
    Then key is derived using HKDF-SHA256
    And domain separation is used ("llorch-seal-key-v1")
```

---

#### 3. 320-error-handling.feature
**Priority:** P0  
**Effort:** 2-3 days  
**Scenarios:** 20-25

```gherkin
Feature: Error Handling
  As a system handling errors
  I want to return structured errors without panicking
  So that system remains stable

  @error @p0
  Scenario: ERR-001 - No unwrap() in production paths
    Given all production code paths
    When code is audited
    Then no unwrap() calls exist in production paths
    And all errors use Result<T, E>

  @error @p0
  Scenario: ERR-002 - Structured error responses
    Given queen-rbee is running
    When error occurs during worker spawn
    Then response is JSON with:
      | field         | value                          |
      | error         | "Worker spawn failed"          |
      | error_code    | "WORKER_SPAWN_FAILED"          |
      | correlation_id| "<uuid>"                       |
      | details       | {"reason": "..."}              |

  @error @p0
  Scenario: ERR-003 - Error correlation IDs
    Given queen-rbee is running
    When error occurs
    Then error response includes correlation_id
    And correlation_id is logged
    And correlation_id can be used for debugging

  @error @p0
  Scenario: ERR-004 - Graceful degradation
    Given model catalog database is unavailable
    When worker spawn is requested
    Then system attempts to continue without catalog
    And returns error with recovery instructions
    And system does NOT panic

  @error @p0
  Scenario: ERR-005 - Safe error messages
    Given authentication fails
    When error is returned
    Then error message does NOT contain:
      | sensitive_data    |
      | Raw API tokens    |
      | File paths        |
      | Internal IPs      |
    And error message is safe for logs
```

---

### P1 - High Priority (Should Have)

#### 4. 330-audit-logging.feature
**Priority:** P1  
**Effort:** 1-2 days  
**Scenarios:** 10-12

```gherkin
Feature: Audit Logging
  As a system tracking security events
  I want to log all critical actions
  So that audit trail exists

  @audit @p1
  Scenario: AUDIT-001 - Log worker spawn
    Given queen-rbee is running with audit logging
    When worker is spawned
    Then audit log contains:
      | field    | value              |
      | action   | worker.spawn       |
      | actor    | queen-rbee         |
      | resource | worker-abc123      |
      | outcome  | success            |
      | metadata | {"model": "..."}   |

  @audit @p1
  Scenario: AUDIT-002 - Tamper-evident hash chain
    Given audit log has 10 entries
    When entry 5 is modified
    Then hash chain verification fails
    And tampering is detected

  @audit @p1
  Scenario: AUDIT-003 - Log authentication events
    Given queen-rbee is running with audit logging
    When authentication fails
    Then audit log contains:
      | field    | value                    |
      | action   | auth.failed              |
      | actor    | token:a3f2c1 (fingerprint)|
      | outcome  | failure                  |
```

---

#### 5. 340-deadline-propagation.feature
**Priority:** P1  
**Effort:** 1-2 days  
**Scenarios:** 8-10

```gherkin
Feature: Deadline Propagation
  As a system handling timeouts
  I want to propagate deadlines through the stack
  So that requests timeout consistently

  @deadline @p1
  Scenario: DEAD-001 - Propagate timeout queen → hive → worker
    Given queen-rbee receives request with timeout 30s
    When queen-rbee forwards to rbee-hive
    Then request includes X-Request-Deadline header
    And rbee-hive forwards to worker with same deadline
    And worker checks deadline before processing

  @deadline @p1
  Scenario: DEAD-002 - Cancel request when deadline exceeded
    Given request has deadline in 5s
    When 6s elapse
    Then request is cancelled
    And response is 408 Request Timeout
    And worker stops processing

  @deadline @p1
  Scenario: DEAD-003 - Deadline inheritance
    Given parent request has deadline T+30s
    When child request is created
    Then child inherits parent deadline
    And child cannot extend deadline
```

---

### P2 - Medium Priority (Nice to Have)

#### 6. 350-metrics-observability.feature
**Priority:** P2  
**Effort:** 2-3 days  
**Scenarios:** 15-20

```gherkin
Feature: Metrics and Observability
  As a system operator
  I want to expose Prometheus metrics
  So that I can monitor system health

  @metrics @p2
  Scenario: MET-001 - Expose /metrics endpoint
    Given queen-rbee is running
    When I GET /metrics
    Then response is Prometheus format
    And response includes worker_count metric
    And response includes request_latency_seconds histogram

  @metrics @p2
  Scenario: MET-002 - Track worker state distribution
    Given 5 workers in various states
    When metrics are scraped
    Then worker_count{state="idle"} = 3
    And worker_count{state="busy"} = 2
    And worker_count{state="loading"} = 0

  @metrics @p2
  Scenario: MET-003 - Track request latency
    Given 100 requests completed
    When metrics are scraped
    Then request_latency_seconds{quantile="0.5"} < 0.1
    And request_latency_seconds{quantile="0.95"} < 0.5
    And request_latency_seconds{quantile="0.99"} < 1.0
```

---

#### 7. 360-configuration-management.feature
**Priority:** P2  
**Effort:** 1-2 days  
**Scenarios:** 8-10

```gherkin
Feature: Configuration Management
  As a system administrator
  I want to manage configuration via files
  So that configuration is version-controlled

  @config @p2
  Scenario: CFG-001 - Load config from TOML file
    Given config file exists at ~/.rbee/config.toml
    When queen-rbee starts
    Then configuration is loaded from file
    And environment variables override file config

  @config @p2
  Scenario: CFG-002 - Validate config on startup
    Given config file has invalid timeout value
    When queen-rbee starts
    Then queen-rbee fails to start
    And displays validation error with line number

  @config @p2
  Scenario: CFG-003 - Hot-reload config on SIGHUP
    Given queen-rbee is running
    When config file is modified
    And SIGHUP is sent to queen-rbee
    Then queen-rbee reloads configuration
    And new config is applied without restart
```

---

## Existing Tests to Expand

### 1. 110-rbee-hive-lifecycle.feature
**Add Scenarios:**
- Parallel worker shutdown (currently sequential)
- Force-kill with PID tracking
- Shutdown timeout enforcement
- Shutdown progress metrics

### 2. 140-input-validation.feature
**Add Scenarios:**
- Path traversal prevention
- Command injection prevention
- SQL injection prevention (if applicable)
- Log injection prevention

### 3. 210-failure-recovery.feature
**Add Scenarios:**
- Worker restart policy (exponential backoff)
- Circuit breaker for failing models
- Restart count tracking
- Max restart attempts

### 4. 230-resource-management.feature
**Add Scenarios:**
- CPU limits (cgroups)
- Memory limits (cgroups)
- Disk space monitoring
- VRAM limits per worker

---

## Test Implementation Priority

### Phase 1 - P0 Security (Week 1)
**Days 1-3:** 300-authentication.feature (15-20 scenarios)  
**Days 4-5:** 310-secrets-management.feature (10-15 scenarios)  
**Days 6-7:** 320-error-handling.feature (20-25 scenarios)

**Deliverable:** Security tests complete, ~50 scenarios

---

### Phase 2 - P0 Lifecycle (Week 2)
**Days 8-9:** Expand 110-rbee-hive-lifecycle.feature (force-kill, parallel shutdown)  
**Days 10-11:** Expand 140-input-validation.feature (all injection types)  
**Days 12-14:** Expand 210-failure-recovery.feature (restart policy)

**Deliverable:** Lifecycle tests complete, ~30 scenarios

---

### Phase 3 - P1 Operations (Week 3)
**Days 15-16:** 330-audit-logging.feature (10-12 scenarios)  
**Days 17-18:** 340-deadline-propagation.feature (8-10 scenarios)  
**Days 19-21:** Expand 230-resource-management.feature (cgroups)

**Deliverable:** Operations tests complete, ~25 scenarios

---

### Phase 4 - P2 Observability (Week 4)
**Days 22-24:** 350-metrics-observability.feature (15-20 scenarios)  
**Days 25-26:** 360-configuration-management.feature (8-10 scenarios)  
**Days 27-28:** Integration and cleanup

**Deliverable:** All tests complete, ~150 total scenarios

---

## Acceptance Criteria

### For Each Feature File
- [ ] All scenarios follow Given-When-Then structure
- [ ] Scenarios are independent (no shared state)
- [ ] Tags are appropriate (@p0, @p1, @p2, @auth, @secrets, etc.)
- [ ] Traceability to RC checklist item
- [ ] Step definitions implemented (not just feature files)
- [ ] Tests pass against real product code
- [ ] No mocks for core functionality

### For Overall Test Suite
- [ ] 100+ scenarios covering all RC items
- [ ] 80%+ code coverage
- [ ] All P0 items have tests
- [ ] All P1 items have tests
- [ ] P2 items have basic tests
- [ ] Chaos tests included
- [ ] Load tests included
- [ ] Security tests included

---

## Integration with RC Checklist

**Rule:** No RC checklist item can be marked complete without:
1. ✅ BDD feature file exists
2. ✅ Step definitions implemented
3. ✅ Tests pass against real code
4. ✅ Coverage > 80% for that feature

**Verification Command:**
```bash
# Run tests for specific RC item
LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features/300-authentication.feature \
  cargo run --bin bdd-runner

# Check coverage
cargo tarpaulin --bin bdd-runner --out Html
```

---

## Team Responsibilities

### BDD Test Team (TEAM-097+)
- Write feature files
- Implement step definitions
- Ensure tests use real product code
- Maintain test documentation
- Report coverage metrics

### Implementation Teams (TEAM-098+)
- Implement features to make tests pass
- Do NOT modify tests to pass
- Fix bugs found by tests
- Add integration tests as needed

---

**Created by:** TEAM-096 | 2025-10-18  
**Status:** Gap analysis complete, ready for test implementation  
**Estimated Effort:** 20-28 days for complete test coverage
