# Release Candidate Checklist (v0.1.0)

**Target:** Production-ready rbee ecosystem  
**Created by:** TEAM-096 | 2025-10-18  
**Status:** 🔴 NOT READY - Critical gaps identified

---

## Executive Summary

**Current State:** Core functionality exists but **critical security and reliability gaps** prevent production deployment.

**Blockers:**
- 🔴 **P0:** Worker lifecycle PID tracking (can't kill hung workers)
- 🔴 **P0:** No authentication (all APIs open)
- 🔴 **P0:** No input validation (injection vulnerabilities)
- 🔴 **P0:** Secrets in environment variables (insecure)
- 🔴 **P0:** Missing BDD tests for RC items (only 40% coverage)

**Estimated Work:** 
- **BDD Tests:** 20-28 days (must be done FIRST)
- **Implementation:** 15-20 days (after tests exist)
- **Total:** 35-48 days to production-ready

**⚠️ CRITICAL:** BDD tests MUST be written BEFORE implementation work begins!

---

## BDD Test Requirements (MUST DO FIRST!)

**⚠️ NO IMPLEMENTATION WORK WITHOUT TESTS ⚠️**

### Current BDD Coverage
- ✅ **Exists:** 3/18 RC items (17%)
- 🟡 **Partial:** 5/18 RC items (28%)
- 🔴 **Missing:** 10/18 RC items (55%)

### Required New Feature Files
1. **300-authentication.feature** (15-20 scenarios) - P0
2. **310-secrets-management.feature** (10-15 scenarios) - P0
3. **320-error-handling.feature** (20-25 scenarios) - P0
4. **330-audit-logging.feature** (10-12 scenarios) - P1
5. **340-deadline-propagation.feature** (8-10 scenarios) - P1
6. **350-metrics-observability.feature** (15-20 scenarios) - P2
7. **360-configuration-management.feature** (8-10 scenarios) - P2

### Required Test Expansions
1. **110-rbee-hive-lifecycle.feature** - Add force-kill, parallel shutdown
2. **140-input-validation.feature** - Add all injection types
3. **210-failure-recovery.feature** - Add restart policy
4. **230-resource-management.feature** - Add cgroups limits

**See:** `test-harness/bdd/BDD_TESTS_FOR_RC_CHECKLIST.md` for complete test plan

**Timeline:** 20-28 days for complete BDD test coverage

---

## P0 - Critical Blockers (MUST FIX)

### 1. Worker Lifecycle - PID Tracking ⚠️ CRITICAL
**Status:** 🔴 NOT IMPLEMENTED  
**Impact:** Cannot force-kill hung workers, system hangs on shutdown  
**Effort:** 2-3 days

**BDD Tests Required:**
- [ ] Expand `110-rbee-hive-lifecycle.feature` with force-kill scenarios
- [ ] Add scenario: Worker ignores shutdown, force-kill after 10s
- [ ] Add scenario: PID tracking across worker lifecycle
- [ ] Add scenario: Process liveness checks (not just HTTP)

**Tasks:**
- [ ] Add `pid: Option<u32>` to `WorkerInfo` struct (`bin/rbee-hive/src/registry.rs`)
- [ ] Store `child.id()` during spawn (`bin/rbee-hive/src/http/workers.rs`)
- [ ] Implement `force_kill_worker()` method (SIGTERM → wait → SIGKILL)
- [ ] Add process liveness checks (not just HTTP health)
- [ ] Add ready timeout (kill if stuck in Loading > 30s)
- [ ] Update shutdown sequence to force-kill after timeout
- [ ] Add tests for force kill scenarios

**Files to Modify:**
- `bin/rbee-hive/src/registry.rs` - Add pid field
- `bin/rbee-hive/src/http/workers.rs` - Store PID on spawn
- `bin/rbee-hive/src/monitor.rs` - Add process checks
- `bin/rbee-hive/src/commands/daemon.rs` - Force kill on shutdown

**Acceptance Criteria:**
- ✅ Can force-kill workers that don't respond to HTTP shutdown
- ✅ Shutdown completes in <30s even with hung workers
- ✅ Workers stuck in Loading are auto-killed after 30s
- ✅ Process liveness checks detect crashes faster than HTTP

---

### 2. Authentication - API Security ⚠️ CRITICAL
**Status:** 🔴 NOT IMPLEMENTED  
**Impact:** All APIs are open, no access control  
**Effort:** 3-4 days

**BDD Tests Required:**
- [ ] Create `300-authentication.feature` (15-20 scenarios)
- [ ] Scenario: Reject request without token (401)
- [ ] Scenario: Reject request with invalid token (401)
- [ ] Scenario: Accept request with valid token (200)
- [ ] Scenario: Timing-safe token comparison (no timing attacks)
- [ ] Scenario: Loopback bind without token (dev mode)
- [ ] Scenario: Public bind requires token

**Tasks:**
- [ ] Integrate `auth-min` into all components
- [ ] Add Bearer token authentication to queen-rbee HTTP API
- [ ] Add Bearer token authentication to rbee-hive HTTP API
- [ ] Add Bearer token authentication to llm-worker-rbee HTTP API
- [ ] Implement token-based hive → worker communication
- [ ] Add token fingerprinting to all logs (never log raw tokens)
- [ ] Add bind policy enforcement (require token for 0.0.0.0)
- [ ] Generate and document token management procedures

**Files to Create/Modify:**
- `bin/queen-rbee/src/http/middleware/auth.rs` - Auth middleware
- `bin/rbee-hive/src/http/middleware/auth.rs` - Auth middleware
- `bin/llm-worker-rbee/src/http/middleware/auth.rs` - Auth middleware
- `bin/queen-rbee/Cargo.toml` - Add auth-min dependency
- `bin/rbee-hive/Cargo.toml` - Add auth-min dependency
- `bin/llm-worker-rbee/Cargo.toml` - Add auth-min dependency

**Acceptance Criteria:**
- ✅ All HTTP endpoints require Bearer token
- ✅ Invalid tokens return 401 Unauthorized
- ✅ Token validation is timing-safe (no timing attacks)
- ✅ Logs show token fingerprints, never raw tokens
- ✅ Loopback binds work without token (dev mode)
- ✅ Public binds require token or fail to start

---

### 3. Input Validation - Injection Prevention ⚠️ CRITICAL
**Status:** 🔴 NOT IMPLEMENTED  
**Impact:** Log injection, path traversal, command injection vulnerabilities  
**Effort:** 2-3 days

**BDD Tests Required:**
- [ ] Expand `140-input-validation.feature` with all injection types
- [ ] Scenario: Prevent log injection (newlines, ANSI codes)
- [ ] Scenario: Prevent path traversal (../../etc/passwd)
- [ ] Scenario: Prevent command injection (shell metacharacters)
- [ ] Scenario: Validate model references
- [ ] Scenario: Validate worker IDs

**Tasks:**
- [ ] Integrate `input-validation` into all HTTP handlers
- [ ] Validate all user inputs before logging
- [ ] Validate all file paths before use
- [ ] Validate all model references
- [ ] Validate all worker IDs
- [ ] Add property-based tests for validation
- [ ] Document validation rules

**Files to Modify:**
- `bin/queen-rbee/src/http/*.rs` - Add validation to all endpoints
- `bin/rbee-hive/src/http/*.rs` - Add validation to all endpoints
- `bin/llm-worker-rbee/src/http/*.rs` - Add validation to all endpoints
- All `Cargo.toml` files - Add input-validation dependency

**Acceptance Criteria:**
- ✅ All user inputs validated before use
- ✅ Log injection attempts rejected
- ✅ Path traversal attempts rejected (../../etc/passwd)
- ✅ Command injection attempts rejected
- ✅ Property tests pass (proptest fuzzing)
- ✅ Invalid inputs return 400 Bad Request with safe error messages

---

### 4. Secrets Management - Credential Security ⚠️ CRITICAL
**Status:** 🔴 NOT IMPLEMENTED  
**Impact:** Secrets visible in process listings, logs, memory dumps  
**Effort:** 2-3 days

**BDD Tests Required:**
- [ ] Create `310-secrets-management.feature` (10-15 scenarios)
- [ ] Scenario: Load API token from file (0600 permissions)
- [ ] Scenario: Reject world-readable secret file (0644)
- [ ] Scenario: Load from systemd credentials
- [ ] Scenario: Memory zeroization on drop
- [ ] Scenario: Derive encryption key from token (HKDF)

**Tasks:**
- [ ] Integrate `secrets-management` into all components
- [ ] Replace `LLORCH_API_TOKEN` env var with file-based loading
- [ ] Add systemd LoadCredential support
- [ ] Implement memory zeroization for all secrets
- [ ] Add file permission validation (reject world-readable)
- [ ] Document secret management procedures
- [ ] Add secret rotation procedures

**Files to Modify:**
- `bin/queen-rbee/src/main.rs` - Load secrets from files
- `bin/rbee-hive/src/main.rs` - Load secrets from files
- `bin/llm-worker-rbee/src/main.rs` - Load secrets from files
- All `Cargo.toml` files - Add secrets-management dependency

**Acceptance Criteria:**
- ✅ No secrets in environment variables
- ✅ Secrets loaded from files with 0600 permissions
- ✅ Secrets zeroized on drop (memory safety)
- ✅ Systemd LoadCredential support works
- ✅ World-readable secret files rejected at startup
- ✅ Secrets never appear in logs or error messages

---

### 5. Error Handling - Production Robustness ⚠️ CRITICAL
**Status:** 🟡 PARTIAL - Many unwrap() and expect() calls  
**Impact:** Panics crash entire process, no graceful degradation  
**Effort:** 3-4 days

**Tasks:**
- [ ] Audit all `unwrap()` and `expect()` calls
- [ ] Replace with proper error handling (Result, Option)
- [ ] Add error recovery for non-fatal errors
- [ ] Implement graceful degradation
- [ ] Add structured error responses (JSON)
- [ ] Add error correlation IDs for debugging
- [ ] Document error handling patterns

**Files to Audit:**
- `bin/queen-rbee/src/**/*.rs` - Replace unwrap/expect
- `bin/rbee-hive/src/**/*.rs` - Replace unwrap/expect
- `bin/llm-worker-rbee/src/**/*.rs` - Replace unwrap/expect
- `bin/rbee-keeper/src/**/*.rs` - Replace unwrap/expect

**Acceptance Criteria:**
- ✅ No unwrap() in production code paths
- ✅ All errors return proper HTTP status codes
- ✅ Error responses include correlation IDs
- ✅ Non-fatal errors don't crash the process
- ✅ Graceful degradation when dependencies fail
- ✅ Error messages are safe (no sensitive data)

---

## P1 - High Priority (Needed for Reliability)

### 6. Worker Restart Policy
**Status:** 🔴 NOT IMPLEMENTED  
**Effort:** 2-3 days

**Tasks:**
- [ ] Implement exponential backoff restart policy
- [ ] Add max restart attempts (e.g., 3 attempts)
- [ ] Add restart cooldown period
- [ ] Track restart count per worker
- [ ] Add circuit breaker for failing models
- [ ] Document restart behavior

**Acceptance Criteria:**
- ✅ Crashed workers auto-restart (up to 3 times)
- ✅ Exponential backoff between restarts (1s, 2s, 4s)
- ✅ Workers that fail 3 times stay dead
- ✅ Circuit breaker prevents restart loops

---

### 7. Heartbeat Mechanism
**Status:** 🔴 NOT IMPLEMENTED  
**Effort:** 1-2 days

**Tasks:**
- [ ] Add heartbeat endpoint to workers
- [ ] Implement 5s heartbeat interval (faster than 30s health)
- [ ] Detect crashes within 10s (2 missed heartbeats)
- [ ] Add heartbeat to worker registry
- [ ] Document heartbeat protocol

**Acceptance Criteria:**
- ✅ Worker crashes detected in <10s
- ✅ Heartbeat failures trigger restart
- ✅ Heartbeat more reliable than HTTP health checks

---

### 8. Audit Logging
**Status:** 🔴 NOT IMPLEMENTED  
**Effort:** 2-3 days

**Tasks:**
- [ ] Integrate `audit-logging` into queen-rbee
- [ ] Integrate `audit-logging` into rbee-hive
- [ ] Log all worker lifecycle events (spawn, shutdown, crash)
- [ ] Log all authentication events (success, failure)
- [ ] Log all configuration changes
- [ ] Implement tamper-evident hash chains
- [ ] Add audit log rotation
- [ ] Document audit log format

**Acceptance Criteria:**
- ✅ All security events logged
- ✅ Tamper detection via hash chains
- ✅ Audit logs survive process restarts
- ✅ Log rotation prevents disk fill
- ✅ Logs include correlation IDs

---

### 9. Deadline Propagation
**Status:** 🔴 NOT IMPLEMENTED  
**Effort:** 1-2 days

**Tasks:**
- [ ] Integrate `deadline-propagation` into request chain
- [ ] Propagate timeouts: queen → hive → worker
- [ ] Implement request cancellation
- [ ] Add timeout headers to all requests
- [ ] Document timeout behavior

**Acceptance Criteria:**
- ✅ Timeouts propagate through entire stack
- ✅ Requests cancelled when deadline exceeded
- ✅ Timeout headers in all HTTP requests
- ✅ Graceful timeout handling (no panics)

---

### 10. Resource Limits
**Status:** 🔴 NOT IMPLEMENTED  
**Effort:** 2-3 days

**Tasks:**
- [ ] Add CPU limits per worker (cgroups)
- [ ] Add memory limits per worker (cgroups)
- [ ] Add disk space monitoring
- [ ] Add VRAM monitoring and limits
- [ ] Implement backpressure when resources exhausted
- [ ] Document resource requirements

**Acceptance Criteria:**
- ✅ Workers can't OOM the system
- ✅ Workers can't starve CPU
- ✅ Disk space monitored, alerts before full
- ✅ VRAM limits enforced per worker
- ✅ Graceful degradation when resources low

---

## P2 - Medium Priority (Operational Excellence)

### 11. Metrics & Observability
**Status:** 🔴 NOT IMPLEMENTED  
**Effort:** 3-4 days

**Tasks:**
- [ ] Add Prometheus metrics to all components
- [ ] Expose `/metrics` endpoint
- [ ] Track worker count, state distribution
- [ ] Track request latency, throughput
- [ ] Track error rates, crash rates
- [ ] Track resource usage (CPU, memory, VRAM)
- [ ] Create Grafana dashboards
- [ ] Document metrics

**Metrics to Add:**
- Worker count by state (Loading, Idle, Busy)
- Request latency (p50, p95, p99)
- Error rate by endpoint
- Crash rate by model
- VRAM usage by worker
- Model download progress
- Health check success rate

---

### 12. Configuration Management
**Status:** 🟡 PARTIAL - Some config via env vars  
**Effort:** 2-3 days

**Tasks:**
- [ ] Create unified config file format (TOML)
- [ ] Add config validation on startup
- [ ] Support config hot-reload (SIGHUP)
- [ ] Add config schema documentation
- [ ] Add config examples for common scenarios
- [ ] Validate config against schema

**Acceptance Criteria:**
- ✅ Single config file per component
- ✅ Config validated on startup (fail-fast)
- ✅ Hot-reload without restart
- ✅ Schema documented
- ✅ Examples provided

---

### 13. Health Checks - Comprehensive
**Status:** 🟡 PARTIAL - Basic HTTP health exists  
**Effort:** 1-2 days

**Tasks:**
- [ ] Add `/health/live` endpoint (liveness)
- [ ] Add `/health/ready` endpoint (readiness)
- [ ] Check all dependencies (DB, Redis, etc.)
- [ ] Add startup probes
- [ ] Document health check behavior

**Acceptance Criteria:**
- ✅ Liveness checks process is alive
- ✅ Readiness checks dependencies are healthy
- ✅ Kubernetes-compatible health endpoints
- ✅ Health checks don't impact performance

---

### 14. Cascading Shutdown - Complete
**Status:** 🟡 PARTIAL (TEAM-030) - rbee-hive complete, queen-rbee scaffolded  
**Effort:** 2-3 days

**Tasks:**
- [ ] Complete queen-rbee → hives SSH shutdown (TEAM-030 scaffold exists)
- [ ] Implement parallel worker shutdown (currently sequential)
- [ ] Add shutdown timeout enforcement (30s total)
- [ ] Implement force-kill after graceful timeout (requires PID tracking)
- [ ] Add shutdown progress metrics
- [ ] Add shutdown audit logging
- [ ] Document shutdown behavior

**Files to Modify:**
- `bin/queen-rbee/src/main.rs` - Complete shutdown cascade to hives
- `bin/rbee-hive/src/commands/daemon.rs` - Parallel shutdown, force-kill
- `bin/rbee-hive/src/registry.rs` - Add PID field for force-kill

**Acceptance Criteria:**
- ✅ queen-rbee → hives shutdown works (SSH SIGTERM)
- ✅ Parallel worker shutdown (all workers shutdown concurrently)
- ✅ Shutdown completes in <30s total
- ✅ Force-kill after 10s if graceful fails
- ✅ Clean resource cleanup (VRAM released)
- ✅ Shutdown metrics logged

---

### 15. Testing - Comprehensive
**Status:** 🟡 PARTIAL - Some unit tests exist  
**Effort:** 5-7 days

**Tasks:**
- [ ] Add integration tests for full stack
- [ ] Add chaos testing (kill workers randomly)
- [ ] Add load testing (1000 concurrent requests)
- [ ] Add security testing (injection, auth bypass)
- [ ] Add property-based tests (proptest)
- [ ] Add smoke tests for deployment
- [ ] Achieve 80%+ code coverage

**Test Scenarios:**
- Worker crashes during inference
- Network partitions (queen ↔ hive)
- Disk full during model download
- OOM during model loading
- Concurrent worker spawns
- Rapid worker churn
- Authentication bypass attempts
- Injection attacks

---

## P3 - Nice to Have (Future Enhancements)

### 16. JWT Authentication (Enterprise)
**Status:** 🔴 NOT IMPLEMENTED  
**Effort:** 3-4 days

**Tasks:**
- [ ] Integrate `jwt-guardian` into queen-rbee
- [ ] Implement user authentication
- [ ] Add token revocation (Redis)
- [ ] Add role-based access control (RBAC)
- [ ] Document JWT setup

---

### 17. Model Verification
**Status:** 🔴 NOT IMPLEMENTED  
**Effort:** 2-3 days

**Tasks:**
- [ ] Add SHA256 checksum verification
- [ ] Verify model integrity after download
- [ ] Add model signing (optional)
- [ ] Reject corrupted models

---

### 18. Backup & Recovery
**Status:** 🔴 NOT IMPLEMENTED  
**Effort:** 2-3 days

**Tasks:**
- [ ] Add registry backup (SQLite)
- [ ] Add registry restore
- [ ] Add disaster recovery procedures
- [ ] Document backup procedures

---

## Release Criteria

### Must Have (P0)
- ✅ Worker PID tracking and force-kill
- ✅ Authentication on all APIs
- ✅ Input validation on all endpoints
- ✅ Secrets loaded from files (not env vars)
- ✅ No unwrap/expect in production paths

### Should Have (P1)
- ✅ Worker restart policy
- ✅ Heartbeat mechanism
- ✅ Audit logging
- ✅ Deadline propagation
- ✅ Resource limits

### Nice to Have (P2)
- ✅ Metrics & observability
- ✅ Configuration management
- ✅ Comprehensive health checks
- ✅ Complete graceful shutdown
- ✅ Comprehensive testing

---

## Estimated Timeline

### Phase 1 - Critical Security (Week 1-2)
**Days 1-5:** Worker PID tracking, force-kill  
**Days 6-10:** Authentication, input validation, secrets management  
**Days 11-15:** Error handling audit and fixes

### Phase 2 - Reliability (Week 3)
**Days 16-18:** Worker restart policy, heartbeat  
**Days 19-21:** Audit logging, deadline propagation

### Phase 3 - Operations (Week 4)
**Days 22-24:** Resource limits, metrics  
**Days 25-28:** Testing, documentation

**Total:** 20-28 days to production-ready

---

## Risk Assessment

### High Risk
- 🔴 **Worker lifecycle gaps** - Can't kill hung workers (blocks shutdown)
- 🔴 **No authentication** - APIs completely open
- 🔴 **No input validation** - Injection vulnerabilities

### Medium Risk
- 🟡 **No restart policy** - Crashed workers stay dead
- 🟡 **Slow crash detection** - 30-90s delay
- 🟡 **No resource limits** - Workers can OOM system

### Low Risk
- 🟢 **No metrics** - Operational blind spots
- 🟢 **Basic config** - Not ideal but functional

---

## Success Metrics

### Security
- ✅ 0 open APIs (all require auth)
- ✅ 0 injection vulnerabilities
- ✅ 0 secrets in env vars or logs

### Reliability
- ✅ 99.9% uptime
- ✅ <10s crash detection
- ✅ <30s shutdown time
- ✅ 0 hung worker incidents

### Performance
- ✅ <100ms API latency (p95)
- ✅ 1000+ concurrent requests
- ✅ <1% error rate

---

## Sign-Off Checklist

### Security Review
- [ ] All P0 security items complete
- [ ] Security audit passed
- [ ] Penetration testing passed
- [ ] No known vulnerabilities

### Reliability Review
- [ ] All P0 reliability items complete
- [ ] Chaos testing passed
- [ ] Load testing passed
- [ ] Graceful degradation verified

### Operations Review
- [ ] Metrics and monitoring in place
- [ ] Runbooks documented
- [ ] Backup/recovery tested
- [ ] On-call procedures defined

### Documentation Review
- [ ] Architecture documented
- [ ] API documentation complete
- [ ] Deployment guide complete
- [ ] Troubleshooting guide complete

---

**Status:** 🔴 NOT READY FOR PRODUCTION  
**Blockers:** 5 critical P0 items  
**Estimated Effort:** 15-20 days  
**Next Steps:** Start with worker PID tracking (highest impact)

---

**Created by:** TEAM-096 | 2025-10-18  
**Purpose:** Production readiness assessment and release planning
