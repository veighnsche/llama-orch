# Week 2 Checklist: Reliability Features

**Week:** 2 of 4  
**Goal:** Wire existing libraries, add worker lifecycle features  
**Duration:** 5-6 days  
**Target:** ~130-150/300 tests passing (43-50%)

**Reference:** See `ORCHESTRATOR_STANDARDS.md` for what rbee already does

---

## ✅ COMPLETED by TEAM-113

### Task 0.1: Audit Logging to queen-rbee ✅
- [x] Add audit-logging dependency to queen-rbee/Cargo.toml
- [x] Add deadline-propagation dependency to queen-rbee/Cargo.toml
- [x] Initialize AuditLogger in queen-rbee startup
- [x] Configure for home lab mode (disabled by default)
- [x] Add env var support (LLORCH_AUDIT_MODE=local)
- [x] Verify compilation

**Time:** 1 hour  
**Status:** ✅ COMPLETE

---

## 📋 Priority 1: Complete Audit Logging (3-4 hours)

### Task 1.1: Wire Audit Logging to rbee-hive
- [ ] Verify audit-logging dependency exists in rbee-hive/Cargo.toml (already exists!)
- [ ] Initialize AuditLogger in rbee-hive startup (bin/rbee-hive/src/main.rs)
- [ ] Use same config pattern as queen-rbee (disabled by default)
- [ ] Add env var support (LLORCH_AUDIT_MODE=local)
- [ ] Verify compilation

**Files to modify:**
- `bin/rbee-hive/src/main.rs` - Add AuditLogger initialization

### Task 1.2: Add Audit Events to Auth Middleware
- [ ] Log AuthSuccess events in queen-rbee auth middleware
- [ ] Log AuthFailure events in queen-rbee auth middleware
- [ ] Log AuthSuccess events in rbee-hive auth middleware
- [ ] Log AuthFailure events in rbee-hive auth middleware
- [ ] Include token fingerprint (never raw token!)
- [ ] Include IP address, path, service_id

**Files to modify:**
- `bin/queen-rbee/src/http/middleware/auth.rs` - Add audit events
- `bin/rbee-hive/src/http/middleware/auth.rs` - Add audit events

**Example:**
```rust
// On auth success
if let Some(ref logger) = audit_logger {
    logger.emit(AuditEvent::AuthSuccess {
        timestamp: Utc::now(),
        actor: ActorInfo {
            user_id: format!("token:{}", token_fp6(&token)),
            ip: Some(ip_addr),
            auth_method: AuthMethod::BearerToken,
            session_id: None,
        },
        method: AuthMethod::BearerToken,
        path: req.uri().path().to_string(),
        service_id: "queen-rbee".to_string(),
    });
}

// On auth failure
if let Some(ref logger) = audit_logger {
    logger.emit(AuditEvent::AuthFailure {
        timestamp: Utc::now(),
        attempted_user: None,
        reason: "Invalid token".to_string(),
        ip: ip_addr,
        path: req.uri().path().to_string(),
        service_id: "queen-rbee".to_string(),
    });
}
```

### Task 1.3: Add Audit Events to Worker Lifecycle
- [ ] Log worker spawn events (WorkerStarted)
- [ ] Log worker shutdown events (WorkerStopped)
- [ ] Log worker crash events (if detected)
- [ ] Log worker restart events
- [ ] Include worker_id, model_ref, backend, device

**Files to modify:**
- `bin/rbee-hive/src/http/workers.rs` - Add audit events

### Task 1.4: Add Audit Events to Configuration
- [ ] Log config reload events (if SIGHUP implemented)
- [ ] Log config validation failures
- [ ] Include changed fields (sanitized)

**Files to modify:**
- `bin/rbee-hive/src/config.rs` - Add audit events
- `bin/queen-rbee/src/config.rs` - Add audit events

### Task 1.5: Pass AuditLogger Through AppState
- [ ] Add audit_logger: Option<Arc<AuditLogger>> to AppState
- [ ] Pass through create_router()
- [ ] Access in middleware and handlers

**Files to modify:**
- `bin/queen-rbee/src/http/routes.rs` - Add to AppState
- `bin/rbee-hive/src/http/routes.rs` - Add to AppState

**Impact:** ✅ Compliance features enabled, security audit trail

---

## 📋 Priority 2: Wire Deadline Propagation (1 day)

### Task 2.1: Add Deadline Propagation to rbee-hive
- [ ] Add deadline-propagation dependency to rbee-hive/Cargo.toml
- [ ] Review deadline-propagation library API
- [ ] Understand deadline header format

**Files to modify:**
- `bin/rbee-hive/Cargo.toml` - Add dependency

### Task 2.2: Add Deadline Headers (queen-rbee → rbee-hive)
- [ ] Extract deadline from incoming request (if present)
- [ ] Create deadline for new requests (default: 30s)
- [ ] Add X-Request-Deadline header to rbee-hive requests
- [ ] Propagate deadline through HTTP client

**Files to modify:**
- `bin/queen-rbee/src/http/inference.rs` - Add deadline headers

### Task 2.3: Add Deadline Headers (rbee-hive → workers)
- [ ] Extract deadline from incoming request
- [ ] Calculate remaining time
- [ ] Add X-Request-Deadline header to worker requests
- [ ] Reject request if deadline already expired

**Files to modify:**
- `bin/rbee-hive/src/http/workers.rs` - Add deadline headers

### Task 2.4: Implement Timeout Cancellation
- [ ] Check deadline before starting inference
- [ ] Cancel inference if deadline expires
- [ ] Return 504 Gateway Timeout
- [ ] Clean up worker state

**Files to modify:**
- `bin/rbee-hive/src/http/workers.rs` - Add timeout logic
- `bin/llm-worker-rbee/src/http/infer.rs` - Add timeout logic

### Task 2.5: Add Deadline Tracking to Registry
- [ ] Add deadline field to WorkerInfo (optional)
- [ ] Track active request deadlines
- [ ] Clean up expired deadlines

**Files to modify:**
- `bin/rbee-hive/src/registry.rs` - Add deadline tracking

### Task 2.6: Testing
- [ ] Test: Deadline propagates through chain
- [ ] Test: Request cancelled when deadline expires
- [ ] Test: 504 returned on timeout
- [ ] Test: Worker state cleaned up after timeout

**Impact:** ✅ Timeout handling, better request cancellation

---

## 📋 Priority 3: Wire Auth to llm-worker-rbee (1 day)

### Task 3.1: Copy Auth Middleware
- [ ] Copy auth.rs from queen-rbee to llm-worker-rbee
- [ ] Update imports (change queen_rbee to llm_worker_rbee)
- [ ] Update service_id to "llm-worker-rbee"
- [ ] Verify compilation

**Files to create:**
- `bin/llm-worker-rbee/src/http/middleware/auth.rs` - NEW FILE (copy from queen-rbee)
- `bin/llm-worker-rbee/src/http/middleware/mod.rs` - NEW FILE

### Task 3.2: Add Auth to Worker Routes
- [ ] Add auth-min dependency to llm-worker-rbee/Cargo.toml
- [ ] Update routes.rs to use auth middleware
- [ ] Split routes into public and protected
- [ ] Public: /health, /metrics
- [ ] Protected: /infer, /shutdown

**Files to modify:**
- `bin/llm-worker-rbee/Cargo.toml` - Add auth-min dependency
- `bin/llm-worker-rbee/src/http/routes.rs` - Add auth middleware

### Task 3.3: Load API Token in Worker
- [ ] Add expected_token to worker config
- [ ] Load from environment variable (LLORCH_API_TOKEN)
- [ ] Pass to create_router()
- [ ] Log if auth enabled/disabled

**Files to modify:**
- `bin/llm-worker-rbee/src/main.rs` - Load token, pass to router

### Task 3.4: Update rbee-hive to Send Token
- [ ] Add API token to rbee-hive config
- [ ] Send Authorization header when calling workers
- [ ] Handle 401 Unauthorized responses

**Files to modify:**
- `bin/rbee-hive/src/http/workers.rs` - Add Authorization header

### Task 3.5: Testing
- [ ] Test: Worker accepts valid token
- [ ] Test: Worker rejects invalid token (401)
- [ ] Test: Worker rejects missing token (401)
- [ ] Test: Public endpoints work without token

**Impact:** ✅ Complete authentication coverage (100%)

---

## 📋 Priority 4: Worker Restart Policy (2-3 days)

### Task 4.1: Design Restart Policy
- [ ] Define restart policy config structure
- [ ] Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, max 60s
- [ ] Max restart attempts: default 3
- [ ] Circuit breaker: stop after N failures in M minutes
- [ ] Document restart policy in architecture docs

**Files to modify:**
- `bin/rbee-hive/src/config.rs` - Add RestartPolicy config

### Task 4.2: Implement Exponential Backoff
- [ ] Calculate backoff delay: min(2^attempt * 1s, 60s)
- [ ] Add jitter to prevent thundering herd (±20%)
- [ ] Sleep before restart attempt
- [ ] Log backoff delays

**Files to create:**
- `bin/rbee-hive/src/restart.rs` - NEW FILE - Restart policy logic

### Task 4.3: Track Restart Count
- [ ] Increment WorkerInfo.restart_count on restart (field already exists!)
- [ ] Update WorkerInfo.last_restart timestamp (field already exists!)
- [ ] Reset restart_count after successful run (e.g., 5 minutes uptime)
- [ ] Persist restart history (optional)

**Files to modify:**
- `bin/rbee-hive/src/registry.rs` - Add restart tracking methods

### Task 4.4: Implement Circuit Breaker
- [ ] Track failures in time window (e.g., last 5 minutes)
- [ ] Stop restarting if > N failures in window
- [ ] Log circuit breaker activation
- [ ] Reset circuit breaker after cooldown period

**Files to modify:**
- `bin/rbee-hive/src/restart.rs` - Add circuit breaker logic

### Task 4.5: Integrate with Worker Lifecycle
- [ ] Detect worker crash (process exit, health check failure)
- [ ] Trigger restart policy
- [ ] Apply backoff delay
- [ ] Attempt restart
- [ ] Update metrics

**Files to modify:**
- `bin/rbee-hive/src/monitor.rs` - Add crash detection
- `bin/rbee-hive/src/http/workers.rs` - Add restart logic

### Task 4.6: Add Restart Metrics
- [ ] Add metric: rbee_hive_worker_restarts_total
- [ ] Add metric: rbee_hive_worker_restart_failures_total
- [ ] Add metric: rbee_hive_circuit_breaker_activations_total

**Files to modify:**
- `bin/rbee-hive/src/metrics.rs` - Add restart metrics

### Task 4.7: Testing
- [ ] Test: Worker restarts after crash
- [ ] Test: Exponential backoff works correctly
- [ ] Test: Max attempts enforced
- [ ] Test: Circuit breaker activates after threshold
- [ ] Test: Circuit breaker resets after cooldown
- [ ] Test: Restart count tracked correctly

**Impact:** ✅ Resilient workers, automatic recovery

---

## 📊 Week 2 Deliverables

- [ ] Audit logging active on all components (queen-rbee, rbee-hive)
- [ ] Audit events logged for auth, worker lifecycle, config changes
- [ ] Deadline propagation working end-to-end
- [ ] Timeout cancellation implemented
- [ ] Authentication on all 3 components (100%)
- [ ] Worker restart policy implemented
- [ ] Exponential backoff working
- [ ] Circuit breaker working
- [ ] Restart metrics exposed
- [ ] ~130-150/300 tests passing (43-50%)

---

## 🎯 Success Criteria

### Functional
- [ ] Audit events logged for all security-relevant actions
- [ ] Deadlines propagate through entire request chain
- [ ] Requests cancelled when deadline expires
- [ ] Workers require authentication
- [ ] Workers restart automatically after crash
- [ ] Circuit breaker prevents restart loops

### Performance
- [ ] Audit logging overhead < 1% (when disabled)
- [ ] Deadline propagation overhead < 5ms
- [ ] Auth overhead < 2ms per request

### Quality
- [ ] All new code has unit tests
- [ ] Integration tests pass
- [ ] No new unwrap/expect in production code
- [ ] Proper error handling throughout

---

## 📝 Notes for Next Team

### Quick Wins
- Audit logging is well-designed, easy to integrate
- Auth middleware can be copied directly
- Restart policy is mostly configuration

### Challenges
- Deadline propagation requires careful time calculations
- Circuit breaker state management can be tricky
- Testing restart scenarios requires mocking

### Testing Strategy
- Use mock audit logger for tests
- Use short deadlines for timeout tests
- Use mock workers for restart tests

### Code Examples Provided
- Auth middleware: `bin/queen-rbee/src/http/middleware/auth.rs`
- Audit events: See Task 1.2 above
- Restart policy: See Priority 4 tasks

---

## 🔗 Related Documents

- `WEEK_3_CHECKLIST.md` - Next week's tasks
- `WEEK_4_CHECKLIST.md` - Final week tasks
- `RELEASE_CANDIDATE_CHECKLIST_UPDATED.md` - Overall progress

---

**Created by:** TEAM-113  
**Date:** 2025-10-18  
**For:** Week 2 implementation team  
**Status:** 10% complete (audit logging to queen-rbee done)
