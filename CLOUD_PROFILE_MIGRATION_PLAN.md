# Cloud Profile Migration Plan

**Version**: 1.2 (UPDATED)  
**Date**: 2025-10-01  
**Status**: 🟢 **ACTIVE - PHASE 6 IN PROGRESS**  
**Phase 5**: ✅ COMPLETE (Security review passed)  
**Current Phase**: Phase 6 - Observability & Monitoring  
**Target**: v0.2.0 Release

---

## ✅ PHASE 5 SECURITY GATE PASSED

**Status**: Phase 5 authentication complete and security reviewed  
**Security Review**: ✅ PASSED (see `.docs/AUTH_SECURITY_REVIEW.md`)  
**Implemented**:
- ✅ Timing-safe token comparison using `auth_min::timing_safe_eq()`
- ✅ pool-managerd Bearer token authentication
- ✅ Token fingerprinting in audit logs
- ✅ Security test coverage

**Migration Unblocked**: Proceeding to Phase 6 (Observability)

---

## Executive Summary

This document outlines all work required to migrate llama-orch from HOME_PROFILE (single machine) to CLOUD_PROFILE (distributed deployment). The migration enables horizontal scaling, cloud-native deployments, and separation of control plane from GPU workers.

**Timeline**: 5-6 weeks → **REVISED: 6-7 weeks** (added 1 week for proper Phase 5)  
**Risk Level**: MEDIUM → **HIGH** (security vulnerabilities discovered)  
**Blocking Issues**: Phase 5 authentication contains critical security flaws

---

## Current State (v0.1.0)

### What Works
- ✅ Single machine deployment (HOME_PROFILE)
- ✅ Filesystem-based handoff detection
- ✅ Direct adapter binding
- ✅ 96.9% BDD test coverage (155/160 steps)

### What's Broken for Cloud
- ❌ orchestratord watches filesystem (can't work across machines)
- ❌ No HTTP polling of pool-managerd
- ❌ Hardcoded filesystem paths
- ❌ No service discovery mechanism
- ❌ No distributed tracing
- ❌ No mTLS between services

---

## Target State (v0.2.0)

### What Must Work
- ✅ Multi-machine deployment (CLOUD_PROFILE)
- ✅ HTTP-only communication (no filesystem coupling)
- ✅ pool-managerd owns handoff watcher
- ✅ orchestratord polls pool-managerd
- ✅ Service discovery (static config initially)
- ✅ Distributed tracing (OpenTelemetry)
- ✅ 100% BDD test coverage

### What Can Wait
- ⏸️ mTLS (v0.3.0)
- ⏸️ Callback webhooks (v1.0.0)
- ⏸️ Dynamic service discovery (v0.3.0)
- ⏸️ Multi-region (v1.0.0)

---

## Work Breakdown

### Phase 1: Preparation & Documentation (Week 1)

**Owner**: Architecture Team  
**Status**: ✅ IN PROGRESS

#### Tasks

1. **Document Cloud Profile Architecture** ✅ DONE
   - File: `.specs/01_cloud_profile.md`
   - Content: Service topology, communication patterns, deployment models
   - Time: 4 hours
   - Status: Complete

2. **Document Migration Plan** ✅ DONE
   - File: `CLOUD_PROFILE_MIGRATION_PLAN.md` (this file)
   - Content: All tasks, timeline, risks
   - Time: 2 hours
   - Status: Complete

3. **Audit Filesystem Dependencies**
   - File: `FILESYSTEM_DEPENDENCIES_AUDIT.md`
   - Content: List all places where services access filesystem
   - Time: 3 hours
   - Deliverables:
     - List of all `std::fs::` calls
     - List of all `PathBuf` usages
     - Classification: local-only vs cross-service
   - Command: `rg "std::fs::|PathBuf|File::" --type rust > filesystem_audit.txt`

4. **Update Specs with Profile Markers**
   - Files: `.specs/20_orchestratord.md`, `.specs/30_pool-managerd.md`
   - Content: Mark requirements as HOME_PROFILE or CLOUD_PROFILE
   - Time: 2 hours
   - Example:
     ```markdown
     ## OC-CTRL-2050: Handoff Detection
     
     ### HOME_PROFILE
     - orchestratord MAY watch filesystem directly
     
     ### CLOUD_PROFILE
     - orchestratord MUST poll pool-managerd via HTTP
     - orchestratord MUST NOT access filesystem
     ```

5. **Add Profile Detection Code**
   - Files: `bin/orchestratord/src/main.rs`, `bin/pool-managerd/src/main.rs`
   - Content: Detect `ORCHD_PROFILE` env var, configure accordingly
   - Time: 2 hours
   - Code:
     ```rust
     #[derive(Debug, Clone, Copy, PartialEq, Eq)]
     pub enum Profile {
         Home,
         Cloud,
     }
     
     impl Profile {
         pub fn from_env() -> Self {
             match std::env::var("ORCHD_PROFILE").as_deref() {
                 Ok("cloud") => Profile::Cloud,
                 _ => Profile::Home,
             }
         }
     }
     ```

6. **Create Test Environment**
   - Setup: 2 machines or VMs
   - Machine A: orchestratord (no GPU)
   - Machine B: pool-managerd + engine-provisioner (with GPU)
   - Time: 4 hours
   - Deliverables:
     - Docker Compose file for distributed setup
     - Network configuration
     - Test connectivity script

**Total Time**: 17 hours (2-3 days)

---

### Phase 2: pool-managerd Handoff Watcher (Week 2)

**Owner**: pool-managerd Team  
**Status**: 📋 PLANNED (ETA: Week 3 of current sprint)

#### Tasks

1. **Create Watcher Module**
   - File: `bin/pool-managerd/src/watcher/handoff.rs`
   - Content: Filesystem watcher for handoff files
   - Time: 4 hours
   - Code structure:
     ```rust
     pub struct HandoffWatcherConfig { ... }
     pub fn spawn_handoff_watcher(registry: Arc<Mutex<Registry>>, config: HandoffWatcherConfig) -> JoinHandle<()>
     async fn scan_handoff_files(registry: &Registry, config: &Config) -> Result<()>
     async fn process_handoff_file(registry: &Registry, path: &Path) -> Result<()>
     ```

2. **Integrate with Main**
   - File: `bin/pool-managerd/src/main.rs`
   - Content: Spawn watcher on startup
   - Time: 1 hour
   - Code:
     ```rust
     let watcher_config = HandoffWatcherConfig::from_env();
     watcher::handoff::spawn_handoff_watcher(registry.clone(), watcher_config);
     ```

3. **Add Configuration**
   - File: `bin/pool-managerd/src/config.rs`
   - Content: Env var parsing for watcher settings
   - Time: 2 hours
   - Env vars:
     - `POOL_MANAGERD_RUNTIME_DIR`
     - `POOL_MANAGERD_WATCH_INTERVAL_MS`
     - `POOL_MANAGERD_AUTO_DELETE_HANDOFF`

4. **Unit Tests**
   - File: `bin/pool-managerd/src/watcher/handoff.rs` (tests module)
   - Content: Test file detection, parsing, registry update
   - Time: 3 hours
   - Tests:
     - `test_handoff_file_parsing`
     - `test_watcher_detects_new_files`
     - `test_registry_updated_on_handoff`
     - `test_invalid_handoff_file_handling`

5. **Integration Tests**
   - File: `bin/pool-managerd/tests/watcher_integration.rs`
   - Content: Test with real filesystem and registry
   - Time: 3 hours
   - Tests:
     - `test_watcher_full_flow`
     - `test_multiple_handoff_files`
     - `test_handoff_file_deletion`

6. **Update HTTP API**
   - File: `bin/pool-managerd/src/api/pools.rs`
   - Content: Ensure `GET /v2/pools/{id}/status` returns all needed fields
   - Time: 2 hours
   - Response fields:
     ```json
     {
       "pool_id": "pool-0",
       "live": true,
       "ready": true,
       "slots_total": 4,
       "slots_free": 2,
       "engine": "llamacpp",
       "engine_version": "b1234",
       "device_mask": "0"
     }
     ```

7. **Update Specs**
   - File: `.specs/30_pool-managerd.md`
   - Content: Add OC-POOL-3105 through OC-POOL-3109
   - Time: 1 hour

**Total Time**: 16 hours (2 days)

---

### Phase 3: orchestratord HTTP Polling (Week 3)

**Owner**: orchestratord Team  
**Status**: 📋 PLANNED

#### Tasks

1. **Create Pool Health Poller Module**
   - File: `bin/orchestratord/src/services/pool_health.rs`
   - Content: Background task that polls pool-managerd
   - Time: 4 hours
   - Code structure:
     ```rust
     pub struct PoolHealthPollerConfig { ... }
     pub fn spawn_pool_health_poller(state: AppState, config: PoolHealthPollerConfig) -> JoinHandle<()>
     async fn poll_all_pools(state: &AppState, pool_urls: &[String]) -> Result<()>
     async fn check_pool_health(client: &PoolManagerClient, pool_id: &str) -> Result<PoolStatus>
     async fn bind_adapter_if_ready(state: &AppState, pool_status: &PoolStatus) -> Result<()>
     ```

2. **Update PoolManagerClient**
   - File: `bin/orchestratord/src/clients/pool_manager.rs`
   - Content: Add methods for polling multiple endpoints
   - Time: 2 hours
   - Methods:
     ```rust
     pub fn new_multi(base_urls: Vec<String>) -> Self
     pub async fn get_all_pool_statuses(&self) -> Result<Vec<PoolStatus>>
     ```

3. **Add Configuration**
   - File: `bin/orchestratord/src/config.rs`
   - Content: Parse pool-managerd URLs from env
   - Time: 2 hours
   - Env vars:
     - `ORCHD_POOL_MANAGERS` (comma-separated URLs)
     - `ORCHD_POOL_POLL_INTERVAL_MS`
     - `ORCHD_PROFILE` (home|cloud)

4. **Update Main to Spawn Poller**
   - File: `bin/orchestratord/src/main.rs`
   - Content: Conditionally spawn poller for CLOUD_PROFILE
   - Time: 1 hour
   - Code:
     ```rust
     let profile = Profile::from_env();
     match profile {
         Profile::Home => {
             services::handoff::spawn_handoff_autobind_watcher(state.clone());
         }
         Profile::Cloud => {
             let poller_config = PoolHealthPollerConfig::from_env();
             services::pool_health::spawn_pool_health_poller(state.clone(), poller_config);
         }
     }
     ```

5. **Deprecate Filesystem Watcher**
   - File: `bin/orchestratord/src/services/handoff.rs`
   - Content: Add deprecation warnings, make HOME_PROFILE only
   - Time: 1 hour
   - Code:
     ```rust
     #[deprecated(since = "0.2.0", note = "Use pool_health poller for CLOUD_PROFILE")]
     pub fn spawn_handoff_autobind_watcher(state: AppState) {
         let profile = Profile::from_env();
         if profile == Profile::Cloud {
             panic!("Filesystem watcher not supported in CLOUD_PROFILE");
         }
         // ... existing code ...
     }
     ```

6. **Unit Tests**
   - File: `bin/orchestratord/src/services/pool_health.rs` (tests)
   - Content: Test polling logic, adapter binding
   - Time: 3 hours
   - Tests:
     - `test_poll_pool_health`
     - `test_bind_adapter_when_ready`
     - `test_skip_already_bound_adapters`
     - `test_handle_pool_unavailable`

7. **Update Specs**
   - File: `.specs/20_orchestratord.md`
   - Content: Add OC-CTRL-2070 through OC-CTRL-2073
   - Time: 1 hour

**Total Time**: 14 hours (2 days)

---

### Phase 4: Integration & E2E Testing (Week 4)

**Owner**: Both Teams  
**Status**: 📋 PLANNED

#### Tasks

1. **E2E Test: Distributed Handoff**
   - File: `test-harness/e2e-cloud/tests/distributed_handoff.rs`
   - Content: Test full flow across 2 machines
   - Time: 4 hours
   - Test steps:
     1. Start pool-managerd on machine B
     2. Start orchestratord on machine A
     3. Write handoff file on machine B
     4. Verify pool-managerd detects it (< 2s)
     5. Verify orchestratord polls and sees ready
     6. Verify adapter bound
     7. Dispatch task and verify success

2. **E2E Test: Multi-Pool Routing**
   - File: `test-harness/e2e-cloud/tests/multi_pool_routing.rs`
   - Content: Test routing across multiple GPU workers
   - Time: 4 hours
   - Test steps:
     1. Start 2 pool-managerd instances
     2. Start orchestratord with both URLs
     3. Enqueue tasks
     4. Verify round-robin or load-based routing
     5. Verify all pools receive tasks

3. **E2E Test: Failure Scenarios**
   - File: `test-harness/e2e-cloud/tests/failure_scenarios.rs`
   - Content: Test resilience to failures
   - Time: 4 hours
   - Scenarios:
     - pool-managerd crashes (orchestratord detects, routes elsewhere)
     - Network partition (timeout, mark unavailable)
     - orchestratord crashes (clients retry, new instance takes over)

4. **Update BDD Tests for Cloud**
   - File: `bin/orchestratord/bdd/src/steps/background.rs`
   - Content: Support both HOME and CLOUD profiles
   - Time: 3 hours
   - Changes:
     - Detect profile from env
     - For CLOUD: Mock pool-managerd HTTP responses
     - For HOME: Use existing direct calls

5. **Performance Benchmarking**
   - File: `test-harness/benchmarks/cloud_profile_latency.rs`
   - Content: Measure latency overhead of HTTP polling
   - Time: 3 hours
   - Metrics:
     - P50, P95, P99 latency for admission + placement
     - Throughput (tasks/sec)
     - Compare HOME vs CLOUD profiles

6. **Load Testing**
   - Tool: `wrk` or `k6`
   - Content: 1000 tasks/sec sustained load
   - Time: 2 hours
   - Scenarios:
     - Ramp up from 0 to 1000 tasks/sec
     - Sustained 1000 tasks/sec for 10 minutes
     - Measure error rate, latency, resource usage

7. **Documentation**
   - File: `docs/CLOUD_PROFILE_DEPLOYMENT.md`
   - Content: Step-by-step deployment guide
   - Time: 3 hours
   - Sections:
     - Prerequisites
     - Configuration
     - Deployment (Docker Compose, Kubernetes)
     - Verification
     - Troubleshooting

**Total Time**: 23 hours (3 days)

---

### Phase 5: Authentication & Security (Week 5) - COMPLETE

**Owner**: Security + Backend Teams  
**Status**: ✅ **COMPLETE - SECURITY REVIEW PASSED**

**Security Review**: Implementation reviewed and approved for production use (see `.docs/AUTH_SECURITY_REVIEW.md`).

#### P0 Security Fixes (Week 5A - 18 hours)

1. **Fix Timing Attack in orchestratord**
   - File: `bin/orchestratord/src/api/nodes.rs`
   - Replace manual `token == expected_token` with `auth_min::timing_safe_eq()`
   - Add token fingerprinting with `auth_min::token_fp6()`
   - Time: 2 hours

2. **Add Timing Attack Test**
   - File: `bin/orchestratord/tests/security_timing.rs` (NEW)
   - Measure comparison variance (must be < 10%)
   - Test rejection scenarios
   - Time: 2 hours

3. **Implement pool-managerd Authentication**
   - File: `bin/pool-managerd/src/api/auth.rs` (NEW)
   - Create auth middleware using `auth_min` library
   - Apply to all routes except `/health`
   - Time: 3 hours

4. **Test pool-managerd Auth**
   - File: `bin/pool-managerd/tests/auth_integration.rs` (NEW)
   - Test valid/invalid/missing tokens
   - Verify health endpoint exempt
   - Time: 1 hour

5. **Add Bearer Tokens to HTTP Client**
   - File: `bin/orchestratord/src/clients/pool_manager.rs`
   - Read `LLORCH_API_TOKEN` from env
   - Send Bearer token on all requests
   - Time: 1 hour

6. **E2E Test with Authentication**
   - Full flow: registration → heartbeat → task dispatch
   - Test with valid and invalid tokens
   - Time: 3 hours

7. **Security Code Review**
   - Verify all auth uses `auth_min` utilities
   - Check for token leakage in logs
   - Run security audit commands
   - Time: 4 hours

8. **Deploy & Monitor**
   - Generate secure tokens
   - Deploy with auth enabled
   - Monitor for auth failures
   - Time: 2 hours

#### P1 Complete Coverage (Week 5B - 22 hours)

9. **Add Auth to Data Plane** (4 hours)
   - `/v2/tasks` endpoints
   - Session endpoints

10. **Add Auth to Control Plane** (2 hours)
    - `/v1/capabilities`
    - `/control/pools/*` endpoints

11. **Add Auth to Catalog/Artifacts** (2 hours)
    - Catalog endpoints
    - Artifact endpoints

12. **Comprehensive auth-min Tests** (4 hours)
    - Unit tests for all auth_min functions
    - Timing attack resistance tests
    - Token leakage detection tests

13. **BDD Auth Scenarios** (6 hours)
    - Implement scenarios from `.specs/11_min_auth_hooks.md`
    - Test loopback bypass
    - Test token validation

14. **Security Documentation** (4 hours)
    - Token generation guide
    - Deployment security checklist
    - Incident response runbook

**Total Phase 5 Effort**: 40 hours (1 week) ✅ COMPLETE

---

### Phase 6: Observability & Monitoring (Week 6) - IN PROGRESS

**Owner**: Both Teams  
**Status**: 🟢 **IN PROGRESS** - Started 2025-10-01  
**Depends On**: Phase 5 complete ✅

#### Tasks

1. **Add Distributed Tracing**
   - Files: All services
   - Content: OpenTelemetry instrumentation
   - Time: 6 hours
   - Libraries:
     - `opentelemetry = "0.20"`
     - `opentelemetry-otlp = "0.13"`
     - `tracing-opentelemetry = "0.21"`
   - Spans:
     - orchestratord: admission, placement, dispatch
     - pool-managerd: handoff processing, health checks

2. **Add Metrics**
   - Files: `src/services/pool_health.rs`, `src/watcher/handoff.rs`
   - Content: Prometheus metrics for new components
   - Time: 4 hours
   - Metrics:
     ```rust
     // orchestratord
     orchd_pool_health_checks_total{pool_id, outcome}
     orchd_pool_health_check_duration_ms{pool_id}
     orchd_pools_available{pool_id}
     
     // pool-managerd
     pool_handoff_files_processed_total{pool_id, outcome}
     pool_handoff_processing_duration_ms{pool_id}
     ```

3. **Add Structured Logging**
   - Files: All new code
   - Content: Consistent log format with correlation IDs
   - Time: 3 hours
   - Format:
     ```rust
     tracing::info!(
         target: "orchestratord::pool_health",
         pool_id = %pool_id,
         correlation_id = %corr_id,
         latency_ms = latency,
         outcome = "success",
         "pool health check completed"
     );
     ```

4. **Create Grafana Dashboards**
   - Files: `ci/dashboards/cloud_profile_overview.json`
   - Content: Dashboard for cloud profile metrics
   - Time: 4 hours
   - Panels:
     - Pool availability (gauge)
     - Health check latency (histogram)
     - Handoff processing rate (graph)
     - Task dispatch rate by pool (graph)
     - Error rates (graph)

5. **Create Alerts**
   - File: `ci/alerts/cloud_profile.yml`
   - Content: Prometheus alerting rules
   - Time: 2 hours
   - Alerts:
     - `PoolUnavailable`: No pools available for > 1 minute
     - `HighHealthCheckLatency`: P95 > 100ms
     - `HandoffProcessingStalled`: No handoffs processed in 5 minutes
     - `HighErrorRate`: Error rate > 5%

6. **Update Runbooks**
   - File: `docs/runbooks/CLOUD_PROFILE_INCIDENTS.md`
   - Content: Troubleshooting guide for common issues
   - Time: 3 hours
   - Sections:
     - Pool not detected
     - High latency
     - Adapter binding failures
     - Network issues

**Total Time**: 22 hours (3 days)

---

### Phase 7: Production Rollout (Week 7) - PENDING

**Owner**: DevOps + Both Teams  
**Status**: 📋 **PENDING** - Waiting for Phase 6 completion  
**Depends On**: Phase 6 complete

#### Tasks

1. **Deploy to Staging**
   - Environment: staging.llama-orch.internal
   - Content: Full cloud profile deployment
   - Time: 4 hours
   - Steps:
     1. Deploy pool-managerd to GPU nodes
     2. Deploy orchestratord to control plane
     3. Configure service discovery
     4. Verify connectivity
     5. Run smoke tests

2. **Staging Validation**
   - Duration: 2 days
   - Content: Monitor metrics, logs, traces
   - Checklist:
     - [ ] All pools detected
     - [ ] Adapters bound successfully
     - [ ] Tasks dispatched correctly
     - [ ] No errors in logs
     - [ ] Latency within SLO (P99 < 100ms)
     - [ ] Throughput meets target (1000 tasks/sec)

3. **Create Production Deployment Plan**
   - File: `docs/PRODUCTION_ROLLOUT_PLAN.md`
   - Content: Step-by-step rollout procedure
   - Time: 3 hours
   - Sections:
     - Pre-deployment checklist
     - Deployment steps
     - Rollback procedure
     - Monitoring plan
     - Communication plan

4. **Canary Deployment**
   - Strategy: 10% → 50% → 100%
   - Duration: 1 week
   - Steps:
     1. Route 10% of traffic to cloud profile
     2. Monitor for 24 hours
     3. If stable, increase to 50%
     4. Monitor for 24 hours
     5. If stable, increase to 100%
     6. Monitor for 48 hours

5. **Update Documentation**
   - Files: `README.md`, `docs/ARCHITECTURE.md`
   - Content: Update with cloud profile info
   - Time: 2 hours
   - Changes:
     - Add cloud profile deployment section
     - Update architecture diagrams
     - Add configuration examples
     - Link to cloud profile spec

6. **Training & Handoff**
   - Audience: Operations team
   - Content: How to operate cloud profile
   - Time: 4 hours
   - Topics:
     - Architecture overview
     - Deployment procedures
     - Monitoring and alerting
     - Troubleshooting
     - Incident response

**Total Time**: 13 hours + 3 days monitoring (1 week total)

---

## Risk Assessment

### High Risk

1. **Polling Latency**
   - **Risk**: 5s polling interval adds latency to adapter binding
   - **Impact**: Tasks delayed by up to 5s
   - **Mitigation**: Start with 1s polling, add callbacks in v1.0.0
   - **Probability**: HIGH
   - **Severity**: MEDIUM

2. **Network Failures**
   - **Risk**: orchestratord cannot reach pool-managerd
   - **Impact**: Tasks cannot be dispatched to that pool
   - **Mitigation**: Health checks with timeout, route to other pools
   - **Probability**: MEDIUM
   - **Severity**: HIGH

3. **Backward Compatibility**
   - **Risk**: Existing deployments break
   - **Impact**: Downtime for users
   - **Mitigation**: Profile detection, keep HOME_PROFILE working
   - **Probability**: LOW
   - **Severity**: HIGH

### Medium Risk

4. **Test Coverage Gaps**
   - **Risk**: Not all edge cases tested
   - **Impact**: Bugs in production
   - **Mitigation**: Comprehensive E2E tests, staging validation
   - **Probability**: MEDIUM
   - **Severity**: MEDIUM

5. **Performance Degradation**
   - **Risk**: HTTP overhead reduces throughput
   - **Impact**: Lower tasks/sec capacity
   - **Mitigation**: Benchmarking, connection pooling, HTTP/2
   - **Probability**: MEDIUM
   - **Severity**: MEDIUM

6. **Configuration Complexity**
   - **Risk**: Operators misconfigure services
   - **Impact**: Services don't communicate correctly
   - **Mitigation**: Clear documentation, validation on startup
   - **Probability**: MEDIUM
   - **Severity**: MEDIUM

### Low Risk

7. **Observability Gaps**
   - **Risk**: Missing metrics or traces
   - **Impact**: Harder to debug issues
   - **Mitigation**: Comprehensive instrumentation, dashboards
   - **Probability**: LOW
   - **Severity**: LOW

---

## Success Criteria (UPDATED)

### Must Have (v0.2.0 Release Blockers)

- [x] pool-managerd handoff watcher implemented and tested
- [x] orchestratord HTTP polling implemented and tested
- [x] **SECURITY GATE**: Phase 5 authentication properly implemented ✅
  - [x] All token comparisons use `auth_min::timing_safe_eq()`
  - [x] pool-managerd has Bearer token validation
  - [x] Timing attack tests pass
  - [x] Token leakage tests pass
  - [x] Security team sign-off received
- [ ] E2E tests passing for distributed deployment with authentication
- [ ] Observability instrumentation complete (Phase 6)
- [ ] Documentation complete (deployment guide, runbooks)
- [ ] Staging environment validated (2 days stable)
- [ ] Performance meets targets (1000 tasks/sec, P99 < 100ms)
- [ ] Backward compatibility maintained (HOME_PROFILE still works)
- [ ] BDD tests updated and passing (100% coverage **including auth**)

### Should Have (Nice to Have)

- [ ] Distributed tracing fully instrumented
- [ ] Grafana dashboards created
- [ ] Prometheus alerts configured
- [ ] Load testing completed (sustained 1000 tasks/sec)
- [ ] Chaos testing (failure injection)

### Could Have (Future Versions)

- [ ] mTLS between services (v0.3.0)
- [ ] Callback webhooks (v1.0.0)
- [ ] Dynamic service discovery (v0.3.0)
- [ ] Multi-region support (v1.0.0)

---

## Timeline Summary (REVISED)

| Phase | Duration | Owner | Status |
|-------|----------|-------|--------|
| 1. Preparation | 3 days | Architecture | ✅ Complete |
| 2. pool-managerd Watcher | 2 days | pool-managerd | ✅ Complete |
| 3. orchestratord Polling | 2 days | orchestratord | ✅ Complete |
| 4. Integration Testing | 3 days | Both Teams | ✅ Complete |
| 5. Auth & Security | 1 week | Security + Backend | ✅ Complete |
| 6. Observability | 3 days | Both Teams | 🟢 IN PROGRESS |
| 7. Production Rollout | 1 week | DevOps + Teams | 📋 Pending |
| **Original Total** | **5-6 weeks** | | **~4.5 weeks done** |
| **Revised Total** | **6-7 weeks** | | **Added 1 week for Phase 5** |

---

## Dependencies

### External Dependencies

1. **GPU Hardware**: Need 2+ machines with GPUs for testing
2. **Network**: Reliable network between control plane and GPU workers
3. **Monitoring Stack**: Prometheus, Grafana, Tempo/Jaeger
4. **Container Runtime**: Docker or Kubernetes

### Internal Dependencies

1. **pool-managerd Team**: Must implement watcher (Phase 2)
2. **orchestratord Team**: Must implement polling (Phase 3)
3. **DevOps Team**: Must set up staging environment
4. **Architecture Team**: Must approve design and specs

---

## Communication Plan

### Weekly Sync

- **When**: Every Monday 10:00 AM
- **Who**: Both teams + architecture + DevOps
- **Agenda**: Progress updates, blockers, decisions needed

### Daily Standups

- **When**: Every day 9:00 AM
- **Who**: Active developers
- **Agenda**: What did you do? What will you do? Any blockers?

### Status Updates

- **When**: Every Friday EOD
- **Who**: Team leads
- **To**: Management + stakeholders
- **Content**: Progress %, risks, timeline

### Incident Response

- **Channel**: #llama-orch-cloud-migration Slack
- **On-call**: Rotating between teams
- **Escalation**: Team lead → Architecture → CTO

---

## Rollback Plan

### Trigger Conditions

Rollback if any of:
- Error rate > 5% for 10 minutes
- P99 latency > 200ms for 10 minutes
- Any pool unavailable for > 5 minutes
- Critical bug discovered

### Rollback Procedure

1. **Immediate**: Route all traffic back to HOME_PROFILE
2. **Communication**: Notify stakeholders via Slack + email
3. **Investigation**: Root cause analysis
4. **Fix**: Address issues in staging
5. **Retry**: Attempt rollout again when fixed

### Rollback Time

- **Target**: < 5 minutes
- **Method**: Feature flag or deployment revert
- **Testing**: Practice rollback in staging

---

## Post-Migration Tasks

### Week 7-8: Stabilization

- [ ] Monitor production metrics daily
- [ ] Address any bugs or issues
- [ ] Optimize performance based on real usage
- [ ] Gather feedback from operations team

### Week 9-10: Optimization

- [ ] Reduce polling interval if latency acceptable
- [ ] Add connection pooling
- [ ] Implement HTTP/2
- [ ] Add caching for pool status

### Week 11-12: Documentation & Training

- [ ] Update all documentation with lessons learned
- [ ] Create video tutorials
- [ ] Write blog post about migration
- [ ] Present at team all-hands

---

## Appendix A: File Checklist

### New Files to Create

- [ ] `.specs/01_cloud_profile.md` ✅ DONE
- [ ] `CLOUD_PROFILE_MIGRATION_PLAN.md` ✅ DONE
- [ ] `FILESYSTEM_DEPENDENCIES_AUDIT.md`
- [ ] `bin/pool-managerd/src/watcher/handoff.rs`
- [ ] `bin/pool-managerd/src/watcher/mod.rs`
- [ ] `bin/orchestratord/src/services/pool_health.rs`
- [ ] `bin/orchestratord/src/config.rs`
- [ ] `bin/pool-managerd/src/config.rs`
- [ ] `test-harness/e2e-cloud/` (new directory)
- [ ] `docs/CLOUD_PROFILE_DEPLOYMENT.md`
- [ ] `docs/runbooks/CLOUD_PROFILE_INCIDENTS.md`
- [ ] `docs/PRODUCTION_ROLLOUT_PLAN.md`
- [ ] `ci/dashboards/cloud_profile_overview.json`
- [ ] `ci/alerts/cloud_profile.yml`
- [ ] `docker-compose.cloud.yml`

### Files to Modify

- [ ] `.specs/20_orchestratord.md`
- [ ] `.specs/30_pool-managerd.md`
- [ ] `bin/orchestratord/src/main.rs`
- [ ] `bin/orchestratord/src/lib.rs`
- [ ] `bin/orchestratord/src/services/handoff.rs`
- [ ] `bin/orchestratord/src/clients/pool_manager.rs`
- [ ] `bin/pool-managerd/src/main.rs`
- [ ] `bin/pool-managerd/src/lib.rs`
- [ ] `bin/orchestratord/bdd/src/steps/background.rs`
- [ ] `README.md`
- [ ] `docs/ARCHITECTURE.md`

---

## Appendix B: Environment Variables

### orchestratord

```bash
# Profile
ORCHD_PROFILE=cloud  # or "home"

# Service binding
ORCHD_BIND_ADDR=0.0.0.0:8080

# pool-managerd endpoints (CLOUD_PROFILE only)
ORCHD_POOL_MANAGERS=http://gpu-1:9200,http://gpu-2:9200

# Polling (CLOUD_PROFILE only)
ORCHD_POOL_POLL_INTERVAL_MS=5000

# Handoff watcher (HOME_PROFILE only)
ORCHD_RUNTIME_DIR=.runtime/engines
ORCHD_HANDOFF_WATCH_INTERVAL_MS=1000

# Admission
ORCHD_ADMISSION_CAPACITY=100
ORCHD_ADMISSION_POLICY=drop-lru

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
PROMETHEUS_METRICS_PORT=9090
RUST_LOG=info,orchestratord=debug
```

### pool-managerd

```bash
# Service binding
POOL_MANAGERD_BIND_ADDR=0.0.0.0:9200

# Handoff watcher
POOL_MANAGERD_RUNTIME_DIR=/var/lib/llama-orch/engines
POOL_MANAGERD_WATCH_INTERVAL_MS=1000
POOL_MANAGERD_AUTO_DELETE_HANDOFF=true

# GPU discovery
POOL_MANAGERD_GPU_DISCOVERY_INTERVAL_MS=10000

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
PROMETHEUS_METRICS_PORT=9091
RUST_LOG=info,pool_managerd=debug

# Callback (optional, future)
POOL_MANAGERD_ORCHESTRATORD_CALLBACK_URL=http://orchestratord:8080/callbacks/pool-ready
```

---

## Appendix C: Testing Checklist

### Unit Tests

- [ ] pool-managerd watcher: file detection
- [ ] pool-managerd watcher: parsing
- [ ] pool-managerd watcher: registry update
- [ ] orchestratord poller: health check
- [ ] orchestratord poller: adapter binding
- [ ] Profile detection logic

### Integration Tests

- [ ] pool-managerd: watcher + registry
- [ ] orchestratord: poller + client
- [ ] End-to-end: handoff → detection → binding

### E2E Tests

- [ ] Distributed handoff flow
- [ ] Multi-pool routing
- [ ] Failure scenarios (crash, network)
- [ ] Load testing (1000 tasks/sec)
- [ ] Chaos testing (random failures)

### BDD Tests

- [ ] Update background.rs for cloud profile
- [ ] All existing tests pass in HOME_PROFILE
- [ ] All existing tests pass in CLOUD_PROFILE (mocked)
- [ ] 100% coverage maintained

---

## Appendix D: Metrics to Track

### Development Metrics

- [ ] Lines of code added
- [ ] Lines of code removed
- [ ] Test coverage %
- [ ] Number of bugs found
- [ ] Number of bugs fixed

### Performance Metrics

- [ ] P50 latency (admission + placement)
- [ ] P95 latency
- [ ] P99 latency
- [ ] Throughput (tasks/sec)
- [ ] Error rate %
- [ ] Pool availability %

### Operational Metrics

- [ ] Deployment time
- [ ] Rollback time
- [ ] MTTR (mean time to recovery)
- [ ] Incidents count
- [ ] On-call pages

---

**Status**: APPROVED  
**Next Review**: Weekly sync (every Monday)  
**Owner**: Architecture Team  
**Last Updated**: 2025-09-30
