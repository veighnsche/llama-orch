# Week 4 Checklist: Polish & Production Readiness

**Week:** 4 of 4  
**Goal:** Final hardening, documentation, deployment prep  
**Duration:** 5-6 days  
**Target:** ~200+/300 tests passing (67%+), **READY FOR v0.1.0 RELEASE**

---

## 📋 Priority 1: Graceful Shutdown Completion (1-2 days)

### Task 1.1: Integrate Force-Kill into Shutdown
- [ ] Review existing force_kill_worker() method (TEAM-113 implemented)
- [ ] Add shutdown timeout config (default: 30s graceful, then force-kill)
- [ ] Implement shutdown sequence:
  1. Send HTTP shutdown to all workers
  2. Wait up to 30 seconds for graceful shutdown
  3. Force-kill any remaining workers
  4. Clean up worker registry
- [ ] Log all shutdown events

**Files to modify:**
- `bin/rbee-hive/src/main.rs` - Update shutdown handler
- `bin/rbee-hive/src/shutdown.rs` - NEW FILE - Shutdown orchestration
- `bin/queen-rbee/src/main.rs` - Update cascading shutdown

### Task 1.2: Test Shutdown Scenarios
- [ ] Test: All workers shutdown gracefully
- [ ] Test: One worker hangs, gets force-killed
- [ ] Test: Multiple workers hang, all get force-killed
- [ ] Test: Shutdown during active inference
- [ ] Test: Shutdown with no workers running

### Task 1.3: Shutdown Metrics
- [ ] Add metric: rbee_hive_shutdown_duration_seconds
- [ ] Add metric: rbee_hive_workers_force_killed_total
- [ ] Add metric: rbee_hive_workers_graceful_shutdown_total

**Files to modify:**
- `bin/rbee-hive/src/metrics.rs` - Add shutdown metrics

**Impact:** ✅ Clean shutdowns, no orphaned processes

---

## 📋 Priority 2: Configuration Management (1-2 days)

### Task 2.1: Config File Validation
- [ ] Add config file schema validation on startup
- [ ] Validate all required fields present
- [ ] Validate field types and ranges
- [ ] Provide helpful error messages for invalid config
- [ ] Add --validate-config flag to check config without starting

**Files to modify:**
- `bin/rbee-hive/src/config.rs` - Add validation
- `bin/queen-rbee/src/config.rs` - Add validation
- `bin/llm-worker-rbee/src/config.rs` - Add validation

### Task 2.2: Config Reload on SIGHUP
- [ ] Add SIGHUP signal handler
- [ ] Reload config file on SIGHUP
- [ ] Validate new config before applying
- [ ] Apply config changes without restart (where possible)
- [ ] Log config reload events
- [ ] Audit log config changes

**Files to modify:**
- `bin/rbee-hive/src/main.rs` - Add SIGHUP handler
- `bin/queen-rbee/src/main.rs` - Add SIGHUP handler
- `bin/rbee-hive/src/config.rs` - Add reload() method

### Task 2.3: Document All Config Options
- [ ] Create CONFIG.md with all options
- [ ] Document default values
- [ ] Document valid ranges
- [ ] Document environment variable overrides
- [ ] Add examples for common scenarios
- [ ] Add migration guide from old configs

**Files to create:**
- `docs/CONFIG.md` - NEW FILE - Configuration reference

### Task 2.4: Config Validation Tests
- [ ] Test: Valid config loads successfully
- [ ] Test: Invalid config rejected with helpful error
- [ ] Test: Config reload on SIGHUP works
- [ ] Test: Invalid reload rejected, old config retained
- [ ] Test: Environment variables override config file

**Impact:** ✅ Better ops experience, fewer config errors

---

## 📋 Priority 3: Integration Testing (2-3 days)

### Task 3.1: Full Inference Flow Test
- [ ] Test: queen-rbee → rbee-hive → worker → response
- [ ] Test: Streaming SSE response works end-to-end
- [ ] Test: Authentication required at each hop
- [ ] Test: Input validation at each hop
- [ ] Test: Deadline propagation through chain
- [ ] Test: Audit events logged at each step

**Files to modify:**
- `test-harness/bdd/tests/features/160-end-to-end-flows.feature` - Implement steps

### Task 3.2: Multi-Worker Scenarios
- [ ] Test: Spawn 3 workers with same model
- [ ] Test: Load balance across workers
- [ ] Test: Worker failure doesn't affect others
- [ ] Test: Worker restart works correctly
- [ ] Test: Concurrent inference requests

**Files to modify:**
- `test-harness/bdd/tests/features/200-concurrency.feature` - Implement steps

### Task 3.3: Failure Recovery Scenarios
- [ ] Test: Worker crashes during inference
- [ ] Test: Worker hangs, gets force-killed
- [ ] Test: rbee-hive crashes, workers continue
- [ ] Test: queen-rbee crashes, rbee-hive continues
- [ ] Test: Network partition between components
- [ ] Test: Disk full during model download

**Files to modify:**
- `test-harness/bdd/tests/features/320-error-handling.feature` - Implement steps

### Task 3.4: Resource Exhaustion Scenarios
- [ ] Test: Out of memory, worker spawn rejected
- [ ] Test: Out of VRAM, worker spawn rejected
- [ ] Test: Out of disk space, download rejected
- [ ] Test: Too many workers, spawn rejected
- [ ] Test: Resource limits enforced correctly

**Files to modify:**
- `test-harness/bdd/tests/features/230-resource-management.feature` - Implement steps

### Task 3.5: Implement Missing BDD Steps
- [ ] Review 87 missing steps identified in Week 1
- [ ] Implement 30-40 high-value steps
- [ ] Focus on integration scenarios
- [ ] Follow TEAM-112 pattern (no TODOs)
- [ ] Verify tests pass

**Impact:** ✅ Higher confidence in production

---

## 📋 Priority 4: Documentation (1 day)

### Task 4.1: Production Deployment Guide
- [ ] Document system requirements
- [ ] Document installation steps
- [ ] Document configuration steps
- [ ] Document secret management setup
- [ ] Document monitoring setup
- [ ] Document backup procedures
- [ ] Document upgrade procedures

**Files to create:**
- `docs/DEPLOYMENT.md` - NEW FILE

### Task 4.2: Secret Management Documentation
- [ ] Document API token generation
- [ ] Document token file permissions (0600)
- [ ] Document systemd credentials integration
- [ ] Document token rotation procedures
- [ ] Document emergency token revocation

**Files to create:**
- `docs/SECRETS.md` - NEW FILE

### Task 4.3: Monitoring Setup Guide
- [ ] Document Prometheus setup
- [ ] Document Grafana setup
- [ ] Document dashboard import
- [ ] Document alerting rules
- [ ] Document log aggregation
- [ ] Document audit log retention

**Files to create:**
- `docs/MONITORING.md` - NEW FILE

### Task 4.4: Troubleshooting Guide
- [ ] Common issues and solutions
- [ ] Debug logging instructions
- [ ] Health check procedures
- [ ] Performance tuning tips
- [ ] Recovery procedures

**Files to create:**
- `docs/TROUBLESHOOTING.md` - NEW FILE

### Task 4.5: API Documentation
- [ ] Document all HTTP endpoints
- [ ] Document request/response formats
- [ ] Document authentication
- [ ] Document error codes
- [ ] Add curl examples
- [ ] Add client library examples

**Files to create:**
- `docs/API.md` - NEW FILE

### Task 4.6: Update README
- [ ] Add production deployment section
- [ ] Add monitoring section
- [ ] Add troubleshooting section
- [ ] Add API reference link
- [ ] Add architecture diagram
- [ ] Add quick start guide

**Files to modify:**
- `README.md` - Update with production info

**Impact:** ✅ Easier deployment, better support

---

## 📊 Week 4 Deliverables

- [ ] Graceful shutdown complete with force-kill fallback
- [ ] Config validation on startup
- [ ] Config reload on SIGHUP
- [ ] Full config documentation
- [ ] 30-40 new BDD steps implemented
- [ ] Integration tests passing
- [ ] Failure recovery tests passing
- [ ] Resource exhaustion tests passing
- [ ] Production deployment guide
- [ ] Monitoring setup guide
- [ ] Troubleshooting guide
- [ ] API documentation
- [ ] ~200+/300 tests passing (67%+)
- [ ] **READY FOR v0.1.0 RELEASE** 🎉

---

## 🎯 Success Criteria for v0.1.0

### Must Have (All P0 Items)
- [x] Worker PID tracking and force-kill (TEAM-113)
- [x] Authentication on all components
- [x] Input validation on all components
- [x] Secrets loaded from files
- [ ] No unwrap/expect in production paths
- [ ] Graceful shutdown with force-kill fallback

### Should Have (P1 Items)
- [ ] Worker restart policy with exponential backoff
- [ ] Heartbeat mechanism with stale worker detection
- [ ] Audit logging wired to all components
- [ ] Deadline propagation wired end-to-end
- [ ] Resource limits (memory, VRAM, disk)

### Quality Metrics
- [ ] 200+/300 BDD tests passing (67%+)
- [ ] Zero panics in production code paths
- [ ] All HTTP endpoints authenticated
- [ ] All inputs validated
- [ ] Comprehensive error handling

### Documentation
- [ ] Production deployment guide
- [ ] API documentation
- [ ] Troubleshooting guide
- [ ] Monitoring setup guide

---

## 📝 Notes for Release Team

### Pre-Release Checklist
- [ ] All Week 4 tasks complete
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Security audit complete
- [ ] Performance testing complete
- [ ] Upgrade path tested
- [ ] Rollback procedure tested

### Release Process
1. Tag release: `git tag v0.1.0`
2. Build release binaries
3. Generate changelog
4. Update documentation
5. Publish release notes
6. Announce release

### Post-Release
- [ ] Monitor for issues
- [ ] Respond to bug reports
- [ ] Plan v0.2.0 features
- [ ] Gather user feedback

---

## 🎉 Celebration Criteria

**v0.1.0 is production-ready when:**
- ✅ All P0 items complete
- ✅ 200+ tests passing
- ✅ Documentation complete
- ✅ Security audit passed
- ✅ Performance acceptable
- ✅ Team confident in production deployment

**Then:** 🎉 **SHIP IT!** 🚀

---

**Created by:** TEAM-113  
**Date:** 2025-10-18  
**For:** Week 4 implementation team and release preparation
