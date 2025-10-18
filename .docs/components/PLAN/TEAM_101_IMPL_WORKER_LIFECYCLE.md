# TEAM-101: Implementation - Worker Lifecycle

**Phase:** 2 - Implementation  
**Duration:** 3-4 days  
**Priority:** P0 - Critical  
**Status:** 🔴 NOT STARTED

---

## Mission

Implement worker lifecycle features to make BDD tests pass:
1. PID Tracking
2. Force-Kill capability
3. Ready Timeout
4. Process Liveness Checks

**Prerequisite:** TEAM-098 BDD tests MUST be complete and failing

---

## Tasks

### 1. PID Tracking (Day 1)
- [ ] Add `pid: Option<u32>` to `WorkerInfo` struct
- [ ] Store `child.id()` during spawn
- [ ] Update registry methods to handle PID
- [ ] Add PID to worker listing

**Files:**
- `bin/rbee-hive/src/registry.rs`
- `bin/rbee-hive/src/http/workers.rs`

---

### 2. Force-Kill (Day 2)
- [ ] Implement `force_kill_worker(pid)` method
- [ ] SIGTERM → wait 10s → SIGKILL sequence
- [ ] Add force-kill to shutdown sequence
- [ ] Log force-kill events

**Files:**
- `bin/rbee-hive/src/commands/daemon.rs`
- `bin/rbee-hive/src/monitor.rs`

---

### 3. Ready Timeout (Day 3)
- [ ] Add timeout for Loading state (30s)
- [ ] Auto-kill workers stuck in Loading
- [ ] Log timeout events

---

### 4. Process Liveness (Day 4)
- [ ] Add process existence checks
- [ ] Detect crashes faster than HTTP
- [ ] Update health monitoring

---

## Acceptance Criteria

- [ ] All TEAM-098 lifecycle tests pass
- [ ] No regressions in existing tests
- [ ] Code coverage > 80%
- [ ] TEAM-101 signature added to modified files

---

## Checklist

**Implementation:**
- [ ] PID tracking ❌ TODO
- [ ] Force-kill ❌ TODO
- [ ] Ready timeout ❌ TODO
- [ ] Process liveness ❌ TODO

**Testing:**
- [ ] BDD tests pass ❌ TODO
- [ ] No regressions ❌ TODO
- [ ] Coverage > 80% ❌ TODO

**Completion:** 0/4 tasks (0%)

---

**Created by:** TEAM-096 | 2025-10-18  
**Assigned to:** TEAM-101  
**Prerequisite:** TEAM-098 tests complete  
**Next Team:** TEAM-102 (Security Implementation)
