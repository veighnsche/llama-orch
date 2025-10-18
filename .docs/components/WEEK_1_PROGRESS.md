# Week 1 Progress: Error Handling & BDD Steps

**Team:** TEAM-113 (continuing)  
**Week:** 1 of 4  
**Goal:** Eliminate panics, implement easy BDD wins  
**Status:** 🟡 IN PROGRESS

---

## 📋 Tasks Overview

### Priority 1: Error Handling Audit (3-4 days)
**Status:** 🟡 IN PROGRESS

**Audit Results:**
```bash
# Production code unwrap() calls found:
bin/rbee-hive/src/metrics.rs: 1
bin/rbee-keeper/src/pool_client.rs: 3 (test code)
bin/rbee-keeper/src/ssh.rs: 1
bin/rbee-hive/src/http/metrics.rs: 2 (test code)
bin/rbee-hive/src/http/health.rs: 2 (test code)
bin/rbee-hive/src/http/routes.rs: 1 (test code)
bin/rbee-hive/src/http/workers.rs: 9 (mostly test code)
bin/queen-rbee/src/http/routes.rs: 1 (test code)
bin/queen-rbee/src/worker_registry.rs: 2

# Production code expect() calls found:
bin/rbee-hive/src/metrics.rs: 5
bin/queen-rbee/src/preflight/rbee_hive.rs: 1
bin/queen-rbee/src/beehive_registry.rs: 1
bin/shared-crates/auth-min/src/lib.rs: 1
bin/llm-worker-rbee/src/common/error.rs: 9
```

**Analysis:**
- ✅ Most unwrap() calls are in test code (acceptable)
- 🔴 ~10-15 unwrap() calls in production code (need fixing)
- 🔴 ~20 expect() calls in production code (need fixing)
- 🟢 No unwrap() in critical request paths (good!)

**Critical Files to Fix:**
1. `bin/rbee-hive/src/metrics.rs` - 1 unwrap, 5 expect
2. `bin/queen-rbee/src/worker_registry.rs` - 2 unwrap
3. `bin/queen-rbee/src/beehive_registry.rs` - 1 expect
4. `bin/llm-worker-rbee/src/common/error.rs` - 9 expect

---

### Priority 2: Missing BDD Steps (4-6 hours)
**Status:** ⏳ PENDING

**Steps to Implement:**
- [ ] Find missing steps: `cargo test --test cucumber 2>&1 | grep "Step doesn't match"`
- [ ] Implement 20-30 stub functions
- [ ] Follow TEAM-112 pattern (tracing::info + minimal state)
- [ ] No TODO markers

---

## 🎯 Week 1 Goals

**Deliverables:**
- [ ] Zero unwrap/expect in critical paths
- [ ] 20-30 new BDD steps implemented
- [ ] ~110-120/300 tests passing (37-40%)

**Current Status:**
- Tests passing: ~85-90/300 (28-30%)
- Target: 110-120/300 (37-40%)
- Improvement needed: +25-35 tests

---

## 📝 Work Log

### 2025-10-18 - TEAM-113

**Completed:**
- ✅ Audited unwrap() calls in production code
- ✅ Audited expect() calls in production code
- ✅ Identified critical files for fixing

**Next Steps:**
1. Fix metrics.rs unwrap/expect calls
2. Fix worker_registry.rs unwrap calls
3. Fix beehive_registry.rs expect calls
4. Implement missing BDD steps

---

**Updated by:** TEAM-113  
**Date:** 2025-10-18  
**Status:** 🟡 IN PROGRESS
