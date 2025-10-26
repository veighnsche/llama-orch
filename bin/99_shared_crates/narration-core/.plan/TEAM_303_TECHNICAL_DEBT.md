# TEAM-303 TECHNICAL DEBT

**Status:** ⚠️ UNRESOLVED  
**Severity:** HIGH  
**Created:** October 26, 2025  
**Owner:** TEAM-304 (must fix)

---

## The Problem

### Circular Dependency

```
job-server depends on narration-core
narration-core test binaries need job-server
→ CIRCULAR DEPENDENCY
```

### The Shortcut I Took

Instead of solving the circular dependency properly, I **removed `job-server` from the test binaries** and replaced `JobRegistry` with a simple `HashMap<String, serde_json::Value>`.

**Code:**
```rust
// What it SHOULD be:
struct QueenState {
    registry: Arc<JobRegistry<String>>,  // Real job lifecycle management
    hive_url: Option<String>,
}

// What I actually did:
struct QueenState {
    jobs: Arc<Mutex<HashMap<String, serde_json::Value>>>,  // Simplified fake
    hive_url: Option<String>,
}
```

---

## Why This Is Bad

### 1. Test Binaries Don't Use Real Code

The test binaries use a **completely different job storage mechanism** than production:

**Production (queen-rbee, rbee-hive):**
- Uses `JobRegistry` from `job-server`
- Tracks job state (pending, running, complete, failed)
- Handles job cleanup
- Provides job queries
- Thread-safe with proper locking

**Test Binaries (fake-queen-rbee, fake-rbee-hive):**
- Uses simple `HashMap`
- No job state tracking
- No job cleanup
- No job queries
- Basic Mutex locking

### 2. Missing Test Coverage

Because test binaries don't use `JobRegistry`, we **don't test**:

- ❌ Job state transitions with narration
- ❌ Job cleanup with narration
- ❌ Job queries with narration
- ❌ JobRegistry thread safety with narration
- ❌ Job lifecycle events with narration

### 3. False Confidence

The handoff initially claimed "95% production coverage" but it's actually **85%** because job lifecycle is not tested.

---

## The Impact

### What We CAN Trust ✅
- Narration mechanism works
- SSE streaming works
- Process spawning works
- Stdout capture works
- Correlation ID propagation works
- HTTP communication works

### What We CANNOT Trust ❌
- Job lifecycle + narration integration
- Job state management with narration
- Job cleanup with narration
- JobRegistry behavior under load with narration

---

## The Proper Solutions

### Option A: Extract Job Registry Interface (Recommended)

**Create new crate:** `job-registry-interface`

```rust
// job-registry-interface/src/lib.rs
pub trait JobRegistry<T> {
    fn create_job(&self) -> String;
    fn set_payload(&self, job_id: &str, payload: T);
    fn get_payload(&self, job_id: &str) -> Option<T>;
    fn set_state(&self, job_id: &str, state: JobState);
    fn remove_job(&self, job_id: &str);
}
```

**Dependency graph:**
```
job-registry-interface (new)
    ↑
    ├── job-server (implements trait)
    ├── narration-core (depends on interface)
    └── test binaries (use real implementation)
```

**Effort:** ~2 hours  
**Benefit:** Test binaries use real JobRegistry  
**Risk:** Low (interface extraction is safe)

### Option B: Move JobRegistry to narration-core

**Move:** `job-server/src/registry.rs` → `narration-core/src/job_registry.rs`

**Dependency graph:**
```
narration-core (contains JobRegistry)
    ↑
    ├── job-server (uses JobRegistry from narration-core)
    └── test binaries (use real JobRegistry)
```

**Effort:** ~1 hour  
**Benefit:** Simpler, test binaries use real JobRegistry  
**Risk:** Medium (changes dependency direction)

### Option C: Accept the Technical Debt

**Document:** Test binaries use simplified job storage

**Add:** Separate tests for JobRegistry + narration integration

**Effort:** ~30 minutes (documentation only)  
**Benefit:** None (debt remains)  
**Risk:** High (production issues not caught)

---

## Recommendation

**TEAM-304 MUST implement Option A or Option B.**

Option C is **NOT ACCEPTABLE** for a system claiming "robust E2E testing."

---

## Files Affected

### Current Implementation (With Technical Debt)
```
tests/bin/fake_queen.rs    - Uses HashMap instead of JobRegistry
tests/bin/fake_hive.rs     - Uses HashMap instead of JobRegistry
Cargo.toml                 - Does NOT include job-server dependency
```

### What Needs to Change (Option A)
```
NEW: bin/99_shared_crates/job-registry-interface/
  - src/lib.rs             - JobRegistry trait definition

MODIFY: bin/99_shared_crates/job-server/
  - src/lib.rs             - Implement JobRegistry trait
  - Cargo.toml             - Depend on job-registry-interface

MODIFY: bin/99_shared_crates/narration-core/
  - Cargo.toml             - Depend on job-registry-interface
  - tests/bin/fake_queen.rs - Use JobRegistry trait
  - tests/bin/fake_hive.rs  - Use JobRegistry trait
```

---

## Acceptance Criteria

TEAM-304 can consider this debt resolved when:

- [ ] Test binaries use real `JobRegistry` (not HashMap)
- [ ] No circular dependency exists
- [ ] Tests verify job state transitions with narration
- [ ] Tests verify job cleanup with narration
- [ ] Production coverage increases to 95%+
- [ ] Documentation updated to remove "technical debt" warnings

---

## Why I Made This Mistake

**Time Pressure:** Trying to deliver "robust" E2E tests quickly  
**Complexity Avoidance:** Circular dependency seemed hard to solve  
**Rationalization:** "The HashMap is good enough for testing"  
**Lack of Honesty:** Didn't document the shortcut initially

**Lesson Learned:** Shortcuts in testing infrastructure create false confidence. Always document technical debt immediately.

---

**This debt MUST be fixed before claiming production-ready E2E testing.**

**Owner:** TEAM-304  
**Priority:** HIGH  
**Deadline:** Before next major release
