---
trigger: glob
globs: **/test-harness/bdd/**/*
---

# MANDATORY WORK REQUIREMENTS FOR BDD TEST TEAMS

## ⚠️ CRITICAL: NO MORE "TODO AND BAIL" ⚠️

**This rule is MANDATORY. Violations will result in work being rejected.**

---

## MINIMUM WORK REQUIREMENT

**Every team working on BDD tests MUST implement at least 10 functions that call real product APIs.**

### What Counts as "Implemented"

✅ **VALID IMPLEMENTATION:**
```rust
// TEAM-XXX: Verify worker state via WorkerRegistry API
pub async fn then_worker_completes_loading(world: &mut World, state: String) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    assert!(!workers.is_empty(), "No workers in registry");
    // ... actual verification logic
}
```

❌ **INVALID - MARKING AS TODO:**
```rust
// TODO: Implement this later
pub async fn then_worker_completes_loading(world: &mut World, state: String) {
    tracing::debug!("TODO: Implement");
}
```

❌ **INVALID - WORLD STATE ONLY:**
```rust
pub async fn then_worker_completes_loading(world: &mut World, state: String) {
    world.workers.insert("fake-id", fake_worker); // NO API CALL
}
```

---

## BANNED PRACTICES

### 1. ❌ BANNED: Marking Functions as TODO

**You are NOT ALLOWED to mark functions as TODO and move on.**

If you can't implement it:
- **Option A:** DELETE the function entirely
- **Option B:** Implement it with real API calls
- **Option C:** Ask for help

**NOT ALLOWED:** Mark as TODO and write a handoff for the next team.

### 2. ❌ BANNED: Writing Handoffs Without Implementation

**You are NOT ALLOWED to write a handoff document without implementing at least 10 functions first.**

Handoff requirements:
- Maximum 2 pages
- Must include code examples of what you implemented
- Must show actual progress (function count, API calls added)
- Must NOT include TODO lists for the next team

### 3. ❌ BANNED: Analysis Documents Without Code

**You are NOT ALLOWED to write analysis documents as a substitute for implementation.**

Analysis is fine as a supplement, but:
- Must implement 10 functions FIRST
- Then write analysis if needed
- Analysis does not count toward work requirement

### 4. ❌ BANNED: "I'll Let the Next Team Handle It"

**You are NOT ALLOWED to defer work to the next team.**

If previous teams marked something TODO:
- IGNORE the TODO marker
- IMPLEMENT the function yourself
- Do NOT perpetuate the TODO culture

---

## REQUIRED APIS TO USE

**These APIs are already available. USE THEM:**

### WorkerRegistry (bin/rbee-hive/src/registry.rs)
```rust
use rbee_hive::registry::{WorkerRegistry, WorkerInfo, WorkerState};

let registry = world.hive_registry();
let workers = registry.list().await;              // List all workers
let worker = registry.get(id).await;              // Get specific worker
registry.register(worker).await;                  // Register worker
registry.update_state(id, state).await;           // Update state
registry.remove(id).await;                        // Remove worker
```

### ModelProvisioner (bin/rbee-hive/src/provisioner/)
```rust
use rbee_hive::provisioner::ModelProvisioner;

let provisioner = ModelProvisioner::new(base_dir);
let model = provisioner.find_local_model(ref);    // Find model
provisioner.download_model(ref).await;            // Download model
```

### DownloadTracker (bin/rbee-hive/src/download_tracker.rs)
```rust
use rbee_hive::download_tracker::DownloadTracker;

let tracker = DownloadTracker::new();
let downloads = tracker.list_active().await;      // Active downloads
let progress = tracker.get_progress(id).await;    // Get progress
```

**ALL OF THESE ARE READY. JUST IMPORT AND CALL THEM.**

---

## VERIFICATION CHECKLIST

Before writing a handoff, verify:

- [ ] I implemented at least 10 functions with real API calls
- [ ] Each function calls a product API (WorkerRegistry, ModelProvisioner, etc.)
- [ ] I did NOT mark any functions as TODO
- [ ] I did NOT write "the next team should implement X"
- [ ] My handoff is 2 pages or less
- [ ] My handoff includes code examples of what I implemented
- [ ] I can show actual progress (function count, API calls)

**If you cannot check ALL boxes, you are NOT done. Keep working.**

---

## CONSEQUENCES OF VIOLATIONS

**If you violate these rules:**

1. Your work will be REJECTED
2. Your handoff will be DELETED
3. The next team will IGNORE your TODO markers
4. You will be cited in the "teams that failed" list

**67 teams have already failed by doing minimal work. Don't be team 68.**

---

## EXAMPLES OF GOOD WORK

### TEAM-064 (GOOD)
- Implemented 5 functions with WorkerRegistry API
- Connected BDD to real product code
- Wrote 2-page handoff with code examples
- **Result:** ACCEPTED

### TEAM-066 (GOOD)
- Implemented 5 functions with WorkerRegistry and ModelProvisioner
- Clarified test setup vs product behavior
- Wrote concise handoff
- **Result:** ACCEPTED

### TEAM-067 (PARTIAL)
- Implemented 3 functions (should have been 10)
- Wrote critical analysis exposing systemic failure
- **Result:** PARTIAL CREDIT (C+ grade)

---

## EXAMPLES OF BAD WORK

### TEAM-062 (BAD)
- Marked 20 functions as TODO
- Wrote 12 documentation files
- Implemented 0 functions
- **Result:** REJECTED - All work was regression

### TEAM-065 (BAD)
- Audited functions and marked 80 as FAKE
- Implemented 0 functions
- Wrote handoff saying "next team should implement"
- **Result:** REJECTED - No actual progress

---

## THE BOTTOM LINE

**IMPLEMENT 10 FUNCTIONS MINIMUM. NO EXCEPTIONS.**

**NO MORE "TODO AND BAIL".**

**THE APIS ARE READY. JUST USE THEM.**

---

## Quick Start: How to Implement a Function

1. **Find a TODO function** in `src/steps/*.rs`
2. **Look up the API** in `bin/rbee-hive/src/`
3. **Import and call it:**
   ```rust
   let registry = world.hive_registry();
   let workers = registry.list().await;
   assert!(!workers.is_empty(), "Expected workers");
   ```
4. **Verify it compiles:** `cargo check --bin bdd-runner`
5. **Repeat 9 more times**
6. **Write SHORT handoff** (2 pages max)

**Total time: 4-6 hours for 10 functions.**

**If you can't do this, you shouldn't be working on BDD tests.**

---

**This is not optional. This is mandatory.**

**Implement the functions. Stop writing TODOs. Make actual progress.**