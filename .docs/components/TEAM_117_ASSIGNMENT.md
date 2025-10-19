# TEAM-117: Fix Ambiguous Steps

**Priority:** 🚨 CRITICAL  
**Time Estimate:** 4 hours  
**Difficulty:** ⭐⭐ Medium

---

## Your Mission

**Fix all 32 ambiguous step definitions** that are causing "Step match is ambiguous" errors.

**Impact:** This will immediately fix ~32 failing scenarios.

---

## Task List

### Task 1: Identify All Duplicates (30 min)

Run this command to find all duplicate step definitions:

```bash
cd /home/vince/Projects/llama-orch

# Find all step definitions
grep -rn "#\[given\|#\[when\|#\[then" test-harness/bdd/src/steps/ | \
  sed 's/.*expr = "\(.*\)".*/\1/' | \
  sort | uniq -d > /tmp/duplicate_steps.txt

# View duplicates
cat /tmp/duplicate_steps.txt
```

**Deliverable:** List of all duplicate step definitions with file locations

---

### Task 2: Fix Duplicates (2.5 hours)

For each duplicate, choose ONE of these strategies:

#### Strategy A: Rename One (Preferred)
Make the step definitions more specific:

**Before:**
```rust
// File: error_handling.rs
#[then(expr = "queen-rbee attempts SSH connection")]

// File: deadline_propagation.rs  
#[then(expr = "queen-rbee attempts SSH connection")]
```

**After:**
```rust
// File: error_handling.rs
#[then(expr = "queen-rbee attempts SSH connection for error handling")]

// File: deadline_propagation.rs
#[then(expr = "queen-rbee attempts SSH connection with deadline")]
```

#### Strategy B: Consolidate (If identical)
If both implementations are identical, keep one and delete the other:

```rust
// Keep the one in the more appropriate file
// Delete the duplicate
```

#### Strategy C: Parameterize (If similar)
If they differ only slightly, make one parameterized:

**Before:**
```rust
#[given(expr = "rbee-hive is running with 1 worker")]
#[given(expr = "rbee-hive is running with 3 workers")]
#[given(expr = "rbee-hive is running with 5 workers")]
```

**After:**
```rust
#[given(expr = "rbee-hive is running with {int} worker(s)")]
pub async fn given_hive_running_with_workers(world: &mut World, count: usize) {
    // Implementation
}
```

---

### Task 3: Update Feature Files (1 hour)

If you renamed steps, update the feature files:

```bash
# Find which feature files use the old step
grep -r "old step text" test-harness/bdd/tests/features/

# Update them to use the new step text
```

**Important:** Only update if you used Strategy A (rename)

---

### Task 4: Verify (30 min)

```bash
# Compile
cargo build --package test-harness-bdd

# Run tests
cargo xtask bdd:test

# Verify zero ambiguous errors
cargo test --test cucumber 2>&1 | grep "ambiguous" | wc -l
# Should output: 0
```

---

## Specific Duplicates to Fix

### 1. SSH Connection Steps
**Files:** `error_handling.rs`, `deadline_propagation.rs`

```
"queen-rbee attempts SSH connection with 10s timeout"
"queen-rbee attempts SSH connection"
```

**Solution:** Rename to be more specific about context

---

### 2. Model Catalog Steps
**Files:** `model_catalog.rs`, `worker_provisioning.rs`

```
"rbee-hive checks the model catalog"
"rbee-hive inserts model into SQLite catalog"
```

**Solution:** Make one more specific about the operation

---

### 3. Worker State Steps
**Files:** `lifecycle.rs`, `concurrency.rs`, `integration.rs`

```
"rbee-hive is running with 1 worker"
"rbee-hive is running with 3 workers"
"rbee-hive is running with 4 workers"
"rbee-hive is running with 5 workers"
```

**Solution:** Consolidate into parameterized step

---

### 4. Response Validation Steps
**Files:** `authentication.rs`, `error_handling.rs`

```
"the response is:"
"the response format is:"
"response includes details object"
```

**Solution:** Make each more specific about what it validates

---

### 5. Exit Code Steps
**Files:** Multiple files

```
"the exit code is 0"
"the exit code is 1"
"And the exit code is 1"
```

**Solution:** Consolidate into one parameterized step

---

### 6. Worker Processing Steps
**Files:** `deadline_propagation.rs`, `error_handling.rs`, `lifecycle.rs`

```
"worker is processing inference request"
"Given worker is processing inference request"
"worker-001 is processing request"
```

**Solution:** Consolidate or make more specific

---

## Success Criteria

- [ ] All 32 duplicate step definitions resolved
- [ ] Zero "Step match is ambiguous" errors
- [ ] All tests compile successfully
- [ ] Feature files updated if needed
- [ ] No functionality lost
- [ ] Documentation of changes made

---

## Files You'll Modify

Primary files (most duplicates):
- `test-harness/bdd/src/steps/error_handling.rs`
- `test-harness/bdd/src/steps/lifecycle.rs`
- `test-harness/bdd/src/steps/authentication.rs`
- `test-harness/bdd/src/steps/deadline_propagation.rs`
- `test-harness/bdd/src/steps/concurrency.rs`

Secondary files (fewer duplicates):
- `test-harness/bdd/src/steps/integration.rs`
- `test-harness/bdd/src/steps/worker_provisioning.rs`
- `test-harness/bdd/src/steps/model_catalog.rs`

Feature files (if renamed):
- `test-harness/bdd/tests/features/*.feature`

---

## Tips

1. **Start with the easy ones** - Identical implementations can just be deleted
2. **Use grep to find usage** - Before renaming, see where it's used
3. **Test frequently** - Compile after each fix
4. **Document your changes** - Keep notes on what you renamed
5. **Ask for help** - If unsure, check with TEAM-122

---

## Completion Checklist

- [ ] Task 1: All duplicates identified
- [ ] Task 2: All duplicates fixed
- [ ] Task 3: Feature files updated
- [ ] Task 4: Tests verified
- [ ] Branch pushed
- [ ] Completion report created

---

**Status:** 🚀 READY TO START  
**Your Branch:** `fix/team-117-ambiguous-steps`  
**Estimated Time:** 4 hours  
**Impact:** ~32 scenarios fixed immediately
