# TEAM-066 HANDOFF: Fix All FAKE Step Functions

**From:** TEAM-065  
**To:** TEAM-066  
**Date:** 2025-10-11  
**Status:** 🔥 CRITICAL - Fix False Positives

---

## Mission

**Delete or wire up ALL ~80 FAKE step functions** that create false positives. These functions make tests pass without testing product code.

---

## Who Created the False Positives?

### Primary Culprits:

**TEAM-040** - Created World state infrastructure (not fake itself, but enabled fakes)
- Created `World` struct with all the state fields
- Created `background.rs` with 5 FAKE functions

**TEAM-042** - Created MOST false positives (~60 functions)
- Created `happy_path.rs` with ~40 FAKE functions
- Created `edge_cases.rs` (but TEAM-060 fixed most of them)
- Created `model_provisioning.rs` with 2 FAKE functions
- Created `registry.rs` (but TEAM-064 fixed it)
- Created `beehive_registry.rs` with ~8 FAKE functions
- Created `worker_preflight.rs` with ~10 FAKE functions (all TODO)
- Created `pool_preflight.rs` with ~10 FAKE functions (all TODO)
- Created `error_responses.rs` with ~5 FAKE functions (all TODO)

**TEAM-041** - Created beehive_registry.rs infrastructure
- Some functions later fixed by TEAM-043/044/055

**TEAM-053** - Created TODO functions (not fake, just not implemented)
- Created `lifecycle.rs`, `worker_startup.rs`, `worker_health.rs`, `worker_registration.rs`, `inference_execution.rs`
- All are TODO (just debug logs), not FAKE

### Teams That FIXED Things:

**TEAM-043/044** - Fixed CLI commands and some registry functions ✅
**TEAM-055** - Added HTTP retry logic ✅
**TEAM-060** - Fixed edge_cases.rs with real shell commands ✅
**TEAM-062** - Fixed error_handling.rs with real SSH/HTTP tests ✅
**TEAM-064** - Fixed registry.rs with real rbee-hive integration ✅

---

## The Problem

**TEAM-042 created ~60 FAKE functions** that:
1. Update `world.workers`, `world.model_catalog`, `world.beehive_nodes`, etc.
2. Make tests PASS without testing any product code
3. Create FALSE POSITIVES - tests pass but validate nothing

**Example FAKE function:**
```rust
#[then(expr = "the worker registry returns an empty list")]
pub async fn then_registry_returns_empty(world: &mut World) {
    world.workers.clear();  // ← FAKE! Just clears World state
    tracing::info!("✅ Worker registry returns empty list");
}
```

**This makes tests pass even if the real registry is broken!**

---

## Files Requiring Fixes

### Priority 1: FAKE Functions (Must Fix)

**File: `src/steps/happy_path.rs`** - Created by TEAM-042
- ~15 FAKE functions marked by TEAM-065
- Lines: 19, 37, 43, 50, 66, 77, 84, 121, 143, 157, 253, 303, 319, 336, 351, 360, 371, 419
- **Action:** Delete or wire to real products

**File: `src/steps/background.rs`** - Created by TEAM-040
- 5 FAKE functions marked by TEAM-065
- Lines: 18, 36, 43, 52, 66
- **Action:** Delete or wire to real products

**File: `src/steps/beehive_registry.rs`** - Created by TEAM-041, modified by TEAM-042
- ~8 FAKE functions marked by TEAM-065
- Lines: 114, 192, 227, 265, 308, 342
- **Action:** Delete or wire to real products

**File: `src/steps/model_provisioning.rs`** - Created by TEAM-042
- 2 FAKE functions marked by TEAM-065
- Lines: 18, 138
- **Action:** Delete or wire to real products

### Priority 2: Already Good (Keep These)

**File: `src/steps/registry.rs`** - Fixed by TEAM-064 ✅
**File: `src/steps/edge_cases.rs`** - Fixed by TEAM-060 ✅
**File: `src/steps/error_handling.rs`** - Fixed by TEAM-062 ✅
**File: `src/steps/cli_commands.rs`** - Fixed by TEAM-043 ✅

### Priority 3: TODO Functions (Can Wait)

These are NOT fake, just not implemented yet:
- `src/steps/lifecycle.rs` - Created by TEAM-053
- `src/steps/worker_preflight.rs` - Created by TEAM-042
- `src/steps/pool_preflight.rs` - Created by TEAM-042
- `src/steps/worker_health.rs` - Created by TEAM-053
- `src/steps/worker_startup.rs` - Created by TEAM-053
- `src/steps/worker_registration.rs` - Created by TEAM-053
- `src/steps/inference_execution.rs` - Created by TEAM-053
- `src/steps/gguf.rs` - Created by TEAM-036
- `src/steps/error_responses.rs` - Created by TEAM-042

---

## Your Mission: Fix All FAKE Functions

### Option 1: Nuclear Approach (Recommended)

**Delete all FAKE functions immediately:**

```bash
cd /home/vince/Projects/llama-orch/test-harness/bdd

# Find all FAKE functions
grep -n "// FAKE:" src/steps/*.rs

# For each FAKE function, either:
# 1. Delete it entirely, OR
# 2. Convert to todo!() macro
```

**Example deletion:**
```rust
// BEFORE (FAKE)
// FAKE: Only updates World.workers, doesn't test real registry
#[given(expr = "no workers are registered for model {string}")]
pub async fn given_no_workers_for_model(world: &mut World, model_ref: String) {
    world.workers.retain(|_, worker| worker.model_ref != model_ref);
    tracing::debug!("Cleared workers for model: {}", model_ref);
}

// AFTER (DELETED)
// Function removed - was creating false positive
```

**Example conversion to todo:**
```rust
// BEFORE (FAKE)
// FAKE: Only updates World.workers, doesn't test real registry
#[given(expr = "no workers are registered for model {string}")]
pub async fn given_no_workers_for_model(world: &mut World, model_ref: String) {
    world.workers.retain(|_, worker| worker.model_ref != model_ref);
    tracing::debug!("Cleared workers for model: {}", model_ref);
}

// AFTER (TODO)
#[given(expr = "no workers are registered for model {string}")]
pub async fn given_no_workers_for_model(world: &mut World, model_ref: String) {
    todo!("Wire to real rbee-hive registry - was FAKE (only updated World state)");
}
```

### Option 2: Wire to Real Products (Ideal but Slower)

**Follow TEAM-064's pattern from registry.rs:**

```rust
// BEFORE (FAKE)
// FAKE: Only updates World.workers, doesn't test real registry
#[then(expr = "the worker registry returns an empty list")]
pub async fn then_registry_returns_empty(world: &mut World) {
    world.workers.clear();
    tracing::info!("✅ Worker registry returns empty list");
}

// AFTER (REAL)
#[then(expr = "the worker registry returns an empty list")]
pub async fn then_registry_returns_empty(world: &mut World) {
    let registry = world.hive_registry();
    let workers = registry.list().await;
    assert!(workers.is_empty(), "Expected empty registry but found {} workers", workers.len());
    tracing::info!("✅ Verified registry is empty");
}
```

---

## Implementation Plan

### Phase 1: Audit (30 minutes)

```bash
# Count FAKE functions
grep -c "// FAKE:" src/steps/*.rs

# List all FAKE functions with line numbers
grep -n "// FAKE:" src/steps/*.rs > FAKE_FUNCTIONS.txt
```

### Phase 2: Choose Strategy (15 minutes)

**Decision matrix:**
- **Delete:** Fast (2-3 hours), eliminates false positives immediately
- **Convert to todo!():** Fast (2-3 hours), makes it obvious what needs work
- **Wire to products:** Slow (1-2 weeks), but creates real tests

**Recommendation:** Delete or convert to `todo!()` for now. Wire to products incrementally later.

### Phase 3: Execute (2-3 hours for delete/todo, 1-2 weeks for wiring)

**For each FAKE function:**
1. Read the function
2. Decide: Delete or convert to `todo!()`?
3. Make the change
4. Run `cargo check --bin bdd-runner`
5. Move to next function

### Phase 4: Verify (30 minutes)

```bash
# Should return 0 results
grep -c "// FAKE:" src/steps/*.rs

# Verify compilation
cargo check --bin bdd-runner

# Run tests (many will fail/skip, which is GOOD)
cargo run --bin bdd-runner
```

---

## Success Criteria

### Must Complete:
- [ ] All ~30 FAKE functions in marked files are deleted or converted to `todo!()`
- [ ] No `// FAKE:` comments remain in code
- [ ] `cargo check --bin bdd-runner` passes
- [ ] Tests that were passing due to false positives now fail or skip (expected!)

### Optional (for later teams):
- [ ] Wire FAKE functions to real products incrementally
- [ ] Follow TEAM-064's pattern from registry.rs
- [ ] Add real assertions and verifications

---

## Testing Commands

```bash
cd test-harness/bdd

# Check compilation
cargo check --bin bdd-runner

# Run tests (expect failures/skips - that's GOOD!)
cargo run --bin bdd-runner

# Count remaining FAKE functions (should be 0)
grep -c "// FAKE:" src/steps/*.rs
```

---

## Key Lessons

### What Went Wrong:

1. **TEAM-042 created ~60 false positives** by updating World state instead of testing products
2. **Tests passed but validated nothing** - the fraud crisis
3. **Took 20+ teams to discover** the problem

### What Went Right:

1. **TEAM-043/044** fixed CLI commands with real process execution
2. **TEAM-060** fixed edge cases with real shell commands
3. **TEAM-062** fixed error handling with real SSH/HTTP tests
4. **TEAM-064** fixed registry with real rbee-hive integration
5. **TEAM-065** audited and marked all FAKE functions

### For Future Teams:

**Always ask:** "Does this function test product code or just update World state?"

If it only updates World state → **IT'S FAKE** → Delete it or wire it to products.

---

## Files Modified by TEAM-065

- `TEAM_065_HANDOFF.md` - Complete audit
- `src/steps/happy_path.rs` - Marked 15 FAKE functions
- `src/steps/background.rs` - Marked 5 FAKE functions
- `src/steps/beehive_registry.rs` - Marked 8 FAKE functions
- `src/steps/model_provisioning.rs` - Marked 2 FAKE functions

**Total: ~30 FAKE functions marked with `// FAKE:` comments**

---

## Critical Warnings

1. **DO NOT remove the warning headers** - They're protected by TEAM-064
2. **DO NOT create multiple .md files** - This is the ONLY handoff file
3. **DO NOT skip deleting FAKE functions** - They create false positives
4. **DO NOT assume passing tests mean working code** - Most are FAKE

---

## Summary

**The fraud crisis:** TEAM-042 created ~60 FAKE functions that make tests pass without testing products.

**Your job:** Delete or convert all FAKE functions to `todo!()` to eliminate false positives.

**Timeline:** 2-3 hours for delete/todo approach, 1-2 weeks for wiring to products.

**Success:** Zero `// FAKE:` comments remain, tests fail honestly instead of passing falsely.

---

**TEAM-065 signing off. Fix the false positives!**

🎯 **Next team: Delete all FAKE functions or convert to todo!()** 🔥

---

## Appendix: Quick Reference

### FAKE Function Locations

**happy_path.rs (15 FAKE):**
- Line 19: `given_no_workers_for_model`
- Line 37: `given_node_ram`
- Line 43: `given_node_metal_backend`
- Line 50: `given_node_cuda_backend`
- Line 66: `then_query_worker_registry`
- Line 77: `then_registry_returns_empty`
- Line 84: `then_pool_preflight_check`
- Line 121: `then_download_progress_stream`
- Line 143: `then_download_completes`
- Line 157: `then_register_model_in_catalog`
- Line 253: `then_worker_ready_callback`
- Line 303: `then_stream_loading_progress`
- Line 319: `then_worker_completes_loading`
- Line 336: `then_stream_tokens`
- Line 351: `then_display_tokens`
- Line 360: `then_inference_completes`
- Line 371: `then_worker_transitions_to_state`
- Line 419: `then_update_last_connected`

**background.rs (5 FAKE):**
- Line 18: `given_topology`
- Line 36: `given_current_node`
- Line 43: `given_queen_rbee_url`
- Line 52: `given_model_catalog_path`
- Line 66: `given_beehive_registry_path`

**beehive_registry.rs (8 FAKE):**
- Line 114: `given_registry_empty`
- Line 192: `world.beehive_nodes.insert` (in `given_node_in_registry`)
- Line 227: `given_node_not_in_registry`
- Line 265: `then_save_node_to_registry`
- Line 308: `then_display_output`
- Line 342: `then_remove_node_from_registry`

**model_provisioning.rs (2 FAKE):**
- Line 18: `given_model_catalog_contains`
- Line 138: `then_if_retries_fail_return_error`

---

## Signature

**Created by:** TEAM-065  
**Date:** 2025-10-11  
**Task:** Identify teams that created false positives and create handoff to fix them  
**Result:** TEAM-042 created ~60 FAKE functions. TEAM-066 must delete or wire them to products.
