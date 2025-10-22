# ⚠️ CRITICAL: Orchestration Is The Whole Point

**Date:** 2025-10-20  
**Author:** TEAM-159  
**Audience:** All future developers

---

## The Problem

Multiple teams have made the same mistake: **Testing HTTP endpoints instead of orchestration**.

If your test manually spawns rbee-hive, you're NOT testing what matters.

---

## What Is Queen-Rbee?

Queen-rbee is an **ORCHESTRATOR**. Its job is to:

1. Accept commands like "add localhost to catalog"
2. **SSH to the node and spawn rbee-hive daemon**
3. Wait for first heartbeat
4. Trigger device detection
5. Manage the hive lifecycle

If queen can't do step 2, **it's not an orchestrator - it's just a registry with HTTP endpoints**.

---

## The Stupid Mistake

### ❌ What Teams Keep Doing:

```gherkin
Given queen-rbee is running
When test harness spawns rbee-hive manually  # ← WRONG!
Then rbee-hive sends heartbeat to queen
And queen receives it
```

**Result:**
- ✅ Test passes
- ❌ Production fails because queen doesn't know how to spawn hives
- ❌ You wasted time testing HTTP, not orchestration

### ✅ What Should Happen:

```gherkin
Given queen-rbee is running
When user adds localhost to hive catalog via queen's API
Then queen spawns rbee-hive daemon via SSH  # ← CORRECT!
And rbee-hive sends first heartbeat
And queen receives it and does device detection
```

**Result:**
- ✅ Test passes
- ✅ Production works because queen knows how to spawn hives
- ✅ You tested actual orchestration

---

## Ask Yourself

**Before you write any test that spawns rbee-hive:**

> "If the test spawns the hive, how do we know queen can?"

> "What's the point of an orchestrator that doesn't orchestrate?"

If you can't answer these questions, **you're doing it wrong**.

---

## Where The Warnings Are

To prevent this mistake, warnings have been added to:

### 1. Integration Test Steps
**File:** `bin/10_queen_rbee/bdd/src/steps/integration_steps.rs`

```rust
//! ⚠️ CRITICAL: These tests MUST verify that QUEEN SPAWNS THE HIVE
//!
//! The whole point of queen-rbee is ORCHESTRATION. If your test manually spawns
//! rbee-hive, you're NOT testing orchestration - you're just testing HTTP endpoints.
```

### 2. spawn_rbee_hive() Function
**File:** `bin/10_queen_rbee/bdd/src/steps/integration_steps.rs`

```rust
/// ⚠️ WARNING: This function should NOT be used in integration tests!
///
/// TEAM-160: If you're calling this function, you're testing the WRONG thing.
/// The test harness should NOT spawn rbee-hive - QUEEN should spawn it.
///
/// Ask yourself: If the test spawns the hive, how do we know queen can?
/// What's the point of an orchestrator that doesn't orchestrate?
fn spawn_rbee_hive(...) {
    // TODO: DELETE THIS FUNCTION once queen has orchestration logic
}
```

### 3. Integration Test Feature File
**File:** `bin/10_queen_rbee/bdd/tests/features/integration_first_heartbeat.feature`

```gherkin
# ⚠️ CRITICAL WARNING TO FUTURE DEVELOPERS:
#
# This test MUST verify that QUEEN SPAWNS THE HIVE, not the test harness!
#
# If your test manually spawns rbee-hive, you're NOT testing orchestration.
# You're just testing HTTP endpoints, which means production will fail when
# queen tries to spawn hives and doesn't know how.
```

### 4. TEAM-160 Instructions
**File:** `bin/TEAM_160_INSTRUCTIONS.md`

```markdown
## ⚠️ CRITICAL WARNING: Don't Be Stupid About Orchestration

If your test manually spawns rbee-hive, **you're testing the WRONG thing**.

The whole point of queen-rbee is **ORCHESTRATION**.
```

### 5. Production Code
**File:** `bin/10_queen_rbee/src/http/beehives.rs`

```rust
/// ⚠️ CRITICAL TODO: This endpoint should SPAWN rbee-hive on the target node!
///
/// Currently this just adds the node to the catalog and hopes rbee-hive is already running.
/// That's NOT orchestration - that's just a registry.
///
/// What should happen:
/// 1. Validate SSH connection ✅ (we do this)
/// 2. Add node to catalog ✅ (we do this)
/// 3. **SSH to node and spawn rbee-hive daemon** ❌ (WE DON'T DO THIS!)
```

---

## What Needs To Be Implemented

### In Queen-Rbee Production Code:

**File:** `bin/10_queen_rbee/src/http/beehives.rs`

The `handle_add_node()` endpoint needs to:

1. Validate SSH connection (already does this)
2. Add node to catalog (already does this)
3. **SSH to node and spawn rbee-hive** (NOT IMPLEMENTED)
4. Wait for first heartbeat
5. Return success/failure

**Pseudo-code:**
```rust
pub async fn handle_add_node(...) -> impl IntoResponse {
    // Existing validation...
    
    // Add to catalog
    state.catalog.add_hive(...).await?;
    
    // ⚠️ NEW: Spawn rbee-hive on the node
    let spawn_result = ssh_client
        .connect(&req.ssh_host, &req.ssh_user)
        .await?
        .execute(&format!(
            "rbee-hive --port {} --queen-url http://{}:{}",
            req.port, queen_host, queen_port
        ))
        .await?;
    
    if !spawn_result.success {
        return error_response("Failed to spawn rbee-hive");
    }
    
    // Wait for first heartbeat (with timeout)
    let heartbeat_received = wait_for_first_heartbeat(&req.node_name, Duration::from_secs(30)).await;
    
    if !heartbeat_received {
        return error_response("rbee-hive started but didn't send heartbeat");
    }
    
    success_response()
}
```

### In Integration Tests:

**File:** `bin/10_queen_rbee/bdd/src/steps/integration_steps.rs`

Once queen can spawn hives:

1. **DELETE** the `spawn_rbee_hive()` function
2. Update tests to call queen's API instead
3. Verify queen spawns the hive, not the test

**Example:**
```rust
#[when("user adds localhost to hive catalog")]
async fn when_add_localhost(world: &mut BddWorld) {
    let client = reqwest::Client::new();
    let queen_url = format!("http://localhost:{}", world.queen_port.unwrap());
    
    // Call queen's API to add hive
    let response = client
        .post(&format!("{}/v2/registry/beehives/add", queen_url))
        .json(&json!({
            "node_name": "localhost",
            "ssh_host": "localhost",
            "ssh_user": "test",
            "port": 18600
        }))
        .send()
        .await
        .expect("Failed to add hive");
    
    assert!(response.status().is_success(), "Queen failed to add hive");
    
    // Queen should have spawned rbee-hive by now
    // Wait for it to be ready
    wait_for_hive_ready(18600, 10).await.expect("Hive didn't start");
}
```

---

## Timeline

### Phase 1: Implement Orchestration (TEAM-160 or later)
- Implement SSH spawning in `handle_add_node()`
- Add process management
- Add heartbeat waiting logic

### Phase 2: Update Integration Tests (After Phase 1)
- Delete `spawn_rbee_hive()` function
- Update tests to use queen's API
- Verify queen spawns hives

### Phase 3: Verify Production (After Phase 2)
- Deploy to staging
- Add node via queen's API
- Verify rbee-hive spawns automatically
- Verify first heartbeat works

---

## How To Avoid This Mistake

### Before Writing Any Test:

1. **Read this document**
2. **Ask:** "Am I testing orchestration or just HTTP?"
3. **Ask:** "If the test spawns the daemon, how do we know queen can?"
4. **Ask:** "What's the point of an orchestrator that doesn't orchestrate?"

### Red Flags:

- ❌ Test calls `spawn_rbee_hive()` directly
- ❌ Test uses `Command::new("rbee-hive")`
- ❌ Test says "manually start rbee-hive before running"
- ❌ Documentation says "ensure rbee-hive is running"

### Green Flags:

- ✅ Test calls queen's API to add hive
- ✅ Queen spawns rbee-hive via SSH
- ✅ Test waits for heartbeat from queen-spawned hive
- ✅ Test verifies orchestration, not just HTTP

---

## Summary

**The whole point of queen-rbee is ORCHESTRATION.**

If it can't spawn hives, it's not an orchestrator.

If your test spawns hives, you're not testing orchestration.

**Don't be stupid. Test the right thing.**

---

**This document exists because multiple teams made this mistake. Don't be the next one.**
