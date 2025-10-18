# TEAM-071 HANDOFF - NICE! 🐝

**From:** TEAM-070  
**To:** TEAM-071  
**Date:** 2025-10-11  
**Status:** Ready for implementation

---

## Your Mission - NICE!

**Implement at least 10 functions from the remaining TODO list using real API calls.**

TEAM-070 completed 16 functions (160% of minimum). You should aim for at least 10 functions, but feel free to exceed this like TEAM-070 did! NICE!

---

## What TEAM-070 Completed - NICE!

### Priorities 10-12 (100% Complete)
- ✅ Priority 10: Worker Health (7/7)
- ✅ Priority 11: Lifecycle (4/4)
- ✅ Priority 12: Edge Cases (5/5)

**Total: 16 functions with real API calls**

---

## Your Next Priorities - NICE!

### Recommended: Start with Priority 13-16 (12 functions)

#### Priority 13: Error Handling Functions (4 functions) 🎯 START HERE
**File:** `src/steps/error_handling.rs`

Functions to implement:
1. `given_error_condition` - Set up error condition
2. `when_error_occurs` - Trigger error
3. `then_error_propagated` - Verify error propagation
4. `then_cleanup_performed` - Verify cleanup

**API to use:** World state, error handling, cleanup verification

#### Priority 14: CLI Commands Functions (3 functions)
**File:** `src/steps/cli_commands.rs`

Functions to implement:
1. `when_run_cli_command` - Execute CLI command
2. `then_output_contains` - Verify output
3. `then_exit_code_is` - Verify exit code

**API to use:** `tokio::process::Command`, World state

#### Priority 15: GGUF Functions (3 functions)
**File:** `src/steps/gguf.rs`

Functions to implement:
1. `given_gguf_file` - Set up GGUF file
2. `when_parse_gguf` - Parse GGUF file
3. `then_metadata_extracted` - Verify metadata

**API to use:** File system, GGUF parsing (if available)

#### Priority 16: Background Functions (2 functions)
**File:** `src/steps/background.rs`

Functions to implement:
1. `given_system_initialized` - Initialize system
2. `given_clean_state` - Set clean state

**API to use:** World state initialization

---

## Pattern to Follow - NICE!

### 1. Read the Gherkin Step
Look at the feature files to understand what the step should do.

### 2. Use Real APIs
```rust
// TEAM-071: [Description] NICE!
#[given(expr = "an error condition exists")]
pub async fn given_error_condition(world: &mut World) {
    // Set up error condition in World state
    world.last_error = Some(ErrorResponse {
        code: "TEST_ERROR".to_string(),
        message: "Test error condition".to_string(),
        details: Some(serde_json::json!({"test": true})),
    });
    
    tracing::info!("✅ Error condition set up NICE!");
}
```

### 3. Handle Borrow Checker Carefully
**CRITICAL:** When using `world.hive_registry()`, drop the borrow before accessing other World fields.

```rust
// ✅ GOOD: Drop registry borrow in a scope
let existing_id = {
    let registry = world.hive_registry();
    registry.list().await.first().map(|w| w.id.clone())
};
// Now can access world fields
let port = world.next_worker_port;
```

### 4. Add Team Signature
Every function MUST include "NICE!" in the comment:
```rust
// TEAM-071: Execute CLI command NICE!
```

---

## Available APIs - NICE!

### WorkerRegistry (`rbee_hive::registry`)
```rust
let registry = world.hive_registry();

// List all workers
let workers = registry.list().await;

// Get idle workers
let idle = registry.get_idle_workers().await;

// Update state
registry.update_state(&worker_id, WorkerState::Idle).await;

// Remove worker
registry.remove(&worker_id).await;
```

### Process Management (`tokio::process`)
```rust
use tokio::process::Command;

// Spawn process
let child = Command::new("sleep")
    .arg("3600")
    .spawn();

match child {
    Ok(process) => {
        world.worker_processes.push(process);
        tracing::info!("✅ Process started NICE!");
    }
    Err(e) => {
        tracing::warn!("⚠️  Failed to spawn: {}", e);
    }
}
```

### File System Operations
```rust
use std::fs;
use std::io::Write;

// Create file
let mut file = fs::File::create(path)?;
file.write_all(b"content")?;

// Read directory
let entries = fs::read_dir(path)?;
```

### World State
```rust
// Error tracking
world.last_error = Some(ErrorResponse {
    code: "ERROR_CODE".to_string(),
    message: "Error message".to_string(),
    details: Some(serde_json::json!({"key": "value"})),
});
world.last_exit_code = Some(1);

// Command execution
world.last_command = Some("command".to_string());
world.last_stdout = "output".to_string();
world.last_stderr = "error".to_string();
```

---

## Critical Rules - NICE!

### ⚠️ BDD Rules (MANDATORY)
1. ✅ **Implement at least 10 functions** - No exceptions
2. ✅ **Each function MUST call real API** - No `tracing::debug!()` only
3. ❌ **NEVER mark functions as TODO** - Implement or leave for next team
4. ❌ **NEVER delete checklist items** - Update status only
5. ✅ **Handoff must be 2 pages or less** - Be concise
6. ✅ **Include code examples** - Show the pattern

### ⚠️ Dev-Bee Rules (MANDATORY)
1. ✅ **Add team signature** - "TEAM-071: [Description] NICE!"
2. ❌ **Don't remove other teams' signatures** - Preserve history
3. ✅ **Update existing files** - Don't create multiple .md files
4. ✅ **Follow priorities** - Start with highest priority

### ⚠️ Borrow Checker (CRITICAL)
**TEAM-070 learned this the hard way:**

- ❌ **Don't hold registry borrow while accessing World fields**
- ✅ **Drop registry borrow in a scope before accessing World**
- ✅ **Extract data from registry, then use it**

---

## Verification Commands - NICE!

### Check Compilation
```bash
cd test-harness/bdd
cargo check --bin bdd-runner
```

Should output: `Finished \`dev\` profile [unoptimized + debuginfo] target(s)`

### Count Your Functions
```bash
grep -r "TEAM-071:" src/steps/ | wc -l
```

Should be at least 10!

---

## Success Checklist - NICE!

Before creating your handoff, verify:

- [ ] Implemented at least 10 functions
- [ ] Each function calls real API (not just tracing::debug!)
- [ ] All functions have "TEAM-071: ... NICE!" signature
- [ ] `cargo check --bin bdd-runner` passes (0 errors)
- [ ] Updated `TEAM_069_COMPLETE_CHECKLIST.md` with completion status
- [ ] Created `TEAM_071_COMPLETION.md` (2 pages max)
- [ ] No TODO markers added to code
- [ ] No checklist items deleted
- [ ] Honest completion ratios shown

---

## Example Implementation - NICE!

Here's a complete example for Priority 13:

```rust
// TEAM-071: Set up error condition NICE!
#[given(expr = "an error condition exists")]
pub async fn given_error_condition(world: &mut World) {
    // Set up error condition in World state
    world.last_error = Some(crate::steps::world::ErrorResponse {
        code: "TEST_ERROR".to_string(),
        message: "Test error condition".to_string(),
        details: Some(serde_json::json!({
            "test": true,
            "severity": "high"
        })),
    });
    
    tracing::info!("✅ Error condition set up NICE!");
}

// TEAM-071: Trigger error NICE!
#[when(expr = "the error occurs")]
pub async fn when_error_occurs(world: &mut World) {
    // Simulate error occurrence
    if let Some(ref error) = world.last_error {
        world.last_exit_code = Some(1);
        world.last_stderr = error.message.clone();
        tracing::info!("✅ Error triggered: {} NICE!", error.code);
    } else {
        tracing::warn!("⚠️  No error condition set up");
    }
}

// TEAM-071: Verify error propagation NICE!
#[then(expr = "the error is propagated")]
pub async fn then_error_propagated(world: &mut World) {
    assert!(world.last_error.is_some(), "Expected error to be set");
    assert_eq!(world.last_exit_code, Some(1), "Expected exit code 1");
    
    tracing::info!("✅ Error propagation verified NICE!");
}
```

---

## Summary - NICE!

**Current Progress:**
- TEAM-068: 43 functions
- TEAM-069: 21 functions
- TEAM-070: 16 functions
- **Total: 80 functions (77% complete)**

**Your Goal:**
- Implement at least 10 functions
- Use real APIs
- Follow the pattern
- Be honest about progress

**Recommended Start:**
- Priority 13: Error Handling (4 functions)
- Priority 14: CLI Commands (3 functions)
- Priority 15: GGUF (3 functions)
- This gives you 10 functions to complete!

---

**TEAM-070 says: You got this! NICE! 🐝**

**Good luck, TEAM-071!**
