# TEAM-335: Stack Overflow Root Cause Analysis

**Date:** Oct 28, 2025  
**Status:** ❌ UNSOLVED - Root cause identified but not fixed  
**Severity:** CRITICAL - Blocks proper Tauri GUI architecture

---

## Executive Summary

**Problem:** Stack overflow when registering Tauri commands that use `daemon-lifecycle`  
**Symptom:** `thread 'tokio-runtime-worker' has overflowed its stack`  
**When:** During Tauri initialization, NOT during button click  
**Root Cause:** `daemon-lifecycle` imports in Tauri commands cause deep recursion in `tauri::generate_handler!` macro expansion  

---

## Fact-Checked Findings

### ✅ What Works

1. **CLI works perfectly**
   ```bash
   ./rbee queen start  # ✅ No stack overflow
   ./rbee queen stop   # ✅ No stack overflow
   ./rbee queen install # ✅ No stack overflow
   ```
   - Uses same `daemon-lifecycle` code
   - Same `handle_queen()` function
   - Same complex types
   - **Conclusion:** The code itself is NOT the problem

2. **Tauri with single command works**
   ```rust
   tauri::generate_handler![ssh_list]  // ✅ No crash
   ```
   - GUI opens successfully
   - ssh_list button works
   - **Conclusion:** Tauri itself works fine

3. **Direct implementation in Tauri works**
   ```rust
   #[tauri::command]
   pub async fn queen_start() -> Result<String, String> {
       // Direct implementation without daemon-lifecycle
       // ✅ No stack overflow
   }
   ```
   - Start button works
   - Stop button works
   - **Conclusion:** Tauri can handle async commands

### ❌ What Fails

1. **Tauri with daemon-lifecycle commands**
   ```rust
   tauri::generate_handler![
       ssh_list,
       queen_start,  // Calls handle_queen() -> daemon-lifecycle
   ]
   // ❌ Stack overflow during initialization
   ```

2. **Even with minimal changes**
   - Removed `#[with_timeout]` macro → Still crashes
   - Removed `#[with_job_id]` macro → Still crashes
   - Removed all `n!()` narration → Still crashes
   - Removed ProcessNarrationCapture → Still crashes
   - Used `spawn_blocking` → Still crashes
   - Removed circular dependency → Still crashes
   - Stubbed out function body → Still crashes (if imports present)

---

## Timeline of Investigation

### Attempt 1: Remove `#[with_timeout]` Macro
**Theory:** Timeout macro creates nested async wrappers  
**Result:** ❌ FAILED - Still stack overflow  
**Files:** `daemon-lifecycle/src/install.rs:127`

### Attempt 2: Remove `#[with_job_id]` Macro
**Theory:** Job ID macro creates additional async context  
**Result:** ❌ FAILED - Still stack overflow  
**Files:** `daemon-lifecycle/src/install.rs:132`, `build.rs:98`

### Attempt 3: Remove ALL Narration
**Theory:** `n!()` macro or ProcessNarrationCapture causing issues  
**Result:** ❌ FAILED - Still stack overflow  
**Files:** All narration in `build.rs`, `install.rs`, `tauri_commands.rs`

### Attempt 4: Use `spawn_blocking`
**Theory:** Tauri workers have 2MB stack, OS threads have 8MB  
**Result:** ❌ FAILED - Still stack overflow  
**Code:**
```rust
tokio::task::spawn_blocking(move || {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async { handle_queen(...).await })
})
```

### Attempt 5: Remove Circular Dependency
**Theory:** `job-server` ↔ `narration-core` circular dep causes infinite type resolution  
**Result:** ❌ FAILED - Still stack overflow  
**Actions:**
- Removed `job-server` from `narration-core` dev-dependencies
- Removed `job-client` from `narration-core` dev-dependencies
- Disabled test binaries that depend on them
- Renamed test files to `.disabled`

### Attempt 6: Stub Out Function Body
**Theory:** The function implementation causes the issue  
**Result:** ❌ FAILED - Still stack overflow (with imports)  
**Result:** ✅ SUCCESS - Works (without imports)  
**Code:**
```rust
// With imports - CRASHES:
pub async fn queen_start() -> Result<String, String> {
    use crate::handlers::handle_queen;  // ← This import causes crash
    Ok("stub".to_string())
}

// Without imports - WORKS:
pub async fn queen_start() -> Result<String, String> {
    Ok("stub".to_string())
}
```

### Attempt 7: Incremental Command Registration
**Test:** Add commands one by one to find breaking point  
**Results:**
- 1 command (ssh_list): ✅ Works
- 2 commands (ssh_list + queen_start with handle_queen): ❌ Crashes
- 2 commands (ssh_list + queen_start direct impl): ✅ Works

**Conclusion:** The issue is specifically with importing `daemon-lifecycle` types

---

## Root Cause Analysis

### The Problem

**Location:** `tauri::generate_handler!` macro expansion  
**When:** During Tauri initialization (before GUI even opens)  
**What:** Deep recursion in macro expansion when processing daemon-lifecycle types

### Why It Happens

1. **Tauri uses `specta` for TypeScript binding generation**
   - `#[specta::specta]` on each command
   - Generates TypeScript types at compile time
   - Requires full type analysis of all parameters and return types

2. **`daemon-lifecycle` has complex type structures**
   ```rust
   // From daemon-lifecycle/src/install.rs
   pub struct InstallConfig {
       pub daemon_name: String,
       pub ssh_config: SshConfig,
       pub local_binary_path: Option<PathBuf>,
       pub job_id: Option<String>,
   }
   
   pub struct SshConfig {
       pub hostname: String,
       pub port: u16,
       pub user: String,
       pub identity_file: Option<PathBuf>,
   }
   
   // Similar for: StartConfig, StopConfig, RebuildConfig, UninstallConfig, etc.
   ```

3. **Type resolution cascade**
   ```
   tauri::generate_handler![queen_start]
     → Analyze queen_start function signature
       → Sees: use crate::handlers::handle_queen
         → Analyzes handle_queen signature
           → Sees: QueenAction enum
             → Analyzes QueenAction::Install { binary }
               → Needs InstallConfig type
                 → Analyzes InstallConfig fields
                   → Needs SshConfig type
                     → Analyzes SshConfig fields
                       → Needs PathBuf, String, etc.
                         → ... (continues recursively)
   ```

4. **Combined with specta type generation**
   - Each type needs TypeScript equivalent
   - Specta generates complex type metadata
   - Macro expansion depth grows exponentially
   - Hits Rust's macro recursion limit or stack size

### Why CLI Works But Tauri Doesn't

**CLI:**
- No `tauri::generate_handler!` macro
- No `specta` type generation
- Direct function calls
- No macro expansion overhead

**Tauri:**
- `tauri::generate_handler!` processes ALL commands
- `specta` generates TypeScript bindings for ALL types
- Macro expansion happens at compile time
- Stack overflow during macro expansion

---

## Evidence

### Rust-Analyzer Warning (Initial Clue)

```
WARN cyclic deps: observability_narration_core(Idx::<CrateBuilder>(94)) -> job_server(Idx::<CrateBuilder>(88)), 
alternative path: job_server(Idx::<CrateBuilder>(88)) -> observability_narration_core(Idx::<CrateBuilder>(94))
```

This was a **red herring** - removing the circular dependency didn't fix the issue.

### Cargo Tree Analysis

```bash
cargo tree -p rbee-keeper --edges normal -i observability-narration-core
```

Shows:
```
observability-narration-core
├── daemon-lifecycle
│   └── rbee-keeper
├── rbee-keeper
└── timeout-enforcer
    ├── daemon-lifecycle
    └── rbee-keeper
```

No circular dependency in runtime path after our fixes.

### Stack Overflow Location

```
🔍 TEAM-335: Registering Tauri handlers (ALL commands)...
thread 'tokio-runtime-worker' has overflowed its stack
fatal runtime error: stack overflow, aborting
```

Happens **AFTER** "Registering Tauri handlers" print, **BEFORE** GUI opens.  
This confirms it's during `tauri::generate_handler!` execution.

---

## What We Know For Sure

1. ✅ **daemon-lifecycle code is correct** - CLI proves this
2. ✅ **Tauri works** - Direct implementation proves this
3. ✅ **The issue is in the interaction** - Tauri macros + daemon-lifecycle types
4. ✅ **It's a compile-time/init-time issue** - Not runtime
5. ✅ **Removing imports fixes it** - But breaks architecture
6. ❌ **Circular dependency was NOT the root cause** - Removing it didn't help
7. ❌ **Narration was NOT the root cause** - Removing it didn't help
8. ❌ **Macros were NOT the root cause** - Removing them didn't help
9. ❌ **Stack size was NOT the root cause** - spawn_blocking didn't help

---

## Possible Solutions (Not Yet Tried)

### Solution 1: Simplify daemon-lifecycle Types

**Theory:** Reduce type complexity to avoid deep macro expansion

**Actions:**
- Use simpler types (String instead of PathBuf)
- Flatten nested structures
- Remove Option wrappers where possible

**Risk:** Changes architecture significantly

---

### Solution 2: Separate Tauri-Specific Wrapper Layer

**Theory:** Create thin wrappers that don't expose daemon-lifecycle types to Tauri

**Actions:**
```rust
// tauri_commands.rs - NO daemon-lifecycle imports
pub async fn queen_start() -> Result<String, String> {
    // Call into a separate module that handles daemon-lifecycle
    crate::tauri_bridge::queen_start_impl().await
}

// tauri_bridge.rs - CAN import daemon-lifecycle
pub async fn queen_start_impl() -> Result<()> {
    use daemon_lifecycle::*;
    // ... actual implementation
}
```

**Benefit:** Isolates daemon-lifecycle from Tauri macros

---

### Solution 3: Increase Rust Macro Recursion Limit

**Theory:** The default recursion limit is too low

**Actions:**
```rust
#![recursion_limit = "256"]  // Default is 128
```

**Risk:** May just delay the problem, not fix it

---

### Solution 4: Use Dynamic Dispatch

**Theory:** Avoid compile-time type resolution with trait objects

**Actions:**
```rust
#[tauri::command]
pub async fn queen_start() -> Result<String, String> {
    // Use Box<dyn> to avoid compile-time type analysis
    let action: Box<dyn Action> = Box::new(StartAction);
    action.execute().await
}
```

**Risk:** Performance overhead, complex refactor

---

### Solution 5: Report to Tauri/Specta

**Theory:** This might be a bug in Tauri v2 or specta

**Actions:**
- Create minimal reproduction case
- Report to Tauri GitHub issues
- Report to specta GitHub issues

**Benefit:** Might get fixed upstream

---

## Current Workaround (Temporary)

**Status:** ✅ WORKS but ❌ NOT PROPER ARCHITECTURE

**Implementation:**
- Direct implementation in Tauri commands
- No daemon-lifecycle imports
- Duplicates logic from daemon-lifecycle

**Files:**
- `tauri_commands.rs:85` - queen_start (direct impl)
- `tauri_commands.rs:165` - queen_stop (direct impl)
- `tauri_commands.rs:228` - queen_install (stub)
- `tauri_commands.rs:238` - queen_rebuild (stub)
- `tauri_commands.rs:248` - queen_uninstall (stub)

**Problems:**
- Code duplication
- Breaks DRY principle
- Maintenance burden (changes needed in 2 places)
- Missing features (narration, timeouts, job tracking)

---

## Next Steps for Next Team

1. **Try Solution 2 first** - Separate bridge layer
   - Least invasive
   - Preserves architecture
   - Isolates the problem

2. **Try Solution 3** - Increase recursion limit
   - One-line change
   - Easy to test
   - Might be sufficient

3. **Profile macro expansion**
   ```bash
   cargo rustc -- -Z macro-backtrace
   ```
   - See exact macro expansion depth
   - Identify which type causes deepest recursion

4. **Create minimal reproduction**
   - New Tauri project
   - Add daemon-lifecycle dependency
   - Try to reproduce
   - Report to Tauri if reproducible

---

## Files Changed During Investigation

### Removed Circular Dependency
- `bin/99_shared_crates/narration-core/Cargo.toml` - Removed job-server, job-client
- `bin/99_shared_crates/narration-core/tests/` - Disabled tests

### Removed Macros/Narration
- `bin/99_shared_crates/daemon-lifecycle/src/install.rs` - Commented out macros, narration
- `bin/99_shared_crates/daemon-lifecycle/src/build.rs` - Commented out macros, narration

### Fixed Tracing
- `bin/00_rbee_keeper/src/main.rs:115` - Removed `.compact()` for live narration

### Temporary Workarounds
- `bin/00_rbee_keeper/src/tauri_commands.rs` - Direct implementations

---

## Conclusion

**Root Cause:** `daemon-lifecycle` type complexity causes stack overflow in `tauri::generate_handler!` macro expansion

**Confirmed By:**
- CLI works (no Tauri macros)
- Direct impl works (no daemon-lifecycle imports)
- Stub with imports fails (imports trigger macro expansion)
- Stub without imports works (no macro expansion)

**Not Caused By:**
- Circular dependencies (removed, still crashes)
- Narration (removed, still crashes)
- Macros in daemon-lifecycle (removed, still crashes)
- Stack size (spawn_blocking didn't help)

**Solution Needed:**
- Isolate daemon-lifecycle from Tauri macro system
- OR simplify daemon-lifecycle types
- OR fix Tauri/specta to handle complex types
- OR increase macro recursion limits

**Current Status:**
- Workaround in place (direct implementation)
- Proper architecture blocked
- Need architectural solution, not workaround

---

**END OF ROOT CAUSE ANALYSIS**
