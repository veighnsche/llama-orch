# TEAM-336: Stack Overflow Fix - Double Macro Application

**Status:** ✅ FIXED

**Date:** Oct 28, 2025

## Problem

Stack overflow when pressing "Queen Start" button in Tauri GUI:

```
thread 'tokio-runtime-worker' has overflowed its stack
fatal runtime error: stack overflow, aborting
```

## Root Cause

**Infinite recursion caused by double macro application.**

The `start_daemon()` function in `daemon-lifecycle/src/start.rs` had **both** macros applied:

```rust
#[with_job_id(config_param = "start_config")]
#[with_timeout(secs = 120, label = "Start daemon")]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
    // ... function body ...
}
```

### The Recursion Chain

1. `#[with_timeout]` macro expands first (outer macro)
2. Creates wrapper function that calls `__start_daemon_inner()`
3. **BUT**: The inner function still has `#[with_job_id]` attribute attached
4. `#[with_job_id]` macro expands on the inner function
5. Creates another wrapper that calls another inner function
6. This creates infinite nesting
7. **BOOM**: Stack overflow

### Why This Happened

The `#[with_timeout]` macro was copying **all attributes** from the original function to the inner function:

```rust
// OLD CODE (broken)
let attrs = &func.attrs;  // Copies ALL attributes including #[with_job_id]

let expanded = quote! {
    #(#attrs)*  // ← This includes #[with_job_id]!
    #vis #sig {
        async fn #inner_name #generics ( #inputs ) #output #where_clause {
            #body
        }
        // ...
    }
};
```

## The Fix

**Filter out `#[with_job_id]` from the inner function** to prevent double-wrapping.

### Code Change

File: `bin/99_shared_crates/timeout-enforcer-macros/src/lib.rs`

```rust
// TEAM-336: Filter out #[with_job_id] from inner function to prevent infinite recursion
let attrs: Vec<_> = func.attrs.iter()
    .filter(|attr| !attr.path().is_ident("with_job_id"))
    .collect();
```

### How It Works Now

1. `#[with_timeout]` macro expands first
2. Creates wrapper function
3. **Strips `#[with_job_id]` from inner function**
4. Inner function is plain async function (no more macros)
5. No recursion, no stack overflow ✅

## Affected Functions

All functions in `daemon-lifecycle` crate that use both macros:

- `start_daemon()` - **PRIMARY ISSUE** (called from Tauri GUI)
- `poll_daemon_health()`
- `build_daemon()`
- `install_daemon()`
- `rebuild_daemon()`
- `shutdown_daemon()`
- `uninstall_daemon()`

## Testing

✅ Compilation: `cargo build --package rbee-keeper` - SUCCESS
✅ Tests: `cargo test --package daemon-lifecycle` - PASS
✅ Manual test: Tauri GUI "Queen Start" button - **FIXED**

## Lessons Learned

### Macro Composition is Dangerous

When stacking multiple procedural macros, **each macro must be aware of the others** to prevent:

1. **Infinite recursion** (this bug)
2. **Attribute duplication**
3. **Unexpected expansions**

### Pre-1.0 Breaking Changes Are Good

This bug existed because we tried to maintain backwards compatibility with the old `with_job_id()` API. 

**RULE ZERO saved us:**
- We deleted the old API (breaking change)
- Compiler found all call sites
- Fixed them properly
- No more entropy from dual APIs

If we had kept both APIs, this bug would have been **much harder to find**.

## Related Issues

- TEAM-330: Universal context propagation (removed `with_job_id()` method)
- TEAM-330: Timeout enforcer refactor
- TEAM-335: Queen lifecycle commands in Tauri GUI

## Files Changed

1. `bin/99_shared_crates/timeout-enforcer-macros/src/lib.rs` (+3 lines)
   - Added attribute filtering to prevent double macro application

## Verification

To verify the fix works:

```bash
# Build rbee-keeper
cargo build --package rbee-keeper

# Run Tauri GUI
cd bin/00_rbee_keeper
cargo tauri dev

# Click "Queen Start" button
# Should work without stack overflow ✅
```

## Prevention

To prevent similar issues in the future:

1. **Document macro interactions** in macro crate README
2. **Test macro composition** (add tests for stacked macros)
3. **Lint for dangerous patterns** (multiple proc macros on same function)
4. **Keep macros simple** (avoid complex attribute copying)

---

**TEAM-336 Signature:** Fixed stack overflow caused by double macro application
