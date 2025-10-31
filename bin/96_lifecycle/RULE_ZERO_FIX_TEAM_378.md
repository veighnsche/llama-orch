# RULE ZERO Fix - TEAM-378

**Date:** Oct 31, 2025  
**Status:** ✅ COMPLETE

## Problem

RULE ZERO violation in `lifecycle-local/src/utils/binary.rs`:

```rust
// ❌ ENTROPY PATTERN
pub async fn check_binary_exists(daemon_name: &str) -> bool { ... }
pub async fn check_binary_actually_installed(daemon_name: &str) -> bool { ... }
```

The name "actually" implies the first function is defective. This is exactly the pattern RULE ZERO forbids:

> ❌ **BANNED - Entropy Patterns:**
> - Creating `function_v2()`, `function_new()`, `function_with_options()` to avoid breaking `function()`

## Solution

Added `CheckMode` enum parameter to consolidate both functions:

```rust
// ✅ RULE ZERO COMPLIANT
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckMode {
    /// Check ALL locations: target/debug/, target/release/, ~/.local/bin/
    Any,
    /// Check ONLY ~/.local/bin/ (for "is this actually installed?" checks)
    InstalledOnly,
}

pub async fn check_binary_exists(daemon_name: &str, mode: CheckMode) -> bool { ... }
```

## Changes

### Files Modified

1. **bin/96_lifecycle/lifecycle-local/src/utils/binary.rs**
   - Added `CheckMode` enum
   - Updated `check_binary_exists()` to accept `mode` parameter
   - Deleted `check_binary_actually_installed()` (87 LOC removed)
   - Added TEAM-378 signatures

2. **bin/96_lifecycle/lifecycle-local/src/utils/mod.rs**
   - Removed `check_binary_actually_installed` export
   - Removed deprecated `check_binary_installed` alias
   - Added `CheckMode` export

3. **bin/96_lifecycle/lifecycle-local/src/status.rs**
   - Updated import: `use crate::utils::{check_binary_exists, CheckMode};`
   - Updated call: `check_binary_exists(daemon_name, CheckMode::InstalledOnly).await`

4. **bin/96_lifecycle/lifecycle-local/src/install.rs**
   - Updated import: `use crate::utils::{check_binary_exists, CheckMode};`
   - Updated call: `check_binary_exists(daemon_name, CheckMode::InstalledOnly).await`

### Usage Pattern

```rust
// Check ANY location (dev builds + installed)
if check_binary_exists("rbee-hive", CheckMode::Any).await {
    println!("Binary exists somewhere");
}

// Check ONLY ~/.local/bin/ (for "Uninstall" button state)
if check_binary_exists("rbee-hive", CheckMode::InstalledOnly).await {
    println!("Binary is installed");
}
```

## Verification

✅ **Compilation:** `cargo check -p lifecycle-local` - PASS  
✅ **Unit Tests:** `cargo test -p lifecycle-local` - 11 tests PASS  
⚠️ **Doc Tests:** 4 failures (unrelated - reference old crate names)

## Impact

- **Code Removed:** 87 LOC (deleted function + deprecated alias)
- **Entropy Eliminated:** No more "actually" vs "exists" confusion
- **API Clarity:** Single function with explicit mode parameter
- **Compiler Enforcement:** All call sites updated, no backwards compatibility

## Why This Matters

**Before (Entropy):**
- Two functions doing similar things
- Unclear which one to use
- Name "actually" implies the other is wrong
- Future developers confused
- Bugs need fixing in 2 places

**After (Clean):**
- One function, one way to do it
- Explicit mode parameter makes intent clear
- Compiler finds all call sites
- Single source of truth

## RULE ZERO Compliance

✅ **Update existing functions** - Changed signature of `check_binary_exists()`  
✅ **Delete deprecated code** - Removed `check_binary_actually_installed()`  
✅ **Fix compilation errors** - Updated all 2 call sites  
✅ **One way to do things** - Single function with mode parameter  

**Breaking changes are temporary. Entropy is forever.**

---

## Part 2: SSH Architecture Fix

### The Problem

**lifecycle-ssh** was checking 3 locations on remote machines:
```bash
test -f target/debug/{daemon} || \
test -f target/release/{daemon} || \
test -f ~/.local/bin/{daemon}
```

**But the SSH workflow is:**
1. Build binary **locally**
2. SCP upload to **~/.local/bin/** on remote
3. **Never** build on remote machine

**Why check target/debug and target/release on remote?** → **We don't.**

### The Fix

Simplified SSH version to only check ~/.local/bin/:

```rust
// TEAM-378: RULE ZERO - Only check ~/.local/bin/ (we never build remotely)
let check_cmd = format!("test -f ~/.local/bin/{}", daemon_name);

let is_installed = match ssh_exec(ssh_config, &check_cmd).await {
    Ok(_) => true,   // Exit code 0 = file exists
    Err(_) => false, // Exit code 1 = file doesn't exist
};
```

### Files Modified

**bin/96_lifecycle/lifecycle-ssh/src/utils/binary.rs**
- Removed target/debug and target/release checks
- Simplified to single `test -f ~/.local/bin/{daemon}` command
- Updated documentation to explain why
- Removed unnecessary string parsing (just check exit code)

### Architecture Clarity

**lifecycle-local (CheckMode makes sense):**
- `CheckMode::Any` → Check target/debug, target/release, ~/.local/bin/
- `CheckMode::InstalledOnly` → Check only ~/.local/bin/
- **Why?** We might run dev builds OR installed binaries locally

**lifecycle-ssh (No CheckMode needed):**
- Always checks only ~/.local/bin/
- **Why?** We never build remotely, only SCP pre-built binaries

### Verification

✅ **Compilation:** `cargo check -p lifecycle-ssh` - PASS  
✅ **Logic:** Matches actual SSH workflow (build local → SCP → install)  
✅ **Simplicity:** 15 LOC → 5 LOC (66% reduction in check logic)

---

**TEAM-378 Signature:** All changes marked with `// TEAM-378: RULE ZERO`
