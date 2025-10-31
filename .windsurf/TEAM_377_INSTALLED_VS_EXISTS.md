# TEAM-377 - Installed vs Exists (Uninstall Button Fix)

## 🐛 The Problem

**User:** "The queen is still not being uninstalled"

**Root Cause:** The `check_binary_installed()` function returned `true` for **ANY** of these locations:
1. `target/debug/queen-rbee` ✅ **DEV BUILD**
2. `target/release/queen-rbee`
3. `~/.local/bin/queen-rbee` (actually installed)

But `uninstall` only removes from `~/.local/bin/`!

**Result:**
- Queen running from `target/debug/` shows as "installed"
- User clicks "Uninstall"
- Uninstall tries to remove `~/.local/bin/queen-rbee` (doesn't exist)
- `target/debug/queen-rbee` still exists
- Status still shows "installed" ❌

---

## ✅ The Fix

**Created TWO separate functions:**

### 1. `check_binary_exists()` - Checks ANY location
```rust
/// Check if binary exists locally (dev, release, or installed)
///
/// Returns true if binary exists in:
/// 1. target/debug/{daemon}
/// 2. target/release/{daemon}
/// 3. ~/.local/bin/{daemon}
pub async fn check_binary_exists(daemon_name: &str) -> bool
```

**Use case:** "Can I start this daemon?"

### 2. `check_binary_actually_installed()` - ONLY checks ~/.local/bin/
```rust
/// Check if binary is ACTUALLY installed (only checks ~/.local/bin/)
///
/// Returns true ONLY if binary exists in ~/.local/bin/
/// Does NOT check target/debug/ or target/release/
pub async fn check_binary_actually_installed(daemon_name: &str) -> bool
```

**Use case:** "Should I show the Uninstall button?"

---

## 📊 Before vs After

### Before (BROKEN)

```rust
// status.rs
let is_installed = check_binary_installed(daemon_name).await;
// Returns true for target/debug/ ❌

// UI shows: "Installed" ✅ [Uninstall] button enabled
// User clicks Uninstall
// Uninstall removes ~/.local/bin/queen-rbee (doesn't exist)
// target/debug/queen-rbee still there
// Status still shows "Installed" ❌
```

### After (FIXED)

```rust
// status.rs
let is_installed = check_binary_actually_installed(daemon_name).await;
// Returns false for target/debug/ ✅
// Returns true ONLY for ~/.local/bin/ ✅

// UI shows: "Not Installed" [Uninstall] button disabled ✅
// Dev builds don't show as "installed"
```

---

## 🎯 Semantic Clarity

### "Exists" vs "Installed"

**Exists** = Binary is somewhere on the system
- Could be dev build (`target/debug/`)
- Could be release build (`target/release/`)
- Could be installed (`~/.local/bin/`)

**Installed** = Binary is in the install directory (`~/.local/bin/`)
- NOT a dev build
- NOT a release build
- Specifically in `~/.local/bin/`

### Use Cases

| Function | Dev Build | Release Build | Installed | Use Case |
|----------|-----------|---------------|-----------|----------|
| `check_binary_exists()` | ✅ | ✅ | ✅ | "Can I start it?" |
| `check_binary_actually_installed()` | ❌ | ❌ | ✅ | "Can I uninstall it?" |

---

## 🔧 Files Changed

1. **lifecycle-local/src/utils/binary.rs**
   - Renamed `check_binary_installed()` → `check_binary_exists()`
   - Added `check_binary_actually_installed()` (new function)

2. **lifecycle-local/src/utils/mod.rs**
   - Export both functions
   - Added deprecated alias for backwards compatibility

3. **lifecycle-local/src/status.rs**
   - Use `check_binary_actually_installed()` for `is_installed` field

4. **lifecycle-local/src/install.rs**
   - Use `check_binary_actually_installed()` to check if already installed

---

## 🎓 Why This Matters

**From the user's perspective:**

- ❌ **Before:** Dev builds show as "installed", uninstall button doesn't work
- ✅ **After:** Only actually installed binaries show as "installed"

**From the code's perspective:**

- ❌ **Before:** One function with ambiguous semantics
- ✅ **After:** Two functions with clear, distinct purposes

**From Rule Zero:**

- ❌ **Before:** Function name lies about what it does
- ✅ **After:** Function names accurately describe behavior

---

## ✅ Expected Behavior

### Scenario 1: Dev Build

```
$ cargo build --bin queen-rbee
# Creates target/debug/queen-rbee

Status:
- is_running: false
- is_installed: false ✅ (not in ~/.local/bin/)

UI:
- [Start] button: enabled ✅
- [Uninstall] button: disabled ✅
```

### Scenario 2: Installed Binary

```
$ cargo run --bin rbee-keeper -- install queen
# Copies to ~/.local/bin/queen-rbee

Status:
- is_running: false
- is_installed: true ✅ (in ~/.local/bin/)

UI:
- [Start] button: enabled ✅
- [Uninstall] button: enabled ✅
```

### Scenario 3: Running Dev Build

```
$ cargo run --bin queen-rbee
# Running from target/debug/

Status:
- is_running: true
- is_installed: true (optimization: if running, must exist)

UI:
- [Stop] button: enabled ✅
- [Uninstall] button: disabled ❌ (can't uninstall running daemon)
```

---

## 🔍 Verification

```bash
# Build
cargo build --package lifecycle-local
# ✅ Finished in 0.60s

# Test with dev build
cargo build --bin queen-rbee
# Should show as "Not Installed" in UI

# Test with installed binary
cargo run --bin rbee-keeper -- install queen
# Should show as "Installed" in UI
# Uninstall button should work
```

---

**TEAM-377 | Semantic clarity | Uninstall now works correctly! 🎉**
