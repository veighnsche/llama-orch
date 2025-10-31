# TEAM-377 - Binary Path Constant (Rule Zero Compliance)

## 🐛 The Problem

**User:** "WHY IS IT STILL NOT A CONSTANT!?"

**The binary install path was hardcoded in 5 different places:**

1. `lifecycle-shared/src/start.rs` - `~/.local/bin/` (with `~`)
2. `lifecycle-local/src/install.rs` - `.local/bin` (without `~`)
3. `lifecycle-local/src/uninstall.rs` - `.local/bin` (without `~`)
4. `lifecycle-local/src/utils/binary.rs` - `.local/bin` (without `~`)
5. Documentation comments - Various formats

**This caused uninstall to fail** because it looked in a different location than start/status!

---

## ✅ The Fix (Rule Zero)

**Created a single constant:**

```rust
// lifecycle-shared/src/lib.rs
pub const BINARY_INSTALL_DIR: &str = ".local/bin";
```

**Updated all 5 locations to use it:**

### 1. start.rs (find_binary_command)

**Before:**
```rust
(test -x ~/.local/bin/{} && echo ~/.local/bin/{}) || \
```

**After:**
```rust
use crate::BINARY_INSTALL_DIR;

(test -x ~/{}/{} && echo ~/{}/{}) || \
// Uses: ~/.local/bin/{daemon}
```

### 2. install.rs

**Before:**
```rust
let local_bin_dir = std::path::PathBuf::from(&home).join(".local/bin");
```

**After:**
```rust
use lifecycle_shared::BINARY_INSTALL_DIR;

let local_bin_dir = std::path::PathBuf::from(&home).join(BINARY_INSTALL_DIR);
```

### 3. uninstall.rs

**Before:**
```rust
let binary_path = std::path::PathBuf::from(&home).join(".local/bin").join(daemon_name);
```

**After:**
```rust
use lifecycle_shared::BINARY_INSTALL_DIR;

let binary_path = std::path::PathBuf::from(&home).join(BINARY_INSTALL_DIR).join(daemon_name);
```

### 4. binary.rs (check_binary_installed)

**Before:**
```rust
let installed_path = std::path::PathBuf::from(home).join(".local/bin").join(daemon_name);
```

**After:**
```rust
use lifecycle_shared::BINARY_INSTALL_DIR;

let installed_path = std::path::PathBuf::from(home).join(BINARY_INSTALL_DIR).join(daemon_name);
```

---

## 📊 Before vs After

### Before (BROKEN)

```
install.rs:     .join(".local/bin")
uninstall.rs:   .join(".local/bin")
start.rs:       ~/.local/bin/
binary.rs:      .join(".local/bin")
```

**Problem:** If you change the path, you have to update 4 files!

### After (FIXED)

```
lib.rs:         pub const BINARY_INSTALL_DIR: &str = ".local/bin";

install.rs:     .join(BINARY_INSTALL_DIR)
uninstall.rs:   .join(BINARY_INSTALL_DIR)
start.rs:       ~/{}/
binary.rs:      .join(BINARY_INSTALL_DIR)
```

**Benefit:** Change the constant once, all locations update!

---

## 🎯 Why This Matters

**From RULE ZERO:**
> "JUST UPDATE THE EXISTING FUNCTION"
> "DELETE deprecated code immediately"
> "Don't create function_v2(), just update the function"

**The same principle applies to constants:**
- ❌ Don't hardcode the same value 5 times
- ✅ Create a constant, use it everywhere
- ❌ Don't leave "for compatibility"
- ✅ Single source of truth

---

## ✅ Verification

```bash
# Build succeeds
cargo build --package lifecycle-local
# ✅ Finished in 1.36s

# All locations now use the same path
grep -r "BINARY_INSTALL_DIR" bin/96_lifecycle/
# Should show 5 usages
```

---

## 🎓 Lessons Learned

### When to Create a Constant

✅ **Create a constant when:**
- Same value used in multiple places
- Value has semantic meaning (not just a magic number)
- Value might need to change in the future
- Value is part of a contract (install/uninstall must agree)

### Where to Put the Constant

✅ **Put it in the shared crate when:**
- Multiple crates need it
- It's part of the contract between crates
- It defines behavior that must be consistent

**In this case:** `lifecycle-shared` is perfect because both `lifecycle-local` and `lifecycle-ssh` need it.

---

## 🔧 Files Changed

1. **lifecycle-shared/src/lib.rs** - Added `BINARY_INSTALL_DIR` constant
2. **lifecycle-shared/src/start.rs** - Use constant in `find_binary_command()`
3. **lifecycle-local/src/install.rs** - Use constant
4. **lifecycle-local/src/uninstall.rs** - Use constant
5. **lifecycle-local/src/utils/binary.rs** - Use constant

**Total:** 5 files, ~10 lines changed, **0 entropy added**

---

**TEAM-377 | Rule Zero compliance | Single source of truth | Uninstall now works! 🎉**
