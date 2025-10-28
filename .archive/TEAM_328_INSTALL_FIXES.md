# TEAM-328: install.rs Critical Edge Case Fixes

**Status:** ✅ COMPLETE

**Mission:** Fix 3 critical edge cases in `daemon-lifecycle/src/install.rs`

## Problems Fixed

### 1. ✅ Partial Copy Failure (Atomicity)

**Problem:**
```rust
// OLD CODE
std::fs::copy(&source, &install_path)?;  // ✅ Success
// ... later ...
std::fs::set_permissions(&install_path, perms)?;  // ❌ FAILS
// Binary is installed but not executable!
// User can't reinstall (already exists check blocks it)
```

**Fix:**
```rust
// NEW CODE - TEAM-328
if let Err(e) = std::fs::set_permissions(&install_path, perms) {
    n!("install_cleanup", "⚠️  chmod failed, removing partially installed binary");
    let _ = std::fs::remove_file(&install_path);  // Cleanup!
    return Err(e.into());
}
```

**Benefit:** Maintains atomicity - either fully installed or not at all

---

### 2. ✅ Source == Destination Check

**Problem:**
```rust
// OLD CODE
// User tries: install ~/.local/bin/queen-rbee to ~/.local/bin/queen-rbee
// Gets confusing error: "already installed at ..."
// But it's not "already installed", it's the SOURCE!
```

**Fix:**
```rust
// NEW CODE - TEAM-328
let source_canonical = source.canonicalize()?;
let dest_canonical = install_path.canonicalize().ok();

if let Some(dest) = dest_canonical {
    if source_canonical == dest {
        anyhow::bail!(
            "Source and destination are the same: {}. Binary is already installed.",
            install_path.display()
        );
    }
}
```

**Benefit:** Clear error message when source == destination

---

### 3. ✅ Source Binary Validation

**Problem:**
```rust
// OLD CODE
// No validation of source binary
// Could copy non-executable or corrupted binary
// User gets installed binary that can't run
```

**Fix:**
```rust
// NEW CODE - TEAM-328
#[cfg(unix)]
{
    use std::os::unix::fs::PermissionsExt;
    let source_perms = std::fs::metadata(&source)?.permissions();
    if source_perms.mode() & 0o111 == 0 {
        anyhow::bail!(
            "Source binary is not executable: {} (mode: {:o})",
            source.display(),
            source_perms.mode()
        );
    }
}
```

**Benefit:** Prevents installing non-executable binaries

---

## Implementation Details

### Changes Made

**File:** `bin/99_shared_crates/daemon-lifecycle/src/install.rs`

**Lines added:** +35 LOC  
**Lines modified:** 3 sections

**Sections:**

1. **After finding source binary (lines 67-79):**
   - Validate source is executable (Unix only)
   - Show permission mode in error message

2. **After determining install path (lines 90-102):**
   - Canonicalize both paths
   - Check if they're the same
   - Clear error message

3. **During chmod (lines 122-135):**
   - Wrap `set_permissions` in error handling
   - Cleanup copied binary on failure
   - Emit narration for cleanup action

### Error Messages

**Before:**
```
Error: already installed at ~/.local/bin/queen-rbee. Uninstall first or use rebuild.
```

**After (source == dest):**
```
Error: Source and destination are the same: ~/.local/bin/queen-rbee. Binary is already installed.
```

**New (non-executable source):**
```
Error: Source binary is not executable: target/debug/queen-rbee (mode: 644)
```

**New (chmod failure with cleanup):**
```
⚠️  chmod failed, removing partially installed binary
Error: Permission denied (os error 13)
```

---

## Testing Scenarios

### Scenario 1: Partial Install Failure
```bash
# Make install dir read-only
chmod 555 ~/.local/bin

# Try to install
./rbee queen install

# Result: Binary NOT left in broken state
# Old: Binary copied but not executable, can't reinstall
# New: Binary removed on chmod failure, can retry
```

### Scenario 2: Source == Destination
```bash
# Binary already installed
ls ~/.local/bin/queen-rbee

# Try to install from same location
./rbee queen install --binary ~/.local/bin/queen-rbee

# Result: Clear error message
# Old: "already installed at ..." (confusing)
# New: "Source and destination are the same..." (clear)
```

### Scenario 3: Non-Executable Source
```bash
# Build binary without execute permission
cargo build --bin queen-rbee
chmod 644 target/debug/queen-rbee

# Try to install
./rbee queen install

# Result: Install rejected
# Old: Copies non-executable binary, user can't run it
# New: Error before copy, shows permission mode
```

---

## Compilation

✅ `cargo check -p daemon-lifecycle` - PASS  
✅ `cargo build --bin rbee-keeper` - PASS

---

## Code Signatures

All changes marked with `// TEAM-328:`

---

## What's Still Not Covered

**Medium priority (not critical):**
- Install directory is a file (not directory)
- Path traversal attack (`binary_name = "../../evil"`)
- Symlink handling edge case

**Low priority:**
- Version check timeout
- Empty binary name validation
- Windows `.exe` extension

These can be addressed in future work if needed.

---

**Result:** install.rs now handles critical edge cases that could leave system in broken state.

**Key improvements:**
1. **Atomicity** - All-or-nothing installation
2. **Validation** - Source must be executable
3. **Clear errors** - Better messages for edge cases
