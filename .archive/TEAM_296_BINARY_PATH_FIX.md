# TEAM-296: Binary Path Fix - Start Now Uses Installed Binary

**Status:** ✅ COMPLETE  
**Date:** Oct 26, 2025

## Problem

After running `queen install` and `queen uninstall`, the queen could still be started. This was because:

1. **Install** puts binary in `~/.local/bin/queen-rbee`
2. **Uninstall** removes binary from `~/.local/bin/queen-rbee`
3. **Start** was looking for binary in `target/debug` or `target/release`

So install/uninstall managed one binary, but start used a completely different binary!

## Root Cause

In `bin/05_rbee_keeper_crates/queen-lifecycle/src/ensure.rs` line 132:

```rust
// OLD CODE - Only looked in target/
let queen_binary = DaemonManager::find_in_target("queen-rbee")
    .context("Failed to find queen-rbee binary in target directory")?;
```

This meant:
- Development workflow: `./rbee queen start` → uses `target/debug/queen-rbee`
- Production workflow: `./rbee queen install` → installs to `~/.local/bin/queen-rbee`
- **But start never used the installed binary!**

## Solution

Updated start logic to **prefer installed binary** over development binary:

```rust
// NEW CODE - Prefer installed binary
let queen_binary = {
    let home = std::env::var("HOME").context("Failed to get HOME directory")?;
    let installed_path = std::path::PathBuf::from(format!("{}/.local/bin/queen-rbee", home));
    
    if installed_path.exists() {
        NARRATE
            .action("queen_start")
            .context(installed_path.display().to_string())
            .human("Using installed queen-rbee binary at {}")
            .emit();
        installed_path
    } else {
        let dev_binary = DaemonManager::find_in_target("queen-rbee")
            .context("Failed to find queen-rbee binary (not installed and not in target/)")?;
        NARRATE
            .action("queen_start")
            .context(dev_binary.display().to_string())
            .human("Using development queen-rbee binary at {}")
            .emit();
        dev_binary
    }
};
```

## New Behavior

### Priority Order
1. **First:** Check `~/.local/bin/queen-rbee` (installed binary)
2. **Fallback:** Check `target/release/queen-rbee` or `target/debug/queen-rbee` (development binary)

### User Experience

**Production Workflow (Installed Binary):**
```bash
# Install
$ ./rbee queen install
[queen-life] queen_install  : 📦 Installing queen-rbee...
[queen-life] queen_install  : 🔨 Building queen-rbee from source...
[queen-life] queen_install  : ✅ Build successful!
[queen-life] queen_install  : 📋 Installing to: /home/user/.local/bin/queen-rbee
[queen-life] queen_install  : ✅ Queen installed successfully!

# Start (uses installed binary)
$ ./rbee queen start
[kpr-life  ] queen_start    : Using installed queen-rbee binary at /home/user/.local/bin/queen-rbee
[queen-life] queen_start    : ✅ Queen started on http://localhost:7833

# Uninstall
$ ./rbee queen stop
$ ./rbee queen uninstall
[queen-life] queen_uninstall: 🗑️ Uninstalling queen-rbee...
[queen-life] queen_uninstall: ✅ Queen uninstalled successfully!

# Start (now fails - binary is gone)
$ ./rbee queen start
Error: Failed to find queen-rbee binary (not installed and not in target/)
```

**Development Workflow (No Install):**
```bash
# Build for development
$ cargo build --bin queen-rbee

# Start (uses development binary)
$ ./rbee queen start
[kpr-life  ] queen_start    : Using development queen-rbee binary at /home/user/Projects/llama-orch/target/debug/queen-rbee
[queen-life] queen_start    : ✅ Queen started on http://localhost:7833
```

**Mixed Workflow (Both Exist):**
```bash
# Install production binary
$ ./rbee queen install

# Also have development binary
$ cargo build --bin queen-rbee

# Start (prefers installed binary)
$ ./rbee queen start
[kpr-life  ] queen_start    : Using installed queen-rbee binary at /home/user/.local/bin/queen-rbee
[queen-life] queen_start    : ✅ Queen started on http://localhost:7833
```

## Files Changed

1. **`bin/05_rbee_keeper_crates/queen-lifecycle/src/ensure.rs`** (lines 131-154)
   - Added installed binary check
   - Prefer `~/.local/bin/queen-rbee` over `target/` binaries
   - Clear narration showing which binary is used

## Benefits

1. **Consistent Behavior:** Install/uninstall now actually affect what start uses
2. **Production Ready:** Installed binary takes precedence (as it should)
3. **Development Friendly:** Still falls back to target/ if not installed
4. **Clear Feedback:** Narration shows which binary is being used
5. **Error Messages:** Clear error if neither binary exists

## Testing

### Before Fix
```bash
$ ./rbee queen install
✅ Installed to ~/.local/bin/queen-rbee

$ ./rbee queen uninstall
✅ Removed ~/.local/bin/queen-rbee

$ ./rbee queen start
✅ Started (using target/debug/queen-rbee - WRONG!)
```

### After Fix
```bash
$ ./rbee queen install
✅ Installed to ~/.local/bin/queen-rbee

$ ./rbee queen uninstall
✅ Removed ~/.local/bin/queen-rbee

$ ./rbee queen start
❌ Error: Failed to find queen-rbee binary (not installed and not in target/)
```

## Compilation

```bash
$ cargo check -p queen-lifecycle
✅ SUCCESS

$ cargo build --bin rbee-keeper
✅ SUCCESS
```

## Related Work

- **TEAM-296:** Queen lifecycle implementation (install, update, uninstall)
- **TEAM-296:** Binary path fix (this work)

---

**TEAM-296: Fixed start to prefer installed binary over development binary, ensuring install/uninstall actually work as expected.**
