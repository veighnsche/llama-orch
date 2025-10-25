# TEAM-296: Install Required Fix - No Development Fallback

**Status:** ✅ COMPLETE  
**Date:** Oct 26, 2025

## Problem

After uninstalling queen, it could still be started because the code fell back to the development binary in `target/`:

```bash
$ ./rbee queen uninstall
✅ Queen uninstalled successfully!

$ ./rbee queen start
✅ Queen started  # ← WRONG! Should fail
[kpr-life] queen_start: Using development queen-rbee binary at /home/vince/Projects/llama-orch/target/release/queen-rbee
```

This made install/uninstall meaningless - queen could always run from target/.

## Root Cause

The binary resolution logic had a fallback:

```rust
// OLD CODE
let queen_binary = {
    let installed_path = PathBuf::from("~/.local/bin/queen-rbee");
    
    if installed_path.exists() {
        installed_path  // Use installed binary
    } else {
        DaemonManager::find_in_target("queen-rbee")?  // ← Fallback to target/
    }
};
```

This was **too flexible** - it allowed running queen even when not installed.

## Solution

Removed the fallback to development binary. Now start **requires** queen to be installed:

```rust
// NEW CODE - TEAM-296
let queen_binary = {
    let home = std::env::var("HOME")?;
    let installed_path = PathBuf::from(format!("{}/.local/bin/queen-rbee", home));
    
    if installed_path.exists() {
        NARRATE
            .action("queen_start")
            .context(installed_path.display().to_string())
            .human("Using installed queen-rbee binary at {}")
            .emit();
        installed_path
    } else {
        NARRATE
            .action("queen_start")
            .human("❌ Queen not installed. Run 'rbee queen install' first.")
            .error_kind("not_installed")
            .emit();
        anyhow::bail!(
            "Queen not installed at {}. Run 'rbee queen install' to install from source.",
            installed_path.display()
        );
    }
};
```

## New Behavior

### Install → Start → Uninstall → Start Fails

```bash
# 1. Install queen
$ ./rbee queen install
[queen-life] queen_install  : 📦 Installing queen-rbee...
[queen-life] queen_install  : 🔨 Building queen-rbee from source...
[queen-life] queen_install  : ✅ Build successful!
[queen-life] queen_install  : 📋 Installing to: /home/vince/.local/bin/queen-rbee
[queen-life] queen_install  : ✅ Queen installed successfully!

# 2. Start queen (uses installed binary)
$ ./rbee queen start
[kpr-life  ] queen_start    : Using installed queen-rbee binary at /home/vince/.local/bin/queen-rbee
[queen-life] queen_start    : ✅ Queen started on http://localhost:7833

# 3. Stop queen
$ ./rbee queen stop
[queen-life] queen_stop     : ✅ Queen stopped

# 4. Uninstall queen
$ ./rbee queen uninstall
[queen-life] queen_uninstall: 🗑️ Uninstalling queen-rbee...
[dmn-life  ] daemon_uninstalled: ✅ Daemon 'queen-rbee' uninstalled successfully!
[queen-life] queen_uninstall: ✅ Queen uninstalled successfully!

# 5. Try to start queen (NOW FAILS - CORRECT!)
$ ./rbee queen start
[kpr-life  ] queen_start    : ❌ Queen not installed. Run 'rbee queen install' first.
Error: Queen not installed at /home/vince/.local/bin/queen-rbee. Run 'rbee queen install' to install from source.
```

### Start Without Install

```bash
# Fresh system, no queen installed
$ ./rbee queen start
[kpr-life  ] queen_start    : ❌ Queen not installed. Run 'rbee queen install' first.
Error: Queen not installed at /home/vince/.local/bin/queen-rbee. Run 'rbee queen install' to install from source.

# Install first
$ ./rbee queen install
[queen-life] queen_install  : ✅ Queen installed successfully!

# Now start works
$ ./rbee queen start
[kpr-life  ] queen_start    : Using installed queen-rbee binary at /home/vince/.local/bin/queen-rbee
[queen-life] queen_start    : ✅ Queen started on http://localhost:7833
```

## Files Changed

1. **`bin/05_rbee_keeper_crates/queen-lifecycle/src/ensure.rs`** (lines 131-156)
   - Removed fallback to `DaemonManager::find_in_target()`
   - Now errors if `~/.local/bin/queen-rbee` doesn't exist
   - Clear error message with instructions

## Benefits

1. **Install/Uninstall Actually Work:** They now control whether queen can run
2. **Clear User Feedback:** Error message tells user exactly what to do
3. **Production Ready:** Forces proper installation workflow
4. **No Confusion:** Can't accidentally run development binary

## Trade-offs

### Before (Development Friendly)
- ✅ Can run queen from target/ without installing
- ✅ Flexible for development
- ❌ Install/uninstall don't actually control queen
- ❌ Confusing behavior (uninstall doesn't prevent start)

### After (Production Focused)
- ✅ Install/uninstall actually work
- ✅ Clear error messages
- ✅ Enforces proper workflow
- ❌ Must install to run (even for development)

## Development Workflow

For developers who want to test changes without installing:

### Option 1: Use Install/Uninstall
```bash
# Make changes
$ vim bin/10_queen_rbee/src/main.rs

# Install (builds and copies to ~/.local/bin)
$ ./rbee queen install

# Test
$ ./rbee queen start

# Uninstall when done
$ ./rbee queen uninstall
```

### Option 2: Run Queen Directly
```bash
# Build
$ cargo build --release --bin queen-rbee

# Run directly (bypass rbee wrapper)
$ ./target/release/queen-rbee --port 7833
```

### Option 3: Temporary Symlink
```bash
# Build
$ cargo build --release --bin queen-rbee

# Create symlink
$ ln -s $(pwd)/target/release/queen-rbee ~/.local/bin/queen-rbee

# Now rbee will use it
$ ./rbee queen start

# Remove symlink when done
$ rm ~/.local/bin/queen-rbee
```

## Testing

### Compilation
```bash
$ cargo build --bin rbee-keeper
✅ SUCCESS
```

### Manual Testing
```bash
# Test 1: Start without install (should fail)
$ ./rbee queen start
❌ Error: Queen not installed

# Test 2: Install
$ ./rbee queen install
✅ Queen installed successfully!

# Test 3: Start after install (should work)
$ ./rbee queen start
✅ Queen started

# Test 4: Uninstall
$ ./rbee queen stop
$ ./rbee queen uninstall
✅ Queen uninstalled successfully!

# Test 5: Start after uninstall (should fail)
$ ./rbee queen start
❌ Error: Queen not installed
```

All tests pass! ✅

## Related Work

- **TEAM-296:** Queen lifecycle implementation (install, update, uninstall)
- **TEAM-296:** Binary path fix (prefer installed over development)
- **TEAM-296:** Install required fix (this work)

---

**TEAM-296: Removed development binary fallback. Now start REQUIRES queen to be installed to ~/.local/bin. Install/uninstall actually work as expected.**
