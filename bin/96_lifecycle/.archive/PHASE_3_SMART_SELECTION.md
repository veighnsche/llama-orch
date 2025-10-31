# Phase 3: Smart Binary Selection

**Goal:** Update binary selection logic to prefer production binaries when installed.

**Status:** Ready to implement  
**Estimated Time:** 30 minutes

---

## What We're Building

Smart binary selection that:
1. Checks if `~/.local/bin/{daemon}` exists
2. If yes and it's release mode → use it (production install)
3. Otherwise → use `target/debug/{daemon}` (development)

---

## Current vs New Behavior

### **Current (Ambiguous)**
```
find_binary_command():
  1. Check target/debug/{daemon}
  2. Check target/release/{daemon}
  3. Check ~/.local/bin/{daemon}
  → Returns FIRST match (wrong!)
```

### **New (Smart)**
```
find_binary_with_mode_preference():
  1. Check ~/.local/bin/{daemon}
     - If exists AND is release → USE IT ✅
  2. Fallback to target/debug/{daemon}
  3. Fallback to target/release/{daemon}
  → Returns BEST match based on install mode
```

---

## Implementation Checklist

### **Step 1: Create smart binary finder**

**File:** `bin/96_lifecycle/lifecycle-shared/src/lib.rs` (MODIFY)

**Add new function (RULE ZERO: Replace old logic):**

```rust
use std::path::PathBuf;
use anyhow::{Context, Result};
use observability_narration_core::n;

/// Find binary with smart mode-aware selection
///
/// # Logic
/// 1. If ~/.local/bin/{daemon} exists AND is release mode → use it (production install)
/// 2. Otherwise, try target/debug/{daemon} (development)
/// 3. Otherwise, try target/release/{daemon} (fallback)
///
/// # Arguments
/// * `daemon_name` - Name of the daemon binary
///
/// # Returns
/// * `Ok(PathBuf)` - Path to the binary to use
/// * `Err(_)` - No suitable binary found
///
/// # Example
/// ```rust,ignore
/// let binary_path = find_binary_smart("queen-rbee")?;
/// // If production installed: ~/.local/bin/queen-rbee
/// // Otherwise: target/debug/queen-rbee
/// ```
pub fn find_binary_smart(daemon_name: &str) -> Result<PathBuf> {
    use lifecycle_local::utils::binary::get_binary_mode;
    
    // Step 1: Check if production binary is installed
    let home = std::env::var("HOME").ok();
    if let Some(home_dir) = home {
        let installed_path = PathBuf::from(&home_dir)
            .join(BINARY_INSTALL_DIR)
            .join(daemon_name);

        if installed_path.exists() {
            // Check if it's a release binary
            match get_binary_mode(&installed_path) {
                Ok(mode) if mode == "release" => {
                    n!(
                        "binary_found_prod",
                        "✅ Using production binary: {}",
                        installed_path.display()
                    );
                    return Ok(installed_path);
                }
                Ok(mode) => {
                    n!(
                        "binary_found_wrong_mode",
                        "⚠️  Found {} in ~/.local/bin but it's {} mode, not release",
                        daemon_name,
                        mode
                    );
                }
                Err(e) => {
                    n!(
                        "binary_check_failed",
                        "⚠️  Could not check mode of ~/.local/bin/{}: {}",
                        daemon_name,
                        e
                    );
                }
            }
        }
    }

    // Step 2: Try target/debug (development)
    let debug_path = PathBuf::from(format!("target/debug/{}", daemon_name));
    if debug_path.exists() {
        n!(
            "binary_found_dev",
            "✅ Using development binary: {}",
            debug_path.display()
        );
        return Ok(debug_path);
    }

    // Step 3: Try target/release (fallback)
    let release_path = PathBuf::from(format!("target/release/{}", daemon_name));
    if release_path.exists() {
        n!(
            "binary_found_release_fallback",
            "✅ Using release binary (fallback): {}",
            release_path.display()
        );
        return Ok(release_path);
    }

    // Step 4: Not found anywhere
    anyhow::bail!(
        "Binary '{}' not found. Tried:\n  - ~/.local/bin/{}\n  - target/debug/{}\n  - target/release/{}",
        daemon_name,
        daemon_name,
        daemon_name,
        daemon_name
    );
}
```

---

### **Step 2: Update start.rs to use smart finder**

**File:** `bin/96_lifecycle/lifecycle-local/src/start.rs` (MODIFY)

**Replace the find binary logic:**

**OLD (DELETE):**
```rust
// Step 1: Find binary on local machine
n!("find_binary", "🔍 Locating {} binary locally...", daemon_name);

let find_cmd = find_binary_command(daemon_name);
let binary_path = local_exec(&find_cmd).await.context("Failed to find binary locally")?;

let binary_path = binary_path.trim();

if binary_path == "NOT_FOUND" || binary_path.is_empty() {
    n!("binary_not_found", "❌ Binary '{}' not found locally", daemon_name);
    anyhow::bail!(
        "Binary '{}' not found locally. Install it first with install_daemon()",
        daemon_name
    );
}

n!("found_binary", "✅ Found binary at: {}", binary_path);
```

**NEW (REPLACE):**
```rust
// Step 1: Find binary on local machine (smart mode-aware selection)
n!("find_binary", "🔍 Locating {} binary locally...", daemon_name);

use lifecycle_shared::find_binary_smart;
let binary_path = find_binary_smart(daemon_name)
    .context("Failed to find binary locally")?;

n!("found_binary", "✅ Found binary at: {}", binary_path.display());
```

---

### **Step 3: Deprecate old find_binary_command (RULE ZERO)**

**File:** `bin/96_lifecycle/lifecycle-shared/src/lib.rs` (MODIFY)

**DELETE the old function entirely:**

```rust
// DELETED - RULE ZERO: Replaced with find_binary_smart()
// pub fn find_binary_command(daemon_name: &str) -> String { ... }
```

**Or mark as deprecated if needed for SSH:**
```rust
#[deprecated(
    since = "0.2.0",
    note = "Use find_binary_smart() instead for mode-aware selection"
)]
pub fn find_binary_command(daemon_name: &str) -> String {
    // ... existing implementation ...
}
```

---

## Testing Phase 3

### **Test 1: Production binary preferred**

```bash
# Build both debug and release
cargo build --bin queen-rbee
cargo build --release --bin queen-rbee

# Install production
cp target/release/queen-rbee ~/.local/bin/

# Test: Should use ~/.local/bin (release)
# (Run through start_daemon and check narration)
```

### **Test 2: Development fallback**

```bash
# Remove production binary
rm ~/.local/bin/queen-rbee

# Test: Should use target/debug
# (Run through start_daemon and check narration)
```

### **Test 3: Release fallback**

```bash
# Remove both production and debug
rm ~/.local/bin/queen-rbee
rm target/debug/queen-rbee

# Test: Should use target/release
# (Run through start_daemon and check narration)
```

### **Test 4: Error when nothing exists**

```bash
# Remove all binaries
rm ~/.local/bin/queen-rbee
rm target/debug/queen-rbee
rm target/release/queen-rbee

# Test: Should error with helpful message
# (Run through start_daemon and check error)
```

---

## Success Criteria

- ✅ `find_binary_smart()` function implemented
- ✅ Production binary preferred when installed
- ✅ Development binary used as fallback
- ✅ Clear narration messages for each case
- ✅ Old `find_binary_command()` deleted or deprecated
- ✅ `start.rs` updated to use new logic
- ✅ All tests pass

---

## Files Modified

### **MODIFIED Files**
- `bin/96_lifecycle/lifecycle-shared/src/lib.rs`
- `bin/96_lifecycle/lifecycle-local/src/start.rs`

### **DELETED Functions (RULE ZERO)**
- `find_binary_command()` - Replaced with `find_binary_smart()`

---

## Next Phase

After Phase 3 is complete and tested, proceed to:
**`PHASE_4_INSTALL_UPDATE.md`** - Update install logic for dev/prod

---

**Ready to implement!** 🎯
