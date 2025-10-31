# Phase 2: Binary Mode Detection

**Goal:** Create function to read build mode from any binary.

**Status:** Ready to implement  
**Estimated Time:** 20 minutes

---

## What We're Building

A utility function that:
1. Executes a binary with `--build-info` flag
2. Captures stdout
3. Returns "debug" or "release"

---

## Implementation Checklist

### **Step 1: Add mode detection function**

**File:** `bin/96_lifecycle/lifecycle-local/src/utils/binary.rs` (MODIFY)

**Add new function after existing functions:**

```rust
use std::process::Command;
use std::path::Path;
use anyhow::{Context, Result};

/// Get build mode of a binary by executing it with --build-info
///
/// # Arguments
/// * `binary_path` - Path to the binary to check
///
/// # Returns
/// * `Ok("debug")` - Binary was built in debug mode
/// * `Ok("release")` - Binary was built in release mode
/// * `Err(_)` - Binary doesn't support --build-info or execution failed
///
/// # Example
/// ```rust,no_run
/// use std::path::PathBuf;
/// use lifecycle_local::utils::binary::get_binary_mode;
///
/// # async fn example() -> anyhow::Result<()> {
/// let path = PathBuf::from("target/release/queen-rbee");
/// let mode = get_binary_mode(&path)?;
/// assert_eq!(mode, "release");
/// # Ok(())
/// # }
/// ```
pub fn get_binary_mode(binary_path: &Path) -> Result<String> {
    // Execute binary with --build-info flag
    let output = Command::new(binary_path)
        .arg("--build-info")
        .output()
        .with_context(|| format!("Failed to execute {} --build-info", binary_path.display()))?;

    // Check if command succeeded
    if !output.status.success() {
        anyhow::bail!(
            "Binary {} does not support --build-info flag",
            binary_path.display()
        );
    }

    // Parse output
    let mode = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // Validate mode
    if mode != "debug" && mode != "release" {
        anyhow::bail!(
            "Invalid build mode '{}' from binary {}",
            mode,
            binary_path.display()
        );
    }

    Ok(mode)
}

/// Check if a binary is a release build
///
/// # Arguments
/// * `binary_path` - Path to the binary to check
///
/// # Returns
/// * `Ok(true)` - Binary is a release build
/// * `Ok(false)` - Binary is a debug build
/// * `Err(_)` - Could not determine mode
pub fn is_release_binary(binary_path: &Path) -> Result<bool> {
    let mode = get_binary_mode(binary_path)?;
    Ok(mode == "release")
}
```

---

### **Step 2: Export new functions**

**File:** `bin/96_lifecycle/lifecycle-local/src/utils/binary.rs` (MODIFY)

Make sure functions are public and exported in the module.

**File:** `bin/96_lifecycle/lifecycle-local/src/lib.rs` (CHECK)

Ensure `utils::binary` module is public:
```rust
pub mod utils {
    pub mod binary;
    // ... other modules ...
}
```

---

## Testing Phase 2

### **Test 1: Detect debug binary**

```bash
# Build debug binary
cargo build --bin queen-rbee

# Test detection (manual)
./target/debug/queen-rbee --build-info
# Expected: debug
```

**Rust test:**
```rust
#[test]
fn test_detect_debug_binary() {
    let path = PathBuf::from("target/debug/queen-rbee");
    if path.exists() {
        let mode = get_binary_mode(&path).unwrap();
        assert_eq!(mode, "debug");
        assert!(!is_release_binary(&path).unwrap());
    }
}
```

### **Test 2: Detect release binary**

```bash
# Build release binary
cargo build --release --bin queen-rbee

# Test detection (manual)
./target/release/queen-rbee --build-info
# Expected: release
```

**Rust test:**
```rust
#[test]
fn test_detect_release_binary() {
    let path = PathBuf::from("target/release/queen-rbee");
    if path.exists() {
        let mode = get_binary_mode(&path).unwrap();
        assert_eq!(mode, "release");
        assert!(is_release_binary(&path).unwrap());
    }
}
```

### **Test 3: Handle missing binary**

```rust
#[test]
fn test_missing_binary() {
    let path = PathBuf::from("/nonexistent/binary");
    assert!(get_binary_mode(&path).is_err());
}
```

### **Test 4: Handle binary without --build-info**

```rust
#[test]
fn test_binary_without_flag() {
    // Use a system binary that doesn't have --build-info
    let path = PathBuf::from("/bin/ls");
    assert!(get_binary_mode(&path).is_err());
}
```

---

## Success Criteria

- ✅ `get_binary_mode()` function implemented
- ✅ `is_release_binary()` helper function implemented
- ✅ Functions exported from module
- ✅ Debug binaries return "debug"
- ✅ Release binaries return "release"
- ✅ Invalid binaries return error
- ✅ All tests pass

---

## Files Modified

### **MODIFIED Files**
- `bin/96_lifecycle/lifecycle-local/src/utils/binary.rs`
- `bin/96_lifecycle/lifecycle-local/src/lib.rs` (if needed for exports)

---

## Next Phase

After Phase 2 is complete and tested, proceed to:
**`PHASE_3_SMART_SELECTION.md`** - Choose correct binary based on mode

---

**Ready to implement!** 🔍
