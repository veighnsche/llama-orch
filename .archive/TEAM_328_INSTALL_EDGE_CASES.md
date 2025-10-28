# TEAM-328: install.rs Edge Case Analysis

## Current Coverage

### ✅ Covered Edge Cases

1. **Source binary not found** (Line 59-61)
   ```rust
   if !p.exists() {
       anyhow::bail!("Binary not found at: {}", path);
   }
   ```

2. **HOME not set** (Line 71)
   ```rust
   let home = std::env::var("HOME").context("HOME environment variable not set")?;
   ```

3. **Already installed** (Line 77-83)
   ```rust
   if install_path.exists() {
       anyhow::bail!("{} already installed at {}. Uninstall first or use rebuild.", ...);
   }
   ```

4. **Directory creation failure** (Line 86-87)
   ```rust
   std::fs::create_dir_all(&install_dir)
       .context("Failed to create ~/.local/bin directory")?;
   ```

5. **Copy failure** (Line 91-92)
   ```rust
   std::fs::copy(&source, &install_path)
       .context(format!("Failed to copy {} to {}", ...))?;
   ```

6. **Permission setting (Unix)** (Line 95-101)
   ```rust
   #[cfg(unix)]
   {
       let mut perms = std::fs::metadata(&install_path)?.permissions();
       perms.set_mode(0o755);
       std::fs::set_permissions(&install_path, perms)?;
   }
   ```

7. **Version check graceful failure** (Line 108-112)
   ```rust
   let version = if let Ok(out) = output {
       String::from_utf8_lossy(&out.stdout).trim().to_string()
   } else {
       "unknown".to_string()
   };
   ```

---

## ❌ Missing Edge Cases

### Critical Issues

#### 1. **Partial Copy Failure (Atomicity)**
**Problem:** If copy succeeds but chmod fails, you have a non-executable binary installed.

**Current behavior:**
```rust
std::fs::copy(&source, &install_path)?;  // ✅ Success
// ... later ...
std::fs::set_permissions(&install_path, perms)?;  // ❌ FAILS
// Binary is installed but not executable!
```

**Impact:** User has broken installation, can't run binary, can't reinstall (already exists check blocks it)

**Fix needed:**
```rust
// If chmod fails, delete the copied binary
#[cfg(unix)]
{
    if let Err(e) = std::fs::set_permissions(&install_path, perms) {
        // Cleanup: remove partially installed binary
        let _ = std::fs::remove_file(&install_path);
        return Err(e.into());
    }
}
```

#### 2. **Source and Destination are the Same**
**Problem:** If user tries to install from `~/.local/bin/queen-rbee` to `~/.local/bin/queen-rbee`

**Current behavior:**
```rust
// Already exists check would catch this, but with wrong error message
anyhow::bail!("{} already installed at {}. Uninstall first or use rebuild.", ...)
```

**Impact:** Confusing error message (it's not "already installed", it's the source!)

**Fix needed:**
```rust
// After determining source and install_path
if source.canonicalize()? == install_path.canonicalize().unwrap_or_default() {
    anyhow::bail!("Source and destination are the same: {}", install_path.display());
}
```

#### 3. **Source Binary is Not Executable**
**Problem:** Source binary might not have execute permissions

**Current behavior:** Copies non-executable binary, sets permissions on copy, but doesn't verify source

**Impact:** Could copy a corrupted/incomplete build

**Fix needed:**
```rust
// After finding source, verify it's executable
#[cfg(unix)]
{
    use std::os::unix::fs::PermissionsExt;
    let source_perms = std::fs::metadata(&source)?.permissions();
    if source_perms.mode() & 0o111 == 0 {
        anyhow::bail!("Source binary is not executable: {}", source.display());
    }
}
```

#### 4. **Insufficient Disk Space**
**Problem:** Copy might fail due to disk space, but error message is generic

**Current behavior:**
```rust
std::fs::copy(&source, &install_path)
    .context(format!("Failed to copy {} to {}", binary_name, install_path.display()))?;
```

**Impact:** User gets "Failed to copy" but doesn't know why (disk full? permissions? network drive?)

**Fix needed:** Already handled by `.context()`, but could be more specific:
```rust
std::fs::copy(&source, &install_path)
    .context(format!(
        "Failed to copy {} to {} (check disk space and permissions)",
        binary_name,
        install_path.display()
    ))?;
```

---

### Medium Priority Issues

#### 5. **Install Directory is a File**
**Problem:** If `~/.local/bin` exists as a file (not directory)

**Current behavior:**
```rust
std::fs::create_dir_all(&install_dir)  // Will fail
```

**Impact:** Confusing error from `create_dir_all`

**Fix needed:**
```rust
if install_dir.exists() && !install_dir.is_dir() {
    anyhow::bail!("Install directory exists but is not a directory: {}", install_dir.display());
}
std::fs::create_dir_all(&install_dir)?;
```

#### 6. **Binary Name Contains Path Separators**
**Problem:** `binary_name` could be `"../../evil"` (path traversal)

**Current behavior:**
```rust
let install_path = install_dir.join(binary_name);  // Could escape install_dir!
```

**Impact:** Security issue - could install binary outside intended directory

**Fix needed:**
```rust
// Validate binary_name doesn't contain path separators
if binary_name.contains('/') || binary_name.contains('\\') {
    anyhow::bail!("Binary name cannot contain path separators: {}", binary_name);
}
```

#### 7. **Symlink Handling**
**Problem:** What if install_path is a symlink to the source?

**Current behavior:**
```rust
if install_path.exists() {  // symlink.exists() follows the link
    anyhow::bail!("already installed");
}
```

**Impact:** Could have symlink pointing to source, but check passes

**Fix needed:**
```rust
// Check both regular file and symlink
if install_path.exists() || install_path.symlink_metadata().is_ok() {
    anyhow::bail!("{} already exists at {}", binary_name, install_path.display());
}
```

---

### Low Priority Issues

#### 8. **Version Check Timeout**
**Problem:** `--version` command could hang indefinitely

**Current behavior:**
```rust
let output = std::process::Command::new(&install_path)
    .arg("--version")
    .output();  // No timeout!
```

**Impact:** Install command hangs if binary is broken

**Fix needed:**
```rust
use std::time::Duration;
use tokio::time::timeout;

let version = match timeout(Duration::from_secs(5), async {
    tokio::process::Command::new(&install_path)
        .arg("--version")
        .output()
        .await
}).await {
    Ok(Ok(out)) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
    _ => "unknown".to_string(),
};
```

#### 9. **Empty Binary Name**
**Problem:** `binary_name` could be empty string

**Current behavior:** Would create file named "" in install_dir

**Fix needed:**
```rust
if binary_name.is_empty() {
    anyhow::bail!("Binary name cannot be empty");
}
```

#### 10. **Windows Executable Extension**
**Problem:** On Windows, binaries need `.exe` extension

**Current behavior:** Doesn't add `.exe` on Windows

**Fix needed:**
```rust
#[cfg(windows)]
let binary_name = if !binary_name.ends_with(".exe") {
    format!("{}.exe", binary_name)
} else {
    binary_name.to_string()
};
```

---

## Summary

### Critical (Must Fix)
1. ❌ **Partial copy failure** - No cleanup on chmod failure
2. ❌ **Source == destination** - Confusing error message
3. ❌ **Source not executable** - No validation

### Medium Priority (Should Fix)
4. ⚠️ **Install dir is file** - Better error message
5. ⚠️ **Path traversal** - Security issue
6. ⚠️ **Symlink handling** - Edge case

### Low Priority (Nice to Have)
7. 💡 **Version check timeout** - Prevent hangs
8. 💡 **Empty binary name** - Input validation
9. 💡 **Windows .exe** - Platform compatibility

### Already Covered ✅
- Source not found
- HOME not set
- Already installed
- Directory creation failure
- Copy failure
- Permission setting (Unix)
- Version check graceful failure

---

## Recommendation

**Fix the 3 critical issues immediately:**
1. Add cleanup on chmod failure (atomicity)
2. Check if source == destination
3. Validate source is executable

**Consider the medium priority issues:**
4. Check install_dir is actually a directory
5. Validate binary_name doesn't contain path separators

The low priority issues can be addressed later or as bugs are reported.

---

**TEAM-328 Assessment:** install.rs covers basic cases but missing critical atomicity and validation
