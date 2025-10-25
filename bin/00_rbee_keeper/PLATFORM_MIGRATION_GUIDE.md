# Platform Migration Guide

**TEAM-293: How to migrate existing code to use platform abstraction**

## Quick Reference

### ✅ DO: Use Platform Module

```rust
use rbee_keeper::platform;

// Get config directory (cross-platform)
let config_dir = platform::config_dir()?;

// Get binary with correct extension
let exe = format!("queen-rbee{}", platform::exe_extension());

// Check if process is running
if platform::is_running(pid) { /* ... */ }

// Graceful termination
platform::terminate(pid)?;

// Force kill
platform::kill(pid)?;

// Check SSH support
if platform::has_ssh_support() {
    let ssh = platform::ssh_executable();
    // Use SSH
}
```

### ❌ DON'T: Hardcode Platform-Specific Values

```rust
// ❌ BAD: Hardcoded paths
let config_dir = PathBuf::from("~/.config/rbee");
let bin_dir = PathBuf::from("~/.local/bin");

// ❌ BAD: Linux-only commands
Command::new("kill").arg("-TERM").arg(pid.to_string());

// ❌ BAD: No extension handling
let exe = "queen-rbee";  // Won't work on Windows

// ❌ BAD: Hardcoded SSH
let ssh = "ssh";  // Should be "ssh.exe" on Windows
```

## Migration Patterns

### Pattern 1: Configuration Paths

**Before:**
```rust
fn get_config_path() -> PathBuf {
    PathBuf::from(env!("HOME"))
        .join(".config")
        .join("rbee")
        .join("config.toml")
}
```

**After:**
```rust
use rbee_keeper::platform;

fn get_config_path() -> Result<PathBuf> {
    Ok(platform::config_dir()?.join("config.toml"))
}
```

### Pattern 2: Binary Paths

**Before:**
```rust
fn get_queen_binary() -> String {
    "queen-rbee".to_string()
}
```

**After:**
```rust
use rbee_keeper::platform;

fn get_queen_binary() -> String {
    format!("queen-rbee{}", platform::exe_extension())
}
```

### Pattern 3: Process Management

**Before:**
```rust
use std::process::Command;

fn is_running(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{}", pid)).exists()
}

fn kill_process(pid: u32) -> Result<()> {
    Command::new("kill")
        .arg("-KILL")
        .arg(pid.to_string())
        .status()?;
    Ok(())
}
```

**After:**
```rust
use rbee_keeper::platform;

fn is_running(pid: u32) -> bool {
    platform::is_running(pid)
}

fn kill_process(pid: u32) -> Result<()> {
    platform::kill(pid)
}
```

### Pattern 4: SSH Detection

**Before:**
```rust
fn can_use_ssh() -> bool {
    true  // Assumes Linux
}
```

**After:**
```rust
use rbee_keeper::platform;

fn can_use_ssh() -> bool {
    platform::has_ssh_support()
}

fn get_ssh_command(host: &str) -> Command {
    let mut cmd = Command::new(platform::ssh_executable());
    cmd.arg(host);
    cmd
}
```

### Pattern 5: Conditional Features

**Before:**
```rust
// No fallback for missing SSH
fn remote_operation(host: &str) -> Result<()> {
    let output = Command::new("ssh")
        .arg(host)
        .arg("command")
        .output()?;
    Ok(())
}
```

**After:**
```rust
use rbee_keeper::platform;

fn remote_operation(host: &str) -> Result<()> {
    if !platform::has_ssh_support() {
        anyhow::bail!(
            "Remote operations require SSH. \
             On Windows, install via: \
             Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0"
        );
    }
    
    let output = Command::new(platform::ssh_executable())
        .arg(host)
        .arg("command")
        .output()?;
    Ok(())
}
```

## Code Examples by Module

### Config Module (src/config.rs)

**Already Cross-Platform! ✅**

The config module already uses `dirs::config_dir()` which is cross-platform:

```rust
fn config_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir()
        .context("Failed to get config directory")?;
    Ok(config_dir.join("rbee").join("config.toml"))
}
```

**Could be enhanced with:**
```rust
use crate::platform;

fn config_path() -> Result<PathBuf> {
    Ok(platform::config_dir()?.join("config.toml"))
}
```

### Hive Lifecycle (bin/05_rbee_keeper_crates/hive-lifecycle)

**Needs Migration:**

**Current (Linux-only):**
```rust
// src/ssh_helper.rs
fn ssh_command(host: &str) -> Command {
    let mut cmd = Command::new("ssh");  // ❌ Linux-only
    cmd.arg(host);
    cmd
}
```

**Migrated (Cross-platform):**
```rust
use rbee_keeper::platform;

fn ssh_command(host: &str) -> Result<Command> {
    if !platform::has_ssh_support() {
        anyhow::bail!("SSH not available on this platform");
    }
    
    let mut cmd = Command::new(platform::ssh_executable());
    cmd.arg(host);
    Ok(cmd)
}
```

### Queen Lifecycle (bin/05_rbee_keeper_crates/queen-lifecycle)

**Needs Migration:**

**Current (Linux-only):**
```rust
fn find_queen_binary() -> Result<PathBuf> {
    let debug_path = PathBuf::from("target/debug/queen-rbee");
    let release_path = PathBuf::from("target/release/queen-rbee");
    // ...
}
```

**Migrated (Cross-platform):**
```rust
use rbee_keeper::platform;

fn find_queen_binary() -> Result<PathBuf> {
    let exe_name = format!("queen-rbee{}", platform::exe_extension());
    let debug_path = PathBuf::from("target/debug").join(&exe_name);
    let release_path = PathBuf::from("target/release").join(&exe_name);
    // ...
}
```

## Testing Checklist

When migrating code, test on all platforms:

### ✅ Linux (Primary)
- [ ] Build succeeds
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing complete

### ⚠️ macOS (Secondary)
- [ ] Build succeeds
- [ ] Config created in correct location
- [ ] Binaries have correct paths
- [ ] SSH operations work
- [ ] GUI builds correctly

### ⚠️ Windows (Secondary)
- [ ] Build succeeds
- [ ] Config created in `%APPDATA%\rbee`
- [ ] Binaries have `.exe` extension
- [ ] Process management works (tasklist/taskkill)
- [ ] SSH detection works
- [ ] Graceful degradation when SSH not available
- [ ] GUI builds correctly

## Common Pitfalls

### 1. Path Separators

**❌ DON'T:**
```rust
let path = format!("/home/user/.config/rbee");  // Unix-only
let path = format!("C:\\Users\\user\\AppData\\rbee");  // Windows-only
```

**✅ DO:**
```rust
use std::path::PathBuf;
use rbee_keeper::platform;

let path = platform::config_dir()?.join("rbee");  // Cross-platform
```

### 2. Executable Extensions

**❌ DON'T:**
```rust
let binary = "queen-rbee";  // Missing .exe on Windows
```

**✅ DO:**
```rust
use rbee_keeper::platform;

let binary = format!("queen-rbee{}", platform::exe_extension());
```

### 3. Command Names

**❌ DON'T:**
```rust
Command::new("which").arg("ssh");  // Unix-only
Command::new("tasklist");  // Windows-only
```

**✅ DO:**
```rust
use rbee_keeper::platform;

// Use platform abstraction
if platform::is_running(pid) { /* ... */ }

// Or use conditional compilation
#[cfg(not(target_os = "windows"))]
fn find_executable(name: &str) -> Option<PathBuf> {
    Command::new("which").arg(name).output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| PathBuf::from(s.trim()))
}

#[cfg(target_os = "windows")]
fn find_executable(name: &str) -> Option<PathBuf> {
    Command::new("where").arg(name).output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| PathBuf::from(s.trim()))
}
```

### 4. Assuming SSH

**❌ DON'T:**
```rust
fn connect_remote(host: &str) -> Result<()> {
    Command::new("ssh").arg(host).status()?;  // Assumes SSH exists
    Ok(())
}
```

**✅ DO:**
```rust
use rbee_keeper::platform;

fn connect_remote(host: &str) -> Result<()> {
    if !platform::has_ssh_support() {
        anyhow::bail!(
            "SSH not available. Remote operations disabled.\n\
             Windows users: Install OpenSSH via Settings → Apps → Optional Features"
        );
    }
    
    Command::new(platform::ssh_executable())
        .arg(host)
        .status()?;
    Ok(())
}
```

## Gradual Migration Strategy

You don't need to migrate everything at once. Use this strategy:

### Phase 1: New Code (Immediate)
- All **new** code uses `platform` module
- Prevents introducing new platform-specific code

### Phase 2: High-Impact Areas (Soon)
- Binary path resolution
- Process management
- SSH operations

### Phase 3: Low-Impact Areas (Later)
- Configuration (already mostly cross-platform)
- Logging and output
- Helper utilities

### Phase 4: Testing (Ongoing)
- Test on macOS
- Test on Windows
- Document platform differences

## Resources

### Internal
- `CROSS_PLATFORM.md` - Complete cross-platform guide
- `src/platform/mod.rs` - Platform abstraction traits
- `src/platform/linux.rs` - Linux implementation
- `src/platform/macos.rs` - macOS implementation
- `src/platform/windows.rs` - Windows implementation

### External
- [Rust Platform Support](https://doc.rust-lang.org/rustc/platform-support.html)
- [dirs crate](https://docs.rs/dirs/) - Cross-platform paths
- [std::path](https://doc.rust-lang.org/std/path/) - Path manipulation

---

**TEAM-293: Follow this guide when adding new features or fixing bugs**
