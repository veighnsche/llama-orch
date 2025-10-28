# TEAM-338: Binary Installation Utility

**Date:** Oct 28, 2025  
**Status:** ✅ COMPLETE

## Summary

Extracted `check_binary_installed()` from `status.rs` into reusable utility at `utils/binary.rs`. Now used by `status.rs`, `install.rs`, and `uninstall.rs`.

## Changes

### New File: `/bin/99_shared_crates/daemon-lifecycle/src/utils/binary.rs`

**Purpose:** Reusable binary installation checks across all operations

**Function:**
```rust
pub async fn check_binary_installed(
    daemon_name: &str,
    ssh_config: &SshConfig
) -> bool
```

**Implementation:**
- **Localhost:** Direct filesystem check (`~/.local/bin/{daemon_name}`)
- **Remote:** SSH command (`test -f ~/.local/bin/{daemon_name} && echo 'EXISTS'`)

**Returns:**
- `true` if binary exists
- `false` if binary doesn't exist or check fails

### Updated Files

#### 1. `/bin/99_shared_crates/daemon-lifecycle/src/utils/mod.rs`

**Added:**
```rust
pub mod binary; // TEAM-338: Binary installation checks
pub use binary::check_binary_installed; // Re-export
```

#### 2. `/bin/99_shared_crates/daemon-lifecycle/src/status.rs`

**Changed:**
- Removed duplicate `check_binary_installed()` function
- Added import: `use crate::utils::binary::check_binary_installed;`
- Function now calls utility instead of inline implementation

#### 3. `/bin/99_shared_crates/daemon-lifecycle/src/install.rs`

**Added:**
- Import: `use crate::utils::binary::check_binary_installed;`
- **New check at start of `install_daemon()`:**

```rust
// Step 0: Check if already installed
if check_binary_installed(daemon_name, ssh_config).await {
    n!("already_installed", "⚠️  {} is already installed on {}@{}", 
        daemon_name, ssh_config.user, ssh_config.hostname);
    anyhow::bail!("{} is already installed. Use rebuild to update.", daemon_name);
}
```

**Behavior:**
- Install now fails fast if binary already exists
- Prevents accidental overwrites
- Directs user to use `rebuild` command instead

#### 4. `/bin/99_shared_crates/daemon-lifecycle/src/uninstall.rs`

**No changes needed** - Already uses `check_daemon_health()` which internally calls `check_binary_installed()`

## Usage Examples

### Status Check
```rust
use daemon_lifecycle::{check_daemon_health, SshConfig};

let ssh = SshConfig::localhost();
let status = check_daemon_health(
    "http://localhost:7833/health",
    "queen-rbee",
    &ssh
).await;

// Internally calls check_binary_installed() if not running
```

### Install Check
```rust
use daemon_lifecycle::{install_daemon, InstallConfig, SshConfig};

let config = InstallConfig {
    daemon_name: "queen-rbee".to_string(),
    ssh_config: SshConfig::localhost(),
    local_binary_path: None,
    job_id: None,
};

// Will fail if already installed
install_daemon(config).await?;
```

### Direct Check
```rust
use daemon_lifecycle::utils::check_binary_installed;
use daemon_lifecycle::SshConfig;

let ssh = SshConfig::localhost();
let is_installed = check_binary_installed("queen-rbee", &ssh).await;

if is_installed {
    println!("Binary exists in ~/.local/bin/");
}
```

## Benefits

### 1. Single Source of Truth
- ❌ **Before:** Logic duplicated in `status.rs` and potentially elsewhere
- ✅ **After:** One function in `utils/binary.rs`

### 2. Reusability
**Used by:**
- `status.rs` - Check installation when daemon not running
- `install.rs` - Prevent overwriting existing binary
- `uninstall.rs` - Indirectly via `check_daemon_health()`

**Future use cases:**
- `rebuild.rs` - Verify binary exists before rebuilding
- `upgrade.rs` - Check current installation before upgrading
- Any operation that needs to know if binary exists

### 3. Consistent Behavior
All operations use the same logic:
- Same path: `~/.local/bin/{daemon_name}`
- Same SSH command: `test -f ~/.local/bin/{daemon_name} && echo 'EXISTS'`
- Same error handling: Returns `false` on any error

### 4. Easier Testing
```rust
#[tokio::test]
async fn test_check_binary_installed_localhost() {
    let ssh = SshConfig::localhost();
    
    // Create temp binary
    let home = env::var("HOME").unwrap();
    let bin_dir = PathBuf::from(home).join(".local/bin");
    fs::create_dir_all(&bin_dir).unwrap();
    fs::write(bin_dir.join("test-daemon"), "").unwrap();
    
    // Check
    let is_installed = check_binary_installed("test-daemon", &ssh).await;
    assert!(is_installed);
    
    // Cleanup
    fs::remove_file(bin_dir.join("test-daemon")).unwrap();
}
```

## Install Behavior Change

### Before
```bash
$ rbee-keeper queen install
# Overwrites existing binary without warning
✅ Queen installed successfully
```

### After
```bash
$ rbee-keeper queen install
# Checks first
⚠️  queen-rbee is already installed on vince@localhost
❌ Error: queen-rbee is already installed. Use rebuild to update.

$ rbee-keeper queen rebuild
# Correct command for updating
✅ Queen rebuilt successfully
```

**Why this is better:**
- Prevents accidental overwrites
- Makes user intent explicit (install vs update)
- Follows principle of least surprise
- Matches behavior of package managers (apt, brew, etc.)

## Implementation Details

### Localhost Check
```rust
if ssh_config.is_localhost() {
    let home = std::env::var("HOME")?;
    let binary_path = PathBuf::from(home)
        .join(".local/bin")
        .join(daemon_name);
    binary_path.exists()
}
```

**Advantages:**
- No SSH overhead
- Instant response
- Works offline

### Remote Check
```rust
else {
    let check_cmd = format!("test -f ~/.local/bin/{} && echo 'EXISTS'", daemon_name);
    let output = ssh_exec(ssh_config, &check_cmd).await?;
    output.trim().contains("EXISTS")
}
```

**Why `test -f` + `echo`:**
- `test -f` returns exit code (0 = exists, 1 = doesn't exist)
- `echo 'EXISTS'` provides string output to check
- More reliable than checking exit codes over SSH
- Works with any shell (bash, sh, zsh, etc.)

### Error Handling
```rust
match ssh_exec(ssh_config, &check_cmd).await {
    Ok(output) => output.trim().contains("EXISTS"),
    Err(_) => false,  // Any error = assume not installed
}
```

**Philosophy:** Fail safe
- SSH connection error → `false`
- Permission denied → `false`
- Command not found → `false`
- Better to say "not installed" than crash

## File Structure

```
daemon-lifecycle/
├── src/
│   ├── status.rs          (uses check_binary_installed)
│   ├── install.rs         (uses check_binary_installed)
│   ├── uninstall.rs       (uses check_daemon_health → check_binary_installed)
│   └── utils/
│       ├── mod.rs         (exports check_binary_installed)
│       ├── binary.rs      (NEW - reusable utility)
│       ├── ssh.rs
│       ├── poll.rs
│       └── local.rs
```

## Testing

### Compilation
```bash
cargo check --package daemon-lifecycle
```
**Result:** ✅ PASS

### Manual Test - Install Check
```bash
# Install queen
rbee-keeper queen install
# ✅ Should succeed

# Try to install again
rbee-keeper queen install
# ❌ Should fail with "already installed" message

# Rebuild instead
rbee-keeper queen rebuild
# ✅ Should succeed
```

### Manual Test - Status Check
```bash
# Check status when not running
rbee-keeper queen status
# Should show: ❌ not running
# Should check installation (uses check_binary_installed)

# Start queen
rbee-keeper queen start

# Check status when running
rbee-keeper queen status
# Should show: ✅ running
# Should skip installation check (optimization)
```

## Future Enhancements

Consider adding:

1. **Version check**
   ```rust
   pub async fn get_binary_version(
       daemon_name: &str,
       ssh_config: &SshConfig
   ) -> Option<String>
   ```

2. **Binary hash check**
   ```rust
   pub async fn get_binary_hash(
       daemon_name: &str,
       ssh_config: &SshConfig
   ) -> Option<String>
   ```

3. **Installation date**
   ```rust
   pub async fn get_install_date(
       daemon_name: &str,
       ssh_config: &SshConfig
   ) -> Option<SystemTime>
   ```

All would follow the same localhost/remote pattern.

---

**Pattern:** Extract reusable utilities to `utils/` folder. Single source of truth. Consistent behavior across all operations.
