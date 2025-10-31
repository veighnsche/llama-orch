# TEAM-358: Localhost Bypass Removal

**Date:** Oct 30, 2025  
**Status:** ✅ COMPLETE (lifecycle-ssh fixed, rbee-keeper needs updates)

---

## 🎯 Mission

Remove all localhost bypass logic from `lifecycle-ssh` to ensure clear separation:
- **lifecycle-local** = LOCAL operations only (no SSH, ever)
- **lifecycle-ssh** = REMOTE operations only (always SSH, even for localhost)

---

## ✅ Changes Made to lifecycle-ssh

### **1. Removed localhost methods from SshConfig**
**File:** `lib.rs`

**Deleted:**
```rust
pub fn localhost() -> Self { ... }
pub fn is_localhost(&self) -> bool { ... }
```

**Why:** lifecycle-ssh should ALWAYS use SSH, even when hostname is "localhost"

### **2. Removed localhost bypass from ssh_exec()**
**File:** `utils/ssh.rs`

**Before:**
```rust
pub async fn ssh_exec(ssh_config: &SshConfig, command: &str) -> Result<String> {
    if ssh_config.is_localhost() {
        return local_exec(command).await;  // ← WRONG!
    }
    // ... SSH code
}
```

**After:**
```rust
pub async fn ssh_exec(ssh_config: &SshConfig, command: &str) -> Result<String> {
    // TEAM-358: Always use SSH, even for localhost
    // ... SSH code only
}
```

### **3. Removed localhost bypass from scp_upload()**
**File:** `utils/ssh.rs`

**Before:**
```rust
pub async fn scp_upload(...) -> Result<()> {
    if ssh_config.is_localhost() {
        return local_copy(local_path, remote_path).await;  // ← WRONG!
    }
    // ... SCP code
}
```

**After:**
```rust
pub async fn scp_upload(...) -> Result<()> {
    // TEAM-358: Always use SCP, even for localhost
    // ... SCP code only
}
```

### **4. Removed localhost bypass from check_binary_installed()**
**File:** `utils/binary.rs`

**Before:**
```rust
let is_installed = if ssh_config.is_localhost() {
    // Direct filesystem check
    binary_path.exists()
} else {
    // SSH check
    ssh_exec(...)
};
```

**After:**
```rust
// TEAM-358: Always use SSH, even for localhost
let check_cmd = format!("test -f ~/.local/bin/{} && echo 'EXISTS'", daemon_name);
let is_installed = match ssh_exec(ssh_config, &check_cmd).await { ... };
```

### **5. Updated utils/local.rs documentation**
**File:** `utils/local.rs`

Added **⚠️ WARNING** documentation:
```rust
//! # ⚠️ WARNING - DO NOT USE FOR LOCALHOST BYPASS
//! 
//! TEAM-358: This module exists for historical reasons but should NOT be used
//! to bypass SSH for localhost operations.
//!
//! **If you want local operations, use the `lifecycle-local` crate instead!**
```

### **6. Removed local_exec/local_copy from exports**
**File:** `utils/mod.rs`

**Before:**
```rust
pub use local::{local_copy, local_exec};
```

**After:**
```rust
// TEAM-358: local_copy and local_exec are NOT exported
// Use lifecycle-local for local operations
```

---

## ✅ Compilation Status

```bash
$ cargo check --package lifecycle-ssh
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.89s
```

**Result:** ✅ SUCCESS - lifecycle-ssh compiles with 0 errors (13 warnings)

---

## 🔴 Breaking Changes for Consumers

### **rbee-keeper needs updates:**

**Errors found:**
1. `unresolved import lifecycle_local::SshConfig` - SshConfig doesn't exist in lifecycle-local
2. `struct StartConfig has no field named ssh_config` - Field removed
3. `struct StopConfig has no field named ssh_config` - Field removed
4. `struct RebuildConfig has no field named ssh_config` - Field removed
5. `struct InstallConfig has no field named ssh_config` - Field removed
6. `struct UninstallConfig has no field named ssh_config` - Field removed
7. `no function or associated item named localhost` - Method removed from lifecycle-ssh
8. `check_daemon_health takes 2 arguments but 3 supplied` - Signature changed

### **Fix Required:**

**File:** `bin/00_rbee_keeper/src/handlers/queen.rs`

**Changes needed:**
1. Remove `use lifecycle_local::SshConfig` imports
2. Remove all `ssh_config: ...` fields from config struct initializations
3. Change `SshConfig::localhost()` to... nothing (use lifecycle-local instead)
4. Fix `check_daemon_health()` calls (remove 3rd argument)

**Decision needed:**
- If rbee-keeper is managing LOCAL daemons → use `lifecycle-local` crate
- If rbee-keeper is managing REMOTE daemons → use `lifecycle-ssh` crate with real SSH config

---

## 📋 Architecture Clarification

### **Before (Confusing):**
```
lifecycle-ssh
├── Has localhost bypass logic
├── Sometimes uses SSH, sometimes local
└── Confusing: "SSH crate that doesn't use SSH?"

lifecycle-local
├── Has SSH code
└── Confusing: "Local crate that uses SSH?"
```

### **After (Clear):**
```
lifecycle-local
├── LOCAL operations ONLY
├── No SSH, no SshConfig
└── Use for: localhost, same machine

lifecycle-ssh
├── REMOTE operations ONLY
├── Always SSH, even for localhost
└── Use for: remote machines via SSH
```

---

## 🎯 Key Principles

1. **lifecycle-local = LOCAL only**
   - No SSH code
   - No SshConfig
   - Uses: `local_exec()`, `local_copy()`, `std::fs`

2. **lifecycle-ssh = REMOTE only**
   - Always uses SSH
   - Even when hostname is "localhost"
   - Uses: `ssh_exec()`, `scp_upload()`

3. **No localhost bypass**
   - If you want local operations, use lifecycle-local
   - If you want SSH operations, use lifecycle-ssh (even for localhost)
   - Clear separation = less confusion

---

## 📝 Next Steps

### **For rbee-keeper maintainers:**

1. **Decide:** Is rbee-keeper managing local or remote daemons?
   
2. **If LOCAL:**
   ```rust
   // Use lifecycle-local
   use lifecycle_local::{StartConfig, start_daemon};
   
   let config = StartConfig {
       daemon_config: HttpDaemonConfig::new(...),
       job_id: Some(job_id),
   };
   start_daemon(config).await?;
   ```

3. **If REMOTE:**
   ```rust
   // Use lifecycle-ssh
   use lifecycle_ssh::{SshConfig, StartConfig, start_daemon};
   
   let ssh = SshConfig::new("remote-host".to_string(), "user".to_string(), 22);
   let config = StartConfig {
       ssh_config: ssh,
       daemon_config: HttpDaemonConfig::new(...),
       job_id: Some(job_id),
   };
   start_daemon(config).await?;
   ```

4. **Update all config struct initializations** to remove `ssh_config` field when using lifecycle-local

5. **Fix `check_daemon_health()` calls** - only 2 arguments now (no ssh_config)

---

## ✅ Summary

**TEAM-358 successfully removed all localhost bypass logic from lifecycle-ssh:**

- ✅ Removed `SshConfig::localhost()` and `is_localhost()` methods
- ✅ Removed localhost bypass from `ssh_exec()`
- ✅ Removed localhost bypass from `scp_upload()`
- ✅ Removed localhost bypass from `check_binary_installed()`
- ✅ Updated documentation with warnings
- ✅ Removed local function exports
- ✅ lifecycle-ssh compiles successfully

**Breaking changes:**
- ⚠️ rbee-keeper needs updates (10 compilation errors)
- ⚠️ Any other consumers using `SshConfig::localhost()` need updates

**Architecture is now clear:**
- lifecycle-local = LOCAL only (no SSH)
- lifecycle-ssh = REMOTE only (always SSH)

---

**TEAM-358 signing off! 🚀**
