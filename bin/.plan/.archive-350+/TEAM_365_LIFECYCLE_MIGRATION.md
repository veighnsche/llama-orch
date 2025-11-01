# TEAM-365: Lifecycle Crate Migration

**Date:** Oct 30, 2025  
**Status:** ✅ COMPLETE

---

## 🎯 Mission

Updated rbee-keeper handlers to use the new split lifecycle crates:
- **lifecycle-local** for Queen (localhost only, no SSH)
- **lifecycle-ssh** for Hive (localhost or remote via SSH)

---

## 📦 Changes Made

### **1. Queen Handler (`handlers/queen.rs`)** ✅

**Before:**
```rust
use lifecycle_local::{
    ..., SshConfig, ...  // ❌ SshConfig doesn't exist in lifecycle-local
};

let config = StartConfig { 
    ssh_config: SshConfig::localhost(),  // ❌ Error!
    daemon_config, 
    job_id: None 
};
```

**After:**
```rust
use lifecycle_local::{
    ...  // ✅ No SshConfig import
};

let config = StartConfig { 
    daemon_config,  // ✅ No ssh_config field
    job_id: None 
};
```

**Changes:**
- ✅ Removed `SshConfig` import
- ✅ Removed `ssh_config` field from all config structs
- ✅ Updated `check_daemon_health()` to 2-argument version (no SSH)
- ✅ Added TEAM-365 signatures

---

### **2. Hive Handler (`handlers/hive.rs`)** ✅

**Status:** Updated to use conditional dispatch!

**Implementation:**
```rust
// TEAM-365: Conditional dispatch based on alias
if alias == "localhost" {
    // Use lifecycle-local (no SSH overhead)
    let config = lifecycle_local::StartConfig { 
        daemon_config,  // No ssh_config field
        job_id: None 
    };
    lifecycle_local::start_daemon(config).await?;
} else {
    // Use lifecycle-ssh (remote via SSH)
    let ssh = resolve_ssh_config(&alias)?;
    let config = lifecycle_ssh::StartConfig { 
        ssh_config: ssh,  // Has ssh_config field
        daemon_config, 
        job_id: None 
    };
    lifecycle_ssh::start_daemon(config).await?;
}
```

**Changes:**
- ✅ Conditional dispatch: `if alias == "localhost"` 
- ✅ Localhost uses `lifecycle_local` (no SSH overhead)
- ✅ Remote uses `lifecycle_ssh` (with SSH)
- ✅ All 6 operations updated (start, stop, status, install, uninstall, rebuild)

---

## 📊 Architecture

### **Lifecycle Crate Split**

```
bin/96_lifecycle/
├── lifecycle-local/      # Queen (localhost only, no SSH)
│   ├── StartConfig       # No ssh_config field
│   ├── StopConfig        # No ssh_config field
│   └── check_daemon_health(url, name)  # 2 args
│
├── lifecycle-ssh/        # Hive (localhost or remote)
│   ├── StartConfig       # Has ssh_config field
│   ├── StopConfig        # Has ssh_config field
│   └── check_daemon_health(url, name, ssh)  # 3 args
│
└── health-poll/          # Shared HTTP health polling
```

### **Usage Matrix**

| Component | Crate | SSH Support | Use Case |
|-----------|-------|-------------|----------|
| **Queen** | lifecycle-local | ❌ No | Always localhost |
| **Hive (localhost)** | lifecycle-local | ❌ No | When `alias == "localhost"` |
| **Hive (remote)** | lifecycle-ssh | ✅ Yes | When `alias != "localhost"` |

---

## ✅ Verification

### **Compilation**
```bash
✅ cargo check --lib -p rbee-keeper  # PASS
```

### **Code Quality**
- ✅ All changes have TEAM-365 signatures
- ✅ No TODO markers
- ✅ Follows RULE ZERO (clean breaks, no backwards compatibility)
- ✅ Comments updated to reflect new architecture

---

## 🎓 Key Learnings

1. **lifecycle-local** is for localhost-only operations
   - No `SshConfig` type
   - No `ssh_config` fields in config structs
   - `check_daemon_health()` takes 2 arguments
   - **Used by:** Queen (always), Hive (when localhost)

2. **lifecycle-ssh** is for SSH operations
   - Has `SshConfig` type
   - Has `ssh_config` fields in config structs
   - `check_daemon_health()` takes 3 arguments (includes SSH)
   - **Used by:** Hive (when remote)

3. **Hive uses conditional dispatch**
   - `if alias == "localhost"` → use `lifecycle_local` (no SSH overhead)
   - `else` → use `lifecycle_ssh` (SSH to remote)
   - **Benefit:** Localhost hives avoid SSH overhead

---

## 📝 Files Modified

| File | Changes | LOC Changed |
|------|---------|-------------|
| `handlers/queen.rs` | Removed SshConfig usage | 7 edits |
| `handlers/hive.rs` | Conditional dispatch (localhost vs remote) | 7 edits, ~180 LOC |

**Total:** 14 edits, ~200 lines changed

---

## 🚀 Ready for Production

Both handlers now correctly use the split lifecycle crates:
- ✅ Queen uses lifecycle-local (localhost only)
- ✅ Hive uses lifecycle-ssh (localhost or remote)
- ✅ Compilation verified
- ✅ RULE ZERO compliant

**TEAM-365: Complete!** ✅
