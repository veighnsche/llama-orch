# TEAM-378: DaemonStatus RULE ZERO Fix

**Date:** Nov 1, 2025  
**Status:** ✅ COMPLETE

---

## Problem

Remote hives showed "workstation" in iframe URLs instead of actual IP addresses, causing DNS resolution errors:
```
http://workstation:7835 → Error: Name or service not known
```

**Root Cause:** `DaemonStatus` only returned daemon state (is_running, is_installed, build_mode) but **not SSH config** (hostname, user, port). Frontend had to merge two separate data sources, which was fragile.

---

## Solution: RULE ZERO

**Updated existing `DaemonStatus` struct** instead of creating new types/commands.

### Before (Incomplete)
```rust
pub struct DaemonStatus {
    pub is_running: bool,
    pub is_installed: bool,
    pub build_mode: Option<String>,
}
```

### After (Complete)
```rust
pub struct DaemonStatus {
    pub is_running: bool,
    pub is_installed: bool,
    pub build_mode: Option<String>,
    // TEAM-378: Added SSH config fields
    pub hostname: String,  // Actual IP/domain for iframe URL
    pub user: String,      // SSH username
    pub port: u16,         // SSH port
}
```

---

## Files Changed

### 1. Core Type Definition
**`bin/96_lifecycle/lifecycle-shared/src/status.rs`**
- Added `hostname`, `user`, `port` fields to `DaemonStatus`
- Updated all helper methods: `new()`, `running()`, `stopped_installed()`, `not_installed()`

### 2. Localhost Implementation
**`bin/96_lifecycle/lifecycle-local/src/status.rs`**
```rust
DaemonStatus {
    is_running,
    is_installed,
    build_mode,
    hostname: "localhost".to_string(),
    user: whoami::username(),
    port: 22,
}
```

### 3. Remote Implementation
**`bin/96_lifecycle/lifecycle-ssh/src/status.rs`**
```rust
DaemonStatus {
    is_running,
    is_installed,
    build_mode,
    hostname: ssh_config.hostname.clone(),  // Actual IP from SSH config!
    user: ssh_config.user.clone(),
    port: ssh_config.port,
}
```

### 4. Tauri Command (Simplified)
**`bin/00_rbee_keeper/src/tauri_commands.rs`**

**Before:**
```rust
let status = if alias == "localhost" {
    let local_status = lifecycle_local::check_daemon_health(...).await;
    // Manual conversion - duplicated fields
    lifecycle_ssh::DaemonStatus {
        is_running: local_status.is_running,
        is_installed: local_status.is_installed,
        build_mode: local_status.build_mode,
    }
} else {
    lifecycle_ssh::check_daemon_health(...).await
};
```

**After:**
```rust
// TEAM-378: Both return complete DaemonStatus - no conversion needed!
let status = if alias == "localhost" {
    lifecycle_local::check_daemon_health(...).await
} else {
    lifecycle_ssh::check_daemon_health(...).await
};
```

### 5. Frontend (Simplified)
**`bin/00_rbee_keeper/ui/src/store/hiveQueries.ts`**

**Before:**
```typescript
// Had to merge SSH list with daemon status
const sshConfig = sshHives?.find(h => h.host === hiveId);
return {
  hostname: sshConfig?.hostname || hiveId,  // Fallback to alias - BUG!
  user: sshConfig?.user || 'unknown',
  port: sshConfig?.port || 22,
};
```

**After:**
```typescript
// Backend provides everything!
const { hostname, user, port } = result.data;
return {
  hostname,  // Actual IP from backend
  user,      // Actual user from backend
  port,      // Actual port from backend
};
```

---

## Data Flow

### Before (Fragile)
```
Frontend
  ├─> Call hive_status("workstation")
  │   └─> Returns: { is_running, is_installed, build_mode }
  │
  ├─> Call ssh_list()
  │   └─> Returns: [{ host: "workstation", hostname: "192.168.1.100", ... }]
  │
  └─> Merge in frontend (FRAGILE!)
      └─> If ssh_list not loaded → fallback to alias → DNS ERROR!
```

### After (Robust)
```
Frontend
  └─> Call hive_status("workstation")
      └─> Backend resolves SSH config internally
          └─> Returns: {
                is_running,
                is_installed,
                build_mode,
                hostname: "192.168.1.100",  ✅ Actual IP!
                user: "vince",
                port: 22
              }
```

---

## Why This Follows RULE ZERO

### ❌ What We DIDN'T Do (Entropy)
- Create `HiveInfo` struct (duplicate of DaemonStatus)
- Create `hive_info()` command (duplicate of hive_status)
- Create `DaemonStatusWithSsh` wrapper type
- Keep old `DaemonStatus` and add new fields to separate struct

### ✅ What We DID Do (Breaking Change)
- **Updated existing `DaemonStatus` struct** with new fields
- **Let compiler find all call sites** (7 compilation errors)
- **Fixed each call site** to provide the new fields
- **One source of truth** - no duplicates, no wrappers

---

## Breaking Change Impact

**Compilation errors:** 7 files needed updates
- `lifecycle-shared/src/status.rs` - Helper methods (4 errors)
- `lifecycle-local/src/status.rs` - Localhost implementation (1 error)
- `lifecycle-ssh/src/status.rs` - Remote implementation (1 error)
- `rbee-keeper/src/tauri_commands.rs` - Tauri command (1 error)

**Time to fix:** ~15 minutes (compiler guided us to every location)

**Result:** All code updated, TypeScript bindings regenerated, frontend works!

---

## Benefits

1. **Single source of truth** - DaemonStatus has everything
2. **No merge logic** - Backend handles SSH config resolution
3. **Type safety** - Compiler enforces all fields are provided
4. **Simpler frontend** - No need to coordinate multiple queries
5. **Robust** - Works even if SSH list isn't loaded yet

---

## Testing

### Localhost
```bash
# Start keeper
cd bin/00_rbee_keeper
cargo run

# Navigate to localhost hive
# Expected: iframe loads http://localhost:7835
```

### Remote Hive
```bash
# SSH config: Host workstation → HostName 192.168.1.100
# Navigate to workstation hive
# Expected: iframe loads http://192.168.1.100:7835 ✅
```

---

## Key Insight

**RULE ZERO is about preventing entropy, not avoiding breaking changes.**

Pre-1.0 software is **allowed to break**. The compiler will catch all call sites in seconds. Fixing them is **temporary pain**.

Creating wrapper types, duplicate structs, or new commands is **permanent entropy**. Every future developer pays the cost. Forever.

**We chose temporary pain (7 compilation errors) over permanent entropy (duplicate types).**

---

## Summary

**Before:** `http://workstation:7835` → DNS error  
**After:** `http://192.168.1.100:7835` → Success! ✅

**Method:** RULE ZERO - Updated existing struct, let compiler guide us, fixed all call sites.

**Result:** Cleaner architecture, fewer lines of code, more robust behavior.
