# TEAM-186 Changes to rbee-operations

## Summary

Updated hive operations to match the new install/uninstall workflow and ensured all operations have proper `hive_id` handling.

---

## Changes Made

### 1. **Renamed Hive Operations**
- ❌ `HiveCreate` → ✅ `HiveInstall`
- ❌ `HiveDelete` → ✅ `HiveUninstall`

**Rationale**: Matches user command `rbee hive install` / `rbee hive uninstall`

### 2. **Updated HiveInstall**
```rust
HiveInstall {
    hive_id: String,
    ssh_host: Option<String>,      // For remote SSH
    ssh_port: Option<u16>,          // For remote SSH
    ssh_user: Option<String>,       // For remote SSH
    #[serde(default = "default_port")]
    port: u16,                      // Defaults to 8600
}
```

**Supports**:
- Localhost: `{"operation":"hive_install","hive_id":"localhost"}`
- Remote SSH: `{"operation":"hive_install","hive_id":"hive-prod-01","ssh_host":"192.168.1.100","ssh_port":22,"ssh_user":"admin"}`

### 3. **Updated HiveUninstall**
```rust
HiveUninstall {
    hive_id: String,
    #[serde(default)]
    catalog_only: bool,  // For unreachable remote hives
}
```

**Supports**:
- Normal uninstall: `{"operation":"hive_uninstall","hive_id":"localhost"}`
- Catalog-only (remote unreachable): `{"operation":"hive_uninstall","hive_id":"hive-prod-01","catalog_only":true}`

### 4. **Updated HiveUpdate**
```rust
HiveUpdate {
    hive_id: String,
    ssh_host: Option<String>,           // Update SSH address
    ssh_port: Option<u16>,              // Update SSH port
    ssh_user: Option<String>,           // Update SSH user
    #[serde(default)]
    refresh_capabilities: bool,         // Refresh device capabilities
}
```

**Supports**:
- Update SSH: `{"operation":"hive_update","hive_id":"hive-prod-01","ssh_host":"192.168.1.101"}`
- Refresh capabilities: `{"operation":"hive_update","hive_id":"localhost","refresh_capabilities":true}`

### 5. **Added hive_id Defaults**
All hive operations now default `hive_id` to `"localhost"`:

```rust
HiveStart {
    #[serde(default = "default_hive_id")]  // Defaults to "localhost"
    hive_id: String,
}

HiveStop {
    #[serde(default = "default_hive_id")]  // Defaults to "localhost"
    hive_id: String,
}

HiveGet {
    #[serde(default = "default_hive_id")]  // Defaults to "localhost"
    hive_id: String,
}
```

**Why?** User can omit `hive_id` for localhost operations:
- ✅ `{"operation":"hive_start"}` → Defaults to localhost
- ✅ `{"operation":"hive_start","hive_id":"hive-prod-01"}` → Explicit hive

### 6. **Updated Constants**
```rust
pub mod constants {
    pub const OP_HIVE_INSTALL: &str = "hive_install";      // Was: hive_create
    pub const OP_HIVE_UNINSTALL: &str = "hive_uninstall";  // Was: hive_delete
    pub const OP_HIVE_UPDATE: &str = "hive_update";
    pub const OP_HIVE_START: &str = "hive_start";
    pub const OP_HIVE_STOP: &str = "hive_stop";
    pub const OP_HIVE_LIST: &str = "hive_list";
    pub const OP_HIVE_GET: &str = "hive_get";
    // ... worker/model/infer constants unchanged
}
```

---

## All Operations Now Have hive_id

### ✅ Hive Operations
- `HiveInstall` - Has `hive_id`
- `HiveUninstall` - Has `hive_id`
- `HiveUpdate` - Has `hive_id`
- `HiveStart` - Has `hive_id` (defaults to "localhost")
- `HiveStop` - Has `hive_id` (defaults to "localhost")
- `HiveGet` - Has `hive_id` (defaults to "localhost")
- `HiveList` - No `hive_id` (lists all hives)

### ✅ Worker Operations (Already had hive_id)
- `WorkerSpawn { hive_id, ... }`
- `WorkerList { hive_id }`
- `WorkerGet { hive_id, ... }`
- `WorkerDelete { hive_id, ... }`

### ✅ Model Operations (Already had hive_id)
- `ModelDownload { hive_id, ... }`
- `ModelList { hive_id }`
- `ModelGet { hive_id, ... }`
- `ModelDelete { hive_id, ... }`

### ✅ Inference Operation (Already had hive_id)
- `Infer { hive_id, ... }`

---

## Example Usage

### Localhost Install
```json
{
  "operation": "hive_install",
  "hive_id": "localhost",
  "port": 8600
}
```

### Remote SSH Install
```json
{
  "operation": "hive_install",
  "hive_id": "hive-prod-01",
  "ssh_host": "192.168.1.100",
  "ssh_port": 22,
  "ssh_user": "admin",
  "port": 8600
}
```

### Update SSH Address
```json
{
  "operation": "hive_update",
  "hive_id": "hive-prod-01",
  "ssh_host": "192.168.1.101"
}
```

### Refresh Capabilities
```json
{
  "operation": "hive_update",
  "hive_id": "localhost",
  "refresh_capabilities": true
}
```

### Uninstall (Normal)
```json
{
  "operation": "hive_uninstall",
  "hive_id": "localhost"
}
```

### Uninstall (Catalog Only - Remote Unreachable)
```json
{
  "operation": "hive_uninstall",
  "hive_id": "hive-prod-01",
  "catalog_only": true
}
```

### Start Hive (Defaults to localhost)
```json
{
  "operation": "hive_start"
}
```

### Start Hive (Explicit)
```json
{
  "operation": "hive_start",
  "hive_id": "hive-prod-01"
}
```

---

## Tests Added

1. ✅ `test_serialize_hive_install` - Localhost install
2. ✅ `test_serialize_hive_install_remote` - Remote SSH install
3. ✅ `test_serialize_hive_uninstall` - Uninstall with catalog_only
4. ✅ `test_serialize_hive_update` - Update with refresh_capabilities
5. ✅ `test_hive_start_defaults_to_localhost` - Default hive_id behavior

**All tests pass!** ✅

---

## Breaking Changes

### ⚠️ API Changes
- `HiveCreate` → `HiveInstall` (renamed)
- `HiveDelete` → `HiveUninstall` (renamed)
- `HiveInstall` now requires `hive_id` and optional SSH fields
- `HiveUpdate` now has SSH fields and `refresh_capabilities` flag
- `HiveStart`, `HiveStop`, `HiveGet` now have `hive_id` (defaults to "localhost")

### Migration Guide

**Old Code:**
```rust
Operation::HiveCreate { host, port }
Operation::HiveDelete { id }
```

**New Code:**
```rust
Operation::HiveInstall {
    hive_id: "localhost".to_string(),
    ssh_host: None,
    ssh_port: None,
    ssh_user: None,
    port: 8600,
}
Operation::HiveUninstall {
    hive_id: "localhost".to_string(),
    catalog_only: false,
}
```

---

## Summary

**All operations now properly handle `hive_id`:**
- Explicit `hive_id` for install/uninstall/update
- Default `hive_id = "localhost"` for start/stop/get
- Worker/Model/Infer operations already had `hive_id` ✅

**New workflow supported:**
- `rbee hive install` (localhost or SSH)
- `rbee hive uninstall` (with catalog-only option)
- `rbee hive update` (SSH address + refresh capabilities)

**Type safety maintained:**
- All operations are strongly typed
- Serde serialization/deserialization works correctly
- Tests verify all scenarios
