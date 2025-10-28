# TEAM-333: SSH List Command Implementation

**Status:** ✅ COMPLETE

**Mission:** Rename `hive_list` to `ssh_list` and implement SSH config parsing to return list of configured SSH hosts.

---

## Changes Made

### 1. Tauri Command (`src/tauri_commands.rs`)

**Created Types:**
- `SshTarget` - SSH target from `~/.ssh/config` with specta support
  - `host`: Host alias from SSH config
  - `host_subtitle`: Optional subtitle
  - `hostname`: IP or domain
  - `user`: SSH username
  - `port`: SSH port
  - `status`: Connection status (Online/Offline/Unknown)
- `SshTargetStatus` - Enum for connection status

**Renamed Command:**
- `hive_list()` → `ssh_list()`

**Implementation:**
```rust
#[tauri::command]
#[specta::specta]
pub async fn ssh_list() -> Result<Vec<SshTarget>, String>
```

**Behavior:**
1. Reads `~/.ssh/config` using `ssh_resolver::parse_ssh_config()`
2. Converts each SSH config entry to `SshTarget`
3. Always includes `localhost` as first entry (status: Online)
4. Returns list of all SSH targets

### 2. SSH Resolver (`src/ssh_resolver.rs`)

**Made Public:**
- `parse_ssh_config()` - Now public for use in tauri_commands

### 3. Tauri Main (`src/tauri_main.rs`)

**Updated Registration:**
- Changed `collect_commands![hive_list]` to `collect_commands![ssh_list]`

### 4. TypeScript Component (`ui/src/components/SshHivesContainer.tsx`)

**Updated API Call:**
- Changed `commands.hiveList()` to `commands.sshList()`

### 5. TypeScript Bindings (`ui/src/generated/bindings.ts`)

**Auto-Generated:**
- `commands.sshList()` - Returns `Result<SshTarget[], string>`
- `SshTarget` type with all fields
- `SshTargetStatus` type as union: `"online" | "offline" | "unknown"`

---

## Example SSH Config

```text
Host workstation
    HostName 192.168.1.100
    User vince
    Port 22

Host server
    HostName example.com
    User admin
    Port 2222
```

**Result:**
```json
[
  {
    "host": "localhost",
    "host_subtitle": "Local machine",
    "hostname": "127.0.0.1",
    "user": "vince",
    "port": 22,
    "status": "online"
  },
  {
    "host": "workstation",
    "host_subtitle": null,
    "hostname": "192.168.1.100",
    "user": "vince",
    "port": 22,
    "status": "unknown"
  },
  {
    "host": "server",
    "host_subtitle": null,
    "hostname": "example.com",
    "user": "admin",
    "port": 2222,
    "status": "unknown"
  }
]
```

---

## Compilation

✅ **All builds successful:**
```bash
cargo test --package rbee-keeper --lib tauri_commands::tests::export_typescript_bindings -- --exact
cargo build --bin rbee-keeper-gui
```

---

## Future Enhancements

**TODO (Optional):**
1. Extract `host_subtitle` from comments in SSH config
2. Implement actual status checking (ping/SSH connection test)
3. Add caching to avoid re-parsing SSH config on every call
4. Support for SSH config includes (`Include ~/.ssh/config.d/*`)

---

## Files Changed

- **MODIFIED:** `bin/00_rbee_keeper/src/tauri_commands.rs` (+60 LOC)
  - Added `SshTarget` and `SshTargetStatus` types
  - Renamed `hive_list()` to `ssh_list()`
  - Implemented SSH config parsing logic
  
- **MODIFIED:** `bin/00_rbee_keeper/src/ssh_resolver.rs` (+1 LOC)
  - Made `parse_ssh_config()` public
  
- **MODIFIED:** `bin/00_rbee_keeper/src/main.rs` (+1 LOC) ⭐ **CRITICAL**
  - Added `ssh_list` to `tauri::generate_handler![]` (actual entry point)
  
- **MODIFIED:** `bin/00_rbee_keeper/src/tauri_main.rs` (+3 LOC)
  - Updated command registration (specta builder)
  - Fixed unused import warning
  
- **MODIFIED:** `bin/00_rbee_keeper/ui/src/components/SshHivesContainer.tsx` (+60 LOC)
  - Added `SshHivesErrorBoundary` component
  - Replaced fallback with proper error state
  - Shows error message with "Try Again" button
  
- **GENERATED:** `bin/00_rbee_keeper/ui/src/generated/bindings.ts`
  - Auto-generated TypeScript bindings

---

## Testing

**Manual Test:**
1. Create/edit `~/.ssh/config` with some hosts
2. Run `cargo build --bin rbee-keeper-gui`
3. Launch GUI and navigate to Services page
4. Verify SSH hosts list appears with localhost + SSH config entries

**Expected Behavior:**
- Localhost always appears first (status: Online)
- All SSH config hosts appear with status: Unknown
- No errors if `~/.ssh/config` doesn't exist (just shows localhost)

---

**TEAM-333 | Oct 28, 2025**
