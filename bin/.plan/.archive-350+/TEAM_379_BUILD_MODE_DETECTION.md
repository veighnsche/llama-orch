# TEAM-379: Build Mode Detection for UI

**Status:** ✅ COMPLETE

**Mission:** Display whether installed binaries are debug or release builds in the Queen and Hive cards.

## Problem

Users couldn't tell if they had debug or release builds installed. This is important because:
- Debug builds are slower (15-20s startup vs 2-3s for release)
- Release builds are production-ready
- Users need to know which version they're running

## Solution

### Backend Changes

1. **Added `build_mode` field to `DaemonStatus`** (`lifecycle-shared/src/status.rs`)
   - Type: `Option<String>` - "debug", "release", or `None`
   - Added to struct and all constructor methods

2. **Local build mode detection** (`lifecycle-local/src/status.rs`)
   - Uses `get_binary_mode()` to execute `~/.local/bin/{daemon} --build-info`
   - Parses output to determine build mode
   - Only checks if binary is installed

3. **Remote build mode detection** (`lifecycle-ssh/src/utils/binary.rs`)
   - New function: `get_remote_binary_mode()` - executes `--build-info` via SSH
   - New function: `is_remote_release_binary()` - helper for release detection
   - Integrated into `check_daemon_health()` in `lifecycle-ssh/src/status.rs`

4. **Feature propagation** (Cargo.toml files)
   - `lifecycle-local/tauri` → `lifecycle-shared/tauri`
   - `lifecycle-ssh/tauri` → `lifecycle-shared/tauri`
   - Ensures `DaemonStatus` derives `specta::Type` for TypeScript bindings

### Frontend Changes

1. **Updated TypeScript types** (`ui/src/store/queenQueries.ts`, `hiveQueries.ts`)
   - Added `buildMode: string | null` to `QueenStatus` and `SshHive` interfaces
   - Mapped `build_mode` from backend to `buildMode` in frontend

2. **UI Display** (`QueenCard.tsx`, `HiveCard.tsx`)
   - Shows build mode below description if available
   - Format: `Build: debug` or `Build: release`
   - Only displayed if `buildMode` is not null

## Implementation Details

### Shadow-rs Integration

All binaries already support `--build-info` flag via shadow-rs:

```rust
// In main.rs
use shadow_rs::shadow;
shadow!(build);

if args.build_info {
    println!("{}", build::BUILD_RUST_CHANNEL); // "debug" or "release"
    std::process::exit(0);
}
```

### Local Detection Flow

```rust
// lifecycle-local/src/status.rs
let build_mode = if is_installed {
    if let Ok(home) = std::env::var("HOME") {
        let installed_path = PathBuf::from(&home)
            .join(BINARY_INSTALL_DIR)
            .join(daemon_name);
        
        get_binary_mode(&installed_path).ok() // Execute --build-info
    } else {
        None
    }
} else {
    None
};
```

### Remote Detection Flow

```rust
// lifecycle-ssh/src/utils/binary.rs
pub async fn get_remote_binary_mode(daemon_name: &str, ssh_config: &SshConfig) -> Result<String> {
    let check_cmd = format!("~/.local/bin/{} --build-info", daemon_name);
    let output = ssh_exec(ssh_config, &check_cmd).await?;
    let mode = output.trim().to_string();
    
    if mode != "debug" && mode != "release" {
        bail!("Invalid build mode: {}", mode);
    }
    
    Ok(mode)
}
```

## Files Changed

### Backend (Rust)
- `bin/96_lifecycle/lifecycle-shared/src/status.rs` - Added `build_mode` field
- `bin/96_lifecycle/lifecycle-local/src/status.rs` - Local detection
- `bin/96_lifecycle/lifecycle-local/src/utils/mod.rs` - Export `get_binary_mode`
- `bin/96_lifecycle/lifecycle-ssh/src/utils/binary.rs` - Remote detection functions
- `bin/96_lifecycle/lifecycle-ssh/src/status.rs` - Integrated remote detection
- `bin/96_lifecycle/lifecycle-local/Cargo.toml` - Feature propagation
- `bin/96_lifecycle/lifecycle-ssh/Cargo.toml` - Feature propagation
- `bin/00_rbee_keeper/src/tauri_commands.rs` - Fixed `hive_status` conversion

### Frontend (TypeScript/React)
- `ui/src/store/queenQueries.ts` - Added `buildMode` field
- `ui/src/store/hiveQueries.ts` - Added `buildMode` field
- `ui/src/components/cards/QueenCard.tsx` - Display build mode
- `ui/src/components/cards/HiveCard.tsx` - Display build mode
- `ui/src/generated/bindings.ts` - Auto-generated TypeScript types

## Testing

### Manual Testing

1. **Install debug build:**
   ```bash
   cargo build --bin queen-rbee
   cargo install --path bin/10_queen_rbee --debug
   ```
   - UI should show: `Build: debug`

2. **Install release build:**
   ```bash
   cargo build --bin queen-rbee --release
   cargo install --path bin/10_queen_rbee
   ```
   - UI should show: `Build: release`

3. **Remote hive:**
   - SSH to remote machine
   - Install rbee-hive (debug or release)
   - UI should show build mode via SSH detection

### Verification

```bash
# Regenerate TypeScript bindings
cargo test --package rbee-keeper export_typescript_bindings

# Check compilation
cargo check --package rbee-keeper
```

## Benefits

- ✅ Users can see if they're running debug or release builds
- ✅ Helps diagnose slow startup times (debug builds take 15-20s)
- ✅ Works for both local and remote (SSH) installations
- ✅ Automatic detection - no manual configuration needed
- ✅ Type-safe - TypeScript bindings auto-generated

## Architecture

**Data Flow:**
1. Binary installed with `--build-info` flag support (shadow-rs)
2. Backend executes `{daemon} --build-info` (local or via SSH)
3. Parses output ("debug" or "release")
4. Stores in `DaemonStatus.build_mode`
5. Tauri command returns to frontend
6. React Query caches result
7. UI displays in card

**Why This Works:**
- All binaries already support `--build-info` (shadow-rs)
- No database or config changes needed
- Works for any daemon (queen, hive, worker)
- Automatic - detects on every status check

## Future Enhancements

Potential improvements (not implemented):
- Show build timestamp
- Show git commit hash
- Show Rust version used for build
- Color-code debug (yellow) vs release (green)

All of these are available via shadow-rs but not currently displayed.

---

**TEAM-379 Complete** - Build mode detection working for both local and remote installations.
