# TEAM-338: RULE ZERO FIX - Eliminated DaemonStatus Duplicates

**Status:** ✅ COMPLETE

## Problem

Found 3 identical structs with `is_running` and `is_installed` fields:
1. `DaemonStatus` in `daemon-lifecycle/src/status.rs` (original)
2. `QueenStatus` in `tauri_commands.rs` (duplicate)
3. `HiveStatus` in `tauri_commands.rs` (duplicate)

**RULE ZERO VIOLATION:** Creating `QueenStatus` and `HiveStatus` instead of using existing `DaemonStatus`.

## Solution

**Single Source of Truth:** Use `daemon_lifecycle::DaemonStatus` everywhere.

### Changes Made

#### 1. `daemon-lifecycle/src/status.rs`
Added serialization and optional Tauri support:

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "tauri", derive(specta::Type))]
pub struct DaemonStatus {
    pub is_running: bool,
    pub is_installed: bool,
}
```

**Key:** `cfg_attr` only enables `specta::Type` when `tauri` feature is active.

#### 2. `daemon-lifecycle/Cargo.toml`
Added optional Tauri feature:

```toml
[features]
tauri = ["dep:specta"]

[dependencies]
specta = { version = "2.0.0-rc", optional = true }
```

**Why optional?** Not all users of `daemon-lifecycle` need Tauri bindings.

#### 3. `rbee-keeper/Cargo.toml`
Enabled Tauri feature:

```toml
daemon-lifecycle = { path = "../99_shared_crates/daemon-lifecycle", features = ["tauri"] }
```

#### 4. `tauri_commands.rs`
**DELETED:**
- `QueenStatus` struct (7 lines)
- `HiveStatus` struct (7 lines)

**UPDATED:**
```rust
// BEFORE
pub async fn queen_status() -> Result<QueenStatus, String> {
    // ... construct QueenStatus from status fields
    Ok(QueenStatus {
        is_running: status.is_running,
        is_installed: status.is_installed,
    })
}

// AFTER
pub async fn queen_status() -> Result<daemon_lifecycle::DaemonStatus, String> {
    // ... return directly
    Ok(check_daemon_health(&health_url, "queen-rbee", &ssh_config).await)
}
```

Same pattern for `hive_status()`.

**TypeScript bindings test:**
```rust
// BEFORE
.typ::<HiveStatus>()

// AFTER
.typ::<daemon_lifecycle::DaemonStatus>()
```

## Results

### Code Reduction
- **Deleted:** 14 lines (2 duplicate structs)
- **Simplified:** 2 functions (no manual struct construction)
- **Single source of truth:** `daemon_lifecycle::DaemonStatus`

### TypeScript Bindings
Generated type (same for both queen and hive):

```typescript
export type DaemonStatus = { 
  /** Is the daemon currently running? */
  is_running: boolean;
  /** Is the daemon binary installed? */
  is_installed: boolean;
}

async queenStatus(): Promise<Result<DaemonStatus, string>>
async hiveStatus(alias: string): Promise<Result<DaemonStatus, string>>
```

### Compilation
✅ `cargo check -p rbee-keeper` passes
✅ `cargo test -p rbee-keeper export_typescript_bindings` passes
✅ TypeScript bindings generated correctly

## Architecture

```
daemon-lifecycle (shared crate)
  └─ DaemonStatus struct
       ├─ Always: Serialize, Deserialize
       └─ Optional (tauri feature): specta::Type

rbee-keeper (Tauri app)
  ├─ Cargo.toml: daemon-lifecycle with "tauri" feature
  ├─ queen_status() → returns DaemonStatus
  ├─ hive_status() → returns DaemonStatus
  └─ TypeScript bindings export DaemonStatus

Frontend (TypeScript)
  ├─ commands.queenStatus() → DaemonStatus
  └─ commands.hiveStatus() → DaemonStatus
```

## Benefits

1. **Zero duplication** - One struct, used everywhere
2. **Compiler-verified** - Change once, updates everywhere
3. **Type safety** - TypeScript gets exact same structure
4. **Optional dependency** - specta only when needed
5. **No manual construction** - Return status directly

## Pattern for Future

**When you need daemon status:**
1. ✅ Use `daemon_lifecycle::DaemonStatus`
2. ❌ Don't create `MyServiceStatus` struct
3. ✅ Return `check_daemon_health()` directly
4. ❌ Don't reconstruct fields manually

**RULE ZERO:** If a type exists, use it. Don't create duplicates.

## Files Modified

1. `/bin/99_shared_crates/daemon-lifecycle/src/status.rs` - Added serde + optional specta
2. `/bin/99_shared_crates/daemon-lifecycle/Cargo.toml` - Added tauri feature
3. `/bin/00_rbee_keeper/Cargo.toml` - Enabled tauri feature
4. `/bin/00_rbee_keeper/src/tauri_commands.rs` - Deleted duplicates, use DaemonStatus
5. `/bin/00_rbee_keeper/ui/src/generated/bindings.ts` - Regenerated (auto)

## Verification

```bash
# Verify compilation
cargo check -p rbee-keeper

# Regenerate TypeScript bindings
cargo test -p rbee-keeper export_typescript_bindings

# Check generated type
grep -A 5 "export type DaemonStatus" bin/00_rbee_keeper/ui/src/generated/bindings.ts
```

All passing ✅
