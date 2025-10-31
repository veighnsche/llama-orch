# Dev/Prod Install Feature - Complete ✅

**Date:** 2025-10-31  
**Status:** COMPLETE

## Summary

Added ability to install Queen and Hive in either **Development** or **Production** mode:
- **Dev mode** (default): Debug build with full logging
- **Prod mode**: Optimized release build

## UI Changes

### Simple Dropdown Pattern
- Main button: "Install" (dev mode)
- Dropdown menu item: "Install (Production)"
- No extra selects - clean and simple!

### Affected Components

**1. InstallHiveCard**
- Main button installs dev version
- Dropdown has "Install (Production)" option
- Both call `install(targetId, buildMode)`

**2. QueenCard**  
- Uses shared `ServiceActionButton` component
- Main button installs dev version
- Dropdown has "Install (Production)" option
- Both call `install()` or `installProd()`

**3. ServiceActionButton** (shared component)
- Added optional `installProd` action
- Shows "Install (Production)" in dropdown when not installed
- Works for Queen, Hive, and any future services

## Backend Changes

### Tauri Commands

**hive_install:**
```rust
pub async fn hive_install(alias: String, binary: Option<String>) -> Result<String, String>
```
- `binary`: `None` or `Some("release")`
- `None` = dev build (default)
- `Some("release")` = production build

**queen_install:** (already had this parameter)
```rust
pub async fn queen_install(binary: Option<String>) -> Result<String, String>
```

### CLI Updates

**HiveLifecycleAction::Install:**
```rust
Install {
    alias: String,
    binary: Option<String>,  // NEW
}
```

Passes `binary` parameter to both:
- `lifecycle-local::InstallConfig`
- `lifecycle-ssh::InstallConfig`

## Frontend Changes

### Query Hooks

**hiveQueries.ts:**
```typescript
install: async (targetId: string, buildMode: "dev" | "prod" = "dev") => {
  await install.mutateAsync({ targetId, buildMode });
}
```

**queenQueries.ts:**
```typescript
install: async () => { await install.mutateAsync(); },
installProd: async () => { 
  await withCommandExecution(
    () => commands.queenInstall("release"),
    async () => {},
    'Queen install (production)',
  );
  queryClient.invalidateQueries({ queryKey: queenKeys.all });
},
```

## Data Flow

### Hive Install (Production)

```
User clicks dropdown → "Install (Production)"
  ↓
install(targetId, "prod")
  ↓
hiveQueries: install.mutateAsync({ targetId, buildMode: "prod" })
  ↓
commands.hiveInstall(targetId, "release")
  ↓
Tauri: hive_install(alias, Some("release"))
  ↓
HiveLifecycleAction::Install { alias, binary: Some("release") }
  ↓
lifecycle-local/ssh::InstallConfig { local_binary_path: Some("release".into()) }
  ↓
Installs optimized release binary
```

### Queen Install (Production)

```
User clicks dropdown → "Install (Production)"
  ↓
installProd()
  ↓
commands.queenInstall("release")
  ↓
Tauri: queen_install(Some("release"))
  ↓
QueenAction::Install { binary: Some("release") }
  ↓
Installs optimized release binary
```

## Files Modified

### Backend (Rust)
- ✅ `bin/00_rbee_keeper/src/tauri_commands.rs` - Added `binary` parameter to `hive_install`
- ✅ `bin/00_rbee_keeper/src/handlers/hive_lifecycle.rs` - Added `binary` field to `Install` variant
- ✅ `bin/00_rbee_keeper/src/handlers/hive_lifecycle.rs` - Pass `binary` to install configs

### Frontend (TypeScript)
- ✅ `bin/00_rbee_keeper/ui/src/store/hiveQueries.ts` - Updated `install` mutation
- ✅ `bin/00_rbee_keeper/ui/src/store/queenQueries.ts` - Added `installProd` function
- ✅ `bin/00_rbee_keeper/ui/src/components/cards/InstallHiveCard.tsx` - Updated to call with buildMode
- ✅ `bin/00_rbee_keeper/ui/src/components/cards/ServiceActionButton.tsx` - Added `installProd` support
- ✅ `bin/00_rbee_keeper/ui/src/components/cards/QueenCard.tsx` - Pass `installProd` action

## Usage

### For Hive

**Dev Install (default):**
1. Select target from dropdown
2. Click "Install Hive" button

**Production Install:**
1. Select target from dropdown
2. Click dropdown arrow
3. Select "Install (Production)"

### For Queen

**Dev Install (default):**
1. Click "Install" button

**Production Install:**
1. Click dropdown arrow
2. Select "Install (Production)"

## Benefits

✅ **Simple UX** - No extra selects, just a dropdown option  
✅ **Consistent** - Same pattern for Queen and Hive  
✅ **Extensible** - `ServiceActionButton` works for all services  
✅ **Type-safe** - Full TypeScript + Rust type safety  
✅ **Default to dev** - Safer for development workflow

## Testing

1. **Build backend:** `cargo build --package rbee-keeper`
2. **Start UI:** `cd bin/00_rbee_keeper/ui && pnpm dev`
3. **Test Hive:**
   - Try dev install (main button)
   - Try prod install (dropdown)
4. **Test Queen:**
   - Try dev install (main button)
   - Try prod install (dropdown)

## Notes

- Dev mode is the default (safer for development)
- Production mode uses `cargo build --release` (optimized)
- The `binary` parameter is passed through to lifecycle crates
- TypeScript bindings auto-generated by Tauri/Specta

---

**Feature complete!** 🎉
