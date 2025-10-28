# TEAM-338: Queen Status Integration

**Date:** Oct 28, 2025  
**Status:** ✅ COMPLETE

## Summary

Connected frontend `fetchStatus` to backend `QueenAction::Status` handler. Queen status now shows real-time data (isRunning, isInstalled) instead of mock data.

## Backend Changes

### `/bin/00_rbee_keeper/src/tauri_commands.rs`

**Added:**
1. `QueenStatus` struct (lines 99-105)
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
   pub struct QueenStatus {
       pub is_running: bool,
       pub is_installed: bool,
   }
   ```

2. `queen_status()` command (lines 107-135)
   - Checks if running via `/health` endpoint
   - Checks if installed by looking for `~/.local/bin/queen-rbee`
   - Returns structured `QueenStatus` instead of string message

**Updated:**
- Added `queen_status` to TypeScript bindings export (line 36)

## Frontend Changes

### `/bin/00_rbee_keeper/ui/src/store/queenStore.ts`

**Updated `fetchStatus()`:**
- Calls `commands.queenStatus()` instead of returning mock data
- Handles `Result<QueenStatus, string>` type from Tauri
- Converts snake_case (`is_running`, `is_installed`) to camelCase (`isRunning`, `isInstalled`)
- Proper error handling for both Result error and exceptions

**Before:**
```typescript
// Mock data
const status: QueenStatus = {
  isRunning: false,
  isInstalled: false,
};
```

**After:**
```typescript
const result = await commands.queenStatus();

if (result.status === "ok") {
  const status: QueenStatus = {
    isRunning: result.data.is_running,
    isInstalled: result.data.is_installed,
  };
  // ... update store
} else {
  // Handle error
}
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Frontend (React)                                             │
│                                                              │
│  QueenDataProvider (Suspense)                               │
│         ↓                                                    │
│  useQueenStore().fetchStatus()                              │
│         ↓                                                    │
│  commands.queenStatus()                                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Tauri Bridge                                                 │
│                                                              │
│  Result<QueenStatus, string>                                │
│  { status: "ok", data: { is_running, is_installed } }      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Backend (Rust)                                               │
│                                                              │
│  tauri_commands::queen_status()                             │
│         ↓                                                    │
│  check_daemon_health("/health")                             │
│         ↓                                                    │
│  check binary exists (~/.local/bin/queen-rbee)              │
│         ↓                                                    │
│  return QueenStatus { is_running, is_installed }            │
└─────────────────────────────────────────────────────────────┘
```

## Backend Implementation Details

### Health Check
```rust
let health_url = format!("{}/health", queen_url);
let is_running = check_daemon_health(&health_url).await;
```

Uses `daemon_lifecycle::check_daemon_health()` which:
- Makes HTTP GET request to `/health`
- Returns `true` if 200 OK
- Returns `false` on any error (connection refused, timeout, etc.)

### Installation Check
```rust
let binary_path = PathBuf::from(home)
    .join(".local/bin/queen-rbee");
let is_installed = binary_path.exists();
```

Simple filesystem check - does the binary exist?

## Type Mapping

| Rust (Backend)     | TypeScript (Bindings) | TypeScript (Store) |
|--------------------|----------------------|-------------------|
| `is_running: bool` | `is_running: boolean` | `isRunning: boolean` |
| `is_installed: bool` | `is_installed: boolean` | `isInstalled: boolean` |

**Why the conversion?**
- Rust uses snake_case (convention)
- TypeScript uses camelCase (convention)
- Tauri bindings preserve Rust naming
- Store converts to TypeScript conventions

## Error Handling

### Backend Errors
```rust
// Config load error
Config::load().map_err(|e| format!("Config error: {}", e))?

// HOME env var missing
std::env::var("HOME").map_err(|_| "HOME environment variable not set")?
```

### Frontend Errors
```typescript
if (result.status === "ok") {
  // Success path
} else {
  // Result error (from Rust)
  state.error = result.error;
}

// Exception handling
catch (error) {
  state.error = error instanceof Error 
    ? error.message 
    : "Failed to fetch Queen status";
}
```

## Testing

### Manual Test
1. Start Tauri app: `pnpm tauri dev`
2. Navigate to Services page
3. Should see Queen card with real status
4. Install queen: `rbee-keeper queen install`
5. Card should show "Install" button
6. Start queen: `rbee-keeper queen start`
7. Card should show "Stop" button (red)

### Verification Commands
```bash
# Check if queen is running
curl http://localhost:7833/health

# Check if queen is installed
ls -la ~/.local/bin/queen-rbee

# Check TypeScript bindings
cat bin/00_rbee_keeper/ui/src/generated/bindings.ts | grep -A 5 "queenStatus"
```

## Related Files

**Backend:**
- `/bin/00_rbee_keeper/src/tauri_commands.rs` (queen_status command)
- `/bin/00_rbee_keeper/src/handlers/queen.rs` (QueenAction::Status CLI handler)
- `/bin/99_shared_crates/daemon-lifecycle/src/lib.rs` (check_daemon_health)

**Frontend:**
- `/bin/00_rbee_keeper/ui/src/store/queenStore.ts` (fetchStatus implementation)
- `/bin/00_rbee_keeper/ui/src/containers/QueenContainer.tsx` (Suspense wrapper)
- `/bin/00_rbee_keeper/ui/src/components/QueenCard.tsx` (UI component)
- `/bin/00_rbee_keeper/ui/src/generated/bindings.ts` (TypeScript types)

## Architecture Notes

### Why Not Use CLI Handler Directly?

The CLI handler (`QueenAction::Status`) only prints to stdout:
```rust
if is_running {
    n!("queen_status", "✅ queen 'localhost' is running on {}", queen_url);
} else {
    n!("queen_status", "❌ queen 'localhost' is not running on {}", queen_url);
}
```

The Tauri command needs structured data for the frontend:
```rust
Ok(QueenStatus {
    is_running,
    is_installed,
})
```

**Pattern:** CLI handlers are for human output, Tauri commands are for structured data.

### Why Check Binary Existence?

The `isInstalled` flag tells the UI which buttons to show:
- Not installed → Show "Install" button
- Installed but not running → Show "Start" button
- Running → Show "Stop" button (red)

Without this check, we'd need to try starting and handle "already installed" errors.

## Next Steps

Consider adding:
1. **Port detection** - Which port is queen running on?
2. **Version info** - Which version is installed?
3. **Uptime** - How long has queen been running?
4. **Health details** - More than just "running" (memory, CPU, etc.)

These would require extending the `/health` endpoint to return JSON instead of just 200 OK.

---

**Pattern Established:** Backend status commands return structured data, frontend converts naming conventions, Suspense handles loading states.
