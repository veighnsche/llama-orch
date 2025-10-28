# TEAM-338: Hive Status Wiring Complete

**Status:** ✅ COMPLETE

## Mission
Make HiveCard status-aware like QueenCard with proper structured status data from backend.

## Changes Made

### 1. Backend: `daemon-lifecycle/src/status.rs`
**Already correct** - `check_daemon_health()` returns `DaemonStatus` with:
- `is_running: bool` (HTTP health check)
- `is_installed: bool` (binary check)

### 2. Tauri Command: `tauri_commands.rs`
**Updated `hive_status` to return structured data:**

```rust
// BEFORE (returned String)
pub async fn hive_status(alias: String) -> Result<String, String>

// AFTER (returns HiveStatus struct)
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
pub struct HiveStatus {
    pub is_running: bool,
    pub is_installed: bool,
}

pub async fn hive_status(alias: String) -> Result<HiveStatus, String>
```

**Implementation:**
- Resolves SSH config for hive alias (localhost or ~/.ssh/config)
- Calls `check_daemon_health()` from daemon-lifecycle crate
- Returns structured status (same pattern as `queen_status`)

**TypeScript bindings:**
- Added `HiveStatus` type export in test
- Generated bindings: `export type HiveStatus = { is_running: boolean; is_installed: boolean }`

### 3. Tauri Registration: `main.rs`
**Already correct** - `hive_status` command registered in invoke_handler (line 102)

### 4. Frontend Store: `hiveStore.ts`
**Added `fetchHiveStatus` method:**

```typescript
export interface SshHive {
  host: string;
  hostname: string;
  user: string;
  port: number;
  status: "online" | "offline" | "unknown";
  isInstalled?: boolean; // NEW: Track installation status
}

// NEW: Fetch individual hive status
fetchHiveStatus: async (hiveId: string) => {
  const result = await commands.hiveStatus(hiveId);
  if (result.status === "ok") {
    const { is_running, is_installed } = result.data;
    set((state) => {
      const hive = state.hives.find((h) => h.host === hiveId);
      if (hive) {
        hive.status = is_running ? "online" : "offline";
        hive.isInstalled = is_installed;
      }
      // Sync installedHives list
      if (is_installed && !state.installedHives.includes(hiveId)) {
        state.installedHives.push(hiveId);
      } else if (!is_installed) {
        state.installedHives = state.installedHives.filter(
          (id) => id !== hiveId
        );
      }
    });
  }
}
```

### 5. Frontend Component: `HiveCard.tsx`
**Made status-aware (same pattern as QueenCard):**

```typescript
// Check installation status from hive object first, fallback to list
const isInstalled = hive?.isInstalled ?? installedHives.includes(hiveId);
const isRunning = hive?.status === "online";

// Compute UI state based on status
const uiState = !isInstalled
  ? { mainAction: install, mainIcon: <Download />, mainLabel: "Install", badgeStatus: "unknown" }
  : isRunning
    ? { mainAction: stop, mainIcon: <Square />, mainLabel: "Stop", badgeStatus: "running" }
    : { mainAction: start, mainIcon: <Play />, mainLabel: "Start", badgeStatus: "stopped" };

// StatusBadge with clickable refresh
<StatusBadge
  status={uiState.badgeStatus}
  onClick={() => fetchHiveStatus(hiveId)}
  isLoading={isLoading}
/>

// Conditional dropdown items based on status
{isInstalled && !isRunning && <Start />}
{isRunning && <Stop />}
{!isInstalled && <Install />}
{isInstalled && <Refresh />}
{isInstalled && <Uninstall />}
```

## Architecture Flow

```
User clicks StatusBadge
  ↓
HiveCard calls fetchHiveStatus(hiveId)
  ↓
hiveStore calls commands.hiveStatus(hiveId)
  ↓
Tauri command hive_status(alias)
  ↓
Resolves SSH config from alias
  ↓
Calls check_daemon_health(health_url, "rbee-hive", ssh_config)
  ↓
Returns HiveStatus { is_running, is_installed }
  ↓
Store updates hive.status and hive.isInstalled
  ↓
HiveCard re-renders with new status
```

## Files Modified

1. `/bin/99_shared_crates/daemon-lifecycle/src/status.rs` - ✅ Already correct (no changes needed)
2. `/bin/00_rbee_keeper/src/tauri_commands.rs` - Updated `hive_status` to return `HiveStatus`
3. `/bin/00_rbee_keeper/src/main.rs` - ✅ Already correct (command registered)
4. `/bin/00_rbee_keeper/ui/src/store/hiveStore.ts` - Added `fetchHiveStatus` method
5. `/bin/00_rbee_keeper/ui/src/components/HiveCard.tsx` - Made status-aware

## Verification

✅ **Rust compilation:** `cargo check -p rbee-keeper` passes
✅ **TypeScript bindings:** Generated correctly with `HiveStatus` type
✅ **Pattern consistency:** HiveCard matches QueenCard pattern exactly

## Key Features

1. **Structured status** - Returns `{ is_running, is_installed }` instead of string
2. **Dynamic UI** - Button changes based on state (Install → Start → Stop)
3. **Conditional actions** - Dropdown only shows relevant operations
4. **Clickable badge** - StatusBadge refreshes status on click
5. **Single source of truth** - Status flows from backend through store to UI

## Pattern Alignment

HiveCard now follows the exact same pattern as QueenCard:
- ✅ StatusBadge in CardAction
- ✅ Structured status from Tauri command
- ✅ Computed uiState based on is_installed/is_running
- ✅ Dynamic button action, icon, label, variant
- ✅ Conditional dropdown items
- ✅ Clickable badge for manual refresh
