# State Management Architecture

## Overview

Conservative approach: **Zustand stores installation state only**. Service states come from backend.

## What Goes Where

### ✅ Zustand (Client-Side Persistent Storage)

**File:** `src/store/hiveStore.ts` (renamed to `useInstallationStore`)

Stores **installation state** that should persist across sessions:

1. **Queen Installation** (`isQueenInstalled: boolean`)
   - Whether queen is installed locally
   - Default: `false`
   - Methods: `setQueenInstalled()`

2. **Installed Hives** (`installedHives: string[]`)
   - Array of SSH target IDs where hive is installed
   - Default: `[]`
   - Methods: `addInstalledHive()`, `removeInstalledHive()`, `isHiveInstalled()`

**Storage:** Browser localStorage via Zustand's `persist` middleware

**Key:** `installation-state`

### ❌ NOT in Zustand (Backend State)

These come from **backend via heartbeats/health checks**:

1. **Queen Status** (on/off/healthy/unhealthy)
   - Source: `http://localhost:7833/health`
   - Fetched on-demand via health check button
   - TODO: Real-time via heartbeat SSE

2. **Hive Status at Each Target** (on/off/healthy/unhealthy)
   - Source: `http://localhost:7835/health` (localhost)
   - Source: SSH health checks (remote targets)
   - Fetched on-demand
   - TODO: Real-time via heartbeat SSE

3. **Available SSH Targets**
   - Source: Tauri command `ssh_list()`
   - Reads from `~/.ssh/config`
   - Not stored in Zustand, fetched from backend

## Why This Architecture?

### Zustand = UI Preferences
- Persists across app restarts
- User's personal settings
- No source of truth conflicts

### Backend = Service State
- Single source of truth
- Real-time updates via SSE
- Consistent across all clients
- Survives app crashes

## Usage Example

```typescript
// In HiveCard.tsx
import { useHiveStore } from "../store/hiveStore";

function HiveCard() {
  // UI preference from Zustand (persistent)
  const { selectedTarget, setSelectedTarget } = useHiveStore();
  
  // Service state from backend (real-time)
  const [status, setStatus] = useState<ServiceStatus>("unknown");
  
  // Change target (saves to localStorage automatically)
  const handleTargetChange = (target: string) => {
    setSelectedTarget(target); // Persists via Zustand
  };
  
  // Check service status (fetches from backend)
  const handleHealthCheck = async () => {
    const response = await fetch("http://localhost:7835/health");
    setStatus(response.ok ? "healthy" : "unhealthy");
  };
}
```

## Future Enhancements

### Phase 1 (Current)
- ✅ Selected target persistence
- ✅ Favorite targets

### Phase 2 (TODO)
- Real-time service status via SSE heartbeats
- Auto-refresh health checks
- Connection state indicators

### Phase 3 (TODO)
- Multiple hive instances per target
- Per-target configuration
- Advanced SSH options

## Storage Location

**Development:**
- Chrome: `localStorage['hive-preferences']`
- Firefox: `localStorage['hive-preferences']`

**Production (Tauri):**
- Same as browser localStorage
- Path: `~/.local/share/rbee-keeper/` (Linux)
- Path: `~/Library/Application Support/rbee-keeper/` (macOS)
- Path: `%APPDATA%\rbee-keeper\` (Windows)

## Migration Path

If we need more robust storage later:

```typescript
// Option 1: Tauri Store Plugin
import { Store } from '@tauri-apps/plugin-store';
const store = new Store('settings.json');

// Option 2: Custom Tauri Commands
await invoke('save_preference', { key, value });
```

But for now, **localStorage is sufficient** for UI preferences.
