# Part 3: Extract Keeper Page to GUI

**TEAM-293: Move keeper functionality to dedicated Tauri GUI**

## Goal

Extract the keeper page from ui-queen-rbee and implement it properly in the keeper GUI with Tauri command integration.

## What to Extract

**From:** `frontend/apps/web-ui/src/pages/KeeperPage.tsx` (now deleted)  
**To:** `bin/00_rbee_keeper/GUI/src/pages/`

**Keeper-specific operations:**
- Start/Stop queen-rbee
- Install/Uninstall queen-rbee
- Update queen-rbee
- Status checking
- Start/Stop hives
- Install/Uninstall hives

## Step 1: Create Keeper Pages Structure

```
bin/00_rbee_keeper/GUI/src/pages/
├── KeeperDashboard.tsx       # Overview of all components
├── QueenLifecycle.tsx         # Queen start/stop/install/uninstall
├── HiveLifecycle.tsx          # Hive start/stop/install/uninstall
└── Settings.tsx               # Keeper settings
```

## Step 2: Create Tauri Commands Wrapper

**File:** `bin/00_rbee_keeper/GUI/src/lib/tauri-commands.ts`

```typescript
import { invoke } from '@tauri-apps/api/tauri';

// ============================================================================
// QUEEN COMMANDS
// ============================================================================

export async function queenStart(): Promise<void> {
  await invoke('queen_start');
}

export async function queenStop(): Promise<void> {
  await invoke('queen_stop');
}

export async function queenStatus(): Promise<string> {
  return await invoke('queen_status');
}

export async function queenInstall(binary?: string): Promise<void> {
  await invoke('queen_install', { binary });
}

export async function queenUninstall(): Promise<void> {
  await invoke('queen_uninstall');
}

export async function queenRebuild(withLocalHive: boolean): Promise<void> {
  await invoke('queen_rebuild', { withLocalHive });
}

export async function queenInfo(): Promise<string> {
  return await invoke('queen_info');
}

// ============================================================================
// HIVE COMMANDS
// ============================================================================

export async function hiveInstall(
  host: string,
  binary?: string,
  installDir?: string
): Promise<void> {
  await invoke('hive_install', { host, binary, installDir });
}

export async function hiveUninstall(
  host: string,
  installDir?: string
): Promise<void> {
  await invoke('hive_uninstall', { host, installDir });
}

export async function hiveStart(
  host: string,
  port: number,
  installDir?: string
): Promise<void> {
  await invoke('hive_start', { host, port, installDir });
}

export async function hiveStop(host: string): Promise<void> {
  await invoke('hive_stop', { host });
}

export async function hiveList(): Promise<string> {
  return await invoke('hive_list');
}

export async function hiveGet(alias: string): Promise<string> {
  return await invoke('hive_get', { alias });
}

export async function hiveStatus(alias: string): Promise<string> {
  return await invoke('hive_status', { alias });
}

export async function hiveRefreshCapabilities(alias: string): Promise<void> {
  await invoke('hive_refresh_capabilities', { alias });
}

// ============================================================================
// STATUS COMMAND
// ============================================================================

export async function getStatus(): Promise<string> {
  return await invoke('get_status');
}
```

## Step 3: Create QueenLifecycle Page

**File:** `bin/00_rbee_keeper/GUI/src/pages/QueenLifecycle.tsx`

```tsx
import React, { useState } from 'react';
import { 
  queenStart, 
  queenStop, 
  queenStatus, 
  queenInstall, 
  queenUninstall 
} from '@/lib/tauri-commands';

export function QueenLifecycle() {
  const [status, setStatus] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const handleStart = async () => {
    setLoading(true);
    try {
      await queenStart();
      alert('Queen started successfully');
    } catch (error) {
      alert(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await queenStop();
      alert('Queen stopped successfully');
    } catch (error) {
      alert(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleInstall = async () => {
    setLoading(true);
    try {
      await queenInstall();
      alert('Queen installed successfully');
    } catch (error) {
      alert(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleUninstall = async () => {
    if (!confirm('Are you sure you want to uninstall queen-rbee?')) {
      return;
    }
    
    setLoading(true);
    try {
      await queenUninstall();
      alert('Queen uninstalled successfully');
    } catch (error) {
      alert(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCheckStatus = async () => {
    setLoading(true);
    try {
      const result = await queenStatus();
      setStatus(result);
    } catch (error) {
      setStatus(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Queen Lifecycle</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Control Panel */}
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Control</h2>
          
          <div className="space-y-3">
            <button
              onClick={handleStart}
              disabled={loading}
              className="btn btn-primary w-full"
            >
              Start Queen
            </button>
            
            <button
              onClick={handleStop}
              disabled={loading}
              className="btn btn-secondary w-full"
            >
              Stop Queen
            </button>
            
            <button
              onClick={handleCheckStatus}
              disabled={loading}
              className="btn btn-secondary w-full"
            >
              Check Status
            </button>
          </div>
        </div>

        {/* Installation Panel */}
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Installation</h2>
          
          <div className="space-y-3">
            <button
              onClick={handleInstall}
              disabled={loading}
              className="btn btn-primary w-full"
            >
              Install Queen
            </button>
            
            <button
              onClick={handleUninstall}
              disabled={loading}
              className="btn btn-danger w-full"
            >
              Uninstall Queen
            </button>
          </div>
        </div>

        {/* Status Display */}
        {status && (
          <div className="card p-6 md:col-span-2">
            <h2 className="text-xl font-semibold mb-4">Status</h2>
            <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-auto">
              {status}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
```

## Step 4: Create HiveLifecycle Page

**File:** `bin/00_rbee_keeper/GUI/src/pages/HiveLifecycle.tsx`

```tsx
import React, { useState } from 'react';
import {
  hiveStart,
  hiveStop,
  hiveInstall,
  hiveUninstall,
  hiveList,
  hiveStatus,
} from '@/lib/tauri-commands';

export function HiveLifecycle() {
  const [host, setHost] = useState('');
  const [port, setPort] = useState(7835);
  const [hives, setHives] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const handleInstall = async () => {
    if (!host) {
      alert('Please enter a host');
      return;
    }

    setLoading(true);
    try {
      await hiveInstall(host);
      alert(`Hive installed on ${host}`);
    } catch (error) {
      alert(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleStart = async () => {
    if (!host) {
      alert('Please enter a host');
      return;
    }

    setLoading(true);
    try {
      await hiveStart(host, port);
      alert(`Hive started on ${host}:${port}`);
    } catch (error) {
      alert(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    if (!host) {
      alert('Please enter a host');
      return;
    }

    setLoading(true);
    try {
      await hiveStop(host);
      alert(`Hive stopped on ${host}`);
    } catch (error) {
      alert(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleList = async () => {
    setLoading(true);
    try {
      const result = await hiveList();
      setHives(result);
    } catch (error) {
      setHives(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Hive Lifecycle</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Control Panel */}
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Control</h2>

          <div className="space-y-3">
            <input
              type="text"
              placeholder="Host (e.g., localhost or user@remote)"
              value={host}
              onChange={(e) => setHost(e.target.value)}
              className="input w-full"
            />

            <input
              type="number"
              placeholder="Port"
              value={port}
              onChange={(e) => setPort(Number(e.target.value))}
              className="input w-full"
            />

            <button
              onClick={handleInstall}
              disabled={loading}
              className="btn btn-primary w-full"
            >
              Install Hive
            </button>

            <button
              onClick={handleStart}
              disabled={loading}
              className="btn btn-primary w-full"
            >
              Start Hive
            </button>

            <button
              onClick={handleStop}
              disabled={loading}
              className="btn btn-secondary w-full"
            >
              Stop Hive
            </button>
          </div>
        </div>

        {/* List Panel */}
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Hive List</h2>

          <button
            onClick={handleList}
            disabled={loading}
            className="btn btn-secondary w-full mb-4"
          >
            Refresh List
          </button>

          {hives && (
            <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-auto text-sm">
              {hives}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
```

## Step 5: Create KeeperDashboard Page

**File:** `bin/00_rbee_keeper/GUI/src/pages/KeeperDashboard.tsx`

```tsx
import React, { useEffect, useState } from 'react';
import { getStatus } from '@/lib/tauri-commands';

export function KeeperDashboard() {
  const [status, setStatus] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  const loadStatus = async () => {
    try {
      const result = await getStatus();
      setStatus(result);
    } catch (error) {
      setStatus(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Keeper Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Queen Status */}
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Queen Status</h2>
          <div className="text-2xl font-bold text-green-500">
            ● Running
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Port: 7833
          </p>
        </div>

        {/* Hives Status */}
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Hives</h2>
          <div className="text-2xl font-bold">
            2 Active
          </div>
          <p className="text-sm text-gray-600 mt-2">
            1 local, 1 remote
          </p>
        </div>

        {/* Workers Status */}
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Workers</h2>
          <div className="text-2xl font-bold">
            5 Running
          </div>
          <p className="text-sm text-gray-600 mt-2">
            3 LLM, 2 ComfyUI
          </p>
        </div>

        {/* System Status */}
        <div className="card p-6 md:col-span-3">
          <h2 className="text-xl font-semibold mb-4">System Status</h2>
          {loading ? (
            <p>Loading...</p>
          ) : (
            <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-auto text-sm">
              {status}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
```

## Step 6: Create Settings Page

**File:** `bin/00_rbee_keeper/GUI/src/pages/Settings.tsx`

```tsx
import React from 'react';

export function Settings() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Settings</h1>

      <div className="card p-6">
        <h2 className="text-xl font-semibold mb-4">Keeper Settings</h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Queen Port
            </label>
            <input
              type="number"
              defaultValue={7833}
              className="input w-full md:w-64"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Default Hive Port
            </label>
            <input
              type="number"
              defaultValue={7835}
              className="input w-full md:w-64"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Auto-refresh Interval (seconds)
            </label>
            <input
              type="number"
              defaultValue={5}
              className="input w-full md:w-64"
            />
          </div>

          <button className="btn btn-primary">
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}
```

## Step 7: Update App.tsx Routes

**File:** `bin/00_rbee_keeper/GUI/src/App.tsx`

```tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { KeeperDashboard } from './pages/KeeperDashboard';
import { QueenLifecycle } from './pages/QueenLifecycle';
import { HiveLifecycle } from './pages/HiveLifecycle';
import { Settings } from './pages/Settings';

export function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<KeeperDashboard />} />
            <Route path="/queen" element={<QueenLifecycle />} />
            <Route path="/hives" element={<HiveLifecycle />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
```

## Step 8: Verify Tauri Commands

Ensure these commands exist in `bin/00_rbee_keeper/src/tauri_commands.rs`:

```rust
#[tauri::command]
pub async fn queen_start() -> Result<String, String> { /* ... */ }

#[tauri::command]
pub async fn queen_stop() -> Result<String, String> { /* ... */ }

#[tauri::command]
pub async fn queen_install(binary: Option<String>) -> Result<String, String> { /* ... */ }

#[tauri::command]
pub async fn hive_install(host: String, binary: Option<String>, install_dir: Option<String>) -> Result<String, String> { /* ... */ }

// ... etc
```

## Verification Checklist

- [ ] `tauri-commands.ts` created with all wrappers
- [ ] `QueenLifecycle.tsx` created
- [ ] `HiveLifecycle.tsx` created
- [ ] `KeeperDashboard.tsx` created
- [ ] `Settings.tsx` created
- [ ] `App.tsx` routes updated
- [ ] All Tauri commands working
- [ ] UI properly calls Rust backend

## Expected Result

```
✅ Keeper GUI has dedicated pages
✅ Queen operations: start/stop/install/uninstall
✅ Hive operations: start/stop/install/uninstall
✅ Dashboard shows system overview
✅ Settings page for configuration
✅ All operations call Tauri commands (no direct HTTP)
```

## Next Steps

1. **Next:** `04_CREATE_HIVE_UI.md` - Create hive UI
2. **Then:** `05_CREATE_WORKER_UIS.md` - Create worker UIs

---

**Status:** 📋 READY TO IMPLEMENT
