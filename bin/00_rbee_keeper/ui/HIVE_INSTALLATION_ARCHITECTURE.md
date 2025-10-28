# Hive Installation Architecture

## Overview

Dynamic hive installation system where users can install hives to multiple SSH targets, and each installed hive gets its own management card.

## Components

### 1. **InstallHiveCard** (`src/components/InstallHiveCard.tsx`)
- Shows SSH target selector (filtered to exclude already installed hives)
- "Install Hive" button
- When clicked → saves to Zustand → triggers install command
- Filters out targets that already have hives installed

### 2. **InstalledHiveCard** (`src/components/InstalledHiveCard.tsx`)
- One card per installed hive
- Shows target name and connection details
- Has full ServiceActionButtons (start, stop, install, update, uninstall)
- When uninstalled → removes from Zustand

### 3. **QueenCard** (`src/components/QueenCard.tsx`)
- Single card for queen service
- Always at the top
- Has ServiceActionButtons

## Page Layout

```
┌─────────────────────────────────────┐
│ QueenCard                           │
│ (Smart API server)                  │
├─────────────────────────────────────┤
│ InstalledHiveCard (localhost)       │ ← If installed
├─────────────────────────────────────┤
│ InstalledHiveCard (infra)           │ ← If installed
├─────────────────────────────────────┤
│ InstalledHiveCard (mac)             │ ← If installed
├─────────────────────────────────────┤
│ InstallHiveCard                     │ ← Always at bottom
│ (Select target + Install button)    │
└─────────────────────────────────────┘
```

## State Management

### Zustand Store (`useInstallationStore`)

```typescript
{
  isQueenInstalled: boolean,
  installedHives: string[], // ["localhost", "infra", "mac"]
  
  setQueenInstalled(installed: boolean),
  addInstalledHive(targetId: string),
  removeInstalledHive(targetId: string),
  isHiveInstalled(targetId: string): boolean
}
```

**Storage:** localStorage key `installation-state`

## Data Flow

### Installing a Hive

1. User selects target from InstallHiveCard dropdown
2. User clicks "Install Hive"
3. `handleInstallHive(targetId)` called
4. Tauri command invoked: `invoke("hive_start", { host: targetId, ... })`
5. On success: `addInstalledHive(targetId)` → saves to Zustand
6. Page re-renders → new InstalledHiveCard appears
7. InstallHiveCard dropdown filters out the newly installed target

### Uninstalling a Hive

1. User clicks uninstall in InstalledHiveCard ServiceActionButtons
2. `handleHiveCommand("hive-uninstall", targetId)` called
3. `removeInstalledHive(targetId)` → removes from Zustand
4. TODO: Actual uninstall command
5. Page re-renders → InstalledHiveCard disappears
6. Target reappears in InstallHiveCard dropdown

## Filtering Logic

### InstallHiveCard Dropdown

```typescript
// Filter out already installed hives
const availableHives = hives.filter(
  (hive) => !installedHives.includes(hive.host)
);

// Check if localhost is already installed
const isLocalhostInstalled = installedHives.includes("localhost");
```

Shows:
- `localhost` (if not installed)
- SSH targets from `~/.ssh/config` (if not installed)

### InstalledHiveCard Rendering

```typescript
{installedHives.map((targetId) => {
  // Find hive data from SSH config
  const hiveData = hives.find((h) => h.host === targetId);
  
  return (
    <InstalledHiveCard
      targetId={targetId}
      targetName={...}
      targetSubtitle={...}
    />
  );
})}
```

## Command Handlers

### `handleCommand(command: string)`
- Handles queen commands only
- Used by QueenCard

### `handleHiveCommand(command: string, targetId: string)`
- Handles commands for installed hives
- Passes targetId to Tauri commands
- On uninstall: removes from Zustand

### `handleInstallHive(targetId: string)`
- Installs hive to target
- On success: adds to Zustand

## SSH Target Data

Comes from `SshHivesDataProvider`:
- Fetches from Tauri: `commands.sshList()`
- Reads `~/.ssh/config`
- Returns: `SshHive[]` with `{ host, user, hostname, port, status }`

## Benefits

1. **Dynamic** - Add/remove hives at runtime
2. **Persistent** - Installation state survives app restarts
3. **Filtered** - Can't install to same target twice
4. **Scalable** - Support unlimited hives
5. **Clean UI** - Each hive gets its own card
6. **Type-safe** - Full TypeScript support

## Future Enhancements

- [ ] Real-time status updates via heartbeats
- [ ] Actual install/uninstall commands (currently TODOs)
- [ ] Health checks for remote hives
- [ ] Bulk operations (install to multiple targets)
- [ ] Installation progress indicators
