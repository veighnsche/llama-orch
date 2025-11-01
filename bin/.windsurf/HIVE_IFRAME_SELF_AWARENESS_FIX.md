# Hive UI Self-Awareness Fix (TEAM-378)

**Date:** Nov 1, 2025  
**Status:** ✅ COMPLETE

---

## Problem

All remote hives showed "localhost" as their name because:

1. **Keeper iframe always loaded `http://localhost:7835`** regardless of which hive was selected
2. **Hive UI used `window.location.hostname`** which returned "localhost" when iframed from Keeper
3. **No extraction of hive metadata** from heartbeat events

---

## Root Cause

### Issue 1: Using SSH alias instead of actual hostname/IP

**CRITICAL BUG in `hiveQueries.ts`:**

```tsx
// ❌ WRONG - Uses SSH alias "workstation" as hostname
async function fetchHiveStatus(hiveId: string): Promise<SshHive> {
  return {
    host: hiveId,
    hostname: hiveId,  // "workstation" - browser can't resolve this!
    user: 'unknown',
    port: 22,
  };
}
```

**Problem:** The browser tried to load `http://workstation:7835` but couldn't resolve "workstation" (SSH alias) to an IP address!

**Fix:** Merge SSH config data (which has the actual IP) with daemon status:

```tsx
// ✅ CORRECT - Merges SSH config to get actual IP
async function fetchHiveStatus(hiveId: string, sshHives?: SshHive[]): Promise<SshHive> {
  const sshConfig = sshHives?.find(h => h.host === hiveId);
  
  return {
    host: hiveId,
    hostname: sshConfig?.hostname || hiveId,  // "192.168.1.100" - browser can resolve!
    user: sshConfig?.user || 'unknown',
    port: sshConfig?.port || 22,
  };
}
```

### Issue 2: Hardcoded localhost in iframe URL

```tsx
// ❌ WRONG - Always loads localhost
const hiveUrl = getIframeUrl('hive', isDev);  // Returns http://localhost:7835

<iframe src={hiveUrl} />
```

**Problem:** When viewing a remote hive (e.g., `192.168.1.100`), the iframe still loaded `http://localhost:7835`, so you were viewing the wrong hive!

### Issue 2: Hive UI couldn't detect its own identity

```tsx
// ❌ WRONG - Returns Keeper's hostname when iframed
const hiveAddress = window.location.hostname  // "localhost" (Keeper's address)
const client = new HiveClient(`http://${hiveAddress}:7835`, hiveAddress)
```

**Problem:** When iframed, `window.location` refers to the parent (Keeper), not the iframe's actual source.

### Issue 3: Metadata available but not displayed

The heartbeat events contained `hive_info` with full metadata, but the UI only extracted `workers`.

---

## Solution

### Fix 1: Use actual hive hostname in iframe URL (TEAM-378)

**File:** `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/ui/src/pages/HivePage.tsx`

```tsx
// ✅ CORRECT - Use actual hive's hostname
const isLocalhost = hive.hostname === 'localhost' || hive.hostname === '127.0.0.1'

const hiveUrl = isLocalhost
  ? getIframeUrl('hive', isDev)  // localhost: use dev (7836) or prod (7835)
  : `http://${hive.hostname}:7835`  // remote: always use prod port 7835

<iframe src={hiveUrl} />
```

**Result:** 
- Localhost hive → `http://localhost:7835` (or `:7836` in dev)
- Remote hive `192.168.1.100` → `http://192.168.1.100:7835`

### Fix 2: Extract and display hive metadata (TEAM-378)

**File:** `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui/app/src/App.tsx`

**Header:**
```tsx
<h1>Hive: {hiveInfo.id}</h1>
<p>{hiveInfo.hostname}:{hiveInfo.port} • Version {hiveInfo.version} • {hiveInfo.operational_status}</p>
```

**Status Card:**
```tsx
<CardDescription>
  {hiveInfo.id} • {hiveInfo.hostname}:{hiveInfo.port} • v{hiveInfo.version}
</CardDescription>
<Badge variant="outline">{hiveInfo.operational_status}</Badge>
```

**Heartbeat subscription:**
```tsx
monitor.start((event: any) => {
  setConnected(true)
  setWorkerCount(event.workers?.length || 0)
  setLastUpdate(new Date().toLocaleTimeString())
  // TEAM-378: Extract hive metadata for self-awareness
  if (event.hive_info) {
    setHiveInfo(event.hive_info)
  }
})
```

---

## Data Flow

### Before (Broken)

```
Keeper UI
  └─> Navigate to /hive/remote-server
      └─> Load iframe: http://localhost:7835  ❌ WRONG!
          └─> Hive UI detects hostname: "localhost"  ❌ WRONG!
              └─> Shows: "Hive: localhost"  ❌ WRONG!
```

### After (Fixed)

```
Keeper UI
  └─> Navigate to /hive/remote-server
      └─> Fetch hive data: { hostname: "192.168.1.100", ... }
          └─> Load iframe: http://192.168.1.100:7835  ✅ CORRECT!
              └─> Hive UI receives SSE heartbeat
                  └─> Extract hive_info: { id: "remote-server", hostname: "192.168.1.100", ... }
                      └─> Shows: "Hive: remote-server"  ✅ CORRECT!
```

---

## Files Changed

### Keeper UI (iframe URL fix)
- `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/ui/src/pages/HivePage.tsx`
  - Use `hive.hostname` instead of hardcoded localhost
  - Detect localhost vs remote hives
  - Removed unused `ExternalLink` import

- `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/ui/src/store/hiveQueries.ts`
  - **CRITICAL FIX:** `fetchHiveStatus()` now merges SSH config data
  - `useHive()` passes SSH hives list from cache to get actual hostname/IP
  - Previously used `hiveId` (SSH alias) as hostname - browser couldn't resolve it!

### Hive UI (self-awareness)
- `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui/app/src/App.tsx`
  - Extract `hive_info` from heartbeat events
  - Display hive metadata in header
  - Display operational status badge
  - Show full details in status card

---

## Testing

### Localhost Hive
1. Start hive: `./rbee hive start`
2. Navigate to Keeper → Hives → localhost
3. **Expected:** Header shows "Hive: localhost" with version and status

### Remote Hive
1. Configure SSH target in `~/.ssh/config`
2. Install and start remote hive
3. Navigate to Keeper → Hives → remote-server
4. **Expected:** 
   - Iframe loads `http://192.168.1.100:7835`
   - Header shows "Hive: remote-server" with actual hostname
   - Status card shows correct metadata

---

## Key Insights

### Why `window.location` doesn't work in iframes

When a page is loaded in an iframe:
- `window.location.hostname` = **parent's hostname** (Keeper's localhost)
- `iframe.src` = **actual source URL** (hive's hostname)

**Solution:** Don't rely on `window.location` in iframed apps. Get identity from backend data (SSE heartbeat).

### Why hardcoded URLs break multi-hive setups

```tsx
// ❌ WRONG - All hives load the same URL
getIframeUrl('hive', isDev)  // Always returns localhost

// ✅ CORRECT - Each hive loads its own URL
`http://${hive.hostname}:7835`
```

### Why SSE heartbeat is the source of truth

The hive backend **knows its own identity** from startup args:
```bash
rbee-hive --hive-id remote-server --queen-url http://queen:7833
```

This metadata flows through SSE heartbeat events, making it the **canonical source** for self-awareness.

---

## Summary

**Before:** All hives showed "localhost" because iframe always loaded localhost URL  
**After:** Each hive shows its actual name/hostname by loading the correct iframe URL and extracting metadata from SSE

**Impact:** Multi-hive networks now work correctly - each hive UI is self-aware of its identity! 🎯
