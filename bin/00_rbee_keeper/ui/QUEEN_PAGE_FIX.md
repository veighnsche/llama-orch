# Queen Page Fix

**Date:** Oct 29, 2025  
**Status:** ✅ FIXED

---

## Issues Found

Looking at the screenshot, several issues were identified:

1. ❌ **Wrong URL** - iframe pointing to `http://localhost:7833` instead of `http://localhost:7833/ui`
2. ❌ **"Disconnected" status** - UI not loading properly
3. ❌ **Empty content** - Iframe showing mostly blank page

---

## Root Cause

The Queen UI is served at the `/ui` endpoint (as per PORT_CONFIGURATION.md), but the iframe was pointing to the root URL which serves the API, not the UI.

---

## Fix Applied

### Updated iframe URL

**File:** `bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx`

**Before:**
```tsx
<iframe
  src="http://localhost:7833"
  className="w-full h-full border-0 rounded-lg"
  title="Queen Web Interface"
  sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
/>
```

**After:**
```tsx
<iframe
  src="http://localhost:7833/ui"
  className="w-full h-full border-0 rounded-lg"
  title="Queen Web Interface"
  sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-modals"
  allow="cross-origin-isolated"
/>
```

### Changes Made

1. ✅ **Updated iframe src** - Changed from `http://localhost:7833` to `http://localhost:7833/ui`
2. ✅ **Updated "Open in new tab" link** - Changed from `http://localhost:7833` to `http://localhost:7833/ui`
3. ✅ **Added `allow-modals` to sandbox** - Allows modal dialogs
4. ✅ **Added `allow="cross-origin-isolated"`** - Improves iframe isolation

---

## URL Structure

According to `PORT_CONFIGURATION.md`:

| Service | Dev Port | Production URL | Purpose |
|---------|----------|----------------|---------|
| **Queen API** | `7833` | `http://localhost:7833` | Backend HTTP API |
| **Queen UI (dev)** | `7834` | `http://localhost:7833/ui` | Frontend (proxied in dev, embedded in prod) |

**Development Mode:**
- Queen binary proxies `/ui` requests to Vite dev server at `http://localhost:7834`
- Keeper iframe loads `http://localhost:7833/ui` which proxies to Vite

**Production Mode:**
- Queen binary serves embedded static files at `/ui`
- Keeper iframe loads `http://localhost:7833/ui` which serves static files

---

## Testing

### Start Queen (Development Mode)

**Terminal 1: Queen Binary**
```bash
cd bin/10_queen_rbee
cargo run
```

**Terminal 2: Queen UI (Vite)**
```bash
cd bin/10_queen_rbee/ui/app
npm run dev  # Runs on port 7834
```

**Terminal 3: Keeper GUI**
```bash
cd bin/00_rbee_keeper/ui
npm run dev  # Runs on port 5173
```

### Access

1. Open Keeper GUI: `http://localhost:5173`
2. Navigate to "Queen" page in sidebar
3. Queen UI should load in iframe at `http://localhost:7833/ui`

---

## Expected Result

After the fix:
- ✅ Queen UI loads correctly in iframe
- ✅ Shows "Queen Dashboard" with heartbeat monitor
- ✅ Shows "Connected" status (green dot)
- ✅ Displays hives and workers
- ✅ Shows RHAI IDE stub

---

## Summary

The Queen page now correctly loads the Queen UI at the `/ui` endpoint. The iframe will display the full Queen Dashboard with heartbeat monitoring and RHAI IDE.

**The fix ensures the Keeper GUI properly embeds the Queen UI!** 🎉
