# Worker Catalog UI Complete

**Date:** 2025-11-01  
**Status:** ✅ COMPLETE

## Overview

Refocused Worker Management UI to prioritize the **installation lifecycle** over spawning. Users must install workers before they can spawn them.

## New Component: WorkerCatalogView

**File:** `src/components/WorkerManagement/WorkerCatalogView.tsx`

### Features

**1. Browse Available Workers**
- Fetches from worker catalog (port 8787)
- Shows CPU, CUDA, and Metal variants
- Platform-aware filtering (only shows compatible workers)

**2. Rich Worker Cards**
- Worker icon (CPU/GPU/Metal)
- Version badge
- Description
- Metadata grid:
  - Type (CPU/CUDA/METAL)
  - Build system (cargo)
  - Supported formats (gguf, safetensors)
  - Streaming support
- Runtime dependencies (gcc, cuda, etc.)
- Build dependencies (rust, cargo, cmake)

**3. Installation Status**
- "Installed" badge for installed workers
- "Install Worker" button for uninstalled workers
- Remove button (trash icon) for installed workers
- Loading states during install/remove

**4. Platform Detection**
- Auto-detects user's platform (linux/macos/windows)
- Shows platform info banner
- Filters out incompatible workers
- Shows warning if no compatible workers

**5. Error Handling**
- Loading spinner while fetching catalog
- Error card if catalog service unavailable
- Helpful error messages with port info

## Updated Components

### WorkerManagement (index.tsx)

**Changes:**
1. Added "Worker Catalog" as **first tab** (default view)
2. Reordered tabs: Catalog → Active → Spawn
3. Added install/remove handlers (TODO: wire to backend)
4. Updated description: "Install workers, monitor performance..."
5. Added comments explaining priority

**Tab Order:**
```
1. Worker Catalog  ← START HERE (install workers)
2. Active Workers  ← Monitor running workers
3. Spawn Worker    ← Requires installed workers + models
```

### types.ts

**Changes:**
- Updated `ViewMode` type: `'catalog' | 'active' | 'spawn'`

## User Flow

### Correct Flow (New)
1. **Install Workers** (Catalog tab)
   - Browse available workers
   - Check platform compatibility
   - Click "Install Worker"
   - Wait for download + build + install
   - Worker binary now at `/usr/local/bin/`

2. **Download Models** (separate concern - not implemented yet)
   - Browse model catalog
   - Download GGUF files
   - Store in models directory

3. **Spawn Workers** (Spawn tab)
   - Select installed worker type
   - Select downloaded model
   - Select device (GPU ID)
   - Click "Spawn Worker"

4. **Monitor Workers** (Active tab)
   - View running workers
   - Check GPU utilization
   - Terminate workers

### Old Flow (Wrong)
- ❌ Tried to spawn workers that weren't installed
- ❌ No way to install workers
- ❌ No way to remove workers
- ❌ Confusing for new users

## Backend Integration (TODO)

### Install Worker

**Endpoint:** `POST /v1/workers/install`

**Request:**
```json
{
  "worker_id": "llm-worker-rbee-cuda"
}
```

**Response:** SSE stream with build progress
```
data: ==> Fetching PKGBUILD...
data: ==> Downloading source...
data: ==> Building worker...
data: Step 1: Compiling...
data: Step 2: Linking...
data: ==> Installing binary...
data: [DONE]
```

### Remove Worker

**Endpoint:** `DELETE /v1/workers/{worker_id}`

**Response:**
```json
{
  "success": true,
  "message": "Worker removed successfully"
}
```

### List Installed Workers

**Endpoint:** `GET /v1/workers/installed`

**Response:**
```json
{
  "workers": [
    {
      "id": "llm-worker-rbee-cpu",
      "version": "0.1.0",
      "installed_at": "2025-11-01T18:00:00Z",
      "binary_path": "/usr/local/bin/llm-worker-rbee-cpu"
    }
  ]
}
```

## Next Steps

### 1. Backend Implementation (Priority 1)
- [ ] Create `POST /v1/workers/install` endpoint
- [ ] Integrate PKGBUILD parser + executor
- [ ] Stream build output via SSE
- [ ] Create `DELETE /v1/workers/{worker_id}` endpoint
- [ ] Create `GET /v1/workers/installed` endpoint

### 2. Frontend Integration (Priority 2)
- [ ] Wire `handleInstallWorker` to backend API
- [ ] Wire `handleRemoveWorker` to backend API
- [ ] Query installed workers on mount
- [ ] Update `installedWorkers` state from backend
- [ ] Show real-time build progress during installation

### 3. Model Management (Priority 3)
- [ ] Create Model Catalog view (similar to Worker Catalog)
- [ ] Add model download functionality
- [ ] Track downloaded models
- [ ] Show model metadata (size, quantization, etc.)

### 4. Polish (Priority 4)
- [ ] Add confirmation dialog for worker removal
- [ ] Add build log viewer (expand to see full output)
- [ ] Add retry button for failed installations
- [ ] Add "View PKGBUILD" button to see build instructions
- [ ] Add estimated build time

## File Structure

```
src/components/WorkerManagement/
├── index.tsx                    # Main component (3 tabs)
├── WorkerCatalogView.tsx        # NEW - Install/remove workers
├── ActiveWorkersView.tsx        # Monitor running workers
├── SpawnWorkerView.tsx          # Spawn new workers
└── types.ts                     # Shared types (updated)
```

## Key Insights

1. **Installation comes first** - Can't spawn what isn't installed
2. **Platform awareness** - Only show compatible workers
3. **Rich metadata** - Help users understand what they're installing
4. **Clear dependencies** - Show what's required to build
5. **Progressive disclosure** - Start simple (catalog), then advanced (spawn)

## Testing

### Manual Test Flow

1. Start catalog service: `cd bin/80-hono-worker-catalog && pnpm dev`
2. Start hive UI: `cd bin/20_rbee_hive/ui/app && pnpm dev`
3. Open http://localhost:7836
4. Navigate to Worker Management
5. Should see "Worker Catalog" tab first
6. Should see 3 workers (CPU, CUDA, Metal)
7. Should see platform detection banner
8. Click "Install Worker" (logs to console for now)
9. Click trash icon to remove (logs to console for now)

---

**Status:** UI Complete ✅  
**Next:** Backend API implementation
