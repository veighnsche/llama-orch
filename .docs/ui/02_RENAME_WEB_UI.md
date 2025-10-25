# Part 2: Rename web-ui to ui-queen-rbee

**TEAM-293: Separate queen UI from keeper UI**

## Goal

Rename `frontend/apps/web-ui` to `frontend/apps/ui-queen-rbee` and remove all keeper-related code from it.

## Current Structure

```
frontend/apps/web-ui/
├── src/
│   ├── components/
│   │   ├── AppSidebar.tsx        # Has keeper + queen items
│   │   └── CommandsSidebar.tsx
│   ├── pages/
│   │   ├── KeeperPage.tsx        # ❌ REMOVE (move to keeper GUI)
│   │   ├── QueuePage.tsx         # ✅ KEEP (queen functionality)
│   │   └── InferencePage.tsx     # ✅ KEEP (queen functionality)
│   └── App.tsx
└── package.json                   # ❌ Uses generic rbee-sdk (DEPRECATED)
```

## New Structure

```
frontend/apps/10_queen_rbee/
├── src/
│   ├── components/
│   │   ├── AppSidebar.tsx        # Only queen items
│   │   └── CommandsSidebar.tsx
│   ├── pages/
│   │   ├── SchedulingPage.tsx    # ✅ NEW (queen scheduling)
│   │   ├── JobQueuePage.tsx      # ✅ NEW (job management)
│   │   ├── AnalyticsPage.tsx     # ✅ NEW (queen analytics)
│   │   └── HiveSelectionPage.tsx # ✅ NEW (routing logic)
│   └── App.tsx
└── package.json                   # ✅ Uses @rbee/queen-rbee-react
```

## Step 1: Rename Directory

```bash
cd /home/vince/Projects/llama-orch/frontend/apps
mv web-ui 10_queen_rbee
```

## Step 2: Update package.json

**File:** `frontend/apps/10_queen_rbee/package.json`

```json
{
  "name": "@rbee/10-queen-ui",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx"
  },
  "dependencies": {
    "@rbee-ui/styles": "workspace:*",
    "@rbee-ui/stories": "workspace:*",
    "@rbee/queen-rbee-react": "workspace:*",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.22.0",
    "zustand": "^4.5.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.56",
    "@types/react-dom": "^18.2.19",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.2.2",
    "vite": "^5.1.4"
  }
}
```

**Changes:**
- ❌ Removed `@rbee/rbee-sdk` dependency (generic, deprecated)
- ✅ Added `@rbee/queen-rbee-react` (specialized for queen API)
- ✅ Uses React hooks: `useJobs()`, `useQueenStatus()`, etc.

## Step 3: Update pnpm-workspace.yaml

**File:** `/home/vince/Projects/llama-orch/pnpm-workspace.yaml`

```yaml
packages:
  - frontend/apps/commercial
  - frontend/apps/user-docs
  - frontend/apps/00_rbee_keeper
  - frontend/apps/10_queen_rbee      # ✅ RENAMED
  - frontend/apps/20_rbee_hive
  - frontend/packages/rbee-ui
  - frontend/packages/tailwind-config
  - frontend/packages/10_queen_rbee/*
  - frontend/packages/20_rbee_hive/*
  - frontend/packages/30_workers/*
```

## Step 4: Remove Keeper Page

**Delete:** `frontend/apps/10_queen_rbee/src/pages/KeeperPage.tsx`

This page will be moved to `frontend/apps/00_rbee_keeper/src/pages/` (see Part 3)

## Step 5: Update AppSidebar.tsx

**File:** `frontend/apps/ui-queen-rbee/src/components/AppSidebar.tsx`

**Before:**
```tsx
const menuItems = [
  { label: 'Keeper', path: '/keeper' },        // ❌ REMOVE
  { label: 'Queue', path: '/queue' },          // ✅ KEEP
  { label: 'Inference', path: '/inference' },  // ✅ KEEP
];
```

**After:**
```tsx
const menuItems = [
  { label: 'Scheduling', path: '/scheduling' },
  { label: 'Job Queue', path: '/queue' },
  { label: 'Analytics', path: '/analytics' },
  { label: 'Hive Selection', path: '/hives' },
];
```

## Step 6: Update App.tsx Routes

**File:** `frontend/apps/ui-queen-rbee/src/App.tsx`

**Before:**
```tsx
<Routes>
  <Route path="/keeper" element={<KeeperPage />} />  {/* ❌ REMOVE */}
  <Route path="/queue" element={<QueuePage />} />
  <Route path="/inference" element={<InferencePage />} />
</Routes>
```

**After:**
```tsx
<Routes>
  <Route path="/" element={<Navigate to="/scheduling" replace />} />
  <Route path="/scheduling" element={<SchedulingPage />} />
  <Route path="/queue" element={<JobQueuePage />} />
  <Route path="/analytics" element={<AnalyticsPage />} />
  <Route path="/hives" element={<HiveSelectionPage />} />
</Routes>
```

## Step 7: Replace rbee-sdk with queen-rbee-react

**Find and replace all instances:**

**Before (DEPRECATED):**
```tsx
import { useHeartbeat } from '@rbee/rbee-sdk';

const { data, loading } = useHeartbeat();
```

**After (NEW):**
```tsx
import { useJobs, useQueenStatus } from '@rbee/queen-rbee-react';

const { jobs, loading } = useJobs();
const { status } = useQueenStatus();
```

**Rationale:** 
- Specialized SDK for queen API only
- Type-safe React hooks
- No coupling with hive/worker APIs
- Easier to maintain and test

## Step 8: Create New Queen-Specific Pages

### SchedulingPage.tsx

**File:** `frontend/apps/ui-queen-rbee/src/pages/SchedulingPage.tsx`

```tsx
import { PageContainer } from '@rbee-ui/stories';

export function SchedulingPage() {
  return (
    <PageContainer>
      <h1 className="text-3xl font-bold mb-6">Inference Scheduling</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Active Jobs</h2>
          {/* Job list */}
        </div>
        
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Hive Selection</h2>
          {/* Hive selection logic */}
        </div>
      </div>
    </PageContainer>
  );
}
```

### JobQueuePage.tsx

**File:** `frontend/apps/ui-queen-rbee/src/pages/JobQueuePage.tsx`

```tsx
import { PageContainer } from '@rbee-ui/stories';

export function JobQueuePage() {
  return (
    <PageContainer>
      <h1 className="text-3xl font-bold mb-6">Job Queue</h1>
      
      <div className="space-y-4">
        {/* Queue visualization */}
        {/* Job status */}
        {/* Historical jobs */}
      </div>
    </PageContainer>
  );
}
```

### AnalyticsPage.tsx

**File:** `frontend/apps/ui-queen-rbee/src/pages/AnalyticsPage.tsx`

```tsx
import { PageContainer } from '@rbee-ui/stories';

export function AnalyticsPage() {
  return (
    <PageContainer>
      <h1 className="text-3xl font-bold mb-6">Analytics</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card p-6">
          <h3 className="text-lg font-semibold mb-2">Jobs Processed</h3>
          {/* Metric */}
        </div>
        
        <div className="card p-6">
          <h3 className="text-lg font-semibold mb-2">Avg Response Time</h3>
          {/* Metric */}
        </div>
        
        <div className="card p-6">
          <h3 className="text-lg font-semibold mb-2">Success Rate</h3>
          {/* Metric */}
        </div>
      </div>
    </PageContainer>
  );
}
```

### HiveSelectionPage.tsx

**File:** `frontend/apps/ui-queen-rbee/src/pages/HiveSelectionPage.tsx`

```tsx
import { PageContainer } from '@rbee-ui/stories';

export function HiveSelectionPage() {
  return (
    <PageContainer>
      <h1 className="text-3xl font-bold mb-6">Hive Selection</h1>
      
      <div className="space-y-4">
        {/* List of available hives */}
        {/* Selection algorithm visualization */}
        {/* Load balancing stats */}
      </div>
    </PageContainer>
  );
}
```

## Step 9: Update vite.config.ts

**File:** `frontend/apps/ui-queen-rbee/vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 7834,  // Different port from keeper (5173)
    proxy: {
      '/api': {
        target: 'http://localhost:7833',  // queen-rbee API
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    emptyOutDir: true,
  },
});
```

## Step 10: Update Documentation

**File:** `frontend/apps/ui-queen-rbee/README.md`

```markdown
# queen-rbee UI

**Type:** React Static Build  
**Hosted by:** queen-rbee binary  
**Port:** 7834 (dev) / 7833/ui (production)  
**Purpose:** Inference scheduling and job management

## Features

- Inference request scheduling
- Job queue visualization
- Hive selection logic
- Performance analytics

## Development

```bash
pnpm dev
```

## Build

```bash
pnpm build
```

Output: `dist/` directory (served by queen-rbee binary)

## NOT Included

- ❌ rbee-keeper operations (moved to keeper GUI)
- ❌ Model management (that's hive UI)
- ❌ Worker management (that's hive UI)

## Architecture

This UI is embedded in queen-rbee binary and displayed in keeper GUI via iframe.
```

## Step 11: Reinstall Dependencies

```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

## Verification Checklist

- [ ] Directory renamed: `web-ui` → `ui-queen-rbee`
- [ ] `package.json` updated (no rbee-sdk)
- [ ] `pnpm-workspace.yaml` updated
- [ ] KeeperPage.tsx removed
- [ ] AppSidebar.tsx updated (no keeper items)
- [ ] App.tsx routes updated
- [ ] All rbee-sdk imports removed
- [ ] New pages created (Scheduling, Queue, Analytics, Hives)
- [ ] vite.config.ts updated (port 7834)
- [ ] README.md updated
- [ ] Dependencies installed
- [ ] Dev server runs: `pnpm --filter @rbee/ui-queen-rbee dev`

## Expected Result

```
✅ ui-queen-rbee runs on http://localhost:7834
✅ No keeper-related pages
✅ No rbee-sdk dependency
✅ Only queen-specific features
✅ Ready to be hosted by queen-rbee binary
```

## Next Steps

1. **Next:** `03_EXTRACT_KEEPER_PAGE.md` - Move keeper page to GUI
2. **Then:** `04_CREATE_HIVE_UI.md` - Create hive UI

---

**Status:** 📋 READY TO IMPLEMENT
