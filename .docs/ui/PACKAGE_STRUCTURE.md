# Package Structure: Specialized SDKs per Binary

**TEAM-293: Each binary gets its own SDK and React hooks**

## Design Principle

**Specialization:** Each binary (except keeper) has its own SDK package for its HTTP API.

**Why:** 
- No generic "rbee-sdk" that tries to do everything
- Each SDK is tailored to its binary's API
- Clear ownership and versioning
- Type-safe per-binary

## Old Structure (DEPRECATED)

```
❌ frontend/packages/rbee-sdk/        # Generic, tried to cover all binaries
❌ frontend/packages/rbee-react/      # Generic React hooks
```

**Problems:**
- One package tried to handle queen + hive + workers
- Tight coupling between unrelated APIs
- Version conflicts (queen API changes break hive UI)
- Unclear ownership

## New Structure

```
bin/
├── 10_queen_rbee/ui/packages/        # Queen packages
│   ├── queen-rbee-sdk/               # HTTP client for queen API
│   └── queen-rbee-react/             # React hooks for queen SDK
│
├── 20_rbee_hive/ui/packages/         # Hive packages
│   ├── rbee-hive-sdk/                # HTTP client for hive API
│   └── rbee-hive-react/              # React hooks for hive SDK
│
└── 30_llm_worker_rbee/ui/packages/   # Worker packages
    ├── llm-worker-sdk/               # HTTP client for LLM worker API
    └── llm-worker-react/             # React hooks for LLM worker SDK

frontend/packages/
├── rbee-ui/                          # ✅ Shared UI components (unchanged)
└── tailwind-config/                  # ✅ Shared Tailwind config (unchanged)
```

## Keeper Exception

**rbee-keeper does NOT have an SDK package.**

**Why:** 
- Keeper has NO HTTP API (only CLI)
- Keeper GUI uses Tauri commands (Rust backend)
- No need for HTTP client

**Keeper GUI uses:**
- Tauri commands directly (`invoke('queen_start')`)
- No SDK package needed

## Package Naming Convention

| Binary | SDK Package | React Package |
|--------|-------------|---------------|
| `bin/10_queen_rbee/` | `@rbee/queen-rbee-sdk` | `@rbee/queen-rbee-react` |
| `bin/20_rbee_hive/` | `@rbee/rbee-hive-sdk` | `@rbee/rbee-hive-react` |
| `bin/30_llm_worker_rbee/` | `@rbee/llm-worker-sdk` | `@rbee/llm-worker-react` |
| `bin/30_comfy_worker_rbee/` | `@rbee/comfy-worker-sdk` | `@rbee/comfy-worker-react` |
| `bin/30_vllm_worker_rbee/` | `@rbee/vllm-worker-sdk` | `@rbee/vllm-worker-react` |

## SDK Responsibilities

### What SDKs Do

✅ **HTTP client for binary's API**
- Type-safe API calls
- Request/response types
- Error handling
- Base URL configuration

✅ **Only HTTP, no WASM**
- Simple `fetch()` calls
- No Rust compilation
- Fast development

### What SDKs Don't Do

❌ No business logic
❌ No state management
❌ No React components
❌ No WASM bindings

## React Package Responsibilities

### What React Packages Do

✅ **React hooks wrapping SDK**
- `useQueenStatus()`, `useHiveModels()`, etc.
- State management (via Zustand)
- Loading/error states
- Auto-refresh logic

### What React Packages Don't Do

❌ No UI components (use `@rbee/rbee-ui` instead)
❌ No HTTP calls (use SDK instead)

## Example: Queen Packages

### queen-rbee-sdk

**File:** `frontend/packages/10_queen_rbee/queen-rbee-sdk/src/index.ts`

```typescript
// HTTP client for queen-rbee API
// Base URL: http://localhost:7833

export interface Job {
  id: string;
  operation: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
}

export async function listJobs(): Promise<Job[]> {
  const response = await fetch('http://localhost:7833/api/jobs');
  return await response.json();
}

export async function getJob(id: string): Promise<Job> {
  const response = await fetch(`http://localhost:7833/api/jobs/${id}`);
  return await response.json();
}

export async function submitInference(prompt: string): Promise<{ job_id: string }> {
  const response = await fetch('http://localhost:7833/api/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });
  return await response.json();
}
```

**package.json:**
```json
{
  "name": "@rbee/queen-rbee-sdk",
  "version": "0.1.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": "./dist/index.js"
  }
}
```

### queen-rbee-react

**File:** `frontend/packages/10_queen_rbee/queen-rbee-react/src/index.ts`

```typescript
import { useState, useEffect } from 'react';
import { listJobs, getJob, Job } from '@rbee/queen-rbee-sdk';

export function useJobs() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const data = await listJobs();
        setJobs(data);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    };

    fetchJobs();
    const interval = setInterval(fetchJobs, 3000); // Auto-refresh
    return () => clearInterval(interval);
  }, []);

  return { jobs, loading, error };
}

export function useJob(id: string) {
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchJob = async () => {
      const data = await getJob(id);
      setJob(data);
      setLoading(false);
    };
    fetchJob();
  }, [id]);

  return { job, loading };
}
```

**package.json:**
```json
{
  "name": "@rbee/queen-rbee-react",
  "version": "0.1.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "peerDependencies": {
    "react": "^18.2.0"
  },
  "dependencies": {
    "@rbee/queen-rbee-sdk": "workspace:*"
  }
}
```

## Example: Hive Packages

### rbee-hive-sdk

**File:** `frontend/packages/20_rbee_hive/rbee-hive-sdk/src/index.ts`

```typescript
// HTTP client for rbee-hive API
// Base URL: http://localhost:7835

export interface Model {
  id: string;
  name: string;
  size_bytes: number;
}

export interface Worker {
  pid: number;
  model: string;
  device: string;
}

export async function listModels(): Promise<Model[]> {
  const response = await fetch('http://localhost:7835/api/models');
  return await response.json();
}

export async function downloadModel(model: string): Promise<void> {
  await fetch('http://localhost:7835/api/models/download', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  });
}

export async function listWorkers(): Promise<Worker[]> {
  const response = await fetch('http://localhost:7835/api/workers');
  return await response.json();
}

export async function spawnWorker(model: string, device: string): Promise<void> {
  await fetch('http://localhost:7835/api/workers/spawn', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, device }),
  });
}
```

### rbee-hive-react

**File:** `frontend/packages/20_rbee_hive/rbee-hive-react/src/index.ts`

```typescript
import { useState, useEffect } from 'react';
import { listModels, listWorkers, Model, Worker } from '@rbee/rbee-hive-sdk';

export function useModels() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModels = async () => {
      const data = await listModels();
      setModels(data);
      setLoading(false);
    };
    fetchModels();
  }, []);

  return { models, loading };
}

export function useWorkers() {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchWorkers = async () => {
      const data = await listWorkers();
      setWorkers(data);
      setLoading(false);
    };

    fetchWorkers();
    const interval = setInterval(fetchWorkers, 2000);
    return () => clearInterval(interval);
  }, []);

  return { workers, loading };
}
```

## UI Usage

### Queen UI

**File:** `frontend/apps/10_queen_rbee/src/pages/JobsPage.tsx`

```tsx
import { useJobs } from '@rbee/queen-rbee-react';

export function JobsPage() {
  const { jobs, loading, error } = useJobs();

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;

  return (
    <div>
      {jobs.map(job => (
        <div key={job.id}>{job.operation}</div>
      ))}
    </div>
  );
}
```

### Hive UI

**File:** `frontend/apps/20_rbee_hive/src/pages/ModelsPage.tsx`

```tsx
import { useModels } from '@rbee/rbee-hive-react';

export function ModelsPage() {
  const { models, loading } = useModels();

  if (loading) return <p>Loading...</p>;

  return (
    <div>
      {models.map(model => (
        <div key={model.id}>{model.name}</div>
      ))}
    </div>
  );
}
```

## pnpm-workspace.yaml

```yaml
packages:
  # Apps
  - frontend/apps/commercial
  - frontend/apps/user-docs
  - frontend/apps/00_rbee_keeper
  - frontend/apps/10_queen_rbee
  - frontend/apps/20_rbee_hive
  - frontend/apps/30_*_worker_rbee
  
  # Shared packages
  - frontend/packages/rbee-ui
  - frontend/packages/tailwind-config
  
  # Specialized packages (per binary)
  - frontend/packages/10_queen_rbee/*
  - frontend/packages/20_rbee_hive/*
  - frontend/packages/30_workers/*
```

## Migration from Old Structure

### Step 1: Delete Old Packages

```bash
rm -rf frontend/packages/rbee-sdk
rm -rf frontend/packages/rbee-react
```

### Step 2: Create New Package Structure

```bash
# Queen packages
mkdir -p frontend/packages/10_queen_rbee/queen-rbee-sdk
mkdir -p frontend/packages/10_queen_rbee/queen-rbee-react

# Hive packages
mkdir -p frontend/packages/20_rbee_hive/rbee-hive-sdk
mkdir -p frontend/packages/20_rbee_hive/rbee-hive-react

# Worker packages
mkdir -p frontend/packages/30_workers/llm-worker-sdk
mkdir -p frontend/packages/30_workers/llm-worker-react
mkdir -p frontend/packages/30_workers/comfy-worker-sdk
mkdir -p frontend/packages/30_workers/comfy-worker-react
```

### Step 3: Update pnpm-workspace.yaml

Remove:
```yaml
- frontend/packages/rbee-sdk
- frontend/packages/rbee-react
```

Add:
```yaml
- frontend/packages/10_queen_rbee/*
- frontend/packages/20_rbee_hive/*
- frontend/packages/30_workers/*
```

### Step 4: Update UI Dependencies

**Queen UI:**
```json
{
  "dependencies": {
    "@rbee/queen-rbee-react": "workspace:*"
  }
}
```

**Hive UI:**
```json
{
  "dependencies": {
    "@rbee/rbee-hive-react": "workspace:*"
  }
}
```

### Step 5: Update Imports

**Before:**
```typescript
import { useHeartbeat } from '@rbee/rbee-react';
```

**After:**
```typescript
import { useJobs } from '@rbee/queen-rbee-react';
import { useModels } from '@rbee/rbee-hive-react';
```

## Benefits

✅ **Clear ownership:** Each binary owns its SDK  
✅ **Independent versioning:** Queen API changes don't affect hive  
✅ **Type safety:** Each SDK has its own types  
✅ **No coupling:** Hive UI can't accidentally call queen API  
✅ **Easier testing:** Test each SDK independently  
✅ **Smaller bundles:** UIs only import what they need

## Summary

| Component | Has SDK? | Why |
|-----------|----------|-----|
| **rbee-keeper** | ❌ No | No HTTP API, uses Tauri commands |
| **queen-rbee** | ✅ Yes | Has HTTP API for job management |
| **rbee-hive** | ✅ Yes | Has HTTP API for models/workers |
| **Workers** | ✅ Yes | Each has HTTP API for inference |

**Key Rule:** One binary = One SDK = One React package (except keeper)

---

**Status:** 📋 STRUCTURE DEFINED  
**Impact:** Complete reorganization of SDK packages for clarity
