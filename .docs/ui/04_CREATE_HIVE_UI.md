# Part 4: Create Hive UI

**TEAM-293: Create dedicated UI for rbee-hive**

## Goal

Create a new React app for hive management (models and workers) that will be hosted by the rbee-hive binary.

## Purpose

**Hive UI is responsible for:**
- Model management (download, list, delete)
- Worker spawning and management
- Resource monitoring (VRAM, GPU usage)
- Hive configuration

**NOT responsible for:**
- ❌ Scheduling (that's queen's job)
- ❌ Hive lifecycle (that's keeper's job)

## Structure

```
frontend/apps/ui-rbee-hive/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── pages/
│   │   ├── ModelsPage.tsx        # Model download/list/delete
│   │   ├── WorkersPage.tsx       # Worker spawn/kill/status
│   │   ├── ResourcesPage.tsx     # GPU/VRAM monitoring
│   │   └── ConfigPage.tsx        # Hive configuration
│   ├── components/
│   │   ├── ModelCard.tsx
│   │   ├── WorkerCard.tsx
│   │   └── ResourceMonitor.tsx
│   └── lib/
│       └── hive-api.ts           # HTTP calls to local hive
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
└── README.md
```

## Step 1: Create Directory

```bash
cd /home/vince/Projects/llama-orch/frontend/apps
mkdir ui-rbee-hive
cd ui-rbee-hive
```

## Step 2: Create package.json

**File:** `frontend/apps/ui-rbee-hive/package.json`

```json
{
  "name": "@rbee/ui-rbee-hive",
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

## Step 3: Create vite.config.ts

**File:** `frontend/apps/ui-rbee-hive/vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 7836,  // Different from queen (7834) and keeper (5173)
    proxy: {
      '/api': {
        target: 'http://localhost:7835',  // rbee-hive API
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

## Step 4: Create Hive API Client

**File:** `frontend/apps/ui-rbee-hive/src/lib/hive-api.ts`

```typescript
// HTTP client for rbee-hive API
// Assumes hive is running on localhost:7835

const API_BASE = '/api';  // Proxied to localhost:7835 in dev

// ============================================================================
// MODELS
// ============================================================================

export interface Model {
  id: string;
  name: string;
  size_bytes: number;
  downloaded: boolean;
}

export async function listModels(): Promise<Model[]> {
  const response = await fetch(`${API_BASE}/models`);
  return await response.json();
}

export async function downloadModel(model: string): Promise<void> {
  const response = await fetch(`${API_BASE}/models/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to download model: ${await response.text()}`);
  }
}

export async function deleteModel(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/models/${id}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to delete model: ${await response.text()}`);
  }
}

// ============================================================================
// WORKERS
// ============================================================================

export interface Worker {
  pid: number;
  model: string;
  device: string;
  status: 'running' | 'stopped';
  vram_used_mb: number;
}

export async function listWorkers(): Promise<Worker[]> {
  const response = await fetch(`${API_BASE}/workers`);
  return await response.json();
}

export async function spawnWorker(model: string, device: string): Promise<void> {
  const response = await fetch(`${API_BASE}/workers/spawn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, device }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to spawn worker: ${await response.text()}`);
  }
}

export async function killWorker(pid: number): Promise<void> {
  const response = await fetch(`${API_BASE}/workers/${pid}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to kill worker: ${await response.text()}`);
  }
}

// ============================================================================
// RESOURCES
// ============================================================================

export interface GPUInfo {
  device_id: number;
  name: string;
  vram_total_mb: number;
  vram_used_mb: number;
  vram_free_mb: number;
  utilization_percent: number;
}

export async function getGPUInfo(): Promise<GPUInfo[]> {
  const response = await fetch(`${API_BASE}/resources/gpu`);
  return await response.json();
}

export interface HiveStatus {
  uptime_seconds: number;
  workers_count: number;
  models_count: number;
}

export async function getHiveStatus(): Promise<HiveStatus> {
  const response = await fetch(`${API_BASE}/status`);
  return await response.json();
}
```

## Step 5: Create ModelsPage

**File:** `frontend/apps/ui-rbee-hive/src/pages/ModelsPage.tsx`

```tsx
import React, { useEffect, useState } from 'react';
import { listModels, downloadModel, deleteModel, Model } from '@/lib/hive-api';

export function ModelsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [newModel, setNewModel] = useState('');

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const data = await listModels();
      setModels(data);
    } catch (error) {
      console.error('Failed to load models:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!newModel) return;

    try {
      await downloadModel(newModel);
      alert(`Model ${newModel} download started`);
      setNewModel('');
      loadModels();
    } catch (error) {
      alert(`Error: ${error}`);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this model?')) {
      return;
    }

    try {
      await deleteModel(id);
      alert('Model deleted');
      loadModels();
    } catch (error) {
      alert(`Error: ${error}`);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Models</h1>

      {/* Download Section */}
      <div className="card p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Download Model</h2>
        <div className="flex gap-3">
          <input
            type="text"
            placeholder="Model name (e.g., llama-2-7b)"
            value={newModel}
            onChange={(e) => setNewModel(e.target.value)}
            className="input flex-1"
          />
          <button onClick={handleDownload} className="btn btn-primary">
            Download
          </button>
        </div>
      </div>

      {/* Models List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {loading ? (
          <p>Loading models...</p>
        ) : models.length === 0 ? (
          <p>No models available</p>
        ) : (
          models.map((model) => (
            <div key={model.id} className="card p-4">
              <h3 className="font-semibold text-lg mb-2">{model.name}</h3>
              <p className="text-sm text-gray-600 mb-4">
                Size: {(model.size_bytes / 1024 / 1024 / 1024).toFixed(2)} GB
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => handleDelete(model.id)}
                  className="btn btn-danger btn-sm flex-1"
                >
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
```

## Step 6: Create WorkersPage

**File:** `frontend/apps/ui-rbee-hive/src/pages/WorkersPage.tsx`

```tsx
import React, { useEffect, useState } from 'react';
import { listWorkers, spawnWorker, killWorker, Worker } from '@/lib/hive-api';

export function WorkersPage() {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);
  const [model, setModel] = useState('');
  const [device, setDevice] = useState('cuda:0');

  useEffect(() => {
    loadWorkers();
    const interval = setInterval(loadWorkers, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadWorkers = async () => {
    try {
      const data = await listWorkers();
      setWorkers(data);
    } catch (error) {
      console.error('Failed to load workers:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSpawn = async () => {
    if (!model) {
      alert('Please enter a model name');
      return;
    }

    try {
      await spawnWorker(model, device);
      alert(`Worker spawned for ${model}`);
      setModel('');
      loadWorkers();
    } catch (error) {
      alert(`Error: ${error}`);
    }
  };

  const handleKill = async (pid: number) => {
    if (!confirm(`Kill worker ${pid}?`)) {
      return;
    }

    try {
      await killWorker(pid);
      alert(`Worker ${pid} killed`);
      loadWorkers();
    } catch (error) {
      alert(`Error: ${error}`);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Workers</h1>

      {/* Spawn Worker */}
      <div className="card p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Spawn Worker</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <input
            type="text"
            placeholder="Model name"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="input"
          />
          <input
            type="text"
            placeholder="Device (cuda:0)"
            value={device}
            onChange={(e) => setDevice(e.target.value)}
            className="input"
          />
          <button onClick={handleSpawn} className="btn btn-primary">
            Spawn
          </button>
        </div>
      </div>

      {/* Workers List */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {loading ? (
          <p>Loading workers...</p>
        ) : workers.length === 0 ? (
          <p>No workers running</p>
        ) : (
          workers.map((worker) => (
            <div key={worker.pid} className="card p-4">
              <div className="flex justify-between items-start mb-3">
                <div>
                  <h3 className="font-semibold text-lg">{worker.model}</h3>
                  <p className="text-sm text-gray-600">PID: {worker.pid}</p>
                </div>
                <span
                  className={`px-2 py-1 rounded text-xs ${
                    worker.status === 'running'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  }`}
                >
                  {worker.status}
                </span>
              </div>
              
              <div className="text-sm space-y-1 mb-4">
                <p>Device: {worker.device}</p>
                <p>VRAM: {worker.vram_used_mb} MB</p>
              </div>

              <button
                onClick={() => handleKill(worker.pid)}
                className="btn btn-danger btn-sm w-full"
              >
                Kill Worker
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
```

## Step 7: Create ResourcesPage

**File:** `frontend/apps/ui-rbee-hive/src/pages/ResourcesPage.tsx`

```tsx
import React, { useEffect, useState } from 'react';
import { getGPUInfo, getHiveStatus, GPUInfo, HiveStatus } from '@/lib/hive-api';

export function ResourcesPage() {
  const [gpus, setGpus] = useState<GPUInfo[]>([]);
  const [status, setStatus] = useState<HiveStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadResources();
    const interval = setInterval(loadResources, 2000);
    return () => clearInterval(interval);
  }, []);

  const loadResources = async () => {
    try {
      const [gpuData, statusData] = await Promise.all([
        getGPUInfo(),
        getHiveStatus(),
      ]);
      setGpus(gpuData);
      setStatus(statusData);
    } catch (error) {
      console.error('Failed to load resources:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Resources</h1>

      {/* Hive Status */}
      {status && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="card p-4">
            <h3 className="text-sm text-gray-600 mb-1">Uptime</h3>
            <p className="text-2xl font-bold">
              {Math.floor(status.uptime_seconds / 3600)}h
            </p>
          </div>
          <div className="card p-4">
            <h3 className="text-sm text-gray-600 mb-1">Workers</h3>
            <p className="text-2xl font-bold">{status.workers_count}</p>
          </div>
          <div className="card p-4">
            <h3 className="text-sm text-gray-600 mb-1">Models</h3>
            <p className="text-2xl font-bold">{status.models_count}</p>
          </div>
        </div>
      )}

      {/* GPU Info */}
      <h2 className="text-xl font-semibold mb-4">GPUs</h2>
      <div className="space-y-4">
        {loading ? (
          <p>Loading GPU info...</p>
        ) : gpus.length === 0 ? (
          <p>No GPUs detected</p>
        ) : (
          gpus.map((gpu) => (
            <div key={gpu.device_id} className="card p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="font-semibold text-lg">{gpu.name}</h3>
                  <p className="text-sm text-gray-600">Device {gpu.device_id}</p>
                </div>
                <span className="text-2xl font-bold">
                  {gpu.utilization_percent}%
                </span>
              </div>

              {/* VRAM Progress Bar */}
              <div className="mb-2">
                <div className="flex justify-between text-sm mb-1">
                  <span>VRAM Used</span>
                  <span>
                    {gpu.vram_used_mb} / {gpu.vram_total_mb} MB
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{
                      width: `${(gpu.vram_used_mb / gpu.vram_total_mb) * 100}%`,
                    }}
                  />
                </div>
              </div>

              {/* Free VRAM */}
              <p className="text-sm text-gray-600">
                Free: {gpu.vram_free_mb} MB
              </p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
```

## Step 8: Create App.tsx

**File:** `frontend/apps/ui-rbee-hive/src/App.tsx`

```tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ModelsPage } from './pages/ModelsPage';
import { WorkersPage } from './pages/WorkersPage';
import { ResourcesPage } from './pages/ResourcesPage';

export function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow">
          <div className="px-6 py-4">
            <h1 className="text-2xl font-bold">rbee Hive UI</h1>
            <nav className="mt-2 flex gap-4">
              <a href="/models" className="text-blue-600 hover:underline">
                Models
              </a>
              <a href="/workers" className="text-blue-600 hover:underline">
                Workers
              </a>
              <a href="/resources" className="text-blue-600 hover:underline">
                Resources
              </a>
            </nav>
          </div>
        </header>

        {/* Main Content */}
        <Routes>
          <Route path="/" element={<Navigate to="/models" replace />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/workers" element={<WorkersPage />} />
          <Route path="/resources" element={<ResourcesPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
```

## Step 9: Update pnpm-workspace.yaml

**File:** `/home/vince/Projects/llama-orch/pnpm-workspace.yaml`

```yaml
packages:
  - frontend/apps/commercial
  - frontend/apps/ui-queen-rbee
  - frontend/apps/ui-rbee-hive      # ✅ ADD THIS
  - frontend/apps/user-docs
  - frontend/packages/rbee-ui
  - frontend/packages/rbee-sdk
  - frontend/packages/rbee-react
  - frontend/packages/tailwind-config
  - bin/00_rbee_keeper/GUI
```

## Step 10: Install & Test

```bash
# Install dependencies
cd /home/vince/Projects/llama-orch
pnpm install

# Start dev server
pnpm --filter @rbee/ui-rbee-hive dev

# Build for production
pnpm --filter @rbee/ui-rbee-hive build
```

## Verification Checklist

- [ ] Directory created: `frontend/apps/ui-rbee-hive`
- [ ] `package.json` created
- [ ] `vite.config.ts` created
- [ ] `hive-api.ts` created (HTTP client)
- [ ] `ModelsPage.tsx` created
- [ ] `WorkersPage.tsx` created
- [ ] `ResourcesPage.tsx` created
- [ ] `App.tsx` created
- [ ] `pnpm-workspace.yaml` updated
- [ ] Dependencies installed
- [ ] Dev server runs on port 7836
- [ ] Build succeeds

## Expected Result

```
✅ Hive UI runs on http://localhost:7836
✅ Models page: download/list/delete
✅ Workers page: spawn/kill/status
✅ Resources page: GPU/VRAM monitoring
✅ Ready to be hosted by rbee-hive binary
```

## Next Steps

1. **Next:** `05_CREATE_WORKER_UIS.md` - Create worker-specific UIs
2. **Then:** `06_IFRAME_INTEGRATION.md` - Integrate all UIs via iframes

---

**Status:** 📋 READY TO IMPLEMENT
