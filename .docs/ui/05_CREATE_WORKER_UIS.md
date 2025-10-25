# Part 5: Create Worker UIs

**TEAM-293: Create specialized UIs for different worker types**

## Goal

Create worker-type-specific UIs that provide specialized interfaces for each worker type (LLM, ComfyUI, vLLM, etc.).

## Worker Types

According to `.arch/09_WORKER_TYPES_PART_10.md`:

1. **LLM Workers** (Candle-based) - Text generation
2. **ComfyUI Workers** - Image generation workflows
3. **vLLM Workers** - High-performance text generation
4. **Future:** Whisper (audio), FLUX (image), etc.

## UI Structure

```
frontend/apps/
├── ui-llm-worker-rbee/        # Text generation UI
├── ui-comfy-worker-rbee/      # ComfyUI workflow UI
└── ui-vllm-worker-rbee/       # vLLM performance UI
```

---

## 1. LLM Worker UI

### Purpose
- Live inference demo
- Text generation parameters
- Performance metrics
- Worker status

### Create Structure

```bash
cd /home/vince/Projects/llama-orch/frontend/apps
mkdir ui-llm-worker-rbee
```

### package.json

**File:** `frontend/apps/ui-llm-worker-rbee/package.json`

```json
{
  "name": "@rbee/ui-llm-worker-rbee",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@rbee-ui/styles": "workspace:*",
    "@rbee-ui/stories": "workspace:*",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
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

### vite.config.ts

**File:** `frontend/apps/ui-llm-worker-rbee/vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 7837,  // Different port for each worker type
    proxy: {
      '/api': {
        target: 'http://localhost:8080',  // Worker API port
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

### Main Page

**File:** `frontend/apps/ui-llm-worker-rbee/src/App.tsx`

```tsx
import React, { useState } from 'react';

export function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(100);

  const handleInfer = async () => {
    setLoading(true);
    setResponse('');

    try {
      const res = await fetch('/api/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          temperature,
          max_tokens: maxTokens,
        }),
      });

      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      setResponse(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">LLM Worker</h1>

        {/* Inference Demo */}
        <div className="card p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Live Inference</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={4}
                className="w-full p-3 border rounded"
                placeholder="Enter your prompt..."
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Temperature: {temperature}
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Max Tokens: {maxTokens}
                </label>
                <input
                  type="range"
                  min="10"
                  max="500"
                  step="10"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            <button
              onClick={handleInfer}
              disabled={loading || !prompt}
              className="btn btn-primary w-full"
            >
              {loading ? 'Generating...' : 'Generate'}
            </button>

            {response && (
              <div className="mt-4 p-4 bg-gray-100 rounded">
                <h3 className="font-semibold mb-2">Response:</h3>
                <p className="whitespace-pre-wrap">{response}</p>
              </div>
            )}
          </div>
        </div>

        {/* Worker Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className="card p-4">
            <h3 className="text-sm text-gray-600 mb-1">Model</h3>
            <p className="text-lg font-semibold">llama-2-7b</p>
          </div>
          <div className="card p-4">
            <h3 className="text-sm text-gray-600 mb-1">Device</h3>
            <p className="text-lg font-semibold">cuda:0</p>
          </div>
          <div className="card p-4">
            <h3 className="text-sm text-gray-600 mb-1">VRAM</h3>
            <p className="text-lg font-semibold">6.2 GB</p>
          </div>
        </div>
      </div>
    </div>
  );
}
```

---

## 2. ComfyUI Worker UI

### Purpose
- Workflow visualization
- Node graph display
- Image generation queue
- Worker-specific settings

### Create Structure

```bash
cd /home/vince/Projects/llama-orch/frontend/apps
mkdir ui-comfy-worker-rbee
```

### package.json

**File:** `frontend/apps/ui-comfy-worker-rbee/package.json`

```json
{
  "name": "@rbee/ui-comfy-worker-rbee",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@rbee-ui/styles": "workspace:*",
    "@rbee-ui/stories": "workspace:*",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
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

### vite.config.ts

**File:** `frontend/apps/ui-comfy-worker-rbee/vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 7838,  // Different port
    proxy: {
      '/api': {
        target: 'http://localhost:8188',  // ComfyUI default port
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

### Main Page

**File:** `frontend/apps/ui-comfy-worker-rbee/src/App.tsx`

```tsx
import React, { useState, useEffect } from 'react';

interface QueueItem {
  id: string;
  workflow: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
}

export function App() {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState('');

  useEffect(() => {
    // Poll queue status
    const interval = setInterval(async () => {
      try {
        const response = await fetch('/api/queue');
        const data = await response.json();
        setQueue(data.queue);
      } catch (error) {
        console.error('Failed to fetch queue:', error);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const handleSubmitWorkflow = async () => {
    try {
      await fetch('/api/prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          workflow: selectedWorkflow,
        }),
      });
      alert('Workflow submitted');
    } catch (error) {
      alert(`Error: ${error}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">ComfyUI Worker</h1>

        <div className="grid grid-cols-2 gap-6">
          {/* Workflow Submission */}
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4">Submit Workflow</h2>
            <div className="space-y-4">
              <textarea
                value={selectedWorkflow}
                onChange={(e) => setSelectedWorkflow(e.target.value)}
                rows={10}
                className="w-full p-3 border rounded font-mono text-sm"
                placeholder="Paste ComfyUI workflow JSON..."
              />
              <button
                onClick={handleSubmitWorkflow}
                className="btn btn-primary w-full"
              >
                Submit
              </button>
            </div>
          </div>

          {/* Queue Status */}
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4">Queue</h2>
            <div className="space-y-3">
              {queue.length === 0 ? (
                <p className="text-gray-500">Queue is empty</p>
              ) : (
                queue.map((item) => (
                  <div key={item.id} className="border p-3 rounded">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-semibold">{item.workflow}</span>
                      <span
                        className={`px-2 py-1 rounded text-xs ${
                          item.status === 'running'
                            ? 'bg-blue-100 text-blue-800'
                            : item.status === 'completed'
                            ? 'bg-green-100 text-green-800'
                            : item.status === 'failed'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {item.status}
                      </span>
                    </div>
                    {item.status === 'running' && (
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full"
                          style={{ width: `${item.progress}%` }}
                        />
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Worker Stats */}
        <div className="grid grid-cols-4 gap-4 mt-6">
          <div className="card p-4">
            <h3 className="text-sm text-gray-600 mb-1">Type</h3>
            <p className="text-lg font-semibold">ComfyUI</p>
          </div>
          <div className="card p-4">
            <h3 className="text-sm text-gray-600 mb-1">Queue Size</h3>
            <p className="text-lg font-semibold">{queue.length}</p>
          </div>
          <div className="card p-4">
            <h3 className="text-sm text-gray-600 mb-1">Running</h3>
            <p className="text-lg font-semibold">
              {queue.filter((q) => q.status === 'running').length}
            </p>
          </div>
          <div className="card p-4">
            <h3 className="text-sm text-gray-600 mb-1">Completed</h3>
            <p className="text-lg font-semibold">
              {queue.filter((q) => q.status === 'completed').length}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
```

---

## 3. vLLM Worker UI

### Purpose
- Performance metrics
- Throughput monitoring
- Batch inference
- Advanced settings

### Create Structure

```bash
cd /home/vince/Projects/llama-orch/frontend/apps
mkdir ui-vllm-worker-rbee
```

### Configuration similar to LLM worker but with vLLM-specific features

**Key differences:**
- Batch inference support
- Throughput graphs
- Request queuing visualization
- KV cache metrics

---

## Update pnpm-workspace.yaml

**File:** `/home/vince/Projects/llama-orch/pnpm-workspace.yaml`

```yaml
packages:
  - frontend/apps/commercial
  - frontend/apps/ui-queen-rbee
  - frontend/apps/ui-rbee-hive
  - frontend/apps/ui-llm-worker-rbee    # ✅ ADD
  - frontend/apps/ui-comfy-worker-rbee  # ✅ ADD
  - frontend/apps/ui-vllm-worker-rbee   # ✅ ADD
  - frontend/apps/user-docs
  - frontend/packages/rbee-ui
  - frontend/packages/rbee-sdk
  - frontend/packages/rbee-react
  - frontend/packages/tailwind-config
  - bin/00_rbee_keeper/GUI
```

## Install & Build

```bash
cd /home/vince/Projects/llama-orch
pnpm install

# Build all worker UIs
pnpm --filter "@rbee/ui-*-worker-rbee" build
```

## Verification Checklist

- [ ] ui-llm-worker-rbee created
- [ ] ui-comfy-worker-rbee created
- [ ] ui-vllm-worker-rbee created
- [ ] pnpm-workspace.yaml updated
- [ ] Dependencies installed
- [ ] All UIs build successfully

## Port Assignments

| UI | Dev Port | Production |
|----|----------|------------|
| keeper GUI | 5173 | Tauri app |
| queen-rbee | 7834 | 7833/ui |
| rbee-hive | 7836 | 7835/ui |
| llm-worker | 7837 | 8080/ui |
| comfy-worker | 7838 | 8188/ui |
| vllm-worker | 7839 | 8000/ui |

## Expected Result

```
✅ 3 worker-specific UIs created
✅ Each has specialized interface
✅ All ready to be hosted by worker binaries
✅ Will be displayed in keeper GUI via iframe
```

## Next Steps

1. **Next:** `06_IFRAME_INTEGRATION.md` - Integrate all UIs via iframes in keeper
2. **Then:** `07_SIDEBAR_IMPLEMENTATION.md` - Dynamic sidebar based on heartbeats

---

**Status:** 📋 READY TO IMPLEMENT
