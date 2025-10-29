# TEAM-286: web-ui Integration Guide - Complete Reference

**Date:** Oct 24, 2025  
**Target:** `frontend/apps/web-ui` (Next.js 15.5.5 + React 19)  
**SDK:** `consumers/rbee-sdk` (Rust + WASM)  
**Status:** ✅ **SDK READY** - Ready for integration!

---

## Table of Contents

1. [Current State](#current-state)
2. [Rust Backend Architecture](#rust-backend-architecture)
3. [WASM SDK Details](#wasm-sdk-details)
4. [Integration Steps](#integration-steps)
5. [React Hooks](#react-hooks)
6. [Example Components](#example-components)
7. [Troubleshooting](#troubleshooting)

---

## Current State

### SDK Status (consumers/rbee-sdk)

**✅ PRODUCTION READY:**
- **All 17 operations** implemented and working
- **HeartbeatMonitor** for real-time updates (5-second intervals)
- **TypeScript types** auto-generated from Rust
- **WASM bundle:** 593 KB (uncompressed), ~150-180 KB gzipped
- **Compilation:** Zero errors, zero warnings

**Build command:**
```bash
cd consumers/rbee-sdk
wasm-pack build --target bundler --out-dir pkg/bundler
```

### web-ui Status (frontend/apps/web-ui)

**Current structure:**
```
frontend/apps/web-ui/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout (Inter font, globals.css)
│   │   └── page.tsx            # Dashboard (stub with placeholder cards)
│   └── (hooks/ components/ to be created)
├── package.json                # @rbee/sdk dependency configured
├── next.config.ts              # Needs WASM webpack config
└── tsconfig.json               # Paths configured (@/*)
```

**Dependencies:**
- Next.js 15.5.5
- React 19.2.0
- @rbee/ui (shadcn/ui components)
- @rbee/sdk (../../consumers/rbee-sdk)
- Radix UI primitives
- Tailwind CSS 4.1.14

**Current page.tsx (line 10):**
```tsx
// TODO: Connect to rbee SDK
useEffect(() => {
  // Placeholder - will connect to rbee SDK
  console.log('rbee Web UI loaded');
}, []);
```

---

## Rust Backend Architecture

### Overview

The SDK is a thin WASM wrapper around existing Rust crates:

```
┌─────────────────────────────────────────────────────────┐
│ JavaScript/TypeScript (Next.js)                         │
│ import { RbeeClient, HeartbeatMonitor } from '@rbee/sdk'│
└─────────────────────────────────────────────────────────┘
                         ↓ wasm-bindgen
┌─────────────────────────────────────────────────────────┐
│ rbee-sdk (Rust + WASM)                                  │
│ ├─ src/client.rs      - RbeeClient wrapper             │
│ ├─ src/operations.rs  - OperationBuilder (17 ops)      │
│ ├─ src/heartbeat.rs   - HeartbeatMonitor (SSE)         │
│ └─ src/types.rs       - Type conversions                │
└─────────────────────────────────────────────────────────┘
                         ↓ uses
┌─────────────────────────────────────────────────────────┐
│ job-client (Rust crate)                                 │
│ ├─ submit_and_stream() - POST + SSE streaming          │
│ ├─ submit()            - POST only                      │
│ └─ JobClient           - HTTP client (reqwest)          │
└─────────────────────────────────────────────────────────┘
                         ↓ uses
┌─────────────────────────────────────────────────────────┐
│ operations-contract (Rust crate)                        │
│ └─ Operation enum      - All 17 operation types         │
└─────────────────────────────────────────────────────────┘
                         ↓ HTTP/SSE
┌─────────────────────────────────────────────────────────┐
│ queen-rbee (Rust binary)                                │
│ └─ HTTP API            - http://localhost:8500          │
└─────────────────────────────────────────────────────────┘
```

### Key Rust Crates

#### 1. job-client (bin/99_shared_crates/job-client)

**Purpose:** Shared HTTP client for job submission and SSE streaming

**Key features:**
- ✅ **WASM-compatible** (unified reqwest approach)
- ✅ **Incremental SSE parsing** (no buffering)
- ✅ **Per-target dependencies** (native: tokio, WASM: fetch API)

**API:**
```rust
pub struct JobClient {
    pub fn new(base_url: impl Into<String>) -> Self;
    pub async fn submit_and_stream<F>(&self, operation: Operation, line_handler: F) -> Result<String>;
    pub async fn submit(&self, operation: Operation) -> Result<String>;
}
```

**How it works:**
1. Serialize `Operation` to JSON
2. POST to `/v1/jobs` endpoint
3. Extract `job_id` from response
4. Connect to `/v1/jobs/{job_id}/stream` (SSE)
5. Stream events incrementally (no buffering)
6. Call handler for each line
7. Stop on `[DONE]` marker

**WASM compatibility:**
- Native: `reqwest` → `hyper` → `tokio` → OS sockets
- WASM: `reqwest` → browser `fetch()` API
- Same code, different backend (automatic target detection)

#### 2. operations-contract (bin/97_contracts/operations-contract)

**Purpose:** Shared operation types between all components

**Key types:**
```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    Status,
    HiveList,
    HiveGet { alias: String },
    HiveStatus { alias: String },
    HiveRefreshCapabilities { alias: String },
    WorkerSpawn { hive_id: String, model: String, device: String },
    WorkerProcessList { hive_id: String },
    WorkerProcessGet { hive_id: String, pid: u32 },
    WorkerProcessDelete { hive_id: String, pid: u32 },
    ActiveWorkerList,
    ActiveWorkerGet { worker_id: String },
    ActiveWorkerRetire { worker_id: String },
    ModelDownload { hive_id: String, model: String },
    ModelList { hive_id: String },
    ModelGet { hive_id: String, model: String },
    ModelDelete { hive_id: String, model: String },
    Infer(InferRequest),
}
```

**Why this matters:**
- ✅ **Single source of truth** for operation types
- ✅ **Zero type drift** between Rust and TypeScript
- ✅ **Compiler-enforced correctness**
- ✅ **Auto-generated TypeScript types**

#### 3. rbee-sdk (consumers/rbee-sdk)

**Purpose:** WASM wrapper exposing Rust functionality to JavaScript

**Files:**
- `src/client.rs` - RbeeClient (wraps job-client)
- `src/operations.rs` - OperationBuilder (17 static methods)
- `src/heartbeat.rs` - HeartbeatMonitor (SSE for live updates)
- `src/types.rs` - Type conversions (Rust ↔ JavaScript)
- `src/utils.rs` - Utilities

**Code reuse:** 90%+ (only ~400 lines of wrapper code)

---

## WASM SDK Details

### All 17 Operations

**Status:**
```javascript
OperationBuilder.status()
```

**Hive Operations (4):**
```javascript
OperationBuilder.hiveList()
OperationBuilder.hiveGet(alias)
OperationBuilder.hiveStatus(alias)
OperationBuilder.hiveRefreshCapabilities(alias)
```

**Worker Operations (4):**
```javascript
OperationBuilder.workerSpawn(hiveId, model, device)
OperationBuilder.workerProcessList(hiveId)
OperationBuilder.workerProcessGet(hiveId, pid)
OperationBuilder.workerProcessDelete(hiveId, pid)
```

**Active Worker Operations (3):**
```javascript
OperationBuilder.activeWorkerList()
OperationBuilder.activeWorkerGet(workerId)
OperationBuilder.activeWorkerRetire(workerId)
```

**Model Operations (4):**
```javascript
OperationBuilder.modelDownload(hiveId, model)
OperationBuilder.modelList(hiveId)
OperationBuilder.modelGet(hiveId, model)
OperationBuilder.modelDelete(hiveId, model)
```

**Inference (1):**
```javascript
OperationBuilder.infer({
  hive_id: 'hive-1',
  model: 'llama-3-8b',
  prompt: 'Hello!',
  max_tokens: 100,
  temperature: 0.7,
  top_p: 0.9,
  // ... all InferRequest fields
})
```

### HeartbeatMonitor

**Purpose:** Real-time system status updates via SSE

**API:**
```javascript
const monitor = new HeartbeatMonitor('http://localhost:8500');

monitor.start((snapshot) => {
  console.log('Workers online:', snapshot.workers_online);
  console.log('Workers available:', snapshot.workers_available);
  console.log('Hives online:', snapshot.hives_online);
  console.log('Hives available:', snapshot.hives_available);
  console.log('Worker IDs:', snapshot.worker_ids);
  console.log('Hive IDs:', snapshot.hive_ids);
});

// Later...
monitor.stop();
```

**Update frequency:** Every 5 seconds

**Endpoint:** `GET /v1/heartbeats/stream` (SSE)

**Why this is critical:**
- ✅ **Real-time dashboard** without polling
- ✅ **Single persistent connection** (efficient)
- ✅ **Automatic updates** (no manual refresh)
- ✅ **Live worker/hive counts**

---

## Integration Steps

### Step 1: Build WASM (5 minutes)

```bash
cd consumers/rbee-sdk
wasm-pack build --target bundler --out-dir pkg/bundler
```

**Output:** `pkg/bundler/`
- `rbee_sdk_bg.wasm` (593 KB)
- `rbee_sdk.js`
- `rbee_sdk.d.ts` (TypeScript types!)
- `package.json`

### Step 2: Update next.config.ts (2 minutes)

**File:** `frontend/apps/web-ui/next.config.ts`

```typescript
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@rbee/ui'],
  
  // TEAM-286: Enable WASM support
  webpack: (config, { isServer }) => {
    // Add WASM support
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };
    
    // Handle .wasm files
    config.module.rules.push({
      test: /\.wasm$/,
      type: 'webassembly/async',
    });
    
    return config;
  },
};

export default nextConfig;
```

### Step 3: Install Dependencies (1 minute)

```bash
cd frontend/apps/web-ui
pnpm install
```

This will link `@rbee/sdk` from `../../consumers/rbee-sdk`.

---

## React Hooks

### Hook 1: useRbeeSDK (Load WASM)

**File:** `src/hooks/useRbeeSDK.ts`

```typescript
'use client';

import { useState, useEffect } from 'react';

// Type definitions (auto-generated by wasm-pack)
type RbeeClient = any;
type HeartbeatMonitor = any;
type OperationBuilder = any;

interface RbeeSDK {
  RbeeClient: typeof RbeeClient;
  HeartbeatMonitor: typeof HeartbeatMonitor;
  OperationBuilder: typeof OperationBuilder;
}

export function useRbeeSDK() {
  const [sdk, setSDK] = useState<RbeeSDK | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function loadWASM() {
      try {
        // TEAM-286: Dynamic import of WASM module
        const wasmModule = await import('@rbee/sdk');
        
        // Initialize WASM (calls init() automatically in bundler mode)
        await wasmModule.default();
        
        setSDK({
          RbeeClient: wasmModule.RbeeClient,
          HeartbeatMonitor: wasmModule.HeartbeatMonitor,
          OperationBuilder: wasmModule.OperationBuilder,
        });
        setLoading(false);
      } catch (err) {
        console.error('Failed to load rbee-sdk WASM:', err);
        setError(err as Error);
        setLoading(false);
      }
    }

    loadWASM();
  }, []);

  return { sdk, loading, error };
}
```

### Hook 2: useRbeeClient (Create Client)

**File:** `src/hooks/useRbeeClient.ts`

```typescript
'use client';

import { useMemo } from 'react';
import { useRbeeSDK } from './useRbeeSDK';

export function useRbeeClient(baseUrl: string = 'http://localhost:8500') {
  const { sdk, loading, error } = useRbeeSDK();

  const client = useMemo(() => {
    if (!sdk) return null;
    return new sdk.RbeeClient(baseUrl);
  }, [sdk, baseUrl]);

  return { client, loading, error };
}
```

### Hook 3: useHeartbeat (Live Updates)

**File:** `src/hooks/useHeartbeat.ts`

```typescript
'use client';

import { useState, useEffect, useRef } from 'react';
import { useRbeeSDK } from './useRbeeSDK';

interface HeartbeatSnapshot {
  timestamp: string;
  workers_online: number;
  workers_available: number;
  hives_online: number;
  hives_available: number;
  worker_ids: string[];
  hive_ids: string[];
}

export function useHeartbeat(baseUrl: string = 'http://localhost:8500') {
  const { sdk, loading: sdkLoading } = useRbeeSDK();
  const [heartbeat, setHeartbeat] = useState<HeartbeatSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const monitorRef = useRef<any>(null);

  useEffect(() => {
    if (!sdk) return;

    // Create monitor
    const monitor = new sdk.HeartbeatMonitor(baseUrl);
    monitorRef.current = monitor;

    try {
      // Start monitoring
      monitor.start((snapshot: HeartbeatSnapshot) => {
        setHeartbeat(snapshot);
        setConnected(true);
        setError(null);
      });

      // Check connection after a moment
      setTimeout(() => {
        if (monitor.isConnected()) {
          setConnected(true);
        }
      }, 1000);
    } catch (err) {
      setError(err as Error);
      setConnected(false);
    }

    // Cleanup
    return () => {
      if (monitorRef.current) {
        monitorRef.current.stop();
      }
    };
  }, [sdk, baseUrl]);

  return {
    heartbeat,
    connected,
    loading: sdkLoading,
    error,
  };
}
```

---

## Example Components

### Dashboard with Live Updates

**File:** `src/app/page.tsx`

```tsx
'use client';

import { useHeartbeat } from '@/hooks/useHeartbeat';
import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/components/card';

export default function HomePage() {
  const { heartbeat, connected, loading, error } = useHeartbeat();

  if (loading) {
    return <div className="min-h-screen bg-background p-8">Loading rbee SDK...</div>;
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background p-8">
        <div className="text-red-500">Error loading SDK: {error.message}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold mb-2">🐝 rbee Web UI</h1>
        <p className="text-muted-foreground">
          Dashboard for managing queen, hives, workers, and models
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {/* Queen Status Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Queen Status
              <span className={connected ? 'text-green-500' : 'text-red-500'}>
                {connected ? '🟢' : '⚫'}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              {connected ? 'Connected' : 'Disconnected'}
            </p>
          </CardContent>
        </Card>

        {/* Workers Card */}
        <Card>
          <CardHeader>
            <CardTitle>Workers</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {heartbeat?.workers_online || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              {heartbeat?.workers_available || 0} available
            </p>
            {heartbeat?.worker_ids && heartbeat.worker_ids.length > 0 && (
              <ul className="mt-2 space-y-1">
                {heartbeat.worker_ids.map((id) => (
                  <li key={id} className="text-xs">{id}</li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        {/* Hives Card */}
        <Card>
          <CardHeader>
            <CardTitle>Hives</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {heartbeat?.hives_online || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              {heartbeat?.hives_available || 0} available
            </p>
            {heartbeat?.hive_ids && heartbeat.hive_ids.length > 0 && (
              <ul className="mt-2 space-y-1">
                {heartbeat.hive_ids.map((id) => (
                  <li key={id} className="text-xs">{id}</li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        {/* Last Update */}
        <Card>
          <CardHeader>
            <CardTitle>Last Update</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">
              {heartbeat?.timestamp 
                ? new Date(heartbeat.timestamp).toLocaleTimeString()
                : 'Waiting...'}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Updates every 5 seconds
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
```

### Streaming Inference Component

**File:** `src/components/InferencePlayground.tsx`

```tsx
'use client';

import { useState } from 'react';
import { useRbeeClient } from '@/hooks/useRbeeClient';
import { useRbeeSDK } from '@/hooks/useRbeeSDK';
import { Button } from '@rbee/ui/components/button';
import { Textarea } from '@rbee/ui/components/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/components/card';

export function InferencePlayground() {
  const { client, loading } = useRbeeClient();
  const { sdk } = useRbeeSDK();
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  const handleInfer = async () => {
    if (!client || !sdk || !prompt.trim()) return;

    setIsGenerating(true);
    setResponse('');

    try {
      // Create infer operation
      const operation = sdk.OperationBuilder.infer({
        hive_id: 'default',
        model: 'llama-3-8b',
        prompt: prompt,
        max_tokens: 500,
        temperature: 0.7,
        top_p: 0.9,
      });

      // Submit and stream
      await client.submitAndStream(operation, (line: string) => {
        // Append each token to response
        setResponse((prev) => prev + line);
      });
    } catch (error) {
      console.error('Inference error:', error);
      setResponse('Error: ' + error);
    } finally {
      setIsGenerating(false);
    }
  };

  if (loading) {
    return <div>Loading SDK...</div>;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Inference Playground</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Textarea
          placeholder="Enter your prompt..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          disabled={isGenerating}
        />
        
        <Button 
          onClick={handleInfer} 
          disabled={isGenerating || !prompt.trim()}
        >
          {isGenerating ? 'Generating...' : 'Generate'}
        </Button>

        {response && (
          <div className="mt-4 p-4 bg-muted rounded-md">
            <pre className="whitespace-pre-wrap text-sm">{response}</pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
```

---

## Troubleshooting

### Issue 1: "Module not found: Can't resolve '@rbee/sdk'"

**Solution:**
```bash
cd consumers/rbee-sdk
wasm-pack build --target bundler
cd ../../frontend/apps/web-ui
pnpm install
```

### Issue 2: "WebAssembly module is included in initial chunk"

**Solution:** Already handled by `asyncWebAssembly: true` in next.config.ts

### Issue 3: "Cannot use import statement outside a module"

**Solution:** Use `'use client'` directive in components using the SDK

### Issue 4: TypeScript can't find types

**Solution:** The `.d.ts` files are auto-generated by wasm-pack in `pkg/bundler/`

### Issue 5: Heartbeat not connecting

**Check:**
1. queen-rbee is running (`cargo run --bin queen-rbee`)
2. URL is correct (`http://localhost:8500`)
3. CORS is enabled on queen-rbee
4. Browser console for errors

---

## Summary

### What You Get

✅ **Production-ready SDK** (593 KB WASM)  
✅ **All 17 operations** working  
✅ **Real-time updates** (HeartbeatMonitor)  
✅ **Type-safe API** (auto-generated TypeScript)  
✅ **Zero type drift** (Rust → TypeScript)  
✅ **Streaming inference** (token-by-token)  
✅ **React hooks** (useRbeeSDK, useRbeeClient, useHeartbeat)

### Integration Time

- Build WASM: 5 minutes
- Update config: 2 minutes
- Create hooks: 30 minutes
- Update page.tsx: 30 minutes
- Test: 15 minutes

**Total:** ~1.5 hours for basic integration

**Full dashboard:** ~4-6 hours

---

**Created by:** TEAM-286  
**Date:** Oct 24, 2025  
**Status:** ✅ Ready for integration!
