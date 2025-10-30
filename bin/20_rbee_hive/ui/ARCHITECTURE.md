# Hive UI Architecture Study

**Date:** Oct 30, 2025  
**Status:** COMPLETE ANALYSIS  
**Purpose:** Comprehensive architecture documentation for Hive UI MVP

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Job-Based Architecture](#job-based-architecture)
3. [Package Structure](#package-structure)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Key Patterns](#key-patterns)
7. [MVP Scope](#mvp-scope)

---

## System Overview

### What is Hive UI?

**Hive UI** is a React-based web interface for managing **models** and **workers** on a single Hive node. It runs **locally on the Hive** (served at `http://<hive-ip>:7836`) and communicates with the Hive backend via job-based operations.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         HIVE UI                              │
│                  (React + WASM + TanStack Query)             │
│                   http://<hive-ip>:7836                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP + SSE
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      RBEE-HIVE BACKEND                       │
│                   http://<hive-ip>:7835                      │
│                                                              │
│  POST /v1/jobs → job_router.rs → Operation enum             │
│  GET /v1/jobs/{job_id}/stream → SSE narration               │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ↓                   ↓
          ┌──────────────────┐  ┌──────────────────┐
          │  Model Catalog   │  │ Worker Catalog   │
          │  (GGUF files)    │  │ (Binaries + PS)  │
          └──────────────────┘  └──────────────────┘
```

---

## Job-Based Architecture

### Core Pattern (Same as Queen/Hive)

**ALL operations** in the rbee system follow the same pattern:

```
1. POST /v1/jobs (with Operation enum payload)
   → Returns {job_id, sse_url}

2. GET /v1/jobs/{job_id}/stream
   → SSE stream of narration events
   → Ends with [DONE] marker
```

### Why Job-Based?

1. **Consistency** - Same pattern everywhere (Queen, Hive, Worker)
2. **Type Safety** - Uses `operations-contract` enum (single source of truth)
3. **Streaming** - Real-time progress via SSE narration
4. **Async** - Non-blocking operations (downloads, spawns, etc.)
5. **Cancellation** - DELETE /v1/jobs/{job_id} to cancel

### Operations Contract

**Location:** `bin/97_contracts/operations-contract/src/lib.rs`

**Hive Operations (8 total):**

```rust
// Worker Lifecycle (4)
Operation::WorkerSpawn(WorkerSpawnRequest)
Operation::WorkerProcessList(WorkerProcessListRequest)
Operation::WorkerProcessGet(WorkerProcessGetRequest)
Operation::WorkerProcessDelete(WorkerProcessDeleteRequest)

// Model Management (4)
Operation::ModelDownload(ModelDownloadRequest)
Operation::ModelList(ModelListRequest)
Operation::ModelGet(ModelGetRequest)
Operation::ModelDelete(ModelDeleteRequest)
```

**Request Types:**

```rust
// Worker spawn
pub struct WorkerSpawnRequest {
    pub hive_id: String,  // Network address (e.g., "192.168.1.100")
    pub model: String,    // Model ID (e.g., "llama-3.2-1b")
    pub worker: String,   // Worker type ("cpu", "cuda", "metal")
    pub device: u32,      // Device index (0, 1, 2, ...)
}

// Model list
pub struct ModelListRequest {
    pub hive_id: String,  // Network address
}

// etc.
```

---

## Package Structure

### Monorepo Layout

```
bin/20_rbee_hive/ui/
├── app/                          # React application
│   ├── src/
│   │   ├── App.tsx              # Main app component
│   │   ├── main.tsx             # Entry point
│   │   └── index.css            # Global styles
│   ├── package.json             # App dependencies
│   └── vite.config.ts           # Vite config
│
└── packages/                     # Shared packages
    ├── rbee-hive-sdk/           # WASM SDK (Rust → WASM)
    │   ├── src/
    │   │   ├── lib.rs           # WASM entry point
    │   │   ├── client.rs        # HiveClient (wraps JobClient)
    │   │   ├── operations.rs    # OperationBuilder (JS-friendly API)
    │   │   └── conversions.rs   # JS ↔ Rust conversions
    │   ├── Cargo.toml           # Rust dependencies
    │   └── package.json         # NPM package config
    │
    └── rbee-hive-react/         # React hooks
        ├── src/
        │   ├── index.ts         # useModels, useWorkers
        │   └── hooks/
        │       └── useHiveOperations.ts  # spawnWorker, etc.
        └── package.json         # React hooks package
```

### Package Dependencies

```
app (@rbee/rbee-hive-ui)
  ├─→ @rbee/rbee-hive-react (React hooks)
  │     ├─→ @rbee/rbee-hive-sdk (WASM SDK)
  │     │     └─→ job-client (Rust crate)
  │     │     └─→ operations-contract (Rust crate)
  │     ├─→ @rbee/narration-client (SSE parsing)
  │     └─→ @tanstack/react-query (caching)
  │
  ├─→ @rbee/ui (shared UI components)
  ├─→ @rbee/dev-utils (logging)
  └─→ @rbee/shared-config (ports, URLs)
```

---

## Data Flow

### Example: List Models

**1. React Component calls hook:**

```tsx
import { useModels } from '@rbee/rbee-hive-react'

function ModelList() {
  const { models, loading, error, refetch } = useModels()
  
  return (
    <div>
      {models.map(model => <div key={model.id}>{model.name}</div>)}
    </div>
  )
}
```

**2. Hook uses TanStack Query:**

```ts
// packages/rbee-hive-react/src/index.ts
export function useModels() {
  return useQuery({
    queryKey: ['hive-models'],
    queryFn: async () => {
      await ensureWasmInit()  // Initialize WASM once
      const hiveId = client.hiveId  // Get from client
      const op = OperationBuilder.modelList(hiveId)  // Build operation
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') lines.push(line)
      })
      
      const lastLine = lines[lines.length - 1]
      return lastLine ? JSON.parse(lastLine) : []
    },
    staleTime: 30000,  // Cache for 30 seconds
  })
}
```

**3. WASM SDK wraps job-client:**

```rust
// packages/rbee-hive-sdk/src/client.rs
#[wasm_bindgen]
impl HiveClient {
    pub async fn submit_and_stream(
        &self,
        operation: JsValue,
        on_line: js_sys::Function,
    ) -> Result<String, JsValue> {
        let op: Operation = js_to_operation(operation)?;
        
        // Use existing job-client!
        let job_id = self.inner
            .submit_and_stream(op, move |line| {
                let _ = callback.call1(&JsValue::null(), &JsValue::from_str(line));
                Ok(())
            })
            .await
            .map_err(error_to_js)?;
        
        Ok(job_id)
    }
}
```

**4. job-client makes HTTP requests:**

```rust
// bin/99_shared_crates/job-client/src/lib.rs
pub async fn submit_and_stream<F>(
    &self,
    operation: Operation,
    mut line_handler: F,
) -> Result<String> {
    // 1. POST /v1/jobs
    let job_response: serde_json::Value = self.client
        .post(format!("{}/v1/jobs", self.base_url))
        .json(&operation)
        .send().await?
        .json().await?;
    
    let job_id = job_response["job_id"].as_str().unwrap();
    
    // 2. GET /v1/jobs/{job_id}/stream (SSE)
    let stream_url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
    let response = self.client.get(&stream_url).send().await?;
    
    // 3. Stream lines
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        // Parse SSE, strip "data:" prefix, call handler
        line_handler(line)?;
    }
    
    Ok(job_id)
}
```

**5. Hive backend routes operation:**

```rust
// bin/20_rbee_hive/src/job_router.rs
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState,
) -> Result<()> {
    let operation: Operation = serde_json::from_value(payload)?;
    
    match operation {
        Operation::ModelList(request) => {
            let models = state.model_catalog.list();
            
            for model in &models {
                n!("model_list_entry", "  {} | {} | {:.2} GB", 
                   model.id(), model.name(), model.size_gb());
            }
        }
        // ... other operations
    }
    
    Ok(())
}
```

**6. Narration flows back via SSE:**

```
data: {"action":"model_list_start","message":"📋 Listing models on hive 'localhost'"}
data: {"action":"model_list_result","message":"Found 3 model(s)"}
data: {"action":"model_list_entry","message":"  llama-3.2-1b | Llama 3.2 1B | 1.23 GB"}
data: {"action":"model_list_entry","message":"  llama-3.2-3b | Llama 3.2 3B | 3.45 GB"}
data: [DONE]
```

---

## Technology Stack

### Frontend

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | React 19 | UI components |
| **Build Tool** | Vite (Rolldown) | Fast dev server + HMR |
| **State Management** | TanStack Query | Server state caching |
| **WASM Runtime** | wasm-bindgen | Rust → JavaScript bindings |
| **HTTP Client** | reqwest (WASM) | HTTP requests from WASM |
| **Styling** | TailwindCSS | Utility-first CSS |
| **UI Components** | @rbee/ui | Shared component library |

### Backend (Rust)

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Web Framework** | Axum | HTTP server |
| **Job System** | job-server | Job lifecycle management |
| **Narration** | observability-narration-core | SSE streaming |
| **Operations** | operations-contract | Type-safe operation enum |
| **HTTP Client** | job-client | Shared HTTP + SSE client |

### Shared Packages

| Package | Purpose |
|---------|---------|
| `@rbee/rbee-hive-sdk` | WASM SDK (Rust compiled to WASM) |
| `@rbee/rbee-hive-react` | React hooks (useModels, useWorkers) |
| `@rbee/ui` | Shared UI components (Button, Card, etc.) |
| `@rbee/shared-config` | Port/URL configuration |
| `@rbee/narration-client` | SSE parsing utilities |
| `@rbee/dev-utils` | Logging utilities |

---

## Key Patterns

### 1. WASM SDK Pattern (TEAM-286, TEAM-353)

**Problem:** Don't duplicate HTTP client logic in TypeScript

**Solution:** Compile existing Rust crates to WASM

```
Rust Crates                WASM SDK              JavaScript
─────────────────────────────────────────────────────────────
job-client        →    rbee-hive-sdk    →    HiveClient
operations-contract →  (thin wrapper)   →    OperationBuilder
```

**Benefits:**
- ✅ Single source of truth (Rust)
- ✅ Compiler-verified types
- ✅ Zero duplication
- ✅ Auto-generated TypeScript types

### 2. TanStack Query Pattern

**Problem:** Manual state management is complex

**Solution:** Use TanStack Query for server state

```ts
useQuery({
  queryKey: ['hive-models'],  // Cache key
  queryFn: async () => { /* fetch */ },
  staleTime: 30000,  // Cache for 30 seconds
  refetchInterval: 2000,  // Auto-refetch every 2 seconds
  retry: 3,  // Retry on failure
})
```

**Benefits:**
- ✅ Automatic caching
- ✅ Automatic refetching
- ✅ Automatic error handling
- ✅ Loading states
- ✅ Optimistic updates

### 3. Narration Pattern

**Problem:** Need real-time progress updates

**Solution:** SSE streaming with narration events

```
Backend (Rust):
  n!("model_download_progress", "Downloaded {} MB", mb);

Frontend (React):
  useNarration(jobId, (event) => {
    console.log(event.action, event.message);
  });
```

**Benefits:**
- ✅ Real-time progress
- ✅ Structured events (action + message)
- ✅ Job-scoped isolation
- ✅ Automatic cleanup

### 4. Hive ID Pattern

**CRITICAL:** `hive_id` is the **network address** of the Hive, NOT "localhost"!

```ts
// ❌ WRONG
const op = OperationBuilder.modelList("localhost")

// ✅ CORRECT
const hiveAddress = window.location.hostname  // "192.168.1.100"
const op = OperationBuilder.modelList(hiveAddress)
```

**Why?**
- Hive UI runs ON the Hive (same machine)
- But operations need the network address for routing
- Queen uses hive_id to route operations to the correct Hive

### 5. Dual-Call Pattern (TEAM-154)

**All operations use the same pattern:**

```ts
// 1. Submit operation → get job_id
const jobId = await client.submit(operation)

// 2. Stream results via SSE
await client.submitAndStream(operation, (line) => {
  console.log(line)  // Narration events
})
```

**Or combined:**

```ts
const jobId = await client.submitAndStream(operation, (line) => {
  console.log(line)
})
```

---

## MVP Scope

### Phase 1: Model Management ✅

**Features:**
- ✅ List models (with size, status)
- ✅ View model details
- ✅ Delete model
- ⏳ Download model (waiting for TEAM-269)

**UI Components:**
- Model list table (name, size, status)
- Model detail card
- Delete confirmation dialog
- Download progress bar (future)

### Phase 2: Worker Management ⏳

**Features:**
- ⏳ List workers (PID, model, device)
- ⏳ Spawn worker (select model + device)
- ⏳ Kill worker (SIGTERM → SIGKILL)
- ⏳ View worker logs (future)

**UI Components:**
- Worker list table (PID, model, device, status)
- Spawn worker form (model dropdown, device selector)
- Kill confirmation dialog
- Worker status badges

### Phase 3: Real-Time Updates ⏳

**Features:**
- ⏳ Auto-refresh worker list (every 2 seconds)
- ⏳ Live narration feed (SSE events)
- ⏳ Toast notifications (spawn success, errors)
- ⏳ Progress indicators (downloads, spawns)

**UI Components:**
- Narration feed panel (bottom drawer)
- Toast notifications (top-right)
- Progress bars (inline + global)
- Status indicators (pulsing dots)

### Phase 4: Polish ⏳

**Features:**
- ⏳ Error boundaries
- ⏳ Loading skeletons
- ⏳ Empty states
- ⏳ Keyboard shortcuts
- ⏳ Dark mode

---

## Next Steps

### Immediate (MVP)

1. **Create basic layout**
   - Header with Hive ID
   - Two-column layout (Models | Workers)
   - Narration feed drawer (bottom)

2. **Implement Model List**
   - Use `useModels()` hook
   - Table with columns: Name, Size, Status
   - Delete button per row

3. **Implement Worker List**
   - Use `useWorkers()` hook
   - Table with columns: PID, Model, Device, Status
   - Kill button per row

4. **Add Spawn Worker Form**
   - Model dropdown (from `useModels()`)
   - Device selector (0, 1, 2, ...)
   - Worker type selector (cpu, cuda, metal)
   - Submit button

5. **Wire up operations**
   - Use `useHiveOperations()` hook
   - Call `spawnWorker(modelId)`
   - Show toast on success/error
   - Refetch lists after operations

### Future Enhancements

- Model download with progress
- Worker logs viewer
- GPU utilization metrics
- Model search/filter
- Worker health checks
- Batch operations
- Keyboard shortcuts
- Mobile responsive

---

## Key Files Reference

### Rust Backend

```
bin/20_rbee_hive/
├── src/
│   ├── job_router.rs          # Operation routing
│   ├── http/
│   │   ├── jobs.rs            # POST /v1/jobs, GET /v1/jobs/{id}/stream
│   │   └── mod.rs             # HTTP routes
│   └── main.rs                # Server entry point
```

### WASM SDK

```
bin/20_rbee_hive/ui/packages/rbee-hive-sdk/
├── src/
│   ├── lib.rs                 # WASM entry point
│   ├── client.rs              # HiveClient (wraps JobClient)
│   ├── operations.rs          # OperationBuilder (JS API)
│   └── conversions.rs         # JS ↔ Rust conversions
├── Cargo.toml                 # Rust dependencies
└── package.json               # NPM package
```

### React Hooks

```
bin/20_rbee_hive/ui/packages/rbee-hive-react/
├── src/
│   ├── index.ts               # useModels, useWorkers
│   └── hooks/
│       └── useHiveOperations.ts  # spawnWorker, etc.
└── package.json
```

### React App

```
bin/20_rbee_hive/ui/app/
├── src/
│   ├── App.tsx                # Main component
│   ├── main.tsx               # Entry point
│   └── index.css              # Global styles
├── package.json
└── vite.config.ts
```

### Shared Crates

```
bin/99_shared_crates/
├── job-client/                # HTTP + SSE client
└── operations-contract/       # Operation enum

bin/97_contracts/
└── operations-contract/       # Type definitions
```

---

## Summary

**Hive UI** is a simple React app that:

1. **Runs locally on the Hive** (port 7836)
2. **Uses WASM SDK** to call Hive backend (port 7835)
3. **Follows job-based architecture** (POST /v1/jobs → SSE stream)
4. **Manages models and workers** (list, spawn, delete)
5. **Shows real-time progress** via SSE narration

**Key Insight:** The architecture is **already complete**. We just need to build the UI components and wire them up to the existing hooks.

**Next:** Build the MVP UI with Model List, Worker List, and Spawn Worker form.
