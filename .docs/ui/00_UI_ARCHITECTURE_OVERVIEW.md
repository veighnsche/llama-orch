# UI Architecture Overview

**TEAM-293: Hierarchical UI Architecture**  
**Status:** 📋 PLANNED  
**Date:** October 25, 2025

## Vision

Transform rbee from a monolithic web UI into a **hierarchical, distributed UI system** where each component (keeper, queen, hives, workers) has its own specialized interface.

## Current State (Before)

```
frontend/apps/web-ui/
├── Combines rbee-keeper + queen-rbee functionality
├── Uses rbee-sdk (WASM)
└── Single monolithic React app

bin/00_rbee_keeper/
└── Tauri GUI with basic React pages
```

**Problems:**
- ❌ Keeper and Queen UI mixed together
- ❌ No UI for individual hives
- ❌ No UI for individual workers
- ❌ Not hierarchical/composable

## Future State (After)

```
Architecture:
┌─────────────────────────────────────────────────┐
│ rbee-keeper (Tauri GUI)                         │
│ ├── Sidebar Navigation                          │
│ ├── PageContainer with iframes                  │
│ └── Manages: start/stop/status/install/uninstall│
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │ iframe: queen-rbee UI (if alive)          │  │
│  │ - Scheduling                              │  │
│  │ - Job queue                               │  │
│  │ - Inference routing                       │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │ iframe: hive-001 UI                       │  │
│  │ - Model management                        │  │
│  │ - Worker spawning                         │  │
│  │ - Resource monitoring                     │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │ iframe: llm-worker-001 UI                 │  │
│  │ - Inference parameters                    │  │
│  │ - Performance metrics                     │  │
│  │ - Live inference demo                     │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## New UI Structure

### 1. rbee-keeper (Tauri GUI)
**Location:** `frontend/apps/00_rbee_keeper/`  
**Type:** Tauri (Rust + React)  
**Purpose:** Top-level orchestration UI

**Features:**
- Sidebar with hierarchical navigation
- iframe host for child UIs
- Lifecycle management (start/stop/install/uninstall)
- Status monitoring via heartbeats

**Does NOT include:**
- Scheduling (that's queen's job)
- Model management (that's hive's job)
- Worker-specific features (that's worker's job)

### 2. queen-rbee UI
**Location:** `frontend/apps/10_queen_rbee/`  
**Type:** React (static build)  
**Purpose:** Scheduling and job management

**Features:**
- Job queue visualization
- Inference request routing
- Hive selection logic
- Performance analytics

**Hosted by:** queen-rbee binary (static files)  
**Accessed via:** rbee-keeper iframe

### 3. rbee-hive UI
**Location:** `frontend/apps/20_rbee_hive/`  
**Type:** React (static build)  
**Purpose:** Model and worker management

**Features:**
- Model download/list/delete
- Worker spawn/kill/status
- Resource monitoring (VRAM, GPU)
- Hive configuration

**Hosted by:** rbee-hive binary (static files)  
**Accessed via:** rbee-keeper iframe

### 4. Worker UIs (Per Type)
**Locations:** 
- `frontend/apps/30_llm_worker_rbee/`
- `frontend/apps/30_comfy_worker_rbee/`
- `frontend/apps/30_vllm_worker_rbee/`
- etc.

**Type:** React (static builds)  
**Purpose:** Worker-specific interfaces

**Features (vary by type):**
- Live inference demo
- Performance metrics
- Worker-specific parameters
- Health status

**Hosted by:** Worker binaries (static files)  
**Accessed via:** rbee-keeper iframe

## Communication Flow

### Heartbeat-Based Discovery

```
rbee-keeper listens to SSE heartbeats
  ↓
Queen sends heartbeat → Shows in sidebar → iframe available
  ↓
Hive sends heartbeat → Shows in sidebar → iframe available
  ↓
Worker sends heartbeat → Shows in sidebar → iframe available
```

### iframe Communication

```
rbee-keeper (parent)
  ↓ postMessage
Child UI (iframe)
  ↓ postMessage (response)
rbee-keeper (receives)
```

**Security:** Same-origin policy, postMessage API for cross-frame communication

## Technology Stack

- **Keeper GUI:** Tauri (Rust + React + Vite) - Uses Tauri commands, NO SDK
- **Child UIs:** React + Vite (static builds)
- **Shared Components:** `@rbee-ui/stories` (Vue components)
- **SDKs:** Specialized per binary (HTTP clients only)
  - `@rbee/queen-rbee-sdk` + `@rbee/queen-rbee-react`
  - `@rbee/rbee-hive-sdk` + `@rbee/rbee-hive-react`
  - `@rbee/llm-worker-sdk` + `@rbee/llm-worker-react`
  - etc.
- **Styling:** TailwindCSS

## Sidebar Structure

```
rbee Keeper
├─ Dashboard (keeper status)
├─ Install/Uninstall
└─ Settings

queen rbee (dynamic, based on heartbeat)
├─ Scheduling
├─ Job Queue
└─ Analytics

rbee-hives (dynamic, based on heartbeats)
├─ hive-001
│   ├─ Models
│   ├─ Workers
│   └─ Resources
├─ hive-002
└─ [+ Add Hive]

worker-rbees (dynamic, based on heartbeats)
├─ LLM Workers
│   ├─ llm-001 (hive-001)
│   └─ llm-002 (hive-001)
├─ ComfyUI Workers
│   └─ comfy-001 (hive-002)
└─ vLLM Workers
```

## Benefits

### For Users
- ✅ Specialized UI per component
- ✅ Clear separation of concerns
- ✅ Can access any level directly (if needed)
- ✅ Unified experience via keeper

### For Developers
- ✅ Independent UI development
- ✅ No coupling between UIs
- ✅ Easy to add new worker types
- ✅ Clear ownership boundaries

### For System
- ✅ Distributed: Each binary serves its own UI
- ✅ Resilient: If queen dies, hive UIs still work
- ✅ Scalable: Add new hives/workers without keeper changes
- ✅ Hierarchical: Natural system structure

## Migration Path

See individual implementation files:
1. `01_KEEPER_GUI_SETUP.md` - Set up keeper GUI
2. `02_RENAME_WEB_UI.md` - Rename web-ui to ui-queen-rbee
3. `03_EXTRACT_KEEPER_PAGE.md` - Move keeper page to GUI
4. `04_CREATE_HIVE_UI.md` - Create new hive UI
5. `05_CREATE_WORKER_UIS.md` - Create worker UIs
6. `06_IFRAME_INTEGRATION.md` - Wire up iframe system
7. `07_SIDEBAR_IMPLEMENTATION.md` - Implement dynamic sidebar
8. `08_STATIC_FILE_SERVING.md` - Set up static file hosting in binaries
9. `09_TAURI_ROOT_COMMAND.md` - Enable `cargo tauri run` from root
10. `10_TESTING_STRATEGY.md` - Test the entire system

---

**Next:** Read `01_KEEPER_GUI_SETUP.md` to start implementation
