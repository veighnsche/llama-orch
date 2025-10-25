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
┌───────────────────────────────────────────────## Folder Structure

```
bin/
├── 00_rbee_keeper/
│   └── ui/                    # Desktop GUI (Tauri) - no app subfolder
├── 10_queen_rbee/
│   └── ui/
│       ├── app/               # Queen web UI
│       └── packages/
│           ├── queen-rbee-sdk/    # HTTP client for queen API
│           └── queen-rbee-react/  # React hooks for queen
├── 20_rbee_hive/
│   └── ui/
│       ├── app/               # Hive web UI
│       └── packages/
│           ├── rbee-hive-sdk/     # HTTP client for hive API
│           └── rbee-hive-react/   # React hooks for hive
└── 30_llm_worker_rbee/
    └── ui/
        ├── app/               # Worker web UI
        └── packages/
            ├── llm-worker-sdk/    # HTTP client for worker API
            └── llm-worker-react/  # React hooks for worker

frontend/
├── apps/
│   ├── commercial/          # Marketing site
│   ├── user-docs/           # Documentation site
│   └── web-ui/              # DEPRECATED (old monolithic UI)
│
└── packages/
    ├── rbee-ui/               # Shared Storybook components
    ├── rbee-sdk/              # DEPRECATED (generic SDK)
    ├── rbee-react/            # DEPRECATED (generic hooks)
    └── tailwind-config/       # Shared Tailwind config
```

## New UI Structure

### 1. Keeper GUI (`bin/00_rbee_keeper/ui/`)

**Technology:** Tauri (Rust + React)
**Purpose:** Desktop application for managing the entire rbee system
**Port:** N/A (desktop app)
**Location:** `bin/00_rbee_keeper/ui/` (no app subfolder - keeper doesn't need packages)

**Features:**
- System overview dashboard
- Queen management (start/stop/configure)
- Hive registry and health monitoring
- Worker pool visualization
- System logs and diagnostics
- Configuration management

### 2. Queen UI (`bin/10_queen_rbee/ui/app/`)

**Technology:** Vite + React
**Purpose:** Web interface for queen-rbee orchestrator
**Port:** 5174
**API:** `http://localhost:7833`
**Packages:** `bin/10_queen_rbee/ui/packages/` (queen-rbee-sdk, queen-rbee-react)

**Features:**
- Job queue visualization
- Inference request routing
- Hive selection logic
- Performance analytics

**Hosted by:** queen-rbee binary (static files)  
**Accessed via:** rbee-keeper iframe

### 3. Hive UI (`bin/20_rbee_hive/ui/app/`)

**Technology:** Vite + React
**Purpose:** Web interface for rbee-hive manager
**Port:** 5175
**API:** `http://localhost:7835`
**Packages:** `bin/20_rbee_hive/ui/packages/` (rbee-hive-sdk, rbee-hive-react)

**Features:**
- Model download/list/delete
- Worker spawn/kill/status
- Resource monitoring (VRAM, GPU)
- Hive configuration

**Hosted by:** rbee-hive binary (static files)  
**Accessed via:** rbee-keeper iframe

### 4. Worker UIs (`bin/30_llm_worker_rbee/ui/app/`)

**Technology:** Vite + React
**Purpose:** Web interface for individual LLM workers
**Port:** 5176 (base), 5177, 5178, etc.
**API:** `http://localhost:8080` (base), 8081, 8082, etc.
**Packages:** `bin/30_llm_worker_rbee/ui/packages/` (llm-worker-sdk, llm-worker-react)

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
