# UI Architecture Documentation

**TEAM-293: Hierarchical, Distributed UI System**

## Overview

This directory contains the complete plan for transforming rbee into a hierarchical UI system where each component (keeper, queen, hives, workers) has its own specialized interface.

## Reading Order

Read these documents in sequence:

### Core Architecture

1. **[00_UI_ARCHITECTURE_OVERVIEW.md](./00_UI_ARCHITECTURE_OVERVIEW.md)**  
   Start here. Explains the vision, architecture, and benefits.

2. **[00_FOLDER_PARITY_SUMMARY.md](./00_FOLDER_PARITY_SUMMARY.md)**  
   Folder structure: `bin/` ↔ `frontend/apps/` parity.

3. **[FOLDER_STRUCTURE.md](./FOLDER_STRUCTURE.md)**  
   Complete guide to numbered folder structure.

4. **[00_PACKAGE_MIGRATION_SUMMARY.md](./00_PACKAGE_MIGRATION_SUMMARY.md)**  
   Package changes: generic → specialized SDKs.

5. **[PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md)**  
   Complete guide to specialized SDK packages.

### Implementation Steps

6. **[01_KEEPER_GUI_SETUP.md](./01_KEEPER_GUI_SETUP.md)**  
   Set up the keeper Tauri GUI in `frontend/apps/00_rbee_keeper/`.

7. **[02_RENAME_WEB_UI.md](./02_RENAME_WEB_UI.md)**  
   Rename web-ui to `10_queen_rbee` and use specialized SDK.

8. **[03_EXTRACT_KEEPER_PAGE.md](./03_EXTRACT_KEEPER_PAGE.md)**  
   Move keeper functionality to dedicated Tauri GUI.

9. **[04_CREATE_HIVE_UI.md](./04_CREATE_HIVE_UI.md)**  
   Create UI for hive management (models + workers).

10. **[05_CREATE_WORKER_UIS.md](./05_CREATE_WORKER_UIS.md)**  
    Create worker-type-specific UIs (LLM, ComfyUI, vLLM).

7. **[06_IFRAME_INTEGRATION.md](./06_IFRAME_INTEGRATION.md)** ⚠️ TODO  
   Wire up iframe hosting in keeper GUI.

8. **[07_SIDEBAR_IMPLEMENTATION.md](./07_SIDEBAR_IMPLEMENTATION.md)** ⚠️ TODO  
   Implement dynamic sidebar based on heartbeats.

9. **[08_STATIC_FILE_SERVING.md](./08_STATIC_FILE_SERVING.md)** ⚠️ TODO  
   Set up static file serving in Rust binaries.

10. **[09_TAURI_ROOT_COMMAND.md](./09_TAURI_ROOT_COMMAND.md)** ⚠️ TODO  
    Configure `cargo tauri run` to work from repository root.

11. **[10_TESTING_STRATEGY.md](./10_TESTING_STRATEGY.md)** ⚠️ TODO  
    Test the entire hierarchical UI system.

## Quick Reference

### File Structure After Implementation

```
frontend/apps/00_rbee_keeper/        # Tauri GUI (keeper)
frontend/apps/10_queen_rbee/         # Queen UI (scheduling)
frontend/apps/20_rbee_hive/          # Hive UI (models + workers)
frontend/apps/30_llm_worker_rbee/    # LLM worker UI
frontend/apps/30_comfy_worker_rbee/  # ComfyUI worker UI
frontend/apps/30_vllm_worker_rbee/   # vLLM worker UI

frontend/packages/10_queen_rbee/     # Queen SDK packages
frontend/packages/20_rbee_hive/      # Hive SDK packages
frontend/packages/30_workers/        # Worker SDK packages
```

### Port Assignments

| Component | Dev Port | Production |
|-----------|----------|------------|
| keeper GUI | 5173 | Tauri app |
| queen-rbee UI | 7834 | 7833/ui |
| hive UI | 7836 | 7835/ui |
| LLM worker UI | 7837 | 8080/ui |
| ComfyUI worker UI | 7838 | 8188/ui |
| vLLM worker UI | 7839 | 8000/ui |

### Responsibilities

| UI | Manages |
|----|---------|
| **keeper** | Start/stop, install/uninstall (queen + hives) |
| **queen** | Scheduling, job queue, inference routing |
| **hive** | Model management, worker spawning, resources |
| **workers** | Worker-specific live demos and metrics |

## Implementation Status

- ✅ **00-05:** Documentation complete
- ⚠️ **06-10:** Documentation in progress
- ❌ **Implementation:** Not started

## Key Design Decisions

1. **Hierarchical:** Keeper → Queen → Hives → Workers
2. **Distributed:** Each binary hosts its own UI
3. **iframe-based:** Keeper hosts child UIs in iframes
4. **Heartbeat-driven:** Sidebar updates based on SSE heartbeats
5. **Specialized SDKs:** Each binary (except keeper) has its own SDK package
   - `@rbee/queen-rbee-sdk` + `@rbee/queen-rbee-react`
   - `@rbee/rbee-hive-sdk` + `@rbee/rbee-hive-react`
   - `@rbee/llm-worker-sdk` + `@rbee/llm-worker-react`
6. **Keeper exception:** Uses Tauri commands, NO SDK (no HTTP API)

## Common Tasks

### Install Dependencies

```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

### Run Keeper GUI in Dev Mode

```bash
# Terminal 1: Frontend
cd frontend/apps/00_rbee_keeper
pnpm dev

# Terminal 2: Tauri
cd bin/00_rbee_keeper
cargo tauri dev
```

### Build All UIs

```bash
pnpm --filter "@rbee/ui-*" build
pnpm --filter "@rbee/keeper-gui" build
```

### Run from Root (After Setup)

```bash
cd /home/vince/Projects/llama-orch
cargo tauri dev  # Runs keeper GUI
```

## Questions?

- **Architecture:** See `00_UI_ARCHITECTURE_OVERVIEW.md`
- **Keeper Setup:** See `01_KEEPER_GUI_SETUP.md`
- **iframe Integration:** See `06_IFRAME_INTEGRATION.md` (TODO)
- **Sidebar Logic:** See `07_SIDEBAR_IMPLEMENTATION.md` (TODO)

---

**Status:** 📋 DOCUMENTATION IN PROGRESS (5/10 complete)
