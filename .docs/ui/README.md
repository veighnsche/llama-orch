# UI Architecture Documentation

**TEAM-293: Hierarchical, Distributed UI System**

## Overview

This directory contains the complete plan for transforming rbee into a hierarchical UI system where each component (keeper, queen, hives, workers) has its own specialized interface.

## 🚀 How to Use This Documentation

### If You're New Here
1. Start with **[CURRENT_STRUCTURE.md](./CURRENT_STRUCTURE.md)** to see what actually exists now
2. Read **[00_UI_ARCHITECTURE_OVERVIEW.md](./00_UI_ARCHITECTURE_OVERVIEW.md)** to understand the vision
3. Check **[UPDATE_SUMMARY.md](./UPDATE_SUMMARY.md)** for recent changes

### If You're Implementing
1. Follow the **Implementation Guides** (01-05) in order
2. Check **[DOCUMENTATION_CHECKLIST.md](./DOCUMENTATION_CHECKLIST.md)** for status
3. Refer to **Structure & Organization** docs as needed

### If You're Looking for Something Specific
- **Folder structure?** → FOLDER_STRUCTURE.md
- **Package structure?** → PACKAGE_STRUCTURE.md  
- **Port assignments?** → PORT_CONFIGURATION_UPDATE.md
- **Dev proxy setup?** → DEVELOPMENT_PROXY_STRATEGY.md
- **Current status?** → CURRENT_STRUCTURE.md or TURBO_DEV_SUCCESS.md

## 📚 Complete Document Index

### 🎯 Start Here (Essential Reading)

1. **[CURRENT_STRUCTURE.md](./CURRENT_STRUCTURE.md)** 🆕  
   **START HERE.** Shows the actual current state of the project - what exists vs what doesn't.

2. **[00_UI_ARCHITECTURE_OVERVIEW.md](./00_UI_ARCHITECTURE_OVERVIEW.md)**  
   The vision, architecture, and benefits of the hierarchical UI system.

3. **[UPDATE_SUMMARY.md](./UPDATE_SUMMARY.md)** 🆕  
   Quick summary of recent documentation updates (2025-01-25).

### 📁 Structure & Organization

4. **[00_FOLDER_PARITY_SUMMARY.md](./00_FOLDER_PARITY_SUMMARY.md)**  
   UIs co-located with binaries in `bin/` - the co-location strategy.

5. **[FOLDER_STRUCTURE.md](./FOLDER_STRUCTURE.md)**  
   Complete guide to numbered folder structure and navigation.

6. **[00_PACKAGE_MIGRATION_SUMMARY.md](./00_PACKAGE_MIGRATION_SUMMARY.md)**  
   Package changes: generic → specialized SDKs.

7. **[PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md)**  
   Complete guide to specialized SDK packages per binary.

### 🔧 Implementation Guides (Step-by-Step)

8. **[01_KEEPER_GUI_SETUP.md](./01_KEEPER_GUI_SETUP.md)**  
   Set up the keeper Tauri GUI in `bin/00_rbee_keeper/ui/`.

9. **[02_RENAME_WEB_UI.md](./02_RENAME_WEB_UI.md)**  
   Rename web-ui to `10_queen_rbee` and use specialized SDK.

10. **[03_EXTRACT_KEEPER_PAGE.md](./03_EXTRACT_KEEPER_PAGE.md)**  
    Move keeper functionality to dedicated Tauri GUI.

11. **[04_CREATE_HIVE_UI.md](./04_CREATE_HIVE_UI.md)**  
    Create UI for hive management (models + workers).

12. **[05_CREATE_WORKER_UIS.md](./05_CREATE_WORKER_UIS.md)**  
    Create worker-type-specific UIs (LLM, ComfyUI, vLLM).

### 🚧 Future Implementation (TODO)

13. **06_IFRAME_INTEGRATION.md** ⚠️ NOT YET CREATED  
    Wire up iframe hosting in keeper GUI.

14. **07_SIDEBAR_IMPLEMENTATION.md** ⚠️ NOT YET CREATED  
    Implement dynamic sidebar based on heartbeats.

15. **08_STATIC_FILE_SERVING.md** ⚠️ NOT YET CREATED  
    Set up static file serving in Rust binaries.

16. **09_TAURI_ROOT_COMMAND.md** ⚠️ NOT YET CREATED  
    Configure `cargo tauri run` to work from repository root.

17. **10_TESTING_STRATEGY.md** ⚠️ NOT YET CREATED  
    Test the entire hierarchical UI system.

### 📊 Status & Summary Documents

18. **[TEAM_293_BIN_COLOCATION_COMPLETE.md](./TEAM_293_BIN_COLOCATION_COMPLETE.md)** 🆕  
    Summary of documentation updates for bin/ co-location.

19. **[DOCUMENTATION_CHECKLIST.md](./DOCUMENTATION_CHECKLIST.md)** 🆕  
    Complete checklist of what's been updated and what needs review.

20. **[TURBO_DEV_SUCCESS.md](./TURBO_DEV_SUCCESS.md)**  
    Status: All frontend dev servers running successfully with turbo.

21. **[WORKSPACE_SETUP_COMPLETE.md](./WORKSPACE_SETUP_COMPLETE.md)**  
    Status: Workspace packages and SDK setup complete.

22. **[COMPLETE_REORGANIZATION_SUMMARY.md](./COMPLETE_REORGANIZATION_SUMMARY.md)**  
    Historical: Summary of the complete UI reorganization.

23. **[IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)**  
    Historical: Implementation progress and decisions.

### 🌐 Development & Configuration

24. **[DEVELOPMENT_PROXY_STRATEGY.md](./DEVELOPMENT_PROXY_STRATEGY.md)**  
    Strategy for proxying requests during development.

25. **[DEV_PROXY_SUMMARY.md](./DEV_PROXY_SUMMARY.md)**  
    Quick summary of dev proxy configuration.

26. **[PORT_CONFIGURATION_UPDATE.md](./PORT_CONFIGURATION_UPDATE.md)**  
    Port assignments for all UI components.

## Quick Reference

### File Structure After Implementation

```
bin/00_rbee_keeper/ui/               # Tauri GUI (keeper) - no app subfolder
bin/10_queen_rbee/ui/
  ├── app/                           # Queen UI (scheduling)
  └── packages/                      # Queen SDK packages
bin/20_rbee_hive/ui/
  ├── app/                           # Hive UI (models + workers)
  └── packages/                      # Hive SDK packages
bin/30_llm_worker_rbee/ui/
  ├── app/                           # LLM worker UI
  └── packages/                      # Worker SDK packages

frontend/packages/
  ├── rbee-ui/                       # Shared Storybook components
  └── tailwind-config/               # Shared Tailwind config
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
cd bin/00_rbee_keeper/ui
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

## 📋 Document Summary

**Total Documents:** 27 (22 exist, 5 planned)

### By Category
- **Essential Reading:** 3 documents
- **Structure & Organization:** 4 documents  
- **Implementation Guides:** 5 documents (exist) + 5 documents (planned)
- **Status & Summary:** 6 documents
- **Development & Configuration:** 3 documents

### By Status
- ✅ **Complete & Current:** 17 documents
- 🆕 **Recently Added (2025-01-25):** 4 documents
- 📜 **Historical/Reference:** 2 documents
- ⚠️ **Planned (Not Yet Created):** 5 documents

## Recent Updates

- **2025-01-25:** ✅ All core documentation updated for bin/ co-location structure
  - Added 4 new documents: CURRENT_STRUCTURE, UPDATE_SUMMARY, TEAM_293_BIN_COLOCATION_COMPLETE, DOCUMENTATION_CHECKLIST
  - Updated 5 core documents: Architecture, Folder Parity, Folder Structure, Package Structure, README
  - See [UPDATE_SUMMARY.md](./UPDATE_SUMMARY.md) for complete details
  - See [CURRENT_STRUCTURE.md](./CURRENT_STRUCTURE.md) for actual current state

## 🗺️ Document Relationship Map

```
CURRENT_STRUCTURE.md (START HERE)
    ↓
00_UI_ARCHITECTURE_OVERVIEW.md (The Vision)
    ↓
    ├─→ 00_FOLDER_PARITY_SUMMARY.md (Co-location Strategy)
    │       ↓
    │   FOLDER_STRUCTURE.md (Complete Structure)
    │
    ├─→ 00_PACKAGE_MIGRATION_SUMMARY.md (Package Strategy)
    │       ↓
    │   PACKAGE_STRUCTURE.md (Complete Packages)
    │
    └─→ Implementation Guides (01-05)
            ↓
        Future Guides (06-10) [TODO]

Supporting Documents:
├─→ TEAM_293_BIN_COLOCATION_COMPLETE.md (Update Summary)
├─→ DOCUMENTATION_CHECKLIST.md (Status Tracking)
├─→ UPDATE_SUMMARY.md (Recent Changes)
├─→ TURBO_DEV_SUCCESS.md (Dev Status)
├─→ WORKSPACE_SETUP_COMPLETE.md (Setup Status)
├─→ DEVELOPMENT_PROXY_STRATEGY.md (Dev Proxy)
├─→ DEV_PROXY_SUMMARY.md (Proxy Quick Ref)
├─→ PORT_CONFIGURATION_UPDATE.md (Port Assignments)
├─→ COMPLETE_REORGANIZATION_SUMMARY.md (Historical)
└─→ IMPLEMENTATION_SUMMARY.md (Historical)
```

---

**Status:** 📋 CORE DOCUMENTATION COMPLETE | IMPLEMENTATION GUIDES NEED REVIEW  
**Last Updated:** 2025-01-25 by TEAM-293
