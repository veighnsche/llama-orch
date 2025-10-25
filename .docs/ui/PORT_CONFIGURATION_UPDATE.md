# Port Configuration Update Summary

**TEAM-293: Updated PORT_CONFIGURATION.md for hierarchical UI**

## What Changed

Updated `/home/vince/Projects/llama-orch/PORT_CONFIGURATION.md` from version 1.0 to 2.0 to reflect the new hierarchical UI architecture.

## New Ports Added

### Backend APIs (5 total)
- `7833` - queen-rbee (was 8500)
- `7835` - rbee-hive (was 9000)
- `8080` - llm-worker (new)
- `8188` - comfy-worker (new)
- `8000` - vllm-worker (new)

### Frontend Development (9 total)
- `5173` - keeper GUI (Tauri dev server)
- `7834` - queen-rbee UI dev server (new)
- `7836` - rbee-hive UI dev server (new)
- `7837` - llm-worker UI dev server (new)
- `7838` - comfy-worker UI dev server (new)
- `7839` - vllm-worker UI dev server (new)
- `6006` - Storybook (unchanged)
- `7822` - commercial (unchanged)
- `7811` - user-docs (unchanged)

### Deprecated
- `5179` - web-ui (old monolithic UI)

## Key Additions

### 1. Port Map Visualization
Added visual diagram at the top showing all ports at a glance.

### 2. Hierarchical UI Architecture Section
New section explaining:
- Development vs production port mapping
- How each binary hosts its own UI at `/ui`
- iframe integration in keeper GUI

### 3. Updated Files to Update Section
Comprehensive list of all files that need updating when ports change:
- Backend service files (main.rs, config.rs)
- Frontend vite.config.ts files
- SDK package base URLs
- All package.json dev scripts

### 4. Updated Quick Reference
New commands for:
- Starting all backend services
- Starting all frontend dev servers
- Health checks for all services
- Production UI URLs

## Port Assignment Strategy

**Backend APIs:**
- `7833-7839` - rbee services (sequential, grouped)
- `8000, 8080, 8188` - Worker APIs (standard ports)

**Frontend Dev:**
- `5173` - Keeper (Vite default for Tauri)
- `6006` - Storybook (standard)
- `7811, 7822` - Existing Next.js apps
- `7834-7839` - Component UIs (sequential)

## Production Deployment

In production, UIs are served by their respective binaries:

```
http://localhost:7833/ui  ← queen-rbee binary serves queen UI
http://localhost:7835/ui  ← rbee-hive binary serves hive UI
http://localhost:8080/ui  ← llm-worker binary serves worker UI
```

Development servers (7834, 7836, 7837) are only used during development.

## Total Ports Tracked

**Before (v1.0):** 7 ports  
**After (v2.0):** 14 ports

**Breakdown:**
- Backend APIs: 5
- Frontend Dev: 9
- Deprecated: 1

## Related Documentation

Updated links to:
- `.docs/ui/00_UI_ARCHITECTURE_OVERVIEW.md`
- `.docs/ui/FOLDER_STRUCTURE.md`
- `.docs/ui/PACKAGE_STRUCTURE.md`

## Changelog Entry

```
| 2025-10-25 | 2.0 | Updated for hierarchical UI architecture (keeper, queen, hive, workers) |
```

---

**Status:** ✅ COMPLETE  
**Document:** PORT_CONFIGURATION.md updated to v2.0  
**Impact:** Now tracks 14 ports (7 more than before)
