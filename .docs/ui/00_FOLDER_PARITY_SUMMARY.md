# Folder Parity: bin/ ↔ frontend/apps/

**TEAM-293: Aligned folder structure for clarity**

## Quick Summary

**Before:** UIs scattered, inconsistent naming  
**After:** Clear 1:1 mapping between binaries and UIs

## New Structure

```
bin/                          frontend/apps/
├── 00_rbee_keeper/     ↔    ├── 00_rbee_keeper/
├── 10_queen_rbee/      ↔    ├── 10_queen_rbee/
├── 20_rbee_hive/       ↔    ├── 20_rbee_hive/
└── 30_llm_worker_rbee/ ↔    └── 30_llm_worker_rbee/
```

## Key Changes

### 1. Keeper GUI Moved
**From:** `bin/00_rbee_keeper/GUI/`  
**To:** `frontend/apps/00_rbee_keeper/`

**Why:** Keeps all UIs in one place, matches numbering

### 2. Queen UI Renamed
**From:** `frontend/apps/web-ui/`  
**To:** `frontend/apps/10_queen_rbee/`

**Why:** Matches binary name and number

### 3. Hive UI Created
**New:** `frontend/apps/20_rbee_hive/`

**Why:** Matches binary name and number

### 4. Worker UIs Created
**New:** 
- `frontend/apps/30_llm_worker_rbee/`
- `frontend/apps/30_comfy_worker_rbee/`
- `frontend/apps/30_vllm_worker_rbee/`

**Why:** Matches binary prefix (all workers are 30_)

## pnpm-workspace.yaml

```yaml
packages:
  # Commercial & Docs
  - frontend/apps/commercial
  - frontend/apps/user-docs
  
  # rbee UIs (numbered)
  - frontend/apps/00_rbee_keeper
  - frontend/apps/10_queen_rbee
  - frontend/apps/20_rbee_hive
  - frontend/apps/30_*_worker_rbee
  
  # Shared packages
  - frontend/packages/*
```

## Package Names

| UI | Package Name |
|----|--------------|
| Keeper | `@rbee/00-keeper-gui` |
| Queen | `@rbee/10-queen-ui` |
| Hive | `@rbee/20-hive-ui` |
| LLM Worker | `@rbee/30-llm-worker-ui` |
| ComfyUI Worker | `@rbee/30-comfy-worker-ui` |
| vLLM Worker | `@rbee/30-vllm-worker-ui` |

## Tauri Configuration

**File:** `bin/00_rbee_keeper/tauri.conf.json`

```json
{
  "build": {
    "devPath": "http://localhost:5173",
    "distDir": "../../../frontend/apps/00_rbee_keeper/dist"
  }
}
```

**Path explanation:**
- From: `bin/00_rbee_keeper/`
- Go up 3 levels: `../../../`
- Enter: `frontend/apps/00_rbee_keeper/dist`

## Migration Commands

```bash
# 1. Move keeper GUI
mkdir -p frontend/apps/00_rbee_keeper
mv bin/00_rbee_keeper/GUI/* frontend/apps/00_rbee_keeper/
rmdir bin/00_rbee_keeper/GUI

# 2. Rename queen UI
mv frontend/apps/web-ui frontend/apps/10_queen_rbee

# 3. Create hive UI
mkdir frontend/apps/20_rbee_hive

# 4. Create worker UIs
mkdir frontend/apps/30_llm_worker_rbee
mkdir frontend/apps/30_comfy_worker_rbee
mkdir frontend/apps/30_vllm_worker_rbee

# 5. Update tauri.conf.json
# Change distDir to: ../../../frontend/apps/00_rbee_keeper/dist

# 6. Update pnpm-workspace.yaml
# Add numbered app entries

# 7. Reinstall
pnpm install
```

## Benefits

✅ **Clear correspondence:** Same number = related components  
✅ **Easy navigation:** Jump between binary and UI  
✅ **Consistent naming:** No more confusion  
✅ **Scalable:** Easy to add new components  
✅ **Professional:** Organized structure

## Documentation Updated

All documentation files updated to reflect new structure:
- ✅ `00_UI_ARCHITECTURE_OVERVIEW.md`
- ✅ `01_KEEPER_GUI_SETUP.md`
- ⚠️ `02_RENAME_WEB_UI.md` (needs update)
- ⚠️ `03_EXTRACT_KEEPER_PAGE.md` (needs update)
- ⚠️ `04_CREATE_HIVE_UI.md` (needs update)
- ⚠️ `05_CREATE_WORKER_UIS.md` (needs update)

## Next Steps

1. Read `FOLDER_STRUCTURE.md` for complete details
2. Follow migration commands above
3. Update remaining documentation files
4. Test that everything works

---

**Status:** 📋 STRUCTURE DEFINED  
**Impact:** All UI locations changed for better organization
