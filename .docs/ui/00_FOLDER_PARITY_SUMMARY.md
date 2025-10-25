# Folder Co-location: UIs in bin/

**TEAM-293: UIs co-located with their binaries**

## Quick Summary

**Before:** UIs in `frontend/apps/`, separate from binaries  
**After:** UIs in `bin/*/ui/`, co-located with their binaries

## New Structure

```
bin/
├── 00_rbee_keeper/
│   ├── src/              # Binary source
│   └── ui/               # Tauri GUI (no app subfolder)
├── 10_queen_rbee/
│   ├── src/              # Binary source
│   └── ui/
│       ├── app/          # Queen UI
│       └── packages/     # Queen SDK packages
├── 20_rbee_hive/
│   ├── src/              # Binary source
│   └── ui/
│       ├── app/          # Hive UI
│       └── packages/     # Hive SDK packages
└── 30_llm_worker_rbee/
    ├── src/              # Binary source
    └── ui/
        ├── app/          # Worker UI
        └── packages/     # Worker SDK packages
```

## Key Changes

### 1. Keeper GUI Co-located
**From:** `frontend/apps/00_rbee_keeper/`  
**To:** `bin/00_rbee_keeper/ui/`

**Why:** Co-locate UI with binary, no app subfolder (keeper doesn't need packages)

### 2. Queen UI Co-located
**From:** `frontend/apps/10_queen_rbee/`  
**To:** `bin/10_queen_rbee/ui/app/`

**Why:** Co-locate UI with binary, packages in `ui/packages/`

### 3. Hive UI Co-located
**From:** `frontend/apps/20_rbee_hive/`  
**To:** `bin/20_rbee_hive/ui/app/`

**Why:** Co-locate UI with binary, packages in `ui/packages/`

### 4. Worker UIs Co-located
**From:** `frontend/apps/30_llm_worker_rbee/`  
**To:** `bin/30_llm_worker_rbee/ui/app/`

**Why:** Co-locate UI with binary, packages in `ui/packages/`

## pnpm-workspace.yaml

```yaml
packages:
  # Commercial & Docs
  - frontend/apps/commercial
  - frontend/apps/user-docs
  - frontend/apps/web-ui  # DEPRECATED
  
  # rbee UIs (co-located in bin/)
  - bin/00_rbee_keeper/ui
  - bin/10_queen_rbee/ui/app
  - bin/10_queen_rbee/ui/packages/*
  - bin/20_rbee_hive/ui/app
  - bin/20_rbee_hive/ui/packages/*
  - bin/30_llm_worker_rbee/ui/app
  - bin/30_llm_worker_rbee/ui/packages/*
  
  # Shared packages
  - frontend/packages/rbee-ui
  - frontend/packages/rbee-sdk      # DEPRECATED
  - frontend/packages/rbee-react    # DEPRECATED
  - frontend/packages/tailwind-config
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
    "distDir": "../ui/dist"
  }
}
```

**Path explanation:**
- From: `bin/00_rbee_keeper/`
- Go up 1 level: `../`
- Enter: `ui/dist`

## Migration Commands

```bash
# 1. Move keeper GUI to bin/
mkdir -p bin/00_rbee_keeper/ui
mv frontend/apps/00_rbee_keeper/* bin/00_rbee_keeper/ui/
rmdir frontend/apps/00_rbee_keeper

# 2. Move queen UI to bin/
mkdir -p bin/10_queen_rbee/ui/app
mv frontend/apps/10_queen_rbee/* bin/10_queen_rbee/ui/app/
rmdir frontend/apps/10_queen_rbee

# 3. Move queen packages to bin/
mkdir -p bin/10_queen_rbee/ui/packages
mv frontend/packages/10_queen_rbee/* bin/10_queen_rbee/ui/packages/
rmdir frontend/packages/10_queen_rbee

# 4. Move hive UI to bin/
mkdir -p bin/20_rbee_hive/ui/app
mv frontend/apps/20_rbee_hive/* bin/20_rbee_hive/ui/app/
rmdir frontend/apps/20_rbee_hive

# 5. Move hive packages to bin/
mkdir -p bin/20_rbee_hive/ui/packages
mv frontend/packages/20_rbee_hive/* bin/20_rbee_hive/ui/packages/
rmdir frontend/packages/20_rbee_hive

# 6. Move worker UI to bin/
mkdir -p bin/30_llm_worker_rbee/ui/app
mv frontend/apps/30_llm_worker_rbee/* bin/30_llm_worker_rbee/ui/app/
rmdir frontend/apps/30_llm_worker_rbee

# 7. Move worker packages to bin/
mkdir -p bin/30_llm_worker_rbee/ui/packages
mv frontend/packages/30_llm_worker_rbee/* bin/30_llm_worker_rbee/ui/packages/
rmdir frontend/packages/30_llm_worker_rbee

# 8. Update tauri.conf.json
# Change distDir to: ../ui/dist

# 9. Update pnpm-workspace.yaml
# Update to use bin/* paths

# 10. Reinstall
pnpm install
```

## Benefits

✅ **Co-location:** Binary and UI in same directory  
✅ **Easy navigation:** `cd bin/10_queen_rbee/src` → `cd ../ui/app`  
✅ **Clear ownership:** Each component owns its UI  
✅ **Scalable:** Add new component = add one directory  
✅ **Self-contained:** Everything for one component in one place

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
