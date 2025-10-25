# Port Configuration Audit - TEAM-294

**Date:** Oct 25, 2025  
**Status:** ✅ COMPLETE

## Mission

Audit all hardcoded ports in the codebase and document them in `PORT_CONFIGURATION.md` for easy maintenance.

## Findings

### Port Mismatches Fixed

1. **rbee-hive:** Was using port **7844**, changed to **7835** to match PORT_CONFIGURATION.md
   - File: `bin/20_rbee_hive/src/main.rs` line 41

### Comprehensive Port Documentation Added

Updated `PORT_CONFIGURATION.md` with complete list of all files containing hardcoded ports:

#### Backend Services

**queen-rbee (Port 7833):**
- 9 files with hardcoded references
- Includes: main.rs, http endpoints, config, lifecycle, documentation

**rbee-hive (Port 7835):**
- 2 files with hardcoded references
- Now correctly using 7835 (was 7844)

**Workers (Ports 8080, 8188, 8000):**
- Default port arguments in main.rs for each worker type

#### Frontend Services

**Development Ports (5173, 7834, 7836, 7837-7839):**
- Vite config files for each UI
- Correct paths documented (in `bin/` not `frontend/apps/`)

#### Documentation & Examples

**Test/Example Code:**
- narration-core examples (uses 8080)
- auth-min tests (uses 8080)
- http server examples (uses 8080)

## Files Modified

1. **`PORT_CONFIGURATION.md`** - Added comprehensive port reference list with line numbers
2. **`bin/20_rbee_hive/src/main.rs`** - Fixed default port (7844 → 7835)

## Benefits

✅ **Single source of truth** - All port locations documented in one place  
✅ **Easy maintenance** - When changing ports, know exactly which files to update  
✅ **Consistency** - All services now use ports according to PORT_CONFIGURATION.md  
✅ **Line numbers** - Exact locations provided for quick navigation  

## Port Reference Summary

| Service | Port | Files with Hardcoded References |
|---------|------|--------------------------------|
| keeper-ui | 5173 | 1 (vite.config.ts) |
| queen-rbee | 7833 | 9 (main, http, config, lifecycle, docs) |
| queen-ui | 7834 | 1 (vite.config.ts) |
| rbee-hive | 7835 | 2 (main, docs) |
| hive-ui | 7836 | 1 (vite.config.ts) |
| llm-worker-ui | 7837 | 1 (vite.config.ts) |
| llm-worker | 8080 | 1 (main) + examples |
| comfy-worker | 8188 | 1 (main) |
| vllm-worker | 8000 | 1 (main) |
| storybook | 6006 | 1 (config) |

## Next Steps

When changing a port in the future:
1. Update `PORT_CONFIGURATION.md` port table
2. Check the "Files to Update" section for that port
3. Update all listed files
4. Update any documentation/examples if needed

## Verification

All ports now match PORT_CONFIGURATION.md specification:
- ✅ queen-rbee: 7833
- ✅ rbee-hive: 7835 (fixed from 7844)
- ✅ All UIs: 5173, 7834, 7836, 7837-7839
- ✅ Workers: 8000, 8080, 8188
