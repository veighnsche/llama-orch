# TEAM-314: Port Configuration Update

**Status:** ✅ COMPLETE  
**Date:** 2025-10-27  
**Purpose:** Update all hardcoded ports to match PORT_CONFIGURATION.md

---

## Changes Made

### Port Updates

Updated all hardcoded port references to match the official port configuration:

- **queen-rbee**: 8500 → **7833**
- **rbee-hive**: 9000 → **7835**

### Files Modified

#### 1. rbee-keeper (bin/00_rbee_keeper/)

**src/handlers/hive.rs:**
- Line 94: Updated `HiveStatus` hive URL from `http://localhost:9000` to `http://localhost:7835`
- Line 97: Updated comment from port 9000 to 7835
- Line 114: Updated `HiveCheck` hive URL from `http://localhost:9000` to `http://localhost:7835`

**src/cli/hive.rs:**
- Line 73-74: Updated `HiveAction::Start` default port from 9000 to 7835

**src/config.rs:**
- Already correct (7833) - no changes needed

#### 2. hive-lifecycle (bin/05_rbee_keeper_crates/hive-lifecycle/)

**src/start.rs:**
- Line 22: Updated documentation comment from "default: 9000" to "default: 7835"
- Line 23: Updated documentation comment from "http://localhost:8500" to "http://localhost:7833"
- Line 34: Updated default queen URL from `http://localhost:8500` to `http://localhost:7833`

**src/lib.rs:**
- Line 34: Updated example from `start_hive("gpu-server", "/usr/local/bin", 9000).await?;` to `start_hive("gpu-server", "/usr/local/bin", 7835, None).await?;`

#### 3. queen-lifecycle (bin/05_rbee_keeper_crates/queen-lifecycle/)

**src/lib.rs:**
- Line 36: Updated example from `http://localhost:8500` to `http://localhost:7833`
- Line 51: Updated example from `http://localhost:8500` to `http://localhost:7833`
- Line 53: Updated example from `http://localhost:8500` to `http://localhost:7833`
- Line 65: Updated example from `http://localhost:8500` to `http://localhost:7833`

**src/status.rs:**
- Line 14: Updated documentation from `http://localhost:8500` to `http://localhost:7833`

**src/start.rs:**
- Line 16: Updated documentation from `http://localhost:8500` to `http://localhost:7833`

**src/stop.rs:**
- Line 16: Updated documentation from `http://localhost:8500` to `http://localhost:7833`

**src/info.rs:**
- Line 15: Updated documentation from `http://localhost:8500` to `http://localhost:7833`

**src/health.rs:**
- Line 16: Updated documentation from `http://localhost:8500` to `http://localhost:7833`

**src/ensure.rs:**
- Line 29: Updated documentation from `http://localhost:8500` to `http://localhost:7833`

#### 4. Already Correct (No Changes Needed)

These files already had the correct ports:

- `bin/10_queen_rbee/src/main.rs` - Already uses 7833
- `bin/20_rbee_hive/src/main.rs` - Already uses 7835 for hive, 7833 for queen
- `bin/00_rbee_keeper/src/config.rs` - Already uses 7833

---

## Verification

✅ **Compilation:** `cargo build --bin rbee-keeper` - SUCCESS  
✅ **Port Consistency:** All source files now match PORT_CONFIGURATION.md  
✅ **Documentation:** All doc comments updated with correct ports

---

## Port Reference (from PORT_CONFIGURATION.md)

### Backend Services (HTTP APIs)

| Service | Port | Description |
|---------|------|-------------|
| **queen-rbee** | `7833` | Orchestrator daemon (HTTP API) |
| **rbee-hive** | `7835` | Hive daemon (HTTP API) |
| **llm-worker** | `8080` | LLM worker (HTTP API) |
| **comfy-worker** | `8188` | ComfyUI worker (HTTP API) |
| **vllm-worker** | `8000` | vLLM worker (HTTP API) |

---

## Impact

### Breaking Changes

⚠️ **Users with existing installations must update:**

1. **Config files:** Update `~/.config/rbee/config.toml` if manually configured
2. **Scripts:** Update any scripts that hardcode port 8500 or 9000
3. **SSH tunnels:** Update any SSH tunnels pointing to old ports

### Migration

**Old commands:**
```bash
curl http://localhost:8500/health  # queen (OLD)
curl http://localhost:9000/health  # hive (OLD)
```

**New commands:**
```bash
curl http://localhost:7833/health  # queen (NEW)
curl http://localhost:7835/health  # hive (NEW)
```

---

## Related Work

- **TEAM-313:** HiveCheck command (uses direct hive connection)
- **TEAM-314:** Keeper manages hive lifecycle directly (not through queen)
- **PORT_CONFIGURATION.md:** Central port registry (v2.0, updated 2025-10-25)

---

## Notes

- All changes follow PORT_CONFIGURATION.md as the single source of truth
- Documentation comments updated to reflect new ports
- Example code in doc comments updated
- No functional changes - only port number updates
- Compilation successful with no errors

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27
