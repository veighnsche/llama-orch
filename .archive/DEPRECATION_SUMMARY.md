# Deprecation Summary

**Date:** 2025-10-19  
**Status:** Active deprecations documented

---

## Overview

This document provides a quick reference for all deprecated crates and their replacements.

---

## Active Deprecations

### ⚠️ DEPRECATED: shared-crates/gpu-info

**Replacement:** `bin/rbee-hive-crates/device-detection`

### Quick Facts

- **Deprecated:** 2025-10-19
- **Reason:** GPU detection is hive-specific, not shared
- **Status:** Marked as deprecated, pending removal

### Why Deprecated?

**Only rbee-hive needs GPU detection:**
- ✅ rbee-hive runs on GPU nodes and detects local devices
- ❌ Workers don't detect GPUs (told which GPU to use)
- ❌ Queen doesn't detect GPUs (receives GPU state from hives)
- ❌ Keeper doesn't detect GPUs (CLI tool)

### Migration

**Old (DEPRECATED):**
```rust
use gpu_info::{detect_gpus, GpuInfo};
let gpus = detect_gpus()?;
```

**New (CORRECT):**
```rust
use rbee_hive_device_detection::{detect_devices, DeviceInfo};
let devices = detect_devices()?;
```

### Documentation

- **Deprecation notice:** `bin/shared-crates/gpu-info/DEPRECATED.md`
- **Migration guide:** `bin/rbee-hive-crates/device-detection/MIGRATION_FROM_GPU_INFO.md`
- **Tracking:** `bin/.plan/DEPRECATIONS.md`

---

### ⚠️ DEPRECATED: shared-crates/hive-core

**Replacement:** `bin/shared-crates/rbee-types`

#### Quick Facts

- **Deprecated:** 2025-10-19
- **Reason:** Renamed to better reflect purpose
- **Status:** Implementation merged into rbee-types

#### Why Renamed?

**Better naming:**
- ✅ `rbee-types` reflects system-wide shared types
- ❌ `hive-core` implies hive-specific (misleading)

#### Migration

**Old (DEPRECATED):**
```rust
use hive_core::{WorkerInfo, Backend};
```

**New (CORRECT):**
```rust
use rbee_types::{WorkerInfo, Backend};
```

#### Documentation

- **Deprecation notice:** `bin/shared-crates/hive-core/DEPRECATED.md`
- **New location:** `bin/shared-crates/rbee-types/`
- **Tracking:** `bin/.plan/DEPRECATIONS.md`

---

## Completed Moves

### ✅ COMPLETED: worker-rbee-crates/heartbeat → shared-crates/heartbeat

**New Location:** `bin/shared-crates/heartbeat`

### Quick Facts

- **Moved:** 2025-10-19
- **Reason:** Both workers AND hives send heartbeats
- **Status:** Complete, old directory deleted

### Why Moved?

**Used by multiple binaries:**
- ✅ Workers send heartbeats to hives (30s interval)
- ✅ Hives send heartbeats to queen (15s interval)
- ✅ Same mechanism, different payloads
- ✅ ~200 LOC reused instead of duplicated

### Documentation

- **Migration notes:** `bin/shared-crates/heartbeat/MIGRATION_NOTES.md`
- **README:** `bin/shared-crates/heartbeat/README.md`
- **Tracking:** `bin/.plan/DEPRECATIONS.md`

---

## Action Items

### For gpu-info Deprecation

- [ ] Remove `shared-crates/gpu-info` from root `Cargo.toml`
- [ ] Update rbee-hive to use `rbee-hive-crates/device-detection`
- [ ] Verify no other crates depend on `gpu-info`
- [ ] Delete `bin/shared-crates/gpu-info/` directory

### For Other Unused Crates

See `bin/.plan/DEPRECATIONS.md` for:
- `shared-crates/hive-core` (unused, duplicate of rbee-types)
- `shared-crates/secrets-management` (declared but never used)

---

## Related Documentation

- **Full tracking:** `bin/.plan/DEPRECATIONS.md`
- **Architecture:** `bin/.plan/TEAM_130G_FINAL_ARCHITECTURE.md`
- **Consolidation:** `bin/.plan/TEAM_130E_CONSOLIDATION_SUMMARY.md`

---

## Quick Reference

| Deprecated Crate | Replacement | Status | Docs |
|------------------|-------------|--------|------|
| `shared-crates/gpu-info` | `rbee-hive-crates/device-detection` | DEPRECATED | [DEPRECATED.md](bin/shared-crates/gpu-info/DEPRECATED.md) |
| `shared-crates/hive-core` | `shared-crates/rbee-types` | DEPRECATED | [DEPRECATED.md](bin/shared-crates/hive-core/DEPRECATED.md) |
| `worker-rbee-crates/heartbeat` | `shared-crates/heartbeat` | COMPLETED | [MIGRATION_NOTES.md](bin/shared-crates/heartbeat/MIGRATION_NOTES.md) |

---

**Last Updated:** 2025-10-19
