# Heartbeat Crate Migration Notes

**Date:** 2025-10-19  
**Action:** Moved heartbeat crate from `worker-rbee-crates/` to `shared-crates/`

---

## Why This Move?

### Original Assumption (WRONG)
- Heartbeat was assumed to be worker-specific
- Located in `bin/worker-rbee-crates/heartbeat/`
- Only workers send heartbeats

### Actual Architecture (CORRECT)
- **TWO-LEVEL HEARTBEAT CHAIN:**
  1. Worker → Hive (30s interval)
  2. Hive → Queen (15s interval)
- Both workers AND hives need to send heartbeats
- Same mechanism, different payloads

### Decision
✅ **Move to `shared-crates/`** because it's used by multiple binaries

---

## What Changed

### File Moves
```
FROM: bin/worker-rbee-crates/heartbeat/
TO:   bin/shared-crates/heartbeat/
```

### Crate Rename
```
FROM: worker-rbee-heartbeat (worker_rbee_heartbeat)
TO:   rbee-heartbeat (rbee_heartbeat)
```

### README Updates
- Removed worker-specific bias
- Added two-level heartbeat chain diagram
- Explained both use cases (worker and hive)
- Documented aggregation logic
- Added generic API design with `HeartbeatConfig<T>`

---

## Implementation Status

### Current State
- ✅ Crate moved to `shared-crates/`
- ✅ README updated (generic, not worker-specific)
- ✅ Cargo.toml updated (renamed to `rbee-heartbeat`)
- ⏳ Implementation still needed (stub only)

### What Exists in `.bak`
- ✅ Worker → Hive heartbeat (in `llm-worker-rbee.bak/src/heartbeat.rs`)
- ❌ Hive → Queen heartbeat (MISSING - needs to be implemented)

### Next Steps
1. Migrate worker heartbeat implementation from `.bak`
2. Make API generic (support both worker and pool payloads)
3. Implement hive → queen heartbeat (new functionality)
4. Update both binaries to use this shared crate

---

## Key Insight: Asynchronous Collection + Synchronous Aggregation

**Workers send heartbeats asynchronously:**
- Each worker has its own 30s timer
- No coordination between workers
- Hive receives heartbeats at different times

**Hive aggregates synchronously:**
- Hive maintains in-memory worker registry
- Every 15s, hive takes snapshot of current state
- Snapshot includes: GPU VRAM + all worker states
- Sent to queen-rbee in single payload

**Why this works:**
- Worker interval (30s) < Hive interval (15s) × 3 = 45s timeout
- Queen always sees fresh data (<45s old)
- No synchronization needed between workers

---

## Architecture Alignment

This move aligns with TEAM-130E/130G consolidation analysis:

**From TEAM-130E_CONSOLIDATION_SUMMARY.md:**
> Expected NEW shared crates:
> - daemon-lifecycle (~500-800 LOC savings)
> - rbee-http-client (~200-400 LOC savings)
> - rbee-types/rbee-http-types (~300-500 LOC savings)

**Heartbeat is similar pattern:**
- Used by 2+ binaries (workers + hives)
- Generic/reusable logic
- ~200 LOC shared instead of duplicated

---

## Related Documentation

- **Main spec:** `bin/.specs/00_llama-orch.md` (SYS-6.2.4, SYS-6.3.1)
- **Architecture:** `bin/.plan/TEAM_130G_FINAL_ARCHITECTURE.md`
- **Consolidation:** `bin/.plan/TEAM_130E_CONSOLIDATION_SUMMARY.md`
- **This README:** `bin/shared-crates/heartbeat/README.md`
