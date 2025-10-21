# TEAM-190: Status Command & Hive Heartbeat Implementation

**Date:** 2025-10-21  
**Status:** ✅ Complete

## Summary

Implemented two major features:
1. **`rbee status` command**: Displays live status of all hives and workers from hive-registry
2. **Hive heartbeat system**: rbee-hive sends heartbeats every 5 seconds to queen-rbee

These features work together to provide real-time visibility into the system's runtime state.

## Changes Overview

### 1. New Top-Level Command

Added `rbee status` as a base command (not a subcommand) that queries the hive-registry for live runtime state.

### 2. Files Modified

#### CLI Layer (`bin/00_rbee_keeper/`)
- **`src/main.rs`**
  - Added `Status` variant to `Commands` enum (line 65)
  - Added routing to `Operation::Status` (lines 280-284)
  - TEAM-190 annotations added

#### Operations Layer (`bin/99_shared_crates/rbee-operations/`)
- **`src/lib.rs`**
  - Added `Status` variant to `Operation` enum (line 37)
  - Added to `name()` method returning "status" (line 182)
  - Does not have `hive_id()` - system-wide operation
  - TEAM-190 annotations added

#### Server Layer (`bin/10_queen_rbee/`)
- **`src/job_router.rs`**
  - Added `HiveRegistry` import (line 31)
  - Added `hive_registry: Arc<HiveRegistry>` to `JobState` struct (line 44)
  - Updated `execute_job()` to clone and pass `hive_registry` (line 81)
  - Updated `route_operation()` signature to accept `hive_registry` parameter (line 95)
  - Implemented `Operation::Status` handler (lines 115-178):
    - Queries `list_active_hives(30_000)` for hives with recent heartbeats
    - Iterates through hives and collects workers
    - Formats as JSON array with columns: hive, worker, state, model, url
    - Uses `narration().table()` for formatted output
    - Shows helpful message if no active hives found

- **`src/http/jobs.rs`**
  - Added `hive_registry` field to `SchedulerState` struct (line 28)
  - Updated `From<SchedulerState>` impl to include `hive_registry` (line 37)

- **`src/main.rs`**
  - Updated `create_router()` to pass `hive_registry` to `job_state` (lines 124-128)

## Architecture

```
rbee-keeper (CLIENT)
  └─ rbee status
     └─ Operation::Status

         ↓ HTTP + SSE

queen-rbee (SERVER)
  ├─ Routes to Status handler
  ├─ Queries HiveRegistry.list_active_hives(30_000)
  ├─ For each hive: get_hive_state()
  ├─ Collects all workers into JSON array
  └─ Uses narration().table() for formatted output

         ↓ Queries

HiveRegistry (RAM)
  └─ Runtime state from heartbeats
     ├─ Hive IDs with recent heartbeats (<30s)
     ├─ Worker info per hive
     └─ NOT from catalog (persistent config)
```

## Key Features

### 1. Live Runtime State
- Queries **hive-registry** (RAM), not hive-catalog (SQLite)
- Only shows hives with recent heartbeats (last 30 seconds)
- Shows actual running workers, not configured workers

### 2. Table Output
Uses `narration().table()` for formatted CLI display:
```
hive      │ worker    │ state │ model        │ url
──────────┼───────────┼───────┼──────────────┼─────────────────────
localhost │ worker-01 │ Idle  │ llama-3-8b   │ http://localhost:9300
localhost │ worker-02 │ Busy  │ llama-3-8b   │ http://localhost:9301
```

### 3. Empty State Handling
Shows helpful message when no active hives:
```
No active hives found.

Hives must send heartbeats to appear here.

To start a hive:

  ./rbee hive start
```

### 4. Comprehensive Narration
- Initial: "📊 Fetching live status from registry"
- Result: "Live Status (X hive(s), Y worker(s)):"
- Table with all hive/worker data

## Usage Examples

### No Active Hives
```bash
$ ./rbee status

[👑 queen-router] 📊 Fetching live status from registry
[👑 queen-router] No active hives found.

Hives must send heartbeats to appear here.

To start a hive:

  ./rbee hive start
```

### With Active Hives (once heartbeats are working)
```bash
$ ./rbee status

[👑 queen-router] 📊 Fetching live status from registry
[👑 queen-router] Live Status (2 hive(s), 3 worker(s)):

hive      │ worker    │ state │ model        │ url
──────────┼───────────┼───────┼──────────────┼─────────────────────
localhost │ worker-01 │ Idle  │ llama-3-8b   │ http://localhost:9300
localhost │ worker-02 │ Busy  │ llama-3-8b   │ http://localhost:9301
remote-01 │ worker-03 │ Idle  │ mistral-7b   │ http://10.0.0.5:9300
```

## Implementation Notes

### Following TEAM-190 Patterns
✅ **4-Layer Pattern**: CLI → Operation → Router → Registry  
✅ **Narration**: Pre-flight, progress, result with table  
✅ **Error Messages**: Actionable guidance with exact commands  
✅ **TEAM Annotations**: All changes marked with TEAM-190  
✅ **Documentation**: Inline comments explaining purpose  

### Key Differences from `hive list`
| Command | Source | Shows |
|---------|--------|-------|
| `rbee hive list` | Catalog (SQLite) | All registered hives (config) |
| `rbee status` | Registry (RAM) | Only live hives with workers |

### Registry vs Catalog
- **Catalog** = Persistent configuration (host, port, SSH, devices)
- **Registry** = Runtime state (workers, VRAM, last heartbeat)
- Status command uses **Registry** to show what's actually running NOW

## Testing

✅ Build successful  
✅ Command routing works  
✅ Empty state shows helpful message  
✅ Table formatting ready (pending heartbeats)  
⏳ Full test with workers pending hive heartbeat implementation  

## Part 2: Hive Heartbeat System

### rbee-hive Changes (`bin/20_rbee_hive/`)

#### `Cargo.toml`
- Added `rbee-heartbeat` dependency
- Added `observability-narration-core` dependency

#### `src/main.rs`
- Added CLI args: `--hive-id` and `--queen-url`
- Implemented `WorkerStateProvider` trait (returns empty vec for now)
- Started heartbeat task on daemon startup:
  - 5 second interval (as requested)
  - Sends `HiveHeartbeatPayload` to queen
  - Non-blocking background task

### Heartbeat Flow
```
rbee-hive → (every 5s) → POST /v1/heartbeat → queen-rbee → hive-registry (RAM)
```

### Testing Results
✅ Hive appears in `rbee status` after heartbeat  
✅ Hive disappears from status 30 seconds after stopping  
✅ Heartbeats sent continuously every 5 seconds  
✅ Multiple hives can send heartbeats simultaneously  

## Metrics

- **Files changed**: 7 (5 for status, 2 for heartbeat)
- **Lines added**: ~165
- **Lines removed**: ~10
- **Net addition**: ~155 LOC
- **New dependencies**: 2 (rbee-heartbeat, observability-narration-core)

## Future Work

- Worker spawning (to populate workers in registry)
- Worker heartbeats (workers → hive → queen aggregation)
- Enhanced table columns (VRAM, health status, uptime)
- Filtering options (--hive-id, --state)
- JSON output format (--format json)
- Authentication tokens for heartbeat endpoint

## Verification Checklist

### Status Command
- [x] CLI command defined
- [x] Operation variant added
- [x] Operation name mapping
- [x] CLI routing implemented
- [x] Server handler implemented
- [x] State threading (hive_registry added to JobState)
- [x] Narration with table output
- [x] Empty state handling
- [x] Build successful
- [x] Tested with live hives

### Hive Heartbeat
- [x] WorkerStateProvider trait implemented
- [x] Heartbeat task started on daemon startup
- [x] 5 second interval configured
- [x] Heartbeat payload sent to queen
- [x] Queen receives and stores in registry
- [x] Status shows live hives
- [x] Dead hives timeout correctly (30s)
- [x] Build successful
- [x] Integration tested

---

**TEAM-190 Complete** ✅
