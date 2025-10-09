# Simplified Architecture (2025-10-09)

**Decision:** pool-managerd daemon is NOT NEEDED  
**Impact:** M1 milestone simplified  
**Status:** NORMATIVE

---

## The Correct Architecture

### Two Daemons (HTTP)
1. **orchestratord** - Routes inference requests (M1)
2. **llorch-candled** - Executes inference (M0 ✅)

### Two CLIs (SSH/Local)
1. **llorch** - Remote control via SSH (M0 ✅)
2. **llorch-pool** - Local pool operations (M0 ✅)

---

## Why This Works

### Pool Manager Doesn't Need to Be a Daemon

**Pool operations are on-demand:**
- Download model → Run once, exits
- Spawn worker → Run once, spawns daemon, exits
- Stop worker → Run once, exits
- List workers → Run once, exits

**No 24/7 daemon needed!**

### What Needs to Be a Daemon

**Orchestrator:**
- ✅ Accepts client requests 24/7
- ✅ Maintains queue state
- ✅ Routes to workers
- ✅ Relays SSE streams

**Workers:**
- ✅ Keep model in VRAM 24/7
- ✅ Accept inference requests
- ✅ Stream tokens via SSE

---

## Communication Flow

### Control Plane (Operator → Pools)
```
Operator
    ↓ runs
llorch (CLI)
    ↓ SSH
llorch-pool (CLI on remote machine)
    ↓ spawns/manages
llorch-candled (daemon)
```

### Data Plane (Client → Inference)
```
Client
    ↓ HTTP POST /v2/tasks
orchestratord (daemon)
    ↓ HTTP POST /execute
llorch-candled (daemon)
    ↓ SSE stream
orchestratord (relay)
    ↓ SSE stream
Client
```

---

## What This Simplifies

### Before (Incorrect)
- 3 daemons: orchestratord, pool-managerd, workers
- 2 CLIs: llorch, llorch-pool
- **Total:** 5 binaries

### After (Correct)
- 2 daemons: orchestratord, workers
- 2 CLIs: llorch, llorch-pool
- **Total:** 4 binaries

**One less daemon = simpler architecture!**

---

## Milestone Changes

### Old Plan
- M0: Workers + CLIs ✅
- M1: Build pool-managerd
- M2: Build orchestratord

### New Plan
- M0: Workers + CLIs ✅
- ~~M1: pool-managerd~~ ❌ NOT NEEDED
- M1: Build orchestratord (moved up)

**M1 is now the only daemon left to build!**

---

## Files Updated

1. `/bin/.specs/ARCHITECTURE_DECISION_NO_POOL_DAEMON.md` (NEW)
2. `/bin/.plan/TEAM_025_HANDOFF.md` (updated)
3. `/ORCHESTRATION_OVERVIEW.md` (updated)
4. `/QUICK_STATUS.md` (updated)
5. `/ARCHITECTURE_SIMPLIFIED.md` (this file)

---

**Decision By:** User (Vince)  
**Documented By:** TEAM-024  
**Date:** 2025-10-09T17:17:00+02:00
