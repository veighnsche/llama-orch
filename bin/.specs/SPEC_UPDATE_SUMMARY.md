# Spec Update Summary - queen-rbee Orchestration

**Date:** 2025-10-10T14:02  
**Updated By:** TEAM-037  
**Status:** COMPLETE

---

## What Was Updated

All spec documents in `/bin/.specs/` have been updated to reflect the correct architecture:

### 🚨 CRITICAL CHANGES 🚨

1. **rbee-keeper is a TESTING TOOL** - NOT for production
2. **queen-rbee orchestrates everything** - uses SSH to control hives
3. **WHENEVER queen-rbee dies, ALL hives and workers die gracefully**

---

## Files Updated

### ✅ FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md
- Added warning: rbee-keeper is testing tool
- Updated architecture diagrams with queen-rbee orchestration
- Added Testing Mode vs Production Mode sections
- Added cascading shutdown flow

### ✅ COMPONENT_RESPONSIBILITIES_FINAL.md
- Added warning: rbee-keeper is testing tool
- Updated binary table with Production? column
- Changed rbee-hive from CLI to Daemon (HTTP)
- Added cascading shutdown rule

### ✅ ARCHITECTURE_MODES.md
- Added warning: rbee-keeper is testing tool
- Updated Ephemeral Mode with full 17-step lifecycle
- Updated Persistent Mode with SDK usage
- Added queen-rbee orchestration flow

### ✅ ARCHITECTURE_UPDATE.md
- Created by TEAM-037
- Documents the architecture change
- Explains rbee-keeper purpose
- Shows correct flow diagrams

### ✅ CRITICAL_RULES.md
- Created by TEAM-037
- P0 rules document
- Testing tool clarification
- Cascading shutdown guarantee

### ✅ LIFECYCLE_CLARIFICATION.md
- Created by TEAM-037
- Detailed lifecycle rules
- Process ownership
- Implementation checklist

### ✅ TEAM_037_COMPLETION_SUMMARY.md
- Created by TEAM-037
- Summary of BDD work
- Lifecycle clarifications
- Test coverage

---

## Key Architecture Changes

### OLD (INCORRECT)
```
rbee-keeper → rbee-hive → worker
```

### NEW (CORRECT)
```
rbee-keeper (TESTING TOOL)
    ↓ spawns
queen-rbee (orchestrator)
    ↓ SSH
rbee-hive (pool manager)
    ↓ spawns
llm-worker-rbee (worker)
```

---

## The 4 Components

| Component | Type | Production? | Purpose |
|-----------|------|-------------|---------|
| **rbee-keeper** | CLI | ❌ NO | **TESTING TOOL** - Integration tester |
| **queen-rbee** | HTTP Daemon | ✅ YES | Orchestrator, controls hives via SSH |
| **rbee-hive** | HTTP Daemon | ✅ YES | Pool manager, spawns workers |
| **llm-worker-rbee** | HTTP Daemon | ✅ YES | Worker, executes inference |

---

## Critical Rules

### RULE 1: rbee-keeper is a TESTING TOOL
- **NOT for production** - use llama-orch SDK instead
- Spawns queen-rbee for testing
- Runs test, then kills everything
- Validates entire system works

### RULE 2: WHENEVER queen-rbee DIES, EVERYTHING DIES
- **NON-NEGOTIABLE** and **ALWAYS TRUE**
- queen-rbee death → ALL hives die (SSH SIGTERM)
- ALL hives die → ALL workers die (shutdown command)
- Guarantees: No orphaned processes, no leaked VRAM

### RULE 3: rbee-keeper ONLY kills queen-rbee IF it spawned it
- Ephemeral mode: rbee-keeper spawns queen-rbee → owns lifecycle
- Persistent mode: operator starts queen-rbee → rbee-keeper doesn't own

---

## Testing Mode Flow

```
Developer: rbee-keeper infer --node mac --model tinyllama --prompt "hello"

1. rbee-keeper spawns queen-rbee
2. queen-rbee SSH starts rbee-hive on mac
3. rbee-hive spawns worker
4. Worker ready → rbee-hive → queen-rbee
5. queen-rbee sends inference to worker
6. Worker streams SSE to queen-rbee
7. queen-rbee relays SSE to rbee-keeper stdout
8. Developer sees result
9. Developer Ctrl+C
10. rbee-keeper kills queen-rbee
11. queen-rbee SSH kills ALL hives
12. Hives kill ALL workers
13. Everything dies gracefully
```

---

## Production Mode Flow

```
Operator: queen-rbee daemon &
Operator: ssh mac "rbee-hive daemon &"

User App: llama-orch-sdk → queen-rbee (HTTP POST /v2/tasks)
    ↓
queen-rbee → worker (HTTP POST /execute)
    ↓
worker → queen-rbee (SSE stream)
    ↓
queen-rbee → SDK (SSE stream)
    ↓
User App gets result

# Daemons continue running
# No rbee-keeper involved
```

---

## Cascading Shutdown Guarantee

```
queen-rbee dies (SIGTERM, crash, kill, etc.)
    ↓
queen-rbee's shutdown handler runs
    ↓
For each rbee-hive that queen-rbee spawned:
    SSH to node
    Send SIGTERM to rbee-hive
    ↓
rbee-hive receives SIGTERM
    ↓
For each worker:
    HTTP POST /v1/admin/shutdown
    ↓
Worker unloads model from VRAM
Worker exits cleanly
    ↓
rbee-hive exits cleanly
    ↓
queen-rbee exits cleanly
    ↓
System completely clean
```

---

## Files That Still Need Updating

### Test Scenarios
- [ ] `/test-harness/bdd/tests/features/test-001.feature`
- [ ] `/test-harness/bdd/tests/features/test-001-mvp.feature`

These need to be updated to include queen-rbee in all flows.

### Other Specs (Lower Priority)
- [ ] `BINARY_ARCHITECTURE_COMPLETE.md`
- [ ] `BINARY_STRUCTURE_CLARIFICATION.md`
- [ ] `COMPLETE_BINARY_ARCHITECTURE.md`
- [ ] `CONTROL_PLANE_ARCHITECTURE_DECISION.md`
- [ ] `ARCHITECTURE_DECISION_NO_POOL_DAEMON.md`

These are older documents that may have outdated information.

---

## Summary

**What changed:**
- rbee-keeper is now clearly documented as a TESTING TOOL
- queen-rbee orchestrates everything via SSH
- Cascading shutdown is guaranteed (queen dies → everything dies)
- Testing mode vs Production mode is clearly distinguished

**Why it matters:**
- Prevents confusion about rbee-keeper's purpose
- Ensures clean shutdown in testing (no orphaned processes)
- Clear separation between testing and production usage
- Proper lifecycle management

**Next steps:**
- Update test scenarios to include queen-rbee
- Implement queen-rbee orchestration (M1)
- Implement cascading shutdown handlers

---

**Updated by:** TEAM-037 (Testing Team)  
**Date:** 2025-10-10T14:02  
**Status:** COMPLETE

---
Verified by Testing Team 🔍
