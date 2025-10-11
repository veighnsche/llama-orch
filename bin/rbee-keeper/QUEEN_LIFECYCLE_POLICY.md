# Queen-rbee Lifecycle Policy

**TEAM-085: When to auto-start queen-rbee**

---

## The Rule

**Only auto-start queen-rbee for commands that REQUIRE orchestration.**

Starting queen-rbee just to read empty logs or query a remote rbee-hive directly is wasteful and confusing.

---

## Commands That NEED Queen-rbee ✅

### `infer`
**Why:** Routes inference requests to remote nodes, manages worker lifecycle
**Auto-start:** YES

### `setup add-node`
**Why:** Registers nodes in queen-rbee's beehive registry
**Auto-start:** YES

### `setup list-nodes`
**Why:** Queries queen-rbee's beehive registry
**Auto-start:** YES (but only if registry has data)

### `setup remove-node`
**Why:** Removes nodes from queen-rbee's beehive registry
**Auto-start:** YES (but only if registry has data)

---

## Commands That DON'T Need Queen-rbee ❌

### `logs`
**Why:** Fetches logs directly from remote node via SSH
**Auto-start:** NO - Direct SSH operation, no orchestration needed

### `workers list`
**Why:** Queries remote rbee-hive directly via HTTP
**Auto-start:** NO - Direct HTTP to rbee-hive, no orchestration

### `workers health`
**Why:** Queries remote rbee-hive directly via HTTP
**Auto-start:** NO - Direct HTTP to rbee-hive, no orchestration

### `workers shutdown`
**Why:** Sends shutdown command directly to remote rbee-hive
**Auto-start:** NO - Direct HTTP to rbee-hive, no orchestration

### `hive` commands (TEAM-085: renamed from "pool")
**Why:** Direct SSH operations to manage remote rbee-hive instances
**Auto-start:** NO - SSH operations, no orchestration

### `install`
**Why:** Local file operations
**Auto-start:** NO - No network operations at all

---

## The Architecture

```
┌─────────────────────────────────────────────────────────┐
│ ORCHESTRATION (needs queen-rbee)                        │
├─────────────────────────────────────────────────────────┤
│ rbee-keeper infer → queen-rbee → rbee-hive → worker    │
│ rbee-keeper setup → queen-rbee registry                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ DIRECT OPERATIONS (no queen-rbee needed)                │
├─────────────────────────────────────────────────────────┤
│ rbee-keeper logs → SSH → remote node                    │
│ rbee-keeper workers → HTTP → rbee-hive                  │
│ rbee-keeper hive → SSH → rbee-hive                      │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation

**File:** `bin/rbee-keeper/src/queen_lifecycle.rs`

Only these commands should call `ensure_queen_rbee_running()`:
- `commands/infer.rs` ✅
- `commands/setup.rs` (only for add-node, list-nodes, remove-node) ✅

**Do NOT call it from:**
- `commands/logs.rs` ❌
- `commands/workers.rs` ❌
- `commands/pool.rs` ❌
- `commands/install.rs` ❌

---

## Why This Matters

**Bad:**
```bash
# User wants to see logs
$ rbee logs --node mac

⚠️  queen-rbee not running, starting...
🚀 Starting queen-rbee daemon...
✓ queen-rbee started successfully
# Then fetches logs via SSH anyway!
```

**Good:**
```bash
# User wants to see logs
$ rbee logs --node mac

# Directly SSHs to mac and streams logs
# No unnecessary daemon startup!
```

---

**Created by:** TEAM-085  
**Date:** 2025-10-11  
**Status:** POLICY DOCUMENT
