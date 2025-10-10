# Lifecycle Clarification for TEST-001

**Created by:** TEAM-037 (Testing Team)  
**Date:** 2025-10-10  
**Updated:** 2025-10-10T13:54 - Added queen-rbee orchestration flow  
**Status:** NORMATIVE - Critical for implementation

---

## Problem Statement

The original test-001.md specification was ambiguous about **when processes die** and **who orchestrates what**. This caused confusion about:

1. Does rbee-keeper talk directly to rbee-hive or through queen-rbee?
2. Does rbee-hive exit after spawning a worker?
3. Does worker exit after completing inference?
4. Does rbee-keeper stay running?
5. Who controls the lifecycle of whom?

## Correct Architecture (UPDATED)

**rbee-keeper is a TESTING TOOL that spawns queen-rbee for integration testing:**

```
rbee-keeper (CLI - TESTING TOOL, NOT PRODUCTION)
    ↓ spawns for testing
queen-rbee (orchestrator daemon)
    ↓ SSH to remote node
rbee-hive (pool manager daemon)
    ↓ spawns
llm-worker-rbee (worker daemon)
    ↓ ready callback via HTTP
rbee-hive
    ↓ notifies via HTTP
queen-rbee
    ↓ sends inference via HTTP
llm-worker-rbee
    ↓ SSE token stream
queen-rbee (relays stream)
    ↓ SSE to stdout
rbee-keeper (displays tokens)
```

## 🚨 CRITICAL CASCADING SHUTDOWN RULE 🚨

**WHENEVER queen-rbee DIES, ALL rbee-hive instances and ALL workers STOP GRACEFULLY!**

This is **NON-NEGOTIABLE** and **ALWAYS TRUE**:

```
queen-rbee dies (SIGTERM, crash, kill, etc.)
    ↓
ALL rbee-hive instances on ALL nodes receive SIGTERM via SSH
    ↓
ALL workers on ALL nodes receive shutdown command
    ↓
ALL workers unload models and exit gracefully
    ↓
ALL rbee-hive instances exit gracefully
    ↓
System is clean
```

**Why this matters:**
- **rbee-keeper is a testing tool** - when test is done, everything must die
- **No orphaned processes** - clean shutdown every time
- **No leaked VRAM** - models are unloaded properly
- **Deterministic testing** - every test starts from clean state

## Critical Discovery

**From architecture specs (FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md, ARCHITECTURE_MODES.md):**

### rbee-hive is a PERSISTENT HTTP DAEMON

```
rbee-hive is NOT a CLI that exits after spawning workers.
rbee-hive is an HTTP daemon that runs continuously.
```

**Evidence:**
- `FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md:17` - "rbee-hive (pool manager) - Pool management API, worker health monitoring"
- `ARCHITECTURE_MODES.md:20-31` - Ephemeral mode lifecycle shows rbee-hive receives SIGTERM to exit
- `COMPONENT_RESPONSIBILITIES_FINAL.md:30-262` - rbee-hive monitors worker health every 30s, enforces idle timeout

### llm-worker-rbee is a PERSISTENT HTTP DAEMON

```
Workers do NOT exit after inference completes.
Workers stay running until idle timeout (5 min) or explicit shutdown.
```

**Evidence:**
- `FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md:84-104` - Worker lifecycle shows model stays in VRAM
- `ARCHITECTURE_MODES.md:43` - "Worker reuse across requests"
- test-001-mvp.md Phase 8 - "Worker state transition: idle → busy → idle"

### queen-rbee is a PERSISTENT HTTP DAEMON

```
queen-rbee is the orchestrator that coordinates everything.
queen-rbee is spawned by rbee-keeper and stays running.
queen-rbee uses SSH to control rbee-hive on remote nodes.
queen-rbee relays SSE streams from workers to rbee-keeper.
```

**Evidence:**
- `bin/queen-rbee/README.md` - "Orchestrator Daemon"
- `COMPONENT_RESPONSIBILITIES_FINAL.md:14` - "queen-rbee - Daemon (HTTP)"
- `FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md:14-16` - "queen-rbee (orchestrator) - Routes inference requests"

### rbee-keeper is EPHEMERAL CLI (TESTING TOOL)

```
rbee-keeper is a TESTING/INTEGRATION TOOL, NOT for production.
rbee-keeper spawns queen-rbee as child process (for testing).
rbee-keeper streams output from queen-rbee to stdout.
rbee-keeper exits after command completes.
rbee-keeper KILLS queen-rbee when done (ONLY if it spawned it).
rbee-keeper does NOT stay running.
```

**Purpose:**
- Integration testing of the entire rbee system
- Quick one-off inferences for development/testing
- Validates that queen-rbee, rbee-hive, and workers work together
- **NOT meant for production workloads**

**Production Use Case:**
- Users should use **llama-orch SDK** (HTTP client) → queen-rbee
- **NOT** rbee-keeper (CLI)

**Evidence:**
- `COMPONENT_RESPONSIBILITIES_FINAL.md:16` - "llorch (rbee-keeper) - CLI (SSH)"
- `FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md:56-61` - rbee-keeper is CLI on blep

## The Two Modes

### Mode 1: Ephemeral (rbee-keeper spawns queen-rbee)

**Use case:** Testing, single inference, clean environment

```
User runs: rbee-keeper infer --node mac --model tinyllama --prompt "hello"

1. rbee-keeper spawns queen-rbee as child process
2. queen-rbee starts HTTP daemon on port 8080
3. queen-rbee uses SSH to start rbee-hive on mac
4. rbee-hive starts HTTP daemon on port 9200
5. rbee-hive spawns llm-worker-rbee
6. Worker loads model and becomes ready
7. Worker sends ready callback to rbee-hive
8. rbee-hive notifies queen-rbee
9. queen-rbee sends inference request to worker
10. Worker streams tokens via SSE to queen-rbee
11. queen-rbee relays SSE stream to rbee-keeper stdout
12. User gets result
13. User sends SIGTERM to rbee-keeper (Ctrl+C)
14. rbee-keeper sends SIGTERM to queen-rbee
15. queen-rbee uses SSH to send SIGTERM to rbee-hive
16. rbee-hive cascades shutdown to all workers
17. All processes exit, VRAM cleaned
```

**Key insight:** rbee-keeper controls queen-rbee lifecycle because it spawned it. queen-rbee controls rbee-hive lifecycle via SSH.

### Mode 2: Persistent (queen-rbee and rbee-hive pre-started)

**Use case:** Production, worker reuse, performance

```
Operator starts: queen-rbee daemon &
Operator starts on mac: rbee-hive daemon &

User runs: rbee-keeper infer --node mac --model tinyllama --prompt "hello"

1. rbee-keeper connects to existing queen-rbee HTTP API
2. queen-rbee connects to existing rbee-hive HTTP API on mac
3. rbee-hive checks worker registry
4. If worker exists and idle, reuse it (fast)
5. If no worker, spawn new one
6. Worker sends ready callback to rbee-hive
7. rbee-hive notifies queen-rbee
8. queen-rbee sends inference request to worker
9. Worker streams tokens via SSE to queen-rbee
10. queen-rbee relays SSE stream to rbee-keeper stdout
11. User gets result
12. rbee-keeper exits
13. queen-rbee continues running
14. rbee-hive continues running
15. Worker continues running (until idle timeout)
```

**Key insight:** rbee-keeper does NOT control queen-rbee lifecycle because it didn't spawn it. queen-rbee does NOT control rbee-hive lifecycle because it didn't spawn it.

## Process Ownership Rules

### RULE 1: Parent Process Owns Child Lifecycle

```
IF rbee-keeper spawned queen-rbee:
    THEN rbee-keeper owns queen-rbee lifecycle
    AND rbee-keeper sends SIGTERM to queen-rbee on exit

IF operator started queen-rbee:
    THEN operator owns queen-rbee lifecycle
    AND rbee-keeper does NOT send SIGTERM to queen-rbee

IF queen-rbee spawned rbee-hive (via SSH):
    THEN queen-rbee owns rbee-hive lifecycle
    AND queen-rbee sends SIGTERM to rbee-hive on exit

IF operator started rbee-hive:
    THEN operator owns rbee-hive lifecycle
    AND queen-rbee does NOT send SIGTERM to rbee-hive
```

### RULE 2: rbee-hive Always Owns Worker Lifecycle

```
rbee-hive spawns workers
rbee-hive monitors worker health
rbee-hive enforces idle timeout
rbee-hive sends shutdown to workers
```

### RULE 3: Workers Never Own Their Own Lifecycle

```
Workers are managed by rbee-hive
Workers do NOT decide when to exit
Workers respond to shutdown commands from rbee-hive
```

### RULE 4: queen-rbee Orchestrates Everything

```
queen-rbee is spawned by rbee-keeper (or pre-started by operator)
queen-rbee uses SSH to control rbee-hive on remote nodes
queen-rbee sends inference requests directly to workers
queen-rbee relays SSE streams from workers to rbee-keeper
queen-rbee does NOT exit after inference (stays running)
```

## Cascading Shutdown

### Trigger: SIGTERM to rbee-keeper (Ephemeral Mode)

```
User presses Ctrl+C
    ↓
SIGTERM → rbee-keeper
    ↓
rbee-keeper sends SIGTERM → queen-rbee
    ↓
queen-rbee uses SSH to send SIGTERM → rbee-hive (on remote node)
    ↓
rbee-hive sends POST /v1/admin/shutdown to all workers
    ↓
rbee-hive waits for acknowledgment (max 5s per worker)
    ↓
Workers unload models from VRAM
    ↓
Workers exit cleanly
    ↓
rbee-hive clears in-memory registry
    ↓
rbee-hive exits cleanly
    ↓
queen-rbee exits cleanly
    ↓
rbee-keeper exits cleanly
    ↓
Model catalog (SQLite) persists on disk
```

**Implementation:** 
- `bin/rbee-keeper/src/main.rs` - SIGTERM handler
- `bin/queen-rbee/src/main.rs` - SIGTERM handler, SSH shutdown
- `bin/rbee-hive/src/commands/daemon.rs` - Worker shutdown cascade

## Worker Idle Timeout

### Trigger: 5 minutes without requests

```
Worker completes inference
    ↓
Worker transitions to state "idle"
    ↓
rbee-hive starts 5-minute timer
    ↓
5 minutes elapse without new requests
    ↓
rbee-hive sends POST /v1/admin/shutdown to worker
    ↓
Worker unloads model from VRAM
    ↓
Worker exits cleanly
    ↓
rbee-hive removes worker from in-memory registry
    ↓
rbee-hive continues running (does NOT exit)
    ↓
VRAM is available for other applications
```

**Implementation:** `bin/rbee-hive/src/timeout.rs` - Idle timeout enforcement

## What Dies When

### After Inference Completes

| Process | Dies? | Why? |
|---------|-------|------|
| **rbee-keeper** | ✅ YES | CLI exits after command completes |
| **queen-rbee** | ❌ NO | Persistent daemon, continues running |
| **rbee-hive** | ❌ NO | Persistent daemon, continues running |
| **llm-worker-rbee** | ❌ NO | Stays idle, waiting for next request |

### After 5 Minutes Idle

| Process | Dies? | Why? |
|---------|-------|------|
| **rbee-keeper** | N/A | Already exited |
| **queen-rbee** | ❌ NO | Persistent daemon, continues running |
| **rbee-hive** | ❌ NO | Persistent daemon, continues running |
| **llm-worker-rbee** | ✅ YES | Idle timeout, VRAM freed |

### After Ctrl+C to rbee-keeper (Ephemeral Mode)

| Process | Dies? | Why? |
|---------|-------|------|
| **rbee-keeper** | ✅ YES | Received SIGTERM, sends SIGTERM to queen-rbee |
| **queen-rbee** | ✅ YES | Received SIGTERM from rbee-keeper, SSH kills rbee-hive |
| **rbee-hive** | ✅ YES | Received SIGTERM from queen-rbee via SSH, cascades shutdown |
| **llm-worker-rbee** | ✅ YES | Receives shutdown from rbee-hive |

### After Ctrl+C to rbee-keeper (Persistent Mode)

| Process | Dies? | Why? |
|---------|-------|------|
| **rbee-keeper** | ✅ YES | Received SIGTERM, exits |
| **queen-rbee** | ❌ NO | Was not spawned by rbee-keeper, continues running |
| **rbee-hive** | ❌ NO | Was not spawned by queen-rbee, continues running |
| **llm-worker-rbee** | ❌ NO | rbee-hive still running, worker stays idle |

## Implementation Checklist

### rbee-keeper (CLI)

- [ ] Detect if queen-rbee is already running (check HTTP endpoint)
- [ ] If not running, spawn queen-rbee as child process
- [ ] If spawned, send SIGTERM to queen-rbee on exit
- [ ] If not spawned, do NOT send SIGTERM to queen-rbee
- [ ] Stream SSE output from queen-rbee to stdout
- [ ] Exit after command completes (always)

### queen-rbee (HTTP Daemon / Orchestrator)

- [ ] Start HTTP server on port 8080
- [ ] Detect if rbee-hive is already running on remote node (SSH check)
- [ ] If not running, use SSH to spawn rbee-hive on remote node
- [ ] If spawned, use SSH to send SIGTERM to rbee-hive on exit
- [ ] If not spawned, do NOT send SIGTERM to rbee-hive
- [ ] Receive worker ready notifications from rbee-hive
- [ ] Send inference requests directly to workers (HTTP)
- [ ] Relay SSE streams from workers to rbee-keeper
- [ ] Cascading shutdown on SIGTERM (SSH to rbee-hive)

### rbee-hive (HTTP Daemon / Pool Manager)

- [x] Start HTTP server on port 9200 ✅ (TEAM-027)
- [x] Monitor worker health every 30s ✅ (TEAM-027)
- [x] Enforce idle timeout (5 min) ✅ (TEAM-027)
- [x] Cascading shutdown on SIGTERM ✅ (TEAM-030)
- [ ] Notify queen-rbee when worker becomes ready
- [ ] Persist model catalog (SQLite) on shutdown
- [ ] Clear in-memory worker registry on shutdown

### llm-worker-rbee (HTTP Daemon / Worker)

- [x] Start HTTP server on port 8001+ ✅ (M0)
- [x] Load model into VRAM ✅ (M0)
- [x] Send ready callback to rbee-hive ✅ (M0)
- [x] Accept inference requests ✅ (M0)
- [x] Transition: idle → busy → idle ✅ (M0)
- [ ] Respond to shutdown command (POST /v1/admin/shutdown)
- [ ] Unload model from VRAM on shutdown
- [ ] Exit cleanly on shutdown

## Testing Strategy

### Unit Tests

- [ ] rbee-keeper spawns rbee-hive correctly
- [ ] rbee-keeper detects existing rbee-hive
- [ ] rbee-keeper sends SIGTERM only when it spawned rbee-hive
- [ ] rbee-hive cascades shutdown to all workers
- [ ] rbee-hive clears registry on shutdown
- [ ] Worker responds to shutdown command

### Integration Tests

- [ ] Ephemeral mode: All processes exit after Ctrl+C
- [ ] Persistent mode: rbee-hive and worker survive rbee-keeper exit
- [ ] Idle timeout: Worker exits after 5 min, rbee-hive continues
- [ ] Cascading shutdown: SIGTERM to rbee-hive kills all workers

### BDD Tests

- [x] MVP-008: Pool manager remains running as persistent daemon ✅
- [x] MVP-009: Worker idle timeout (worker dies, pool lives) ✅
- [x] MVP-010: rbee-keeper exits after inference (CLI dies, daemons live) ✅
- [x] MVP-011: Cascading shutdown when rbee-hive receives SIGTERM ✅
- [x] MVP-012: rbee-hive spawned by rbee-keeper (ephemeral mode) ✅
- [x] MVP-013: rbee-hive pre-started (persistent mode) ✅

## Common Mistakes to Avoid

### ❌ WRONG: rbee-hive exits after spawning worker

```rust
// WRONG
fn spawn_worker() {
    let worker = spawn_worker_process();
    worker.wait_ready();
    // rbee-hive exits here ❌
}
```

### ✅ CORRECT: rbee-hive continues running

```rust
// CORRECT
fn spawn_worker() {
    let worker = spawn_worker_process();
    worker.wait_ready();
    registry.register(worker);
    // rbee-hive continues running ✅
}

fn main() {
    // HTTP server runs forever
    server.run().await;
}
```

### ❌ WRONG: Worker exits after inference

```rust
// WRONG
fn execute_inference() {
    let result = generate_tokens();
    stream_result(result);
    // Worker exits here ❌
}
```

### ✅ CORRECT: Worker stays idle

```rust
// CORRECT
fn execute_inference() {
    let result = generate_tokens();
    stream_result(result);
    self.state = WorkerState::Idle; // Stay running ✅
}

fn main() {
    // HTTP server runs forever
    server.run().await;
}
```

### ❌ WRONG: rbee-keeper always sends SIGTERM to rbee-hive

```rust
// WRONG
fn main() {
    let hive = connect_to_hive();
    run_inference(hive);
    hive.send_sigterm(); // Always kills rbee-hive ❌
}
```

### ✅ CORRECT: rbee-keeper only sends SIGTERM if it spawned rbee-hive

```rust
// CORRECT
fn main() {
    let hive = if hive_exists() {
        connect_to_existing_hive() // Don't own lifecycle
    } else {
        spawn_hive() // Own lifecycle
    };
    
    run_inference(hive);
    
    if hive.spawned_by_us() {
        hive.send_sigterm(); // Only if we spawned it ✅
    }
}
```

## References

- `/bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md` - HTTP daemon architecture
- `/bin/.specs/ARCHITECTURE_MODES.md` - Ephemeral vs Persistent modes
- `/bin/.specs/COMPONENT_RESPONSIBILITIES_FINAL.md` - Component responsibilities
- `/bin/.specs/.gherkin/test-001-mvp.md` - MVP specification
- `/test-harness/bdd/tests/features/test-001-mvp.feature` - BDD scenarios
- `/test-harness/bdd/README.md` - BDD test harness documentation

---

**Created by:** TEAM-037 (Testing Team)  
**Status:** NORMATIVE - Must be followed for implementation  
**Last Updated:** 2025-10-10

---
Verified by Testing Team 🔍
