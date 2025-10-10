# CRITICAL RULES - READ THIS FIRST

**Date:** 2025-10-10T13:58  
**Status:** NORMATIVE - MUST FOLLOW  
**Priority:** P0 - BLOCKS EVERYTHING

---

## 🚨 THE MOST IMPORTANT RULES 🚨

### RULE 1: rbee-keeper is a TESTING TOOL

**rbee-keeper is NOT for production!**

```
✅ USE rbee-keeper FOR:
- Integration testing
- Development/debugging
- Quick one-off inferences
- Validating the system works

❌ DO NOT USE rbee-keeper FOR:
- Production workloads
- User-facing applications
- Long-running services
```

**Production users should use:**
- **llama-orch SDK** (HTTP client) → queen-rbee directly
- **NOT** rbee-keeper

---

### RULE 2: WHENEVER queen-rbee DIES, EVERYTHING DIES

**This is NON-NEGOTIABLE and ALWAYS TRUE:**

```
queen-rbee dies
    ↓
ALL rbee-hive instances on ALL nodes die (via SSH SIGTERM)
    ↓
ALL workers on ALL nodes die (via shutdown command)
    ↓
System is completely clean
```

**Why this matters:**
- rbee-keeper is a testing tool - when test is done, everything must die
- No orphaned processes - clean shutdown every time
- No leaked VRAM - models are unloaded properly
- Deterministic testing - every test starts from clean state

**This happens:**
- When rbee-keeper kills queen-rbee (if it spawned it)
- When queen-rbee crashes
- When queen-rbee receives SIGTERM
- When queen-rbee is killed manually
- **ALWAYS** - no exceptions

---

### RULE 3: rbee-keeper ONLY kills queen-rbee IF it spawned it

```
IF rbee-keeper spawned queen-rbee:
    THEN rbee-keeper sends SIGTERM to queen-rbee on exit
    AND queen-rbee cascades shutdown to ALL hives and workers

IF operator started queen-rbee:
    THEN rbee-keeper does NOT send SIGTERM to queen-rbee
    AND queen-rbee continues running after rbee-keeper exits
```

---

## The 4 Components

**Updated by TEAM-051:** rbee-keeper is the USER INTERFACE, not a testing tool.

| Component | Type | Production? | Purpose |
|-----------|------|-------------|---------|
| **rbee-keeper** | CLI | ✅ YES | **USER INTERFACE** - Manages queen-rbee, hives, workers, SSH config |
| **queen-rbee** | HTTP Daemon | ✅ YES | Orchestrator, controls all hives |
| **rbee-hive** | HTTP Daemon | ✅ YES | Pool manager, spawns workers |
| **llm-worker-rbee** | HTTP Daemon | ✅ YES | Worker, executes inference |

**Future:** Web UI will be added alongside the CLI.

---

## Complete Flow (Production Mode)

### 1. Startup

```
User runs: rbee-keeper infer --node mac --model tinyllama --prompt "hello"

rbee-keeper (CLI UI)
    ↓ manages lifecycle
queen-rbee (starts HTTP daemon :8080)
    ↓ SSH to mac
rbee-hive (starts HTTP daemon :9200)
    ↓ spawns
llm-worker-rbee (starts HTTP daemon :8001)
```

### 2. Worker Ready Callback

```
llm-worker-rbee (HTTP daemon starts on :8001)
    ↓ loads model into VRAM
    ↓ becomes ready
    ↓ HTTP POST /v1/workers/ready to rbee-hive
    ↓ includes: worker_id, url (http://mac:8001), model_ref, backend, device
rbee-hive (registers worker locally in-memory)
    ↓ HTTP POST /v1/orchestrator/worker-ready to queen-rbee
    ↓ includes: worker_id, url, model_ref, backend, device
queen-rbee (adds worker to global worker registry)
```

### 3. Inference Execution

```
queen-rbee (looks up worker in global registry)
    ↓ gets worker URL: http://mac:8001
    ↓ HTTP POST http://mac:8001/execute (DIRECTLY to worker, bypasses hive)
llm-worker-rbee (generates tokens)
    ↓ SSE stream
queen-rbee (relays stream)
    ↓ SSE to stdout
rbee-keeper (displays tokens to user)
```

**CRITICAL:** queen-rbee connects DIRECTLY to worker using URL from registry, bypassing rbee-hive for inference (by design)

### 4. Shutdown (CRITICAL!)

```
Developer presses Ctrl+C
    ↓
rbee-keeper (receives SIGTERM)
    ↓ sends SIGTERM to queen-rbee (because it spawned it)
queen-rbee (receives SIGTERM)
    ↓ SSH: kill rbee-hive on mac
    ↓ SSH: kill rbee-hive on workstation
    ↓ SSH: kill rbee-hive on ALL nodes
rbee-hive on mac
    ↓ HTTP POST /v1/admin/shutdown to ALL workers
llm-worker-rbee (unloads model, exits)
    ↓
rbee-hive on workstation
    ↓ HTTP POST /v1/admin/shutdown to ALL workers
llm-worker-rbee (unloads model, exits)
    ↓
ALL rbee-hive instances exit
    ↓
queen-rbee exits
    ↓
rbee-keeper exits
    ↓
System is completely clean, no orphaned processes
```

---

## Production Flow (Different!)

### Production Setup

```
Operator starts: queen-rbee daemon &
Operator starts on mac: rbee-hive daemon &
Operator starts on workstation: rbee-hive daemon &
```

### Production Usage

```
User application (llama-orch SDK)
    ↓ HTTP POST /v2/tasks
queen-rbee
    ↓ orchestrates
workers
    ↓ SSE stream
queen-rbee
    ↓ SSE stream
User application
```

**Key difference:**
- No rbee-keeper involved
- queen-rbee and rbee-hive are pre-started by operator
- They continue running after requests complete
- Operator manually shuts them down when needed

---

## Cascading Shutdown Guarantee

### What Triggers Cascading Shutdown?

1. **rbee-keeper kills queen-rbee** (if it spawned it)
2. **queen-rbee crashes**
3. **queen-rbee receives SIGTERM**
4. **queen-rbee is killed manually**

### What Happens?

```
queen-rbee dies
    ↓
queen-rbee's shutdown handler runs
    ↓
For each rbee-hive that queen-rbee spawned:
    SSH to node
    Send SIGTERM to rbee-hive
    ↓
rbee-hive receives SIGTERM
    ↓
rbee-hive's shutdown handler runs
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
```

### Guarantees

- ✅ No orphaned rbee-hive processes
- ✅ No orphaned worker processes
- ✅ All models unloaded from VRAM
- ✅ All resources released
- ✅ Clean state for next test

---

## Implementation Checklist

### rbee-keeper (CLI / Testing Tool)

- [ ] Detect if queen-rbee is already running
- [ ] If not running, spawn queen-rbee as child process
- [ ] Track whether we spawned queen-rbee
- [ ] Stream SSE output from queen-rbee to stdout
- [ ] On exit, send SIGTERM to queen-rbee ONLY if we spawned it
- [ ] Exit after command completes

### queen-rbee (HTTP Daemon / Orchestrator)

- [ ] Start HTTP server on port 8080
- [ ] Track which rbee-hive instances we spawned (via SSH)
- [ ] On SIGTERM, use SSH to kill ALL rbee-hive instances we spawned
- [ ] Wait for all hives to acknowledge shutdown (max 5s each)
- [ ] Exit cleanly after all hives are down

### rbee-hive (HTTP Daemon / Pool Manager)

- [ ] Start HTTP server on port 9200
- [ ] Track which workers we spawned
- [ ] On SIGTERM, send shutdown to ALL workers
- [ ] Wait for all workers to acknowledge (max 5s each)
- [ ] Exit cleanly after all workers are down

### llm-worker-rbee (HTTP Daemon / Worker)

- [ ] Respond to shutdown command (POST /v1/admin/shutdown)
- [ ] Unload model from VRAM
- [ ] Exit cleanly

---

## Common Mistakes

### ❌ WRONG: rbee-keeper is for production

```rust
// WRONG - rbee-keeper is NOT for production
fn main() {
    // Production service using rbee-keeper ❌
    loop {
        rbee_keeper::infer(...);
    }
}
```

### ✅ CORRECT: Use SDK for production

```rust
// CORRECT - Use SDK for production
fn main() {
    let client = LlamaOrchClient::new("http://queen-rbee:8080");
    loop {
        client.enqueue(...).await;
    }
}
```

### ❌ WRONG: queen-rbee doesn't cascade shutdown

```rust
// WRONG - queen-rbee must cascade shutdown
impl QueenRbee {
    fn shutdown(&self) {
        // Just exit ❌
        std::process::exit(0);
    }
}
```

### ✅ CORRECT: queen-rbee cascades shutdown

```rust
// CORRECT - queen-rbee cascades to all hives
impl QueenRbee {
    async fn shutdown(&self) {
        // Kill ALL hives we spawned ✅
        for hive in &self.spawned_hives {
            ssh_kill_hive(hive).await;
        }
        // Wait for all to die
        wait_for_hives_to_die().await;
        // Now exit
        std::process::exit(0);
    }
}
```

### ❌ WRONG: rbee-keeper always kills queen-rbee

```rust
// WRONG - only kill if we spawned it
fn main() {
    let queen = connect_to_queen();
    run_inference(queen);
    queen.kill(); // Always kills ❌
}
```

### ✅ CORRECT: rbee-keeper only kills if it spawned

```rust
// CORRECT - only kill if we spawned it
fn main() {
    let queen = if queen_exists() {
        connect_to_existing_queen() // Don't own
    } else {
        spawn_queen() // Own lifecycle
    };
    
    run_inference(queen);
    
    if queen.spawned_by_us() {
        queen.kill(); // Only if we spawned ✅
    }
}
```

---

## Testing Scenarios

### Test 1: Ephemeral Mode Cleanup

```bash
# Start test
rbee-keeper infer --node mac --model tinyllama --prompt "test"

# Verify processes running
ps aux | grep queen-rbee  # Should show 1 process
ps aux | grep rbee-hive   # Should show 1 process
ps aux | grep llm-worker  # Should show 1 process

# Kill rbee-keeper
kill -SIGTERM <rbee-keeper-pid>

# Wait 10 seconds
sleep 10

# Verify ALL processes are gone
ps aux | grep queen-rbee  # Should show NOTHING
ps aux | grep rbee-hive   # Should show NOTHING
ps aux | grep llm-worker  # Should show NOTHING

# Verify VRAM is clean
nvidia-smi  # Should show no model loaded
```

### Test 2: Persistent Mode (queen-rbee survives)

```bash
# Start queen-rbee manually
queen-rbee daemon &

# Start rbee-hive manually on mac
ssh mac "rbee-hive daemon &"

# Run test
rbee-keeper infer --node mac --model tinyllama --prompt "test"

# Kill rbee-keeper
kill -SIGTERM <rbee-keeper-pid>

# Verify queen-rbee still running
ps aux | grep queen-rbee  # Should show 1 process

# Verify rbee-hive still running
ssh mac "ps aux | grep rbee-hive"  # Should show 1 process
```

---

## Summary

1. **rbee-keeper is a TESTING TOOL** - not for production
2. **WHENEVER queen-rbee dies, EVERYTHING dies** - no exceptions
3. **rbee-keeper ONLY kills queen-rbee if it spawned it**
4. **Production uses SDK → queen-rbee directly**
5. **Cascading shutdown is GUARANTEED** - no orphaned processes

---

**Created by:** TEAM-037 (Testing Team)  
**Status:** NORMATIVE - MUST FOLLOW  
**Priority:** P0 - BLOCKS EVERYTHING

---
Verified by Testing Team 🔍
