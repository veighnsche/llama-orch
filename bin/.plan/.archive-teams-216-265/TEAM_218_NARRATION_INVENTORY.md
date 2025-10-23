# TEAM-218: rbee-hive NARRATION INVENTORY

**Component:** `bin/20_rbee_hive`  
**Date:** Oct 22, 2025  
**Status:** ✅ COMPLETE

---

## Summary

rbee-hive uses narration in 1 location:
- **main.rs** - Server lifecycle and capabilities endpoint

**CRITICAL:** rbee-hive narrations have **NO job_id** → All go to **stderr**.

---

## 1. Narration Factory

**Factory:** `pub const NARRATE: NarrationFactory = NarrationFactory::new("hive");`

**Action Constants (narration.rs:17-29):**
HUMAN SAYS: REALLY BAD PATTERN IF YOU SEE THIS PATTERN>
ACTION SHOULD NOT HAVE CONSTANTS
THE NARRATION.action("string literal here")
```rust
pub const ACTION_STARTUP: &str = "startup";
pub const ACTION_HEARTBEAT: &str = "heartbeat";
pub const ACTION_WORKER_SPAWN: &str = "worker_spawn";  // Not yet used
pub const ACTION_WORKER_STOP: &str = "worker_stop";    // Not yet used
pub const ACTION_LISTEN: &str = "listen";
pub const ACTION_READY: &str = "ready";
pub const ACTION_CAPS_REQUEST: &str = "caps_request";
pub const ACTION_CAPS_GPU_CHECK: &str = "caps_gpu_check";
pub const ACTION_CAPS_GPU_FOUND: &str = "caps_gpu_found";
pub const ACTION_CAPS_CPU_ADD: &str = "caps_cpu_add";
pub const ACTION_CAPS_RESPONSE: &str = "caps_response";
```

---

## 2. Startup Narrations (NO job_id)

```rust
// Line 66-72: Server starting
NARRATE
    .action(ACTION_STARTUP)
    .context(&args.port.to_string())
    .context(&args.hive_id)
    .context(&args.queen_url)
    .human("🐝 Starting on port {}, hive_id: {}, queen: {}")
    .emit();

// Line 86-90: Heartbeat task started
NARRATE
    .action(ACTION_HEARTBEAT)
    .context("5s")
    .human("💓 Heartbeat task started ({} interval)")
    .emit();

// Line 100-104: Listening
NARRATE
    .action(ACTION_LISTEN)
    .context(&format!("http://{}", addr))
    .human("✅ Listening on {}")
    .emit();

// Line 109-112: Ready
NARRATE
    .action(ACTION_READY)
    .human("✅ Hive ready")
    .emit();
```

**Behavior:** These narrations go to **stderr** (no job_id = no SSE routing).

---

## 3. Capabilities Endpoint Narrations (NO job_id)

```rust
// Line 145-148: Capabilities request received
NARRATE
    .action(ACTION_CAPS_REQUEST)
    .human("📡 Received capabilities request from queen")
    .emit();

// Line 151-154: GPU detection attempt
NARRATE
    .action(ACTION_CAPS_GPU_CHECK)
    .human("🔍 Detecting GPUs via nvidia-smi...")
    .emit();

// Line 160-168: GPU detection results
NARRATE
    .action(ACTION_CAPS_GPU_FOUND)
    .context(gpu_info.count.to_string())
    .human(if gpu_info.count > 0 {
        "✅ Found {} GPU(s)"
    } else {
        "ℹ️  No GPUs detected, using CPU only"
    })
    .emit();

// Line 183-188: CPU device added
NARRATE
    .action(ACTION_CAPS_CPU_ADD)
    .context(cpu_cores.to_string())
    .context(system_ram_gb.to_string())
    .human("🖥️  Adding CPU-0: {0} cores, {1} GB RAM")
    .emit();

// Line 200-204: Capabilities response sent
NARRATE
    .action(ACTION_CAPS_RESPONSE)
    .context(devices.len().to_string())
    .human("📤 Sending capabilities response ({} device(s))")
    .emit();
```

**Behavior:** These narrations go to **stderr** (no job_id = no SSE routing).

---

## 4. The Missing job_id Problem

### Current Behavior

rbee-hive narrations have **NO job_id** because:
1. rbee-hive is a **daemon** (not a CLI)
2. It doesn't know about queen's job_id
3. Capabilities endpoint is called by queen-rbee (via hive-lifecycle crate)
4. queen-rbee has the job_id, but doesn't pass it to hive

### Why This Matters

**User Experience:**
```
User runs: rbee hive start -a localhost
  ↓
keeper → queen → POST /v1/jobs (creates job_id)
  ↓
queen → hive-lifecycle → execute_hive_start()
  ↓
hive-lifecycle → HTTP GET http://localhost:9000/capabilities
  ↓
rbee-hive → get_capabilities() → Narrates to stderr
  ↓
User CANNOT see GPU detection progress in SSE stream!
```

**What user sees:**
```
✅ Hive started
📊 Fetching device capabilities from hive...
✅ Found 2 device(s)
```

**What user SHOULD see:**
```
✅ Hive started
📊 Fetching device capabilities from hive...
📡 Received capabilities request from queen
🔍 Detecting GPUs via nvidia-smi...
✅ Found 2 GPU(s)
🖥️  Adding CPU-0: 16 cores, 64 GB RAM
📤 Sending capabilities response (3 device(s))
✅ Found 3 device(s)
```

---

## 5. Solution: Pass job_id to Hive

### Option 1: HTTP Header (RECOMMENDED)

**hive-lifecycle → HTTP client:**
```rust
let response = client
    .get(&format!("{}/capabilities", endpoint))
    .header("X-Job-ID", job_id)  // ← Pass job_id via header
    .send()
    .await?;
```

**rbee-hive → Extract from header:**
```rust
async fn get_capabilities(
    headers: axum::http::HeaderMap,
) -> Json<CapabilitiesResponse> {
    let job_id = headers
        .get("X-Job-ID")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    
    if let Some(job_id) = &job_id {
        NARRATE
            .action(ACTION_CAPS_REQUEST)
            .job_id(job_id)  // ← Include job_id for SSE routing
            .human("📡 Received capabilities request from queen")
            .emit();
    }
    // ... rest of function
}
```

### Option 2: Query Parameter

**hive-lifecycle → HTTP client:**
```rust
let response = client
    .get(&format!("{}/capabilities?job_id={}", endpoint, job_id))
    .send()
    .await?;
```

**rbee-hive → Extract from query:**
```rust
async fn get_capabilities(
    Query(params): Query<HashMap<String, String>>,
) -> Json<CapabilitiesResponse> {
    let job_id = params.get("job_id").map(|s| s.to_string());
    // ... use job_id in narrations
}
```

### Option 3: Do Nothing (CURRENT)

Keep narrations going to stderr. User won't see GPU detection progress.

**Pros:**
- No changes needed
- Simpler code

**Cons:**
- Poor user experience
- Defeats purpose of job-scoped SSE
- User can't see what's happening during slow operations (GPU detection can take 1-2 seconds)

---

## 6. Findings

### ✅ Correct Behaviors
1. **Consistent narration style** - All use NARRATE factory with action constants
2. **Comprehensive capabilities narration** - Every step of GPU detection is narrated
3. **Clean separation** - Server lifecycle vs capabilities endpoint

### ❌ Missing Behaviors
1. **NO job_id in capabilities endpoint** - Narrations go to stderr, not SSE
2. **NO worker lifecycle narration** - ACTION_WORKER_SPAWN/STOP defined but not used
3. **NO error narration** - No error handling in main.rs or get_capabilities()

### 📋 Recommendations
1. **HIGH PRIORITY:** Pass job_id to capabilities endpoint (Option 1: HTTP header)
2. **MEDIUM PRIORITY:** Add worker lifecycle narration when worker registry is implemented
3. **LOW PRIORITY:** Add error narration for server startup failures

---

## 7. Code Signatures

All investigated code marked with:
```rust
// TEAM-218: Investigated Oct 22, 2025 - Behavior inventory complete
```

**Files investigated:**
- `bin/20_rbee_hive/src/main.rs` (lines 1-208)
- `bin/20_rbee_hive/src/narration.rs` (lines 1-30)

---

**TEAM-218 COMPLETE** ✅

**CRITICAL FINDING:** rbee-hive capabilities narration goes to stderr (no job_id). Users cannot see GPU detection progress in SSE stream. Recommend passing job_id via HTTP header.
