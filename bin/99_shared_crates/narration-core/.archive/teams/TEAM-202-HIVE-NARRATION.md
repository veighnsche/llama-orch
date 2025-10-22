# TEAM-202: Hive Narration

**Team:** TEAM-202  
**Priority:** MEDIUM  
**Duration:** 3-4 hours  
**Based On:** TEAM-198 Phase 4 + TEAM-197 corrections

---

## Mission

Replace `println!()` in rbee-hive with proper narration that flows through job-scoped SSE. Use the worker's thread-local pattern (NO HTTP ingestion endpoint).

---

## The Problem

### Current State

**Hive uses println!():**
```rust
// bin/20_rbee_hive/src/main.rs
println!("🐝 rbee-hive starting on port {}", args.port);
println!("💓 Heartbeat task started (5s interval)");
```

**When hive runs remotely:**
```
┌──────────────┐         ┌──────────────┐
│   keeper     │ ─SSH─→  │     hive     │
│  (local)     │         │  (remote)    │
└──────────────┘         └──────────────┘
      ↑                         │
      │                    println!() ✗
      │                    (not visible!)
      └─── NEEDS SSE ←──────────┘
```

**User cannot see hive's output on remote machine!**

---

## The Solution (TEAM-197 Recommended)

### NOT This (TEAM-198's Wrong Approach)

❌ **HTTP Ingestion Endpoint:**
```rust
// BAD: Extra network hop, needs auth, needs job_id routing
POST /v1/narration
```

### YES This (Worker Pattern)

✅ **Thread-Local Channel (Like Worker):**
```rust
// GOOD: No network hop, automatic job scoping, proven pattern
NARRATE.action().emit() → narration-core → job-scoped SSE
```

---

## Implementation

### Step 1: Add narration-core Dependency

**File:** `bin/20_rbee_hive/Cargo.toml`

**Add if missing:**
```toml
[dependencies]
observability-narration-core = { path = "../99_shared_crates/narration-core" }
```

---

### Step 2: Create narration Module

**File:** `bin/20_rbee_hive/src/narration.rs` (NEW FILE)

```rust
//! Hive narration configuration
//!
//! TEAM-202: Narration for rbee-hive using job-scoped SSE

use observability_narration_core::NarrationFactory;

// TEAM-202: Narration factory for hive
// Use "hive" as actor to match other components
pub const NARRATE: NarrationFactory = NarrationFactory::new("hive");

// Hive-specific action constants
pub const ACTION_STARTUP: &str = "startup";
pub const ACTION_HEARTBEAT: &str = "heartbeat";
pub const ACTION_WORKER_SPAWN: &str = "worker_spawn";
pub const ACTION_WORKER_STOP: &str = "worker_stop";
pub const ACTION_LISTEN: &str = "listen";
pub const ACTION_READY: &str = "ready";
```

---

### Step 3: Update main.rs to Use Narration

**File:** `bin/20_rbee_hive/src/main.rs`

**Add module declaration (after existing use statements):**
```rust
mod narration;
use narration::{NARRATE, ACTION_STARTUP, ACTION_HEARTBEAT, ACTION_LISTEN, ACTION_READY};
```

**Replace println!() calls:**

**BEFORE:**
```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("🐝 rbee-hive starting on port {}", args.port);
    println!("📡 Hive ID: {}", args.hive_id);
    println!("👑 Queen URL: {}", args.queen_url);

    // ... heartbeat setup ...

    println!("💓 Heartbeat task started (5s interval)");

    // ... router setup ...

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    println!("✅ rbee-hive listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
```

**AFTER:**
```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // TEAM-202: Use narration instead of println!
    // This automatically goes through job-scoped SSE (if in job context)
    NARRATE
        .action(ACTION_STARTUP)
        .context(&args.port.to_string())
        .context(&args.hive_id)
        .context(&args.queen_url)
        .human("🐝 Starting on port {}, hive_id: {}, queen: {}")
        .emit();

    // ... heartbeat setup ...

    NARRATE
        .action(ACTION_HEARTBEAT)
        .context("5s")
        .human("💓 Heartbeat task started ({} interval)")
        .emit();

    // ... router setup ...

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    NARRATE
        .action(ACTION_LISTEN)
        .context(&format!("http://{}", addr))
        .human("✅ Listening on {}")
        .emit();

    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    NARRATE
        .action(ACTION_READY)
        .human("✅ Hive ready")
        .emit();
    
    axum::serve(listener, app).await?;

    Ok(())
}
```

---

### Step 4: Update health_check to Use Narration (Optional)

**File:** `bin/20_rbee_hive/src/main.rs`

**Current:**
```rust
async fn health_check() -> &'static str {
    "ok"
}
```

**Enhanced (optional):**
```rust
use narration::{NARRATE};

async fn health_check() -> &'static str {
    // TEAM-202: Optional narration for health checks
    // (Only enable if you want to see these in logs)
    // NARRATE.action("health_check").human("Health check received").emit();
    "ok"
}
```

**Note:** Health checks are frequent, so narration might be too noisy. Up to you!

---

## How It Works (No HTTP Ingestion!)

### The Magic: Job-Scoped SSE (from TEAM-200)

```
┌─────────────────────────────────────────────────────────────┐
│ HIVE NARRATION FLOW                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Hive calls NARRATE.action().emit()                     │
│     ↓                                                        │
│  2. narration-core::narrate_at_level()                     │
│     ├─ stderr (daemon logs) ✅                              │
│     └─ sse_sink::send(fields)                              │
│         ↓                                                   │
│  3. TEAM-200's job-scoped routing:                         │
│     ├─ If fields.job_id exists:                            │
│     │   └─ Send to job-specific channel ✅                 │
│     └─ Otherwise:                                           │
│         └─ Send to global channel ✅                        │
│                                                             │
│  4. Keeper's job SSE stream receives it                    │
│     └─ Prints to stdout ✅                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight:** You just call `.emit()` - the rest is automatic!

---

## Testing Strategy

### Manual Test 1: Local Hive

```bash
# Terminal 1: Build
cargo build -p rbee-hive

# Terminal 2: Run hive
./target/debug/rbee-hive --port 8600

# Expected output (stderr):
[hive      ] startup        : 🐝 Starting on port 8600, hive_id: localhost, queen: http://localhost:8500
[hive      ] heartbeat      : 💓 Heartbeat task started (5s interval)
[hive      ] listen         : ✅ Listening on http://127.0.0.1:8600
[hive      ] ready          : ✅ Hive ready
```

### Manual Test 2: Via Keeper (The Real Test!)

```bash
# Terminal 1: Queen
./rbee queen start

# Terminal 2: Hive (via keeper)
./rbee hive start

# Expected: Keeper sees hive narration!
[keeper    ] job_submit     : 📋 Job job-xyz submitted
[keeper    ] job_stream     : 📡 Streaming results...
[hive      ] startup        : 🐝 Starting on port 8600, hive_id: localhost, queen: http://localhost:8500
[hive      ] heartbeat      : 💓 Heartbeat task started (5s interval)
[hive      ] listen         : ✅ Listening on http://127.0.0.1:8600
[hive      ] ready          : ✅ Hive ready
[keeper    ] job_complete   : ✅ Complete
```

**Success:** Keeper sees hive's narration even though hive is running as daemon!

---

## Verification Checklist

### Before Starting
- [ ] TEAM-200 has completed job-scoped SSE
- [ ] TEAM-201 has completed centralized formatting
- [ ] Read TEAM-197 OPPORTUNITY 2 (worker pattern)

### Implementation
- [ ] Add narration-core dependency
- [ ] Create narration.rs module
- [ ] Define action constants
- [ ] Replace all println!() with NARRATE
- [ ] Verify no println!() remain (search for "println!")

### Testing
- [ ] Build: `cargo build -p rbee-hive`
- [ ] Run locally, verify stderr output
- [ ] Run via keeper, verify keeper sees narration
- [ ] Test remote hive (if available)

### Code Quality
- [ ] Added TEAM-202 signature
- [ ] No TODO markers
- [ ] Consistent with other binaries' narration

---

## Expected Changes

### Files Created
- `bin/20_rbee_hive/src/narration.rs` (~25 lines)

### Files Modified
- `bin/20_rbee_hive/Cargo.toml` (~1 line added)
- `bin/20_rbee_hive/src/main.rs` (~20 lines changed)

### Impact
- **Lines added:** ~45
- **println!() replaced:** 4-5 occurrences
- **User benefit:** Hive narration visible remotely!

---

## Common Pitfalls

### ❌ WRONG: Using println!() for Narration
```rust
// BAD: Not visible remotely
println!("🐝 rbee-hive starting on port {}", args.port);
```

### ✅ CORRECT: Using NARRATE
```rust
// GOOD: Visible everywhere (stderr + SSE)
NARRATE
    .action(ACTION_STARTUP)
    .context(&args.port.to_string())
    .human("🐝 Starting on port {}")
    .emit();
```

### ❌ WRONG: Forgetting job_id Context
```rust
// BAD: Narration won't route to job stream!
NARRATE.action(ACTION_STARTUP).human("Starting").emit();
```

**Note:** For hive startup, you DON'T have job_id yet. That's OK! The narration goes to global channel. Once hive is running and processes jobs, those narrations will have job_id.

### ✅ CORRECT: Let Narration-Core Handle Routing
```rust
// GOOD: Just emit, narration-core routes correctly
NARRATE.action(ACTION_STARTUP).human("Starting").emit();
// If no job_id → global channel
// If job_id exists → job-specific channel (automatic!)
```

---

## Success Criteria

### Functionality
- ✅ All hive narration visible in daemon logs (stderr)
- ✅ Hive narration visible in keeper (via SSE)
- ✅ No more println!() in hive
- ✅ Format matches other components

### Remote Operation
- ✅ When hive runs remotely, keeper still sees narration
- ✅ No extra network configuration needed
- ✅ Works through queen's job SSE

### Code Quality
- ✅ Consistent with queen/keeper/worker narration style
- ✅ Action constants defined
- ✅ Clean, readable code

---

## Why No HTTP Ingestion? (TEAM-197's Insight)

### Worker Already Proved This Works!

**Worker's narrate_dual() (line 117-135 in worker/narration.rs):**
```rust
pub fn narrate_dual(fields: NarrationFields) {
    // 1. stderr (for daemon logs)
    observability_narration_core::narrate(fields.clone());

    // 2. Thread-local SSE channel (automatic!)
    let sse_event = InferenceEvent::Narration { /* ... */ };
    narration_channel::send_narration(sse_event);
}
```

**Advantages:**
- ✅ No network hop (direct to SSE stream)
- ✅ No authentication needed (internal)
- ✅ Automatic job scoping (thread-local)
- ✅ Proven pattern (worker uses it successfully)

**TEAM-198's Proposed HTTP Ingestion:**
- ❌ Extra network hop (POST to queen)
- ❌ Needs authentication
- ❌ Needs manual job_id routing
- ❌ More complex

**TEAM-197's Verdict:** Use the simpler, proven pattern!

---

## Next Team

**TEAM-203** depends on your work (needs to verify end-to-end narration flow).

Make sure keeper can see hive narration before handing off!

---

## Summary

**Problem:** Hive uses println!(), not visible remotely  
**Solution:** Replace with NARRATE, flows through job-scoped SSE  
**Pattern:** Like worker (no HTTP ingestion needed)  
**Impact:** ~45 lines, hive narration visible everywhere

---

**Created for:** TEAM-202  
**Priority:** MEDIUM  
**Status:** READY TO IMPLEMENT (after TEAM-200 and TEAM-201)

**This completes narration coverage. TEAM-203 will verify everything works end-to-end.**
