# TEAM-152 Narration Output Channels

**Team:** TEAM-152  
**Date:** 2025-10-20  
**Status:** ✅ Tracing Removed - Narration Output Channels Need Configuration

---

## 🎯 Current State

### ✅ Completed
- ❌ **All tracing removed** from daemon-lifecycle, queen-lifecycle, and queen-rbee
- ✅ **All narration inline** with human-readable strings in code
- ✅ **Clean compilation** - no tracing dependencies anywhere
- ✅ **Consistent pattern** - Actor/action/target taxonomy throughout

### ⏳ Next Step: Configure Narration Output

Currently, narration events are **emitted but not visible** because narration-core uses the `tracing` backend by default, and we removed tracing initialization.

---

## 📊 Narration Output Channels

### User's Requirements

> "The narrator needs to pass narration through stdout or stderr until the queen bee server is up then the narration needs to switch to SSE!"

### Three Output Channels

1. **Shell (stdout/stderr)** - For rbee-keeper and queen-rbee startup
2. **File (structured logs)** - For production logging
3. **SSE (Server-Sent Events)** - For real-time streaming to clients

---

## 🔧 Implementation Plan

### Phase 1: Shell Output (Current Priority)

**Goal:** See all narration in the shell during startup

**Solution:** Configure narration-core to output to stdout/stderr

**Options:**

#### Option A: Direct stdout (Simplest)
```rust
// In narration.rs or lib.rs
impl Narration {
    pub fn emit(self) {
        // Print to stdout for shell visibility
        eprintln!("{}", self.fields.human);
        
        // Also emit to tracing backend (if configured)
        // tracing::info!(...);
    }
}
```

#### Option B: Configurable output
```rust
// Set output mode
NarrationConfig::set_output(NarrationOutput::Stdout);

// Narration automatically outputs to configured channel
Narration::new(ACTOR, ACTION, target)
    .human("message")
    .emit(); // Goes to stdout
```

### Phase 2: SSE Output (After HTTP Server Starts)

**Goal:** Switch narration to SSE once queen-rbee HTTP server is up

**Flow:**
```
1. rbee-keeper starts
   ├─ Narration → stdout/stderr
   │
2. rbee-keeper spawns queen-rbee
   ├─ Narration → stdout/stderr (inherited)
   │
3. queen-rbee HTTP server starts
   ├─ Narration → SSE channel
   │
4. rbee-keeper connects to SSE
   ├─ Narration → SSE → rbee-keeper → stdout
```

**Implementation:**
```rust
// In queen-rbee after HTTP server starts
NarrationConfig::set_output(NarrationOutput::SSE {
    endpoint: "/narration/stream",
});

// In rbee-keeper after queen is healthy
let sse_stream = connect_to_queen_sse("http://localhost:8500/narration/stream").await?;

// Stream SSE events to stdout
while let Some(event) = sse_stream.next().await {
    println!("{}", event.data);
}
```

---

## 🎭 Current Narration Events

### rbee-keeper Events
```rust
// Queen already running
Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_CHECK, base_url)
    .human("Queen is already running and healthy")
    .emit();

// Starting queen
Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_START, "queen-rbee")
    .human("⚠️  Queen is asleep, waking queen")
    .emit();

// Queen ready
Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_READY, "queen-rbee")
    .human("✅ Queen is awake and healthy")
    .duration_ms(elapsed_ms)
    .emit();
```

### queen-rbee Events
```rust
// Starting
Narration::new(ACTOR_QUEEN_RBEE, ACTION_START, &port.to_string())
    .human(format!("Queen-rbee starting on port {}", port))
    .emit();

// Listening
Narration::new(ACTOR_QUEEN_RBEE, ACTION_LISTEN, &addr.to_string())
    .human(format!("Listening on http://{}", addr))
    .emit();

// Ready
Narration::new(ACTOR_QUEEN_RBEE, ACTION_READY, "http-server")
    .human("Ready to accept connections")
    .emit();
```

### daemon-lifecycle Events
```rust
// Spawning
Narration::new(ACTOR_DAEMON_LIFECYCLE, ACTION_SPAWN, &binary_path)
    .human(format!("Spawning daemon: {} with args: {:?}", binary_path, args))
    .emit();

// Spawned
Narration::new(ACTOR_DAEMON_LIFECYCLE, ACTION_SPAWN, &pid)
    .human(format!("Daemon spawned with PID: {}", pid))
    .emit();
```

---

## 🚀 Recommended Next Steps for TEAM-153

### 1. Configure Narration Output to Stdout

**Quick Fix:** Add to narration-core or each binary:
```rust
// In main.rs before any narration
std::env::set_var("NARRATION_OUTPUT", "stdout");
```

Or modify narration-core to print to stderr by default:
```rust
// In narration-core/src/lib.rs
pub fn emit(self) {
    eprintln!("{}", self.fields.human); // Always print to stderr
    // ... rest of emit logic
}
```

### 2. Add SSE Endpoint to queen-rbee

```rust
// In queen-rbee/src/http/narration.rs
use axum::response::sse::{Event, Sse};
use futures::stream::Stream;

pub async fn narration_stream() -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = NarrationStream::new();
    Sse::new(stream)
}

// In router
.route("/narration/stream", get(narration::narration_stream))
```

### 3. Connect rbee-keeper to SSE

```rust
// After queen is healthy
let sse_client = reqwest::Client::new();
let mut sse_stream = sse_client
    .get("http://localhost:8500/narration/stream")
    .send()
    .await?
    .bytes_stream();

// Forward SSE to stdout
while let Some(chunk) = sse_stream.next().await {
    let text = String::from_utf8_lossy(&chunk?);
    print!("{}", text);
}
```

---

## 📝 Files Modified

### Completed
- ✅ `daemon-lifecycle/src/lib.rs` - All narration inline, no tracing
- ✅ `daemon-lifecycle/Cargo.toml` - Removed tracing, added narration-core
- ✅ `queen-lifecycle/src/lib.rs` - All narration inline, no tracing, no println
- ✅ `queen-lifecycle/Cargo.toml` - Removed tracing
- ✅ `queen-rbee/src/main.rs` - All narration inline, no tracing
- ✅ `queen-rbee/Cargo.toml` - Removed tracing, added narration-core

### Next
- ⏳ `narration-core/src/lib.rs` - Add stdout output option
- ⏳ `queen-rbee/src/http/narration.rs` - Add SSE endpoint
- ⏳ `rbee-keeper/src/main.rs` - Connect to SSE after queen is healthy

---

## 🎯 Expected Output

### Phase 1: Shell Output Only
```bash
./target/debug/rbee-keeper infer "hello" --model HF:author/minillama

# Expected output (all narration):
⚠️  Queen is asleep, waking queen
Found queen-rbee binary at target/debug/queen-rbee
Spawning daemon: target/debug/queen-rbee with args: ["--port", "8500"]
Daemon spawned with PID: 12345
Queen-rbee starting on port 8500
Listening on http://127.0.0.1:8500
Ready to accept connections
Polling queen health (attempt 1, delay 100ms)
Queen health check succeeded after 1.2s
✅ Queen is awake and healthy
```

### Phase 2: SSE Output (After HTTP Server Up)
```bash
./target/debug/rbee-keeper infer "hello" --model HF:author/minillama

# Output until HTTP server starts:
⚠️  Queen is asleep, waking queen
Found queen-rbee binary at target/debug/queen-rbee
Spawning daemon: target/debug/queen-rbee with args: ["--port", "8500"]
Daemon spawned with PID: 12345

# Then SSE connection established:
🔗 Connected to queen-rbee SSE stream

# All subsequent narration via SSE:
Queen-rbee starting on port 8500
Listening on http://127.0.0.1:8500
Ready to accept connections
✅ Queen is awake and healthy
```

---

## 💡 Key Insights

### Why Narration > Tracing

1. **Multi-channel:** stdout, file, SSE (tracing only does file/console)
2. **Structured:** Actor/action/target taxonomy
3. **Human-readable:** Clear messages for operators
4. **Switchable:** Can change output channel at runtime

### Why SSE for Narration

1. **Real-time:** Clients see events as they happen
2. **One-way:** Server → Client (perfect for narration)
3. **HTTP-based:** No WebSocket complexity
4. **Reconnectable:** Clients can reconnect if disconnected

### Why stdout/stderr First

1. **Startup visibility:** See what's happening during boot
2. **Debugging:** Easy to see issues before HTTP server is up
3. **Fallback:** If SSE fails, still have output

---

## 🎊 Status

**TEAM-152 Mission:** ✅ Tracing Removed, Narration Inline

**Completed:**
- ✅ All tracing removed
- ✅ All narration inline
- ✅ Clean compilation
- ✅ Consistent pattern

**Next (TEAM-153):**
- ⏳ Configure narration stdout output
- ⏳ Add SSE endpoint to queen-rbee
- ⏳ Connect rbee-keeper to SSE
- ⏳ Switch narration channel after HTTP server starts

---

**Signed:** TEAM-152  
**Date:** 2025-10-20  
**Status:** Tracing Removed - Output Channels Need Configuration ✅
