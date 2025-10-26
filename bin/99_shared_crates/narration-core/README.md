# narration-core 🎀

**Ultra-concise, privacy-safe observability for distributed systems**

**Version**: 0.7.0  
**Status**: ✅ Production Ready  
**License**: GPL-3.0-or-later

---

## What is narration-core?

A structured logging system that makes debugging **delightful** by turning cryptic logs into human-readable stories.

**Key Features:**
- 📝 **Ultra-concise API** - `n!()` macro reduces narration from 5 lines to 1
- 🔒 **Privacy-safe** - Multi-tenant isolation, GDPR/SOC 2 compliant
- 🎭 **Three narration modes** - Human (technical), Cute (whimsical), Story (dialogue)
- 🌊 **SSE streaming** - Real-time events to web clients
- 🔄 **Process capture** - Worker stdout flows through SSE
- ✅ **100% test coverage** - 180+ tests passing

---

## Quick Start

### Basic Usage

```rust
use observability_narration_core::n;

// Simple narration (1 line!)
n!("worker_spawn", "Spawning worker {} on device {}", worker_id, device);
```

**Output:**
```
[worker    ] worker_spawn    : Spawning worker gpu-0-r1 on device cuda:0
```

### All Three Narration Modes

```rust
n!("deploy",
    human: "Deploying service {} to production",
    cute: "🚀 Launching {} into the cloud!",
    story: "The orchestrator whispered to {}: 'Time to fly'",
    service_name
);
```

Switch modes at runtime:
```rust
use observability_narration_core::{set_narration_mode, NarrationMode};

set_narration_mode(NarrationMode::Cute);
// All narration now shows cute version (or falls back to human)
```

---

## Core Concepts

### 1. The `n!()` Macro

The primary API for emitting narration events:

```rust
// Simple message
n!("startup", "Worker starting");

// With variables
n!("ready", "Worker {} is ready", worker_id);

// Multiple variables
n!("spawn", "Spawning worker {} on device {}", worker_id, device);

// All 3 modes
n!("action",
    human: "Technical message",
    cute: "🎀 Fun message",
    story: "'Hello', said the system"
);
```

### 2. Narration Modes

**Human** (default) - Technical, precise debugging information  
**Cute** - Whimsical, emoji-enhanced storytelling  
**Story** - Dialogue-focused, screenplay style

### 3. SSE Streaming

Narration events flow through Server-Sent Events (SSE) to web clients:

```rust
// Events automatically route to correct SSE channel via job_id
with_narration_context(
    NarrationContext::new().with_job_id(&job_id),
    async {
        n!("processing", "Processing request");
        // Automatically sent to client via SSE!
    }
).await;
```

### 4. Process Capture

Capture child process stdout and convert to SSE events:

```rust
use observability_narration_core::ProcessNarrationCapture;
use tokio::process::Command;

let capture = ProcessNarrationCapture::new(Some(job_id));
let mut command = Command::new("worker-binary");
let child = capture.spawn(command).await?;
// Worker's stdout narration now flows through SSE!
```

---

## Privacy & Security

### Multi-Tenant Safe

**Problem Solved:** In v0.6.0 and earlier, global stderr output leaked data across users.

**Solution:** Complete removal of stderr output. Narration only goes to:
- **SSE channels** - Job-scoped, isolated per user
- **Capture adapter** - Tests only

**Result:**
- ✅ User A never sees User B's data
- ✅ GDPR data minimization
- ✅ SOC 2 access control
- ✅ Fail-fast security (no job_id = dropped)

### Security Model

```rust
// Without job_id - event is dropped (security)
n!("action", "message");  // ❌ Dropped

// With job_id - event routes to correct SSE channel
with_narration_context(
    NarrationContext::new().with_job_id(&job_id),
    async {
        n!("action", "message");  // ✅ Sent to job's SSE channel
    }
).await;
```

---

## Architecture

### Narration Flow

```
Worker Process
    ↓ stdout
ProcessNarrationCapture (parses & re-emits)
    ↓ SSE
Client (web UI)
```

### Separation of Concerns

**narration-core** (observability):
- Handles observability events
- SSE channel management
- Event formatting and routing
- **Does NOT emit lifecycle signals**

**job-server** (lifecycle):
- Manages job lifecycle
- Emits [DONE]/[ERROR] signals
- Job state tracking

---

## Testing

### Capture Adapter

```rust
use observability_narration_core::CaptureAdapter;

#[test]
fn test_narration() {
    let adapter = CaptureAdapter::install();
    
    n!("test_action", "Test message");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].action, "test_action");
}
```

### Test Coverage

- **180+ tests passing** (100% pass rate)
- narration-core: 106 tests ✅
- job-server: 74 tests ✅
- Privacy isolation: 10 tests ✅
- E2E integration: 5 tests ✅

---

## Migration from Old API

### Pattern 1: Simple Message

```rust
// Old (5 lines):
NARRATE.action("startup")
    .human("Worker starting")
    .emit();

// New (1 line):
n!("startup", "Worker starting");
```

### Pattern 2: With Variables

```rust
// Old:
NARRATE.action("ready")
    .context(&worker_id)
    .human("Worker {} is ready")
    .emit();

// New:
n!("ready", "Worker {} is ready", worker_id);
```

### Pattern 3: Multiple Variables

```rust
// Old:
NARRATE.action("spawn")
    .context(&worker_id)
    .context(&device)
    .human("Spawning worker {} on device {}")
    .emit();

// New:
n!("spawn", "Spawning worker {} on device {}", worker_id, device);
```

---

## Output Format

```
[{actor:<10}] {action:<15}: {message}
```

**Example:**
```
[keeper    ] queen_start    : Starting queen on http://localhost:8500
[queen     ] listen         : Listening on http://127.0.0.1:8500
[qn-router ] job_create     : Job abc123 created
```

**Total prefix**: 30 characters (perfect column alignment)

---

## Advanced Features

### Runtime Mode Configuration

```rust
use observability_narration_core::{set_narration_mode, get_narration_mode, NarrationMode};

// Switch to cute mode
set_narration_mode(NarrationMode::Cute);

// Query current mode
let mode = get_narration_mode();
```

### Full Rust format!() Support

```rust
// Width, precision, debug, hex - all supported!
n!("debug", "Hex: {:x}, Debug: {:?}, Width: {:5}", 255, vec![1,2,3], 42);
```

### Context Propagation

```rust
let ctx = NarrationContext::new()
    .with_job_id(&job_id)
    .with_correlation_id(&correlation_id);

with_narration_context(ctx, async {
    n!("action", "Message");  // job_id auto-injected
    
    tokio::spawn(async {
        // Spawned tasks need explicit context
        let ctx = NarrationContext::new().with_job_id(&job_id);
        with_narration_context(ctx, async {
            n!("nested", "Also has job_id");
        }).await;
    }).await;
}).await;
```

---

## Version History

- **v0.7.0** - Privacy fix, `n!()` macro, SSE optional, process capture, E2E tests
- **v0.6.0** - Architecture fixes, circular dependency resolved, 100% test pass rate
- **v0.5.0** - Fixed-width format, compile-time validation
- **v0.4.0** - Factory pattern, column alignment
- **v0.3.0** - Table formatting
- **v0.2.0** - Builder pattern, Axum middleware
- **v0.1.0** - Initial release

---

## Documentation

- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Team History**: [TEAM_RESPONSIBILITIES.md](TEAM_RESPONSIBILITIES.md)
- **Specification**: [`.specs/00_narration-core.md`](.specs/00_narration-core.md)

---

## Design Philosophy

1. **Human-First** - Logs should be readable by humans, not just machines
2. **Privacy-Safe** - Multi-tenant isolation by design
3. **Ultra-Concise** - Minimal boilerplate (1 line for most cases)
4. **Never Fails** - SSE is optional, narration always works
5. **Context-Rich** - Interpolate values for clarity

---

**Made with 💝 by the rbee team**
