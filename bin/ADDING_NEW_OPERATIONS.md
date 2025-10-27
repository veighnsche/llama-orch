# Adding New Operations - Architecture Guide

**Last Updated:** Oct 27, 2025 (TEAM-323)

## CRITICAL: Two Different Patterns

rbee has **TWO COMPLETELY DIFFERENT** patterns for operations:

### Pattern 1: Lifecycle Operations (Direct)
**Used for:** Start, Stop, Install, Uninstall, Rebuild

These operations are handled **DIRECTLY** by rbee-keeper using `daemon-lifecycle` crate.
**NO job router, NO Operation enum, NO SSE streaming.**

**Why?** These are local system operations that don't need job tracking or streaming.

### Pattern 2: Job-Based Operations (Routed)
**Used for:** Worker spawn, Model download, Inference, Status checks

These operations go through the **job router** and use SSE streaming.
**Requires Operation enum variant, job_id, SSE channel.**

**Why?** These are long-running operations that need progress tracking and streaming output.

---

## Pattern 1: Lifecycle Operations (Direct)

### Example: Queen Operations

```rust
// bin/00_rbee_keeper/src/handlers/queen.rs
pub async fn handle_queen(action: QueenAction, queen_url: &str) -> Result<()> {
    match action {
        QueenAction::Start => start_queen(port).await,
        QueenAction::Stop => stop_http_daemon("queen-rbee", queen_url).await,
        QueenAction::Install { binary } => {
            install_queen("queen-rbee", binary, None).await?;
            Ok(())
        }
        QueenAction::Uninstall => uninstall_queen(queen_url).await,
        QueenAction::Rebuild => {
            let config = RebuildConfig::new("queen-rbee");
            build_daemon_local(config).await?;
            Ok(())
        }
    }
}
```

### Example: Hive Operations

```rust
// bin/00_rbee_keeper/src/handlers/hive.rs
pub async fn handle_hive(action: HiveAction, queen_url: &str) -> Result<()> {
    match action {
        HiveAction::Start { port } => start_hive(port.unwrap_or(7835), queen_url).await,
        HiveAction::Stop { port } => {
            let base_url = format!("http://localhost:{}", port.unwrap_or(7835));
            stop_http_daemon("rbee-hive", &base_url).await
        }
        HiveAction::Install { alias: _ } => {
            daemon_lifecycle::install_to_local_bin("rbee-hive", None, None).await?;
            Ok(())
        }
        HiveAction::Uninstall { alias: _ } => {
            let hive_url = "http://localhost:7835";
            let _ = stop_http_daemon("rbee-hive", hive_url).await;
            daemon_lifecycle::uninstall_from_local_bin("rbee-hive").await?;
            Ok(())
        }
        HiveAction::Rebuild { alias: _ } => {
            let config = RebuildConfig::new("rbee-hive");
            build_daemon_local(config).await?;
            Ok(())
        }
    }
}
```

### Adding a New Lifecycle Operation

**Files to modify:**
1. `bin/00_rbee_keeper/src/cli/{queen,hive}.rs` - Add CLI action
2. `bin/00_rbee_keeper/src/handlers/{queen,hive}.rs` - Add handler
3. (Optional) `bin/05_rbee_keeper_crates/{queen,hive}-lifecycle/` - Add helper function

**DO NOT:**
- ❌ Add to Operation enum
- ❌ Add to job_router.rs
- ❌ Create SSE channels
- ❌ Use job_id

**Example: Adding "Restart" operation**

```rust
// 1. Add CLI action
pub enum QueenAction {
    Restart,
}

// 2. Add handler
QueenAction::Restart => {
    stop_http_daemon("queen-rbee", queen_url).await?;
    start_queen(port).await
}
```

---

## Pattern 2: Job-Based Operations (Routed)

### When to Use This Pattern

Use job-based operations when you need:
- ✅ Progress tracking (long-running operations)
- ✅ SSE streaming (real-time output to client)
- ✅ Job history/logging
- ✅ Operations that run on remote services (hive, worker)

### The 3-File Pattern

Every job-based operation requires updates to **3 files**:

```
1. operations-contract/src/lib.rs  (Contract)
2. queen-rbee/src/job_router.rs    (Router) OR hive/src/job_router.rs
3. rbee-keeper/src/handlers/       (CLI Handler)
```

### Step 1: Define Operation (Contract)

**File:** `bin/97_contracts/operations-contract/src/lib.rs`

```rust
pub enum Operation {
    // Add your operation
    WorkerYourNewOp {
        hive_id: String,
        // your fields
    },
}

impl Operation {
    pub fn name(&self) -> &'static str {
        match self {
            Operation::WorkerYourNewOp { .. } => "worker_your_new_op",
        }
    }
    
    pub fn hive_id(&self) -> Option<&str> {
        match self {
            Operation::WorkerYourNewOp { hive_id } => Some(hive_id),
        }
    }
}
```

### Step 2: Route Operation (Server)

**File:** `bin/10_queen_rbee/src/job_router.rs` OR `bin/20_rbee_hive/src/job_router.rs`

```rust
match operation {
    Operation::WorkerYourNewOp { hive_id } => {
        n!("worker_your_new_op_start", "Starting operation for hive '{}'", hive_id);
        
        // Your implementation here
        
        n!("worker_your_new_op_success", "✅ Operation complete");
    }
}
```

### Step 3: Add CLI Command (Client)

**File:** `bin/00_rbee_keeper/src/cli/worker.rs` and `src/handlers/worker.rs`

```rust
// CLI action
pub enum WorkerAction {
    YourNewOp {
        #[arg(short = 'h', long = "hive")]
        hive_id: String,
    },
}

// Handler
WorkerAction::YourNewOp { hive_id } => {
    let operation = Operation::WorkerYourNewOp { hive_id };
    submit_and_stream_job(queen_url, operation).await
}
```

---

## Decision Tree

```
Is this a lifecycle operation (start/stop/install/uninstall/rebuild)?
├─ YES → Use Pattern 1 (Direct)
│         - Add to CLI action enum
│         - Add handler calling daemon-lifecycle
│         - NO Operation enum
│         - NO job router
│
└─ NO → Does it need progress tracking or streaming?
        ├─ YES → Use Pattern 2 (Job-Based)
        │         - Add to Operation enum
        │         - Add to job router
        │         - Add CLI handler with submit_and_stream_job()
        │
        └─ NO → Use Pattern 1 (Direct)
                  - Simple request/response
                  - No job tracking needed
```

---

## Examples by Category

### Lifecycle Operations (Pattern 1)
- ✅ Queen: Start, Stop, Install, Uninstall, Rebuild, Status, Info
- ✅ Hive: Start, Stop, Install, Uninstall, Rebuild
- ✅ Worker: (none - workers are managed via job-based operations)

### Job-Based Operations (Pattern 2)
- ✅ Worker: Spawn, ProcessList, ProcessGet, ProcessDelete
- ✅ Model: Download, List, Get, Delete
- ✅ Inference: Infer
- ✅ Status: Status (live registry query)
- ✅ Checks: QueenCheck, HiveCheck (narration tests)
- ✅ Hive: Get, Status, RefreshCapabilities (query operations, not lifecycle)

---

## Common Mistakes

### ❌ WRONG: Routing lifecycle through job router

```rust
// DON'T DO THIS
HiveAction::Install { alias } => {
    let operation = Operation::HiveInstall { alias };
    submit_and_stream_job(queen_url, operation).await
}
```

### ✅ RIGHT: Direct daemon-lifecycle call

```rust
// DO THIS
HiveAction::Install { alias: _ } => {
    daemon_lifecycle::install_to_local_bin("rbee-hive", None, None).await?;
    Ok(())
}
```

---

## Why Two Patterns?

**Historical context:** The old architecture routed EVERYTHING through the job router,
including simple lifecycle operations. This created:
- Unnecessary complexity (job_id for local operations)
- Inconsistency (queen used direct calls, hive used job router)
- Confusion (why does install need SSE streaming?)

**TEAM-323 fix:** Lifecycle operations now use `daemon-lifecycle` directly.
This matches how queen operations work and eliminates the inconsistency.

---

## Testing Checklist

### Pattern 1 (Lifecycle)
- [ ] `cargo check -p rbee-keeper` passes
- [ ] CLI help shows command: `./rbee queen --help`
- [ ] Command executes: `./rbee queen install`
- [ ] Binary appears in `~/.local/bin/`

### Pattern 2 (Job-Based)
- [ ] `cargo check -p operations-contract` passes
- [ ] `cargo check -p queen-rbee` passes
- [ ] `cargo check -p rbee-keeper` passes
- [ ] CLI help shows command: `./rbee worker --help`
- [ ] Command executes: `./rbee worker spawn ...`
- [ ] SSE streaming works (real-time output)

---

## Architecture Principles

### Lifecycle Operations (Pattern 1)
- **rbee-keeper**: Calls daemon-lifecycle directly
- **daemon-lifecycle**: Shared crate for all lifecycle operations
- **No server involvement**: Local operations only

### Job-Based Operations (Pattern 2)
- **rbee-keeper**: Thin HTTP client, submits jobs
- **operations-contract**: Type-safe contract
- **job_router**: Dispatcher in queen/hive
- **SSE streaming**: Real-time progress updates

---

## Need Help?

**For lifecycle operations:** See `bin/00_rbee_keeper/src/handlers/queen.rs`
**For job-based operations:** See existing operations in `operations-contract/src/lib.rs`

**Rule of thumb:** If it's start/stop/install/uninstall/rebuild → Pattern 1 (Direct)
Everything else → Pattern 2 (Job-Based)
