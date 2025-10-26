# TEAM-301: Phase 4 - Keeper Lifecycle Management

**Status:** BLOCKED (Requires TEAM-300 completion)  
**Estimated Duration:** 1 week  
**Dependencies:** TEAM-300 (Phase 3 Process Capture)  
**Risk Level:** Medium (keeper changes, SSH)

---

## Mission

Enable rbee-keeper to spawn queen/hive processes and display their stdout to the user in real-time. Keeper subscribes to SSE stream and displays to terminal (single-user). Completes the end-to-end narration flow.

**CRITICAL:** Keeper displays via SSE subscription, NOT via stderr (see PRIVACY_ATTACK_SURFACE_ANALYSIS.md)

---

## ⚠️ CRITICAL: DO YOUR RESEARCH FIRST!

### Required Research

1. **Read TEAM-300 Handoff** - Understand process capture
2. **Study Keeper Structure** - `rbee-keeper/src/handlers/`
3. **Study Queen Lifecycle** - How keeper starts/stops queen
4. **Study Hive Lifecycle** - How keeper starts/stops hives (SSH)
5. **Create Research Summary** - `.plan/TEAM_301_RESEARCH_SUMMARY.md`

**DO NOT CODE UNTIL RESEARCH IS COMPLETE!**

---

## Problem: Keeper Can't See Daemon Startup

```rust
// Keeper starts queen
let child = Command::new("queen-rbee").spawn()?;
// ↑ Queen's stdout invisible! User sees nothing!

// Keeper starts hive via SSH
let child = Command::new("ssh")
    .arg("user@host")
    .arg("rbee-hive")
    .spawn()?;
// ↑ Hive's stdout lost over SSH! User blind!
```

## Solution: Capture and Display

```rust
// Keeper starts queen with stdout capture
let mut command = Command::new("queen-rbee");
command.stdout(Stdio::piped());
let mut child = command.spawn()?;

// Stream to terminal in real-time
tokio::spawn(async move {
    let reader = BufReader::new(child.stdout.take().unwrap());
    let mut lines = reader.lines();
    while let Ok(Some(line)) = lines.next_line().await {
        println!("{}", line);  // User sees it!
    }
});
```

---

## Implementation Tasks

### Task 1: Create Display Module

**New File:** `bin/00_rbee_keeper/src/display.rs`

```rust
use observability_narration_core::sse_sink::NarrationEvent;
use tokio::sync::mpsc;

/// Display narration events to terminal
///
/// TEAM-301: Keeper subscribes to SSE and displays to terminal.
/// This code ONLY exists in keeper (single-user CLI).
/// Daemons do NOT have this code (secure by design).
pub async fn display_narration_stream(mut rx: mpsc::Receiver<NarrationEvent>) {
    while let Some(event) = rx.recv().await {
        // Display to keeper's terminal (single-user, no privacy issue)
        eprintln!("{}", event.formatted);
    }
}
```

### Task 2: Setup Keeper Display in Main

**File:** `bin/00_rbee_keeper/src/main.rs`

```rust
use observability_narration_core::{sse_sink, NarrationContext, with_narration_context};
use crate::display::display_narration_stream;

#[tokio::main]
async fn main() -> Result<()> {
    // TEAM-301: Create keeper's display channel
    let keeper_job_id = "keeper-display";
    sse_sink::create_job_channel(keeper_job_id.to_string(), 1000);
    let rx = sse_sink::take_job_receiver(keeper_job_id)
        .expect("Failed to create keeper display channel");
    
    // TEAM-301: Spawn display task
    tokio::spawn(async move {
        display_narration_stream(rx).await;
    });
    
    // TEAM-301: Set narration context for keeper
    let ctx = NarrationContext::new().with_job_id(keeper_job_id);
    
    // TEAM-301: Run keeper with narration context
    with_narration_context(ctx, async {
        run_keeper().await
    }).await
}
```

### Task 3: Update Keeper Queen Lifecycle

**File:** `bin/00_rbee_keeper/src/handlers/queen.rs` (or similar)

```rust
pub async fn start_queen() -> Result<()> {
    // TEAM-301: Narration goes to keeper's SSE channel, then displayed to terminal
    n!("queen_start", "Starting queen-rbee...");
    
    let mut command = Command::new("queen-rbee");
    command.arg("--port").arg("7833");
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    
    let mut child = command.spawn()?;
    
    // Stream stdout to terminal
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                println!("{}", line);
            }
        });
    }
    
    // Stream stderr too
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("{}", line);
            }
        });
    }
    
    // Wait for health check
    wait_for_health("http://localhost:7833").await?;
    
    n!("queen_ready", "✅ Queen ready");
    
    Ok(())
}
```

### Task 2: Update Keeper Hive Lifecycle (SSH)

**File:** `bin/00_rbee_keeper/src/handlers/hive.rs` (or similar)

```rust
pub async fn start_hive_ssh(config: &HiveConfig) -> Result<()> {
    n!("hive_start_ssh", "Starting hive on {}...", config.host);
    
    let mut command = Command::new("ssh");
    command.arg(format!("{}@{}", config.user, config.host));
    command.arg("rbee-hive");
    command.arg("--port").arg(config.port.to_string());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    
    let mut child = command.spawn()?;
    
    // Stream SSH output to terminal
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                println!("{}", line);  // User sees hive startup over SSH!
            }
        });
    }
    
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("{}", line);
            }
        });
    }
    
    // Wait for health check
    let hive_url = format!("http://{}:{}", config.host, config.port);
    wait_for_health(&hive_url).await?;
    
    n!("hive_ready_ssh", "✅ Hive ready on {}", config.host);
    
    Ok(())
}
```

### Task 3: Add Helper Function (DRY)

**New File:** `bin/00_rbee_keeper/src/process_utils.rs`

```rust
use tokio::process::{Command, Child};
use tokio::io::{AsyncBufReadExt, BufReader};
use std::process::Stdio;

/// Spawn command with stdout/stderr streaming to terminal
pub async fn spawn_with_output_streaming(mut command: Command) -> Result<Child> {
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    
    let mut child = command.spawn()?;
    
    // Stream stdout
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                println!("{}", line);
            }
        });
    }
    
    // Stream stderr
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("{}", line);
            }
        });
    }
    
    Ok(child)
}
```

**Usage:**
```rust
// Simplified usage
pub async fn start_queen() -> Result<()> {
    n!("queen_start", "Starting queen-rbee...");
    
    let command = Command::new("queen-rbee")
        .arg("--port").arg("7833");
    
    let child = spawn_with_output_streaming(command).await?;
    
    wait_for_health("http://localhost:7833").await?;
    n!("queen_ready", "✅ Queen ready");
    
    Ok(())
}
```

### Task 4: Test Stdout Display

**New File:** `bin/00_rbee_keeper/tests/process_output_tests.rs`

```rust
#[tokio::test]
async fn test_keeper_displays_output() {
    // Mock process that outputs text
    let mut command = Command::new("echo");
    command.arg("[queen     ] startup         : Starting");
    
    let child = spawn_with_output_streaming(command).await.unwrap();
    
    // Wait for process
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Output should be displayed (visual check in terminal)
}
```

---

## Verification Checklist

- [ ] Keeper displays queen startup to terminal
- [ ] Keeper displays hive startup (SSH) to terminal
- [ ] stdout streaming works in real-time
- [ ] stderr streaming works
- [ ] No buffering issues
- [ ] Tests pass

---

## Success Criteria

1. **Queen startup visible** - User sees real-time narration
2. **Hive startup visible** - Even over SSH
3. **Real-time streaming** - No delays or buffering
4. **Complete flow** - End-to-end narration works everywhere!

---

## Final Handoff

Document in `.plan/TEAM_301_FINAL_SUMMARY.md`:

### What We Built (All 5 Phases)

**Phase 0 (TEAM-297):**
- Ultra-concise `n!()` macro
- 3 narration modes (human, cute, story)
- Removed `.context()` system
- 80% less code for simple cases

**Phase 1 (TEAM-298):**
- SSE delivery is now optional
- Stdout always works
- No more race conditions
- Resilient to timing issues

**Phase 2 (TEAM-299):**
- Thread-local context everywhere
- 100+ manual `.job_id()` calls removed
- Set once, use everywhere
- Context inheritance works

**Phase 3 (TEAM-300):**
- Process stdout capture
- Worker startup flows through SSE
- Regex parsing for narration
- No lost events

**Phase 4 (TEAM-301):**
- Keeper displays daemon startup
- SSH output capture works
- Real-time streaming
- Complete end-to-end flow

### Migration Guide

**For Simple Cases:**
```rust
// Old:
NARRATE.action("deploy")
    .context(&name)
    .human("Deploying {}")
    .emit();

// New:
n!("deploy", "Deploying {}", name);
```

**For Job Context:**
```rust
// Old:
NARRATE.action("step").job_id(&job_id).emit();

// New:
with_narration_context(ctx.with_job_id(job_id), async {
    n!("step", "Step 1");  // Auto-injected!
}).await
```

**For Rich Narration:**
```rust
n!("deploy",
    human: "Deploying {}",
    cute: "🚀 Launching {}!",
    story: "'Fly,' said the system to {}",
    name
);
```

### Configuration

```bash
# Set narration mode via environment
RBEE_NARRATION_MODE=cute cargo run

# Or in code:
set_narration_mode(NarrationMode::Cute);
```

### Known Limitations

1. Regex parsing assumes narration format (but handles non-narration gracefully)
2. SSH requires stdout/stderr to be line-buffered
3. Very high-frequency narration (>1000/sec) may have performance impact

### Future Improvements

1. Per-request narration mode (HTTP header)
2. Configuration file support
3. Dynamic regex patterns
4. Narration filtering/sampling

---

## Conclusion

🎉 **Narration V2 is COMPLETE!**

**What We Achieved:**
- ✅ 80% less boilerplate
- ✅ 3 narration modes working
- ✅ Resilient (SSE optional)
- ✅ Automatic (context injection)
- ✅ Complete (process boundaries handled)

**Result:** World-class narration system! 🚀
