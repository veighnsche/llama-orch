# TEAM-300: Phase 3 - Process Stdout Capture

**Status:** BLOCKED (Requires TEAM-299 completion)  
**Estimated Duration:** 1 week  
**Dependencies:** TEAM-299 (Phase 2 Context)  
**Risk Level:** High (new functionality, process management)

---

## Mission

Capture child process stdout and convert narration events to SSE. This enables worker startup narration to flow through the job-server SSE channel back to the client.

---

## ⚠️ CRITICAL: DO YOUR RESEARCH FIRST!

### Required Research

1. **Read TEAM-299 Handoff** - Understand context usage
2. **Study Worker Spawning** - `rbee-hive/src/job_router.rs` (WorkerSpawn operation)
3. **Study Worker Startup** - `llm-worker-rbee/src/main.rs` (what it emits)
4. **Understand Narration Format** - `[actor     ] action         : message`
5. **Create Research Summary** - `.plan/TEAM_300_RESEARCH_SUMMARY.md`

**DO NOT CODE UNTIL RESEARCH IS COMPLETE!**

---

## Problem: Worker Startup Narration Lost

```rust
// Worker main() - NO job context!
fn main() {
    n!("startup", "Worker starting");  // → stdout only, hive can't see!
    load_model();
    n!("ready", "Worker ready");  // → Lost!
}

// Hive spawns worker
let child = Command::new("llm-worker-rbee").spawn()?;
// ↑ Worker's stdout goes nowhere! User never sees startup!
```

## Solution: Capture and Convert

```rust
// Hive spawns with capture
let capture = ProcessNarrationCapture::new(Some(job_id));
let child = capture.spawn(Command::new("llm-worker-rbee")).await?;

// Worker stdout:
// "[worker    ] startup         : Worker starting"
// ↓ Captured by hive
// ↓ Parsed as narration
// ↓ Re-emitted with job_id
// ↓ Flows through SSE to client! 🎉
```

---

## Implementation Tasks

### Task 1: Create ProcessNarrationCapture

**New File:** `bin/99_shared_crates/narration-core/src/process_capture.rs`

```rust
use tokio::process::{Command, Child, ChildStdout};
use tokio::io::{AsyncBufReadExt, BufReader};
use regex::Regex;
use once_cell::sync::Lazy;

/// Regex to parse narration output
static NARRATION_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Match: "[actor     ] action         : message"
    Regex::new(r"^\[(.{1,10})\s*\]\s+(.{1,15})\s*:\s+(.+)$")
        .expect("Failed to compile narration regex")
});

/// Captures child process stdout and converts narration to SSE
pub struct ProcessNarrationCapture {
    job_id: Option<String>,
}

impl ProcessNarrationCapture {
    pub fn new(job_id: Option<String>) -> Self {
        Self { job_id }
    }
    
    /// Spawn command with stdout capture
    pub async fn spawn(&self, mut command: Command) -> Result<Child, std::io::Error> {
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        
        let mut child = command.spawn()?;
        
        // Capture stdout
        if let Some(stdout) = child.stdout.take() {
            let job_id = self.job_id.clone();
            tokio::spawn(async move {
                Self::stream_and_parse(stdout, job_id).await;
            });
        }
        
        // Capture stderr (same logic)
        if let Some(stderr) = child.stderr.take() {
            let job_id = self.job_id.clone();
            tokio::spawn(async move {
                Self::stream_and_parse(stderr, job_id).await;
            });
        }
        
        Ok(child)
    }
    
    /// Stream and parse child output
    async fn stream_and_parse(output: ChildStdout, job_id: Option<String>) {
        let reader = BufReader::new(output);
        let mut lines = reader.lines();
        
        while let Ok(Some(line)) = lines.next_line().await {
            // Try to parse as narration event
            if let Some(event) = Self::parse_narration(&line) {
                if let Some(ref jid) = job_id {
                    // Re-emit with job_id for SSE routing
                    // Use macro to ensure proper formatting
                    let ctx = NarrationContext::new().with_job_id(jid);
                    with_narration_context(ctx, async {
                        n!(event.action.as_str(), "{}", event.message);
                    }).await;
                } else {
                    // No job context, just print
                    eprintln!("{}", line);
                }
            } else {
                // Not narration format, just print
                eprintln!("{}", line);
            }
        }
    }
    
    /// Parse narration event from stdout line
    fn parse_narration(line: &str) -> Option<ParsedNarrationEvent> {
        let caps = NARRATION_REGEX.captures(line)?;
        
        Some(ParsedNarrationEvent {
            actor: caps[1].trim().to_string(),
            action: caps[2].trim().to_string(),
            message: caps[3].to_string(),
        })
    }
}

struct ParsedNarrationEvent {
    actor: String,
    action: String,
    message: String,
}
```

### Task 2: Update Hive Worker Spawning

**File:** `bin/20_rbee_hive/src/job_router.rs`

```rust
Operation::WorkerSpawn(request) => {
    use observability_narration_core::process_capture::ProcessNarrationCapture;
    
    n!("worker_spawn_start", "🚀 Spawning worker {} on device {}", 
        request.worker, request.device);
    
    // Get job_id from context
    let job_id = context::get_context()
        .and_then(|ctx| ctx.job_id.clone())
        .expect("job_id must be set in context");
    
    // Create capture helper
    let capture = ProcessNarrationCapture::new(Some(job_id));
    
    // Build command
    let mut command = Command::new("llm-worker-rbee");
    command.arg("--model").arg(&request.model);
    command.arg("--device").arg(&request.device.to_string());
    command.arg("--port").arg(port.to_string());
    
    // Spawn with capture
    let child = capture.spawn(command).await?;
    
    // Worker's stdout is now:
    // 1. Captured by hive
    // 2. Parsed for narration events
    // 3. Re-emitted with job_id
    // 4. Flows through SSE to client! 🎉
    
    n!("worker_spawn_complete", "✅ Worker {} spawned (PID: {})", 
        request.worker, child.id().unwrap_or(0));
}
```

### Task 3: Test Parsing

**New File:** `bin/99_shared_crates/narration-core/tests/process_capture_tests.rs`

```rust
use observability_narration_core::process_capture::*;

#[test]
fn test_parse_narration_line() {
    let line = "[worker    ] startup         : Starting worker";
    let event = ProcessNarrationCapture::parse_narration(line).unwrap();
    
    assert_eq!(event.actor, "worker");
    assert_eq!(event.action, "startup");
    assert_eq!(event.message, "Starting worker");
}

#[test]
fn test_parse_with_extra_spaces() {
    let line = "[worker] startup: Starting worker";
    let event = ProcessNarrationCapture::parse_narration(line).unwrap();
    
    assert_eq!(event.actor, "worker");
    assert_eq!(event.action, "startup");
    assert_eq!(event.message, "Starting worker");
}

#[test]
fn test_parse_non_narration() {
    let line = "Random log message";
    let event = ProcessNarrationCapture::parse_narration(line);
    
    assert!(event.is_none());
}

#[tokio::test]
async fn test_worker_stdout_capture() {
    // Mock worker process that outputs narration
    let capture = ProcessNarrationCapture::new(Some("job-123".to_string()));
    
    let mut command = Command::new("echo");
    command.arg("[worker    ] startup         : Starting");
    
    let child = capture.spawn(command).await.unwrap();
    
    // Wait for process
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Output should be captured and re-emitted
    // (tested via integration test)
}
```

---

## Verification Checklist

- [ ] `ProcessNarrationCapture` created
- [ ] Regex parsing works correctly
- [ ] Hive worker spawning uses capture
- [ ] Worker stdout flows through SSE
- [ ] Non-narration lines printed to stderr
- [ ] Tests pass

---

## Success Criteria

1. **Worker startup visible** - User sees worker narration
2. **SSE delivery works** - Events flow through job channel
3. **Non-narration handled** - Regular logs still work
4. **No lost events** - All narration captured

---

## Handoff to TEAM-301

Document in `.plan/TEAM_300_HANDOFF.md`:
1. How capture works
2. Regex pattern details
3. Integration points
4. Test results
5. Recommendations for Phase 4 (keeper lifecycle)
