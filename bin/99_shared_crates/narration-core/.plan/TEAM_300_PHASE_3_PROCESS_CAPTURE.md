# TEAM-300: Phase 3 - Process Stdout Capture

**Status:** BLOCKED (Requires TEAM-299 completion)  
**Estimated Duration:** 1 week  
**Dependencies:** TEAM-299 (Phase 2)  
**Risk Level:** High (new functionality, process management)

---

## Mission

Capture child process stdout and convert narration events to SSE. This enables worker startup narration to flow through the job-server SSE channel back to the client.

---

## ⚠️ CRITICAL: DO YOUR RESEARCH FIRST!

### Required Research (Complete ALL before coding)

1. **Read TEAM-299 Handoff** - Understand Phase 2 changes
2. **Study Worker Spawning** - `bin/20_rbee_hive/src/job_router.rs` (worker spawn)
3. **Study Process Management** - How hive spawns workers today
4. **Read Worker Startup** - `bin/30_llm_worker_rbee/src/main.rs` (startup flow)
5. **Understand Narration Format** - `[actor     ] action         : message`
6. **Create Research Summary** - Document in `.plan/TEAM_300_RESEARCH_SUMMARY.md`

**DO NOT CODE UNTIL RESEARCH IS COMPLETE!**

---

## Problem: Worker Startup Narration Lost

```rust
// Worker startup (NO job context)
fn main() {
    NARRATE.action("startup").emit();  // → stdout only
    load_model();  // → stdout only
    start_http_server();  // → stdout only
}

// Hive spawns worker
let child = Command::new("llm-worker-rbee").spawn()?;
// ↑ Worker's stdout goes nowhere! Hive can't see it!
```

## Solution: Capture and Convert

```rust
// Hive spawns with capture
let capture = ProcessNarrationCapture::new(Some(job_id));
let child = capture.spawn(Command::new("llm-worker-rbee")).await?;

// Worker's stdout:
// "[worker    ] startup         : Starting worker"
// ↓ Captured by hive
// ↓ Parsed as narration event
// ↓ Re-emitted with job_id
// ↓ Flows through SSE to client!
```

---

## Implementation Tasks

### Task 1: Create ProcessNarrationCapture

**New File:** `bin/99_shared_crates/narration-core/src/process_capture.rs`

```rust
use tokio::process::{Command, ChildStdout};
use tokio::io::{AsyncBufReadExt, BufReader};
use regex::Regex;

pub struct ProcessNarrationCapture {
    job_id: Option<String>,
}

impl ProcessNarrationCapture {
    pub fn new(job_id: Option<String>) -> Self {
        Self { job_id }
    }
    
    pub async fn spawn(&self, mut command: Command) -> Result<tokio::process::Child> {
        command.stdout(std::process::Stdio::piped());
        let mut child = command.spawn()?;
        
        if let Some(stdout) = child.stdout.take() {
            let job_id = self.job_id.clone();
            tokio::spawn(async move {
                Self::stream_and_parse(stdout, job_id).await;
            });
        }
        
        Ok(child)
    }
    
    async fn stream_and_parse(output: ChildStdout, job_id: Option<String>) {
        let reader = BufReader::new(output);
        let mut lines = reader.lines();
        
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(event) = Self::parse_narration(&line) {
                if let Some(ref jid) = job_id {
                    NARRATE.action(event.action.as_str())
                        .job_id(jid)
                        .human(&event.message)
                        .emit();
                }
            }
        }
    }
    
    fn parse_narration(line: &str) -> Option<ParsedEvent> {
        // Match: "[actor     ] action         : message"
        let re = Regex::new(r"\[(.{10})\] (.{15}): (.+)").ok()?;
        let caps = re.captures(line)?;
        
        Some(ParsedEvent {
            actor: caps[1].trim().to_string(),
            action: caps[2].trim().to_string(),
            message: caps[3].to_string(),
        })
    }
}
```

### Task 2: Update Hive Worker Spawning

**File:** `bin/20_rbee_hive/src/job_router.rs`

```rust
Operation::WorkerSpawn(request) => {
    use observability_narration_core::process_capture::ProcessNarrationCapture;
    
    NARRATE.action("worker_spawn_start").emit();
    
    // Create capture helper
    let capture = ProcessNarrationCapture::new(Some(job_id.clone()));
    
    // Build command
    let mut command = Command::new("llm-worker-rbee");
    command.arg("--model").arg(&request.model);
    command.arg("--device").arg(&request.device.to_string());
    
    // Spawn with capture
    let child = capture.spawn(command).await?;
    
    // Worker's stdout now flows through SSE!
    
    NARRATE.action("worker_spawn_complete").emit();
}
```

### Task 3: Test Parsing

```rust
#[test]
fn test_parse_narration_line() {
    let line = "[worker    ] startup         : Starting worker";
    let event = ProcessNarrationCapture::parse_narration(line).unwrap();
    
    assert_eq!(event.actor, "worker");
    assert_eq!(event.action, "startup");
    assert_eq!(event.message, "Starting worker");
}
```

---

## Verification Checklist

- [ ] `ProcessNarrationCapture` created
- [ ] Regex parsing works correctly
- [ ] Hive worker spawning uses capture
- [ ] Worker stdout flows through SSE
- [ ] Non-narration lines handled gracefully
- [ ] Tests pass

---

## Handoff to TEAM-301

Document in `.plan/TEAM_300_HANDOFF.md`:
1. How capture works
2. Regex pattern used
3. Integration points
4. Recommendations for Phase 4
