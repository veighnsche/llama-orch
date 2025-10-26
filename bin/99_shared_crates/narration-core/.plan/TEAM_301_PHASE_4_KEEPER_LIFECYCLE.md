# TEAM-301: Phase 4 - Keeper Lifecycle Management

**Status:** BLOCKED (Requires TEAM-300 completion)  
**Estimated Duration:** 1 week  
**Dependencies:** TEAM-300 (Phase 3)  
**Risk Level:** Medium (keeper changes, SSH handling)

---

## Mission

Enable rbee-keeper to spawn queen/hive processes and display their stdout to the user. This completes the narration flow for all lifecycle operations.

---

## ⚠️ CRITICAL: DO YOUR RESEARCH FIRST!

### Required Research (Complete ALL before coding)

1. **Read TEAM-300 Handoff** - Understand process capture
2. **Study Keeper Structure** - `bin/00_rbee_keeper/src/handlers/`
3. **Study Queen Lifecycle** - How keeper starts/stops queen today
4. **Study Hive Lifecycle** - How keeper starts/stops hives (SSH)
5. **Understand SSH Output** - How to capture SSH stdout
6. **Create Research Summary** - Document in `.plan/TEAM_301_RESEARCH_SUMMARY.md`

**DO NOT CODE UNTIL RESEARCH IS COMPLETE!**

---

## Problem: Keeper Can't See Daemon Startup

```rust
// Keeper starts queen
let child = Command::new("queen-rbee").spawn()?;
// ↑ Queen's stdout goes nowhere! User can't see startup!

// Keeper starts hive via SSH
let child = Command::new("ssh")
    .arg("user@host")
    .arg("rbee-hive")
    .spawn()?;
// ↑ Hive's stdout lost in SSH! User can't see startup!
```

## Solution: Capture and Display

```rust
// Keeper starts queen with capture
let mut command = Command::new("queen-rbee");
command.stdout(Stdio::piped());
let mut child = command.spawn()?;

// Stream to terminal
tokio::spawn(async move {
    let reader = BufReader::new(child.stdout.take().unwrap());
    let mut lines = reader.lines();
    while let Ok(Some(line)) = lines.next_line().await {
        println!("{}", line);  // Show to user!
    }
});
```

---

## Implementation Tasks

### Task 1: Update Keeper Queen Lifecycle

**File:** `bin/00_rbee_keeper/src/handlers/queen.rs` (or similar)

```rust
pub async fn start_queen() -> Result<()> {
    NARRATE.action("queen_start")
        .human("Starting queen-rbee...")
        .emit();
    
    let mut command = Command::new("queen-rbee");
    command.arg("--port").arg("7833");
    command.stdout(Stdio::piped());
    
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
    
    // Wait for health check
    wait_for_health("http://localhost:7833").await?;
    
    NARRATE.action("queen_ready")
        .human("✅ Queen ready")
        .emit();
    
    Ok(())
}
```

### Task 2: Update Keeper Hive Lifecycle (SSH)

**File:** `bin/00_rbee_keeper/src/handlers/hive.rs` (or similar)

```rust
pub async fn start_hive_ssh(config: &HiveConfig) -> Result<()> {
    NARRATE.action("hive_start_ssh")
        .context(&config.host)
        .human("Starting hive on {}...")
        .emit();
    
    let mut command = Command::new("ssh");
    command.arg(format!("{}@{}", config.user, config.host));
    command.arg("rbee-hive");
    command.arg("--port").arg(config.port.to_string());
    command.stdout(Stdio::piped());
    
    let mut child = command.spawn()?;
    
    // Stream SSH output to terminal
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                println!("{}", line);
            }
        });
    }
    
    Ok(())
}
```

### Task 3: Test Stdout Display

```rust
#[tokio::test]
async fn test_keeper_displays_queen_stdout() {
    // Mock queen process that outputs narration
    let mut command = Command::new("echo");
    command.arg("[queen     ] startup         : Starting");
    command.stdout(Stdio::piped());
    
    let mut child = command.spawn().unwrap();
    
    // Capture output
    let mut output = String::new();
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            output.push_str(&line);
        }
    }
    
    assert!(output.contains("Starting"));
}
```

---

## Verification Checklist

- [ ] Keeper displays queen startup to terminal
- [ ] Keeper displays hive startup to terminal (SSH)
- [ ] Stdout streaming works in real-time
- [ ] No buffering issues
- [ ] Tests pass

---

## Final Handoff

Document in `.plan/TEAM_301_HANDOFF.md`:
1. Complete summary of all 4 phases
2. What changed across the codebase
3. Migration guide for users
4. Known limitations
5. Future improvements

---

## Success Criteria (All Phases)

✅ **Phase 1:** SSE optional, stdout always works  
✅ **Phase 2:** Thread-local context, no manual job_id  
✅ **Phase 3:** Worker startup flows through SSE  
✅ **Phase 4:** Keeper displays daemon startup  

**Result:** Narration works everywhere, all the time!
