# TEAM-296: Start Button Flow Documentation

**Status:** ✅ DOCUMENTED  
**Date:** Oct 26, 2025

## Complete Flow: Start Button → Queen Running

### 1. User Clicks Start Button (Tauri UI)

**File:** `bin/00_rbee_keeper/ui/src/pages/ServicesPage.tsx`

```typescript
case "queen-start":
  await invoke("queen_start");
  break;
```

### 2. Tauri Command Handler

**File:** `bin/00_rbee_keeper/src/tauri_commands.rs`

```rust
#[tauri::command]
pub async fn queen_start() -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();  // "http://localhost:7833"
    
    let result = handlers::handle_queen(QueenAction::Start, &queen_url).await;
    to_response_unit(result)
}
```

### 3. Queen Handler

**File:** `bin/00_rbee_keeper/src/handlers/queen.rs`

```rust
pub async fn handle_queen(action: QueenAction, queen_url: &str) -> Result<()> {
    match action {
        QueenAction::Start => start_queen(queen_url).await,
        // ...
    }
}
```

### 4. Start Queen (Queen Lifecycle)

**File:** `bin/05_rbee_keeper_crates/queen-lifecycle/src/start.rs`

```rust
pub async fn start_queen(queen_url: &str) -> Result<()> {
    let queen_handle = ensure_queen_running(queen_url).await?;
    
    NARRATE
        .action("queen_start")
        .context(queen_handle.base_url())
        .human("✅ Queen started on {}")
        .emit();
    
    // Keep queen alive - detach the handle
    drop(queen_handle);
    
    Ok(())
}
```

### 5. Ensure Queen Running (Queen Lifecycle)

**File:** `bin/05_rbee_keeper_crates/queen-lifecycle/src/ensure.rs`

```rust
pub async fn ensure_queen_running(base_url: &str) -> Result<QueenHandle> {
    // TEAM-197: Use TimeoutEnforcer with progress bar
    TimeoutEnforcer::new(Duration::from_secs(30))
        .with_label("Starting queen-rbee")
        .with_countdown()
        .enforce(ensure_queen_running_inner(base_url))
        .await
}

async fn ensure_queen_running_inner(base_url: &str) -> Result<QueenHandle> {
    let health_url = format!("{}/health", base_url);
    
    // TEAM-276: Use shared ensure pattern from daemon-lifecycle
    let handle = ensure_daemon_with_handle(
        "queen-rbee",
        &health_url,
        None,
        || async {
            spawn_queen_with_preflight(base_url).await
        },
        || QueenHandle::already_running(base_url.to_string()),
        || QueenHandle::started_by_us(base_url.to_string(), None),
    ).await?;
    
    // TEAM-292: Fetch queen's actual URL from /v1/info
    match fetch_queen_url(base_url).await {
        Ok(queen_url) => Ok(handle.with_discovered_url(queen_url)),
        Err(_) => Ok(handle),
    }
}
```

### 6. Ensure Daemon With Handle (Daemon Lifecycle)

**File:** `bin/99_shared_crates/daemon-lifecycle/src/ensure.rs`

```rust
pub async fn ensure_daemon_with_handle<F, Fut, H, AF, SF>(
    daemon_name: &str,
    health_url: &str,
    job_id: Option<&str>,
    spawn_fn: F,
    already_running_fn: AF,
    started_by_us_fn: SF,
) -> Result<H>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
    AF: FnOnce() -> H,
    SF: FnOnce() -> H,
{
    // Step 1: Check if daemon is already healthy
    if is_daemon_healthy(health_url, None, None).await {
        NARRATE
            .action("daemon_already_running")
            .context(daemon_name)
            .human("✅ {} is already running")
            .emit();
        return Ok(already_running_fn());
    }
    
    // Step 2: Daemon not running, spawn it
    NARRATE
        .action("daemon_not_running")
        .context(daemon_name)
        .human("⚠️  {} is not running, starting...")
        .emit();
    
    spawn_fn().await?;
    
    Ok(started_by_us_fn())
}
```

### 7. Spawn Queen With Preflight (Queen Lifecycle)

**File:** `bin/05_rbee_keeper_crates/queen-lifecycle/src/ensure.rs`

```rust
async fn spawn_queen_with_preflight(base_url: &str) -> Result<()> {
    // Step 1: Preflight check (no config needed)
    NARRATE
        .action("queen_preflight")
        .human("✅ Localhost-only mode (no config needed)")
        .emit();
    
    // Step 2: Find queen-rbee binary
    // TEAM-296: Prefer installed binary (~/.local/bin) over development binary (target/)
    let queen_binary = {
        let home = std::env::var("HOME")?;
        let installed_path = PathBuf::from(format!("{}/.local/bin/queen-rbee", home));
        
        if installed_path.exists() {
            NARRATE
                .action("queen_start")
                .context(installed_path.display().to_string())
                .human("Using installed queen-rbee binary at {}")
                .emit();
            installed_path
        } else {
            let dev_binary = DaemonManager::find_in_target("queen-rbee")?;
            NARRATE
                .action("queen_start")
                .context(dev_binary.display().to_string())
                .human("Using development queen-rbee binary at {}")
                .emit();
            dev_binary
        }
    };
    
    // Step 3: Extract port from base_url and spawn
    let port = base_url.split(':').last()?.to_string();
    let args = vec!["--port".to_string(), port];
    let manager = DaemonManager::new(queen_binary, args);
    
    let child = manager.spawn().await?;
    
    NARRATE
        .action("queen_spawned")
        .human("Queen-rbee process spawned, polling health...")
        .emit();
    
    // Step 4: Poll health until ready
    poll_until_healthy(
        HealthPollConfig::new(base_url)
            .with_daemon_name("queen-rbee")
            .with_max_attempts(30),
    ).await?;
    
    // Keep child alive - detach from parent
    drop(child);
    
    Ok(())
}
```

### 8. Daemon Manager Spawn (Daemon Lifecycle)

**File:** `bin/99_shared_crates/daemon-lifecycle/src/manager.rs`

```rust
pub async fn spawn(&self) -> Result<Child> {
    // TEAM-296: Auto-update is NOT enabled for queen
    // (auto_update field is None)
    
    // Verify binary exists
    if !self.binary_path.exists() {
        anyhow::bail!("Binary not found: {}", self.binary_path.display());
    }
    
    NARRATE
        .action("spawn")
        .context(self.binary_path.display().to_string())
        .human("Spawning daemon: {} with args: {:?}")
        .emit();
    
    // Spawn process with null stdio (no interference with parent)
    let child = Command::new(&self.binary_path)
        .args(&self.args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    
    NARRATE
        .action("spawned")
        .context(child.id().unwrap_or(0).to_string())
        .human("Daemon spawned with PID: {}")
        .emit();
    
    Ok(child)
}
```

### 9. Poll Until Healthy (Daemon Lifecycle)

**File:** `bin/99_shared_crates/daemon-lifecycle/src/health.rs`

```rust
pub async fn poll_until_healthy(config: HealthPollConfig) -> Result<()> {
    let mut attempt = 1;
    let max_attempts = config.max_attempts;
    
    loop {
        // Exponential backoff: 200ms * attempt
        let delay = Duration::from_millis(200 * attempt as u64);
        tokio::time::sleep(delay).await;
        
        NARRATE
            .action("daemon_health_poll")
            .context(&config.daemon_name)
            .context(&config.base_url)
            .human("⏳ Waiting for {} to become healthy at {}")
            .emit();
        
        if is_daemon_healthy(&config.base_url, None, None).await {
            NARRATE
                .action("daemon_healthy")
                .context(&config.daemon_name)
                .context(attempt.to_string())
                .human("✅ {} is healthy (attempt {})")
                .emit();
            return Ok(());
        }
        
        attempt += 1;
        if attempt > max_attempts {
            anyhow::bail!("{} failed to become healthy after {} attempts", 
                config.daemon_name, max_attempts);
        }
    }
}
```

## Auto-Update Status

**✅ Auto-update is NOT used for queen-rbee**

The `DaemonManager` has auto-update support via `.enable_auto_update()`, but:
- Queen-lifecycle does NOT call `.enable_auto_update()`
- Auto-update is completely disabled for queen
- No dependency on auto-update crate in the flow

**Code Evidence:**
```rust
// daemon-lifecycle/src/manager.rs
let manager = DaemonManager::new(queen_binary, args);
// ❌ NOT calling .enable_auto_update()

let child = manager.spawn().await?;
// ✅ Spawns directly without auto-update
```

## Heartbeat Stream Connection

**⚠️ IMPORTANT: Tauri app does NOT connect to heartbeat stream**

The current flow:
1. User clicks start
2. Tauri invokes `queen_start()`
3. Queen starts
4. Tauri command returns success/error
5. **No heartbeat connection established**

### Why This Matters

The Tauri app should connect to queen's heartbeat stream to:
1. **Real-time status updates** - Know when queen goes down
2. **Service discovery** - Get queen's actual URL
3. **Health monitoring** - Continuous health checks
4. **Event notifications** - Job completions, errors, etc.

### Recommended Flow

```typescript
// ServicesPage.tsx
case "queen-start":
  await invoke("queen_start");
  
  // TEAM-296: Connect to heartbeat stream immediately
  const queenUrl = await invoke("get_queen_url");
  connectToHeartbeat(queenUrl);
  break;

function connectToHeartbeat(queenUrl: string) {
  const eventSource = new EventSource(`${queenUrl}/v1/heartbeat`);
  
  eventSource.addEventListener("heartbeat", (event) => {
    const data = JSON.parse(event.data);
    // Update UI with queen status
    updateQueenStatus(data);
  });
  
  eventSource.addEventListener("error", () => {
    // Queen went down, update UI
    setQueenStatus("offline");
  });
}
```

## Summary

### Complete Flow (10 Steps)

1. **UI Button Click** → `ServicesPage.tsx`
2. **Tauri Command** → `tauri_commands.rs::queen_start()`
3. **Handler Routing** → `handlers/queen.rs::handle_queen()`
4. **Lifecycle Start** → `queen-lifecycle/src/start.rs::start_queen()`
5. **Ensure Running** → `queen-lifecycle/src/ensure.rs::ensure_queen_running()`
6. **Daemon Ensure** → `daemon-lifecycle/src/ensure.rs::ensure_daemon_with_handle()`
7. **Spawn Preflight** → `queen-lifecycle/src/ensure.rs::spawn_queen_with_preflight()`
8. **Binary Resolution** → Prefer `~/.local/bin/queen-rbee` over `target/`
9. **Process Spawn** → `daemon-lifecycle/src/manager.rs::spawn()` (NO auto-update)
10. **Health Poll** → `daemon-lifecycle/src/health.rs::poll_until_healthy()`

### Key Points

- ✅ **No auto-update** - Queen spawns directly without rebuild checks
- ✅ **Binary preference** - Installed binary (`~/.local/bin`) over development (`target/`)
- ✅ **Health polling** - Exponential backoff (200ms * attempt, max 30 attempts)
- ✅ **Timeout enforcement** - 30-second total timeout with countdown
- ⚠️ **No heartbeat connection** - Tauri app should connect to `/v1/heartbeat` after start

### Dependencies

```
ServicesPage.tsx
  ↓
tauri_commands.rs
  ↓
handlers/queen.rs
  ↓
queen-lifecycle (start.rs, ensure.rs)
  ↓
daemon-lifecycle (ensure.rs, manager.rs, health.rs)
  ↓
timeout-enforcer
  ↓
observability-narration-core
```

**No auto-update dependency in the flow!**

---

**TEAM-296: Documented complete start button flow. Auto-update is not used. Tauri app should connect to heartbeat stream after start.**
