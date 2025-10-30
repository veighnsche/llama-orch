# PID Replacement Strategy for Remote Daemons

**Date:** Oct 27, 2025  
**Problem:** PID files don't work well for remote daemons  
**Question:** How do we track remote daemon processes without PID files?

---

## Current PID File Approach (Local Only)

### How It Works Now
```rust
// Local daemon start
let pid = spawn_daemon().await?;
write_pid_file("rbee-hive", pid)?;  // Write to ~/.local/var/run/rbee-hive.pid

// Local daemon stop
let pid = read_pid_file("rbee-hive")?;  // Read from ~/.local/var/run/rbee-hive.pid
kill(pid, SIGTERM)?;
```

### Why PID Files Don't Work for Remote

1. **PID file is on remote machine** - Can't read it without SSH
2. **Stale PID files** - If SSH connection drops, file stays
3. **Extra SSH calls** - Need SSH to read/write/delete PID file
4. **Race conditions** - Multiple clients might write different PIDs

---

## Solution 1: Process Discovery via pgrep (RECOMMENDED)

**Concept:** Query for running process by name instead of storing PID

### Implementation

```rust
// remote-daemon-lifecycle/src/utils/process.rs

/// Find PID of running daemon on remote machine
pub async fn find_remote_pid(
    ssh_config: &SshConfig,
    daemon_name: &str,
) -> Result<Option<u32>> {
    let script = format!("pgrep -f '{}'", daemon_name);
    
    let output = ssh_exec(ssh_config, &script).await?;
    
    if output.trim().is_empty() {
        return Ok(None);  // Not running
    }
    
    // Parse PID (pgrep returns one PID per line)
    let pids: Vec<u32> = output
        .lines()
        .filter_map(|line| line.trim().parse().ok())
        .collect();
    
    match pids.len() {
        0 => Ok(None),
        1 => Ok(Some(pids[0])),
        _ => {
            // Multiple processes found - return first one
            // Or return error if strict matching needed
            Ok(Some(pids[0]))
        }
    }
}
```

### Usage in start_daemon_remote

```rust
pub async fn start_daemon_remote(
    ssh_config: SshConfig,
    daemon_config: HttpDaemonConfig,
) -> Result<u32> {
    // Step 1: Find binary
    let binary_path = find_remote_binary(&ssh_config, &daemon_config.daemon_name).await?;
    
    // Step 2: Start daemon
    let script = format!(
        "nohup {} {} > /dev/null 2>&1 & echo $!",
        binary_path,
        daemon_config.args.join(" ")
    );
    
    let output = ssh_exec(&ssh_config, &script).await?;
    let pid = output.trim().parse::<u32>()?;
    
    // Step 3: Poll health (HTTP, no SSH)
    poll_daemon_health(&daemon_config.health_url).await?;
    
    // NO PID FILE - just return PID
    Ok(pid)
}
```

### Usage in stop_daemon_remote

```rust
pub async fn stop_daemon_remote(
    ssh_config: SshConfig,
    daemon_name: &str,
    health_url: &str,
) -> Result<()> {
    // Try graceful shutdown via HTTP first
    if try_http_shutdown(health_url).await.is_ok() {
        return Ok(());
    }
    
    // Fallback: Find process and kill it
    if let Some(pid) = find_remote_pid(&ssh_config, daemon_name).await? {
        let script = format!("kill -TERM {}", pid);
        ssh_exec(&ssh_config, &script).await?;
        
        // Wait for process to stop
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Check if still running
        if find_remote_pid(&ssh_config, daemon_name).await?.is_some() {
            // Force kill
            let script = format!("kill -KILL {}", pid);
            ssh_exec(&ssh_config, &script).await?;
        }
    }
    
    Ok(())
}
```

### Benefits

✅ **No PID files** - No file management needed  
✅ **Always accurate** - Queries actual running processes  
✅ **No stale state** - Can't have stale PID file  
✅ **Simple** - Just `pgrep -f daemon_name`  

### Downsides

⚠️ **Extra SSH call** - Need SSH to find PID  
⚠️ **Multiple matches** - If multiple instances running  
⚠️ **Pattern matching** - `pgrep -f` matches command line  

---

## Solution 2: HTTP-Based Status (NO PID NEEDED)

**Concept:** Use HTTP health endpoint to determine if daemon is running

### Implementation

```rust
pub async fn stop_daemon_remote(
    ssh_config: SshConfig,
    daemon_name: &str,
    health_url: &str,
) -> Result<()> {
    // Step 1: Check if running via HTTP
    if !check_daemon_status_remote(health_url).await? {
        return Ok(());  // Already stopped
    }
    
    // Step 2: Try graceful shutdown via HTTP
    if try_http_shutdown(health_url).await.is_ok() {
        return Ok(());
    }
    
    // Step 3: Force kill via pkill (no PID needed)
    let script = format!("pkill -f '{}'", daemon_name);
    ssh_exec(&ssh_config, &script).await?;
    
    Ok(())
}
```

### Benefits

✅ **No PID needed** - Use `pkill -f` to kill by name  
✅ **HTTP first** - Try graceful shutdown before SSH  
✅ **Simple** - No PID tracking at all  

### Downsides

⚠️ **Less precise** - `pkill -f` might match wrong process  
⚠️ **No PID return** - Can't return PID from start_daemon  

---

## Solution 3: Hybrid Approach (RECOMMENDED)

**Concept:** Combine both approaches for best results

### Implementation

```rust
// Start: Return PID but don't store it
pub async fn start_daemon_remote(
    ssh_config: SshConfig,
    daemon_config: HttpDaemonConfig,
) -> Result<u32> {
    // ... start daemon ...
    let pid = output.trim().parse::<u32>()?;
    
    // NO PID FILE - just return it
    Ok(pid)
}

// Stop: Try HTTP first, then pkill by name
pub async fn stop_daemon_remote(
    ssh_config: SshConfig,
    daemon_name: &str,
    health_url: &str,
) -> Result<()> {
    // Step 1: Check if running (HTTP, no SSH)
    if !check_daemon_status_remote(health_url).await? {
        return Ok(());
    }
    
    // Step 2: Try graceful shutdown (HTTP, no SSH)
    if try_http_shutdown(health_url).await.is_ok() {
        // Wait and verify it stopped
        tokio::time::sleep(Duration::from_secs(2)).await;
        if !check_daemon_status_remote(health_url).await? {
            return Ok(());
        }
    }
    
    // Step 3: Force kill by name (ONE SSH call)
    let script = format!("pkill -TERM -f '{}'", daemon_name);
    ssh_exec(&ssh_config, &script).await?;
    
    // Step 4: Wait and check
    tokio::time::sleep(Duration::from_secs(2)).await;
    if !check_daemon_status_remote(health_url).await? {
        return Ok(());
    }
    
    // Step 5: SIGKILL if still running (ONE SSH call)
    let script = format!("pkill -KILL -f '{}'", daemon_name);
    ssh_exec(&ssh_config, &script).await?;
    
    Ok(())
}

// Status: HTTP only (no SSH, no PID)
pub async fn check_daemon_status_remote(health_url: &str) -> Result<bool> {
    let client = reqwest::Client::new();
    match client.get(health_url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
    {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),  // Not running or unreachable
    }
}
```

### Benefits

✅ **No PID files** - No file management  
✅ **HTTP first** - Minimize SSH calls  
✅ **Reliable** - Uses process name, not stored PID  
✅ **Graceful degradation** - HTTP → SIGTERM → SIGKILL  

---

## Solution 4: Systemd Integration (FUTURE)

**Concept:** Use systemd to manage daemon lifecycle

### Implementation

```bash
# Install daemon as systemd service
ssh user@remote "cat > /etc/systemd/system/rbee-hive.service << EOF
[Unit]
Description=rbee-hive daemon

[Service]
ExecStart=/home/user/.local/bin/rbee-hive --port 7835
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF"

# Start daemon
ssh user@remote "systemctl start rbee-hive"

# Stop daemon
ssh user@remote "systemctl stop rbee-hive"

# Check status
ssh user@remote "systemctl is-active rbee-hive"
```

### Benefits

✅ **Systemd manages PID** - No manual tracking  
✅ **Auto-restart** - Built-in crash recovery  
✅ **Standard tool** - Well-known interface  

### Downsides

⚠️ **Requires root** - Need sudo for systemd  
⚠️ **Linux only** - Not portable  
⚠️ **Complex setup** - Service file management  

---

## Comparison Matrix

| Approach | SSH Calls | PID Files | Reliability | Complexity |
|----------|-----------|-----------|-------------|------------|
| **pgrep** | 1-2 | None | High | Low |
| **HTTP only** | 0-1 | None | Medium | Very Low |
| **Hybrid** | 0-2 | None | Very High | Low |
| **systemd** | 1 | None | Very High | High |

---

## Recommended Implementation: Hybrid Approach

### Why Hybrid?

1. **HTTP first** - Fast, no SSH for status checks
2. **pkill fallback** - Reliable process termination
3. **No PID files** - No state management
4. **Graceful degradation** - Try gentle methods first

### Implementation Plan

#### Phase 1: HTTP-Based Status (Week 1)
```rust
// Implement check_daemon_status_remote()
// - HTTP GET to health endpoint
// - Return true/false
// - No SSH needed
```

#### Phase 2: HTTP-Based Shutdown (Week 1)
```rust
// Implement try_http_shutdown()
// - POST to /v1/shutdown
// - Return Ok if succeeds
// - No SSH needed
```

#### Phase 3: SSH Fallback (Week 2)
```rust
// Implement pkill-based stop
// - pkill -TERM -f daemon_name
// - Wait 2 seconds
// - pkill -KILL -f daemon_name if needed
```

#### Phase 4: Start Without PID File (Week 2)
```rust
// Modify start_daemon_remote()
// - Return PID but don't store it
// - Caller can discard PID if not needed
```

---

## Handler Integration

### Current (Local with PID files)
```rust
HiveAction::Start { port } => {
    let config = HttpDaemonConfig::new("rbee-hive", url);
    let pid = daemon_lifecycle::start_daemon(config).await?;
    // PID written to ~/.local/var/run/rbee-hive.pid
}

HiveAction::Stop { port } => {
    let config = HttpDaemonConfig::new("rbee-hive", url);
    daemon_lifecycle::stop_daemon(config).await?;
    // Reads PID from ~/.local/var/run/rbee-hive.pid
}
```

### Future (Remote without PID files)
```rust
HiveAction::Start { port } => {
    if alias == "localhost" {
        let config = HttpDaemonConfig::new("rbee-hive", url);
        daemon_lifecycle::start_daemon(config).await?;
        // Uses PID files locally
    } else {
        let ssh = load_ssh_config(&alias)?;
        let config = HttpDaemonConfig::new("rbee-hive", url);
        let _pid = remote_daemon_lifecycle::start_daemon_remote(ssh, config).await?;
        // NO PID file - just returns PID (can be discarded)
    }
}

HiveAction::Stop { port } => {
    if alias == "localhost" {
        let config = HttpDaemonConfig::new("rbee-hive", url);
        daemon_lifecycle::stop_daemon(config).await?;
        // Uses PID files locally
    } else {
        let ssh = load_ssh_config(&alias)?;
        remote_daemon_lifecycle::stop_daemon_remote(ssh, "rbee-hive", url).await?;
        // NO PID file - uses HTTP + pkill
    }
}
```

---

## Key Insights

### Local vs Remote Differences

| Aspect | Local (daemon-lifecycle) | Remote (remote-daemon-lifecycle) |
|--------|-------------------------|----------------------------------|
| PID tracking | PID files (~/.local/var/run/) | No PID files |
| Status check | PID file + kill -0 | HTTP health endpoint |
| Stop method | Read PID → kill PID | HTTP shutdown → pkill by name |
| State | Stateful (PID file) | Stateless (query on demand) |

### Why This Works

1. **HTTP is universal** - Works for local and remote
2. **Process name is stable** - Daemon name doesn't change
3. **pkill is reliable** - Standard Unix tool
4. **No state to manage** - No PID files to sync

---

## Migration Checklist

- [ ] Implement `check_daemon_status_remote()` (HTTP only)
- [ ] Implement `try_http_shutdown()` helper
- [ ] Implement `ssh_exec()` helper
- [ ] Implement `stop_daemon_remote()` with pkill fallback
- [ ] Implement `start_daemon_remote()` without PID file
- [ ] Update handlers to use remote functions when `alias != "localhost"`
- [ ] Test with real remote machine
- [ ] Document PID-less approach in README

---

## Conclusion

**RECOMMENDATION: Hybrid Approach (Solution 3)**

- ✅ No PID files for remote daemons
- ✅ HTTP first for status and shutdown
- ✅ pkill fallback for force stop
- ✅ Simple, reliable, minimal SSH calls

**Implementation:**
1. Status: HTTP only (0 SSH calls)
2. Stop: HTTP → pkill -TERM → pkill -KILL (0-2 SSH calls)
3. Start: Return PID but don't store it (2 SSH calls)

**Total SSH calls per operation:**
- Start: 2 (find binary, start daemon)
- Stop: 0-2 (try HTTP first, fallback to pkill)
- Status: 0 (HTTP only)

This is simpler, more reliable, and requires less state management than PID files.
