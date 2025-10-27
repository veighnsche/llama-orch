# Remote Execution Architecture - Corrected (No Agent Recursion)

**Date:** Oct 27, 2025  
**Problem Identified:** Agent approach has chicken-and-egg problem (who manages the agent?)  
**Key Insight:** rbee-hive IS the remote daemon, we just need to START it remotely

---

## The Real Problem

```
❌ WRONG: Deploy lifecycle-agent to manage rbee-hive
   Question: How do we start/stop lifecycle-agent? Need lifecycle-agent-agent?
   
✅ RIGHT: Use SSH to START rbee-hive binary directly
   Then: Use HTTP to check health (no SSH needed)
```

---

## Correct Architecture: Split Local vs Remote Execution

### Key Principle

**Local Execution:**
- `daemon-lifecycle` spawns process using `tokio::Command`
- Polls health endpoint
- Manages PID files

**Remote Execution:**
- Use SSH to start the binary directly (ONE command)
- Poll health endpoint over HTTP (no SSH)
- No PID files needed (process runs on remote machine)

---

## Solution: Two-Path Strategy with Minimal SSH

### Architecture

```rust
// bin/99_shared_crates/daemon-lifecycle/src/lib.rs

pub mod local;   // Existing code (start.rs, stop.rs, etc.)
pub mod remote;  // NEW: SSH-based remote operations

// Top-level routing function
pub async fn start_daemon_auto(
    config: HttpDaemonConfig,
    location: DaemonLocation,
) -> Result<u32> {
    match location {
        DaemonLocation::Local => {
            // Use existing local implementation (NO CHANGES)
            local::start_daemon(config).await
        }
        DaemonLocation::Remote(ssh_config) => {
            // NEW: Remote implementation (different strategy)
            remote::start_daemon(config, ssh_config).await
        }
    }
}

pub enum DaemonLocation {
    Local,
    Remote(SshConfig),
}

pub struct SshConfig {
    pub hostname: String,
    pub user: String,
    pub port: u16,
}
```

---

## Remote Implementation (NOT Line-by-Line Wrapping)

### remote/start.rs - The Right Way

```rust
// bin/99_shared_crates/daemon-lifecycle/src/remote/start.rs
use anyhow::{Context, Result};
use crate::types::HttpDaemonConfig;
use crate::SshConfig;

/// Start daemon on remote machine via SSH
/// 
/// This does NOT wrap every Command::new() call.
/// Instead, it executes ONE ssh command to start the binary,
/// then polls health over HTTP (no SSH needed).
pub async fn start_daemon(
    config: HttpDaemonConfig,
    ssh: SshConfig,
) -> Result<u32> {
    // Step 1: Find binary path on REMOTE machine (ONE ssh call)
    let binary_path = find_remote_binary(&ssh, &config.daemon_name).await?;
    
    // Step 2: Build start command
    let start_cmd = build_start_command(&binary_path, &config);
    
    // Step 3: Execute start command via SSH (ONE ssh call)
    execute_remote_start(&ssh, &start_cmd).await?;
    
    // Step 4: Poll health endpoint over HTTP (NO SSH)
    let health_url = parse_remote_health_url(&config.health_url, &ssh.hostname)?;
    poll_remote_health(&health_url, &config.daemon_name).await?;
    
    // Step 5: Get PID via SSH (ONE ssh call)
    let pid = get_remote_pid(&ssh, &config.daemon_name).await?;
    
    Ok(pid)
}

/// Find binary on remote machine - ONE ssh call
async fn find_remote_binary(ssh: &SshConfig, daemon_name: &str) -> Result<String> {
    // Try: installed path, then target/debug, then target/release
    let find_script = format!(
        r#"
        if [ -f ~/.local/bin/{daemon} ]; then
            echo ~/.local/bin/{daemon}
        elif [ -f target/debug/{daemon} ]; then
            echo target/debug/{daemon}
        elif [ -f target/release/{daemon} ]; then
            echo target/release/{daemon}
        else
            exit 1
        fi
        "#,
        daemon = daemon_name
    );
    
    ssh_exec(ssh, &find_script).await
}

/// Execute start command - ONE ssh call
async fn execute_remote_start(ssh: &SshConfig, start_cmd: &str) -> Result<()> {
    // Start daemon in background with nohup
    let remote_cmd = format!(
        "nohup {} > /dev/null 2>&1 & echo $!",
        start_cmd
    );
    
    ssh_exec(ssh, &remote_cmd).await?;
    Ok(())
}

/// Poll health endpoint over HTTP (NO SSH)
async fn poll_remote_health(health_url: &str, daemon_name: &str) -> Result<()> {
    use crate::utils::poll::poll_daemon_health;
    use crate::types::HealthPollConfig;
    
    // Reuse existing health polling (works over network)
    let config = HealthPollConfig::new(health_url)
        .with_daemon_name(daemon_name);
    
    poll_daemon_health(config).await
}

/// Get PID of remote process - ONE ssh call
async fn get_remote_pid(ssh: &SshConfig, daemon_name: &str) -> Result<u32> {
    let cmd = format!("pgrep -f {}", daemon_name);
    let output = ssh_exec(ssh, &cmd).await?;
    
    output.trim().parse::<u32>()
        .context("Failed to parse PID from remote process")
}

/// Low-level SSH execution - ONLY place that calls ssh
async fn ssh_exec(ssh: &SshConfig, command: &str) -> Result<String> {
    let output = tokio::process::Command::new("ssh")
        .arg("-p").arg(ssh.port.to_string())
        .arg(format!("{}@{}", ssh.user, ssh.hostname))
        .arg(command)
        .output()
        .await?;
    
    if !output.status.success() {
        anyhow::bail!("SSH command failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn build_start_command(binary_path: &str, config: &HttpDaemonConfig) -> String {
    let args = config.args.join(" ");
    format!("{} {}", binary_path, args)
}

fn parse_remote_health_url(health_url: &str, hostname: &str) -> Result<String> {
    // Replace "localhost" with actual hostname
    Ok(health_url.replace("localhost", hostname))
}
```

---

## Key Differences from Anti-Pattern

### ❌ Anti-Pattern (What was deleted)
```rust
// Wraps EVERY internal operation
let binary_path = ssh_exec("which rbee-hive").await?;      // SSH call 1
ssh_exec(&format!("mkdir -p {}", dir)).await?;              // SSH call 2
ssh_exec(&format!("chmod +x {}", binary_path)).await?;      // SSH call 3
ssh_exec(&format!("{} --port {}", binary, port)).await?;    // SSH call 4
// ... 50 more SSH calls, duplicating all daemon-lifecycle logic
```

### ✅ Correct Pattern (This proposal)
```rust
// Minimal SSH calls, different strategy
let binary_path = find_remote_binary(&ssh, name).await?;    // SSH call 1 (multi-line script)
execute_remote_start(&ssh, &start_cmd).await?;              // SSH call 2 (start daemon)
poll_remote_health(&health_url, name).await?;               // HTTP calls (NO SSH)
let pid = get_remote_pid(&ssh, name).await?;                // SSH call 3 (get PID)

// Total: 3 SSH calls vs 50+ in anti-pattern
```

---

## Remote Stop Implementation

```rust
// bin/99_shared_crates/daemon-lifecycle/src/remote/stop.rs

pub async fn stop_daemon(
    config: HttpDaemonConfig,
    ssh: SshConfig,
) -> Result<()> {
    // Try graceful shutdown via HTTP first (NO SSH)
    let shutdown_url = format!("{}/v1/shutdown", config.health_url.replace("localhost", &ssh.hostname));
    
    match try_graceful_shutdown(&shutdown_url).await {
        Ok(_) => return Ok(()),
        Err(_) => {
            // Fallback: Force kill via SSH (ONE ssh call)
            let kill_cmd = format!("pkill -f {}", config.daemon_name);
            ssh_exec(&ssh, &kill_cmd).await?;
        }
    }
    
    Ok(())
}

async fn try_graceful_shutdown(shutdown_url: &str) -> Result<()> {
    let client = reqwest::Client::new();
    client.post(shutdown_url)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await?;
    Ok(())
}
```

---

## Remote Install Implementation

```rust
// bin/99_shared_crates/daemon-lifecycle/src/remote/install.rs

pub async fn install_daemon(
    daemon_name: &str,
    ssh: SshConfig,
    local_binary_path: Option<PathBuf>,
) -> Result<()> {
    // Step 1: Find or build binary locally
    let local_path = match local_binary_path {
        Some(path) => path,
        None => crate::build::build_daemon(daemon_name, None).await?,
    };
    
    // Step 2: Copy binary via SCP (ONE scp call)
    let remote_path = format!("/home/{}/.local/bin/{}", ssh.user, daemon_name);
    scp_copy(&local_path, &ssh, &remote_path).await?;
    
    // Step 3: Make executable via SSH (ONE ssh call)
    let chmod_cmd = format!("chmod +x {}", remote_path);
    ssh_exec(&ssh, &chmod_cmd).await?;
    
    Ok(())
}

async fn scp_copy(local_path: &Path, ssh: &SshConfig, remote_path: &str) -> Result<()> {
    let status = tokio::process::Command::new("scp")
        .arg("-P").arg(ssh.port.to_string())
        .arg(local_path)
        .arg(format!("{}@{}:{}", ssh.user, ssh.hostname, remote_path))
        .status()
        .await?;
    
    if !status.success() {
        anyhow::bail!("SCP failed");
    }
    
    Ok(())
}
```

---

## Handler Integration

```rust
// bin/00_rbee_keeper/src/handlers/hive.rs

use daemon_lifecycle::{start_daemon_auto, DaemonLocation, SshConfig, HttpDaemonConfig};

pub async fn handle_hive(action: HiveAction, queen_url: &str) -> Result<()> {
    match action {
        HiveAction::Start { port } => {
            let port = port.unwrap_or(7835);
            let config = HttpDaemonConfig::new("rbee-hive", &format!("http://localhost:{}", port))
                .with_args(vec![
                    "--port".to_string(),
                    port.to_string(),
                    "--queen-url".to_string(),
                    queen_url.to_string(),
                ]);
            
            // Detect location
            let location = if alias == "localhost" {
                DaemonLocation::Local
            } else {
                let hive = load_hive_config(&alias)?;
                DaemonLocation::Remote(SshConfig {
                    hostname: hive.hostname,
                    user: hive.ssh_user,
                    port: hive.ssh_port,
                })
            };
            
            // ONE function call - routes internally
            daemon_lifecycle::start_daemon_auto(config, location).await?;
            Ok(())
        }
        
        HiveAction::Stop { port } => {
            let port = port.unwrap_or(7835);
            let config = HttpDaemonConfig::new("rbee-hive", &format!("http://localhost:{}", port));
            
            let location = if alias == "localhost" {
                DaemonLocation::Local
            } else {
                let hive = load_hive_config(&alias)?;
                DaemonLocation::Remote(SshConfig {
                    hostname: hive.hostname,
                    user: hive.ssh_user,
                    port: hive.ssh_port,
                })
            };
            
            daemon_lifecycle::stop_daemon_auto(config, location).await
        }
        
        // ... other actions
    }
}
```

---

## File Structure

```
bin/99_shared_crates/daemon-lifecycle/
├── src/
│   ├── lib.rs              # Re-exports, routing functions
│   ├── local/              # Existing code (renamed from root)
│   │   ├── mod.rs
│   │   ├── start.rs        # Current start_daemon() moved here
│   │   ├── stop.rs         # Current stop_daemon() moved here
│   │   ├── install.rs      # Current install_daemon() moved here
│   │   └── ...
│   ├── remote/             # NEW: Remote execution
│   │   ├── mod.rs          # Re-exports + ssh_exec helper
│   │   ├── start.rs        # Remote start (3 SSH calls)
│   │   ├── stop.rs         # Remote stop (1 HTTP + 1 SSH)
│   │   └── install.rs      # Remote install (1 SCP + 1 SSH)
│   └── types/
│       └── location.rs     # DaemonLocation enum
```

---

## Why This Works

### No Duplication
- Local path: Uses existing `local::start_daemon()` (unchanged)
- Remote path: Different strategy (SSH + HTTP), not line-by-line wrapping

### No Agent Recursion
- Doesn't require deploying an agent
- Uses SSH to start rbee-hive directly
- rbee-hive manages itself once started

### Minimal SSH Calls
- Start: 3 SSH calls (find binary, start, get PID)
- Stop: 1 HTTP + 1 SSH (try graceful, fallback to kill)
- Install: 1 SCP + 1 SSH (copy, chmod)

### Health Checks Use HTTP
- No SSH needed after daemon starts
- Poll `http://remote-host:7835/health` directly
- Same health check code works for local and remote

---

## Comparison to Anti-Pattern

| Aspect | Anti-Pattern (Deleted) | This Proposal |
|--------|------------------------|---------------|
| SSH calls per operation | 50+ | 3-4 |
| Code duplication | 100% of daemon-lifecycle | 0% |
| Strategy | Wrap every Command | Different approach for remote |
| Health checks | Via SSH | Via HTTP |
| Agent needed | No | No |
| Complexity | High (line-by-line) | Low (function-level) |

---

## Implementation Timeline

### Week 1: Remote Module Structure
- Create `src/remote/` directory
- Add `DaemonLocation` enum
- Create routing functions (`start_daemon_auto`, etc.)
- Refactor existing code into `src/local/` (no logic changes)

### Week 2: Remote Start/Stop
- Implement `remote::start_daemon()` (3 SSH calls)
- Implement `remote::stop_daemon()` (HTTP + SSH fallback)
- Test with real remote machine

### Week 3: Remote Install/Uninstall
- Implement `remote::install_daemon()` (SCP + SSH)
- Implement `remote::uninstall_daemon()` (SSH rm)
- Integration with handlers

### Week 4: Polish & Testing
- Error handling
- Edge cases
- Documentation

**Total: 4 weeks for production-ready implementation**

---

## Key Design Decisions

### 1. Why Not Wrap Every Command?
**Answer:** Different strategies for local vs remote
- Local: Spawn process with tokio
- Remote: SSH to start binary, HTTP to monitor

### 2. Why Not Use Agent?
**Answer:** Chicken-and-egg problem
- Agent needs lifecycle management
- Who manages the agent? Another agent?
- rbee-hive IS the daemon, just start it directly

### 3. Why Minimal SSH Calls?
**Answer:** Performance + simplicity
- Each SSH call adds latency
- Bundle operations into scripts when possible
- Use HTTP for monitoring (no SSH needed)

### 4. Why Separate local/remote Paths?
**Answer:** Clean separation of concerns
- Local code unchanged (zero risk)
- Remote code uses different primitives (SSH/SCP)
- Clear which code runs where

---

## Next Steps

1. Review this corrected architecture
2. Confirm approach addresses concerns
3. Start implementation with routing structure
4. Implement remote::start_daemon() first (most complex)
5. Test with actual remote hive

**Estimated delivery: 4 weeks for complete, correct implementation**
