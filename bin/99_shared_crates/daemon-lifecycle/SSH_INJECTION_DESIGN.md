# SSH Injection Design - Modify daemon-lifecycle Directly

**Date:** Oct 27, 2025  
**Approach:** Add SSH support to existing daemon-lifecycle functions, not separate module  
**Key Insight:** SSH is just an execution context, not a different code path

---

## Core Concept

Instead of:
```rust
❌ local::start_daemon()   // One implementation
❌ remote::start_daemon()  // Duplicate implementation
```

Do this:
```rust
✅ start_daemon(config)    // ONE implementation
   ├─ Uses ProcessExecutor trait internally
   ├─ LocalExecutor: tokio::Command
   └─ SshExecutor: ssh commands
```

---

## Design: Execution Context Abstraction

### Step 1: Create Executor Trait

```rust
// bin/99_shared_crates/daemon-lifecycle/src/executor.rs

use async_trait::async_trait;
use std::path::PathBuf;
use anyhow::Result;

/// Abstraction for executing commands (local or remote)
#[async_trait]
pub trait ProcessExecutor: Send + Sync {
    /// Execute a command and return (stdout, stderr, exit_code)
    async fn exec(&self, command: &str, args: &[String]) -> Result<(String, String, i32)>;
    
    /// Spawn a daemon process in background, return PID
    async fn spawn_daemon(&self, command: &str, args: &[String]) -> Result<u32>;
    
    /// Check if a process is running by PID
    async fn is_running(&self, pid: u32) -> Result<bool>;
    
    /// Kill a process by PID
    async fn kill(&self, pid: u32, signal: Signal) -> Result<()>;
    
    /// Find a binary by name (which, target/debug, target/release)
    async fn find_binary(&self, name: &str) -> Result<PathBuf>;
    
    /// Copy a file (for install operations)
    async fn copy_file(&self, local: &PathBuf, remote: &str) -> Result<()>;
}

pub enum Signal {
    Term,  // SIGTERM
    Kill,  // SIGKILL
}
```

---

### Step 2: Local Executor (Existing Behavior)

```rust
// bin/99_shared_crates/daemon-lifecycle/src/executor/local.rs

use super::*;
use tokio::process::Command;
use std::process::Stdio;

pub struct LocalExecutor;

#[async_trait]
impl ProcessExecutor for LocalExecutor {
    async fn exec(&self, command: &str, args: &[String]) -> Result<(String, String, i32)> {
        let output = Command::new(command)
            .args(args)
            .output()
            .await?;
        
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let code = output.status.code().unwrap_or(-1);
        
        Ok((stdout, stderr, code))
    }
    
    async fn spawn_daemon(&self, command: &str, args: &[String]) -> Result<u32> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        
        // Propagate SSH agent
        if let Ok(ssh_auth_sock) = std::env::var("SSH_AUTH_SOCK") {
            cmd.env("SSH_AUTH_SOCK", ssh_auth_sock);
        }
        
        let child = cmd.spawn()?;
        let pid = child.id().ok_or_else(|| anyhow::anyhow!("Failed to get PID"))?;
        
        // Detach
        std::mem::ManuallyDrop::new(child);
        
        Ok(pid)
    }
    
    async fn is_running(&self, pid: u32) -> Result<bool> {
        // Use kill -0 to check if process exists
        let status = Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .await?;
        
        Ok(status.success())
    }
    
    async fn kill(&self, pid: u32, signal: Signal) -> Result<()> {
        let sig = match signal {
            Signal::Term => "-TERM",
            Signal::Kill => "-KILL",
        };
        
        Command::new("kill")
            .args([sig, &pid.to_string()])
            .status()
            .await?;
        
        Ok(())
    }
    
    async fn find_binary(&self, name: &str) -> Result<PathBuf> {
        // Existing find_binary logic from utils/find.rs
        crate::utils::find::find_binary(name)
    }
    
    async fn copy_file(&self, local: &PathBuf, remote: &str) -> Result<()> {
        // For local, this is just a file copy
        tokio::fs::copy(local, remote).await?;
        Ok(())
    }
}
```

---

### Step 3: SSH Executor (Remote Behavior)

```rust
// bin/99_shared_crates/daemon-lifecycle/src/executor/ssh.rs

use super::*;
use tokio::process::Command;

pub struct SshExecutor {
    hostname: String,
    user: String,
    port: u16,
}

impl SshExecutor {
    pub fn new(hostname: String, user: String, port: u16) -> Self {
        Self { hostname, user, port }
    }
    
    /// Low-level SSH command execution
    async fn ssh_exec(&self, command: &str) -> Result<(String, String, i32)> {
        let output = Command::new("ssh")
            .arg("-p").arg(self.port.to_string())
            .arg(format!("{}@{}", self.user, self.hostname))
            .arg(command)
            .output()
            .await?;
        
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let code = output.status.code().unwrap_or(-1);
        
        Ok((stdout, stderr, code))
    }
}

#[async_trait]
impl ProcessExecutor for SshExecutor {
    async fn exec(&self, command: &str, args: &[String]) -> Result<(String, String, i32)> {
        let full_cmd = format!("{} {}", command, args.join(" "));
        self.ssh_exec(&full_cmd).await
    }
    
    async fn spawn_daemon(&self, command: &str, args: &[String]) -> Result<u32> {
        // Start daemon in background with nohup, capture PID
        let full_cmd = format!("{} {}", command, args.join(" "));
        let remote_cmd = format!(
            "nohup {} > /dev/null 2>&1 & echo $!",
            full_cmd
        );
        
        let (stdout, stderr, code) = self.ssh_exec(&remote_cmd).await?;
        
        if code != 0 {
            anyhow::bail!("Failed to spawn daemon: {}", stderr);
        }
        
        let pid = stdout.trim().parse::<u32>()?;
        Ok(pid)
    }
    
    async fn is_running(&self, pid: u32) -> Result<bool> {
        let cmd = format!("kill -0 {}", pid);
        let (_, _, code) = self.ssh_exec(&cmd).await?;
        Ok(code == 0)
    }
    
    async fn kill(&self, pid: u32, signal: Signal) -> Result<()> {
        let sig = match signal {
            Signal::Term => "-TERM",
            Signal::Kill => "-KILL",
        };
        
        let cmd = format!("kill {} {}", sig, pid);
        self.ssh_exec(&cmd).await?;
        Ok(())
    }
    
    async fn find_binary(&self, name: &str) -> Result<PathBuf> {
        // Try multiple locations on remote machine
        let find_script = format!(
            r#"
            if [ -f ~/.local/bin/{name} ]; then
                echo ~/.local/bin/{name}
            elif [ -f target/debug/{name} ]; then
                echo target/debug/{name}
            elif [ -f target/release/{name} ]; then
                echo target/release/{name}
            else
                exit 1
            fi
            "#,
            name = name
        );
        
        let (stdout, stderr, code) = self.ssh_exec(&find_script).await?;
        
        if code != 0 {
            anyhow::bail!("Binary '{}' not found on remote: {}", name, stderr);
        }
        
        Ok(PathBuf::from(stdout.trim()))
    }
    
    async fn copy_file(&self, local: &PathBuf, remote: &str) -> Result<()> {
        // Use SCP to copy file
        let status = Command::new("scp")
            .arg("-P").arg(self.port.to_string())
            .arg(local)
            .arg(format!("{}@{}:{}", self.user, self.hostname, remote))
            .status()
            .await?;
        
        if !status.success() {
            anyhow::bail!("SCP failed");
        }
        
        Ok(())
    }
}
```

---

### Step 4: Modify Existing Functions to Use Executor

```rust
// bin/99_shared_crates/daemon-lifecycle/src/start.rs

use crate::executor::{ProcessExecutor, LocalExecutor};
use crate::types::HttpDaemonConfig;

/// Start daemon with optional executor (defaults to local)
pub async fn start_daemon(config: HttpDaemonConfig) -> Result<u32> {
    start_daemon_with_executor(config, &LocalExecutor).await
}

/// Internal function that accepts any executor
pub async fn start_daemon_with_executor(
    config: HttpDaemonConfig,
    executor: &dyn ProcessExecutor,
) -> Result<u32> {
    // Step 1: Find binary (works for local AND remote via executor)
    let binary_path = executor.find_binary(&config.daemon_name).await?;
    
    n!("spawn", "Spawning daemon: {} with args: {:?}", binary_path.display(), config.args);
    
    // Step 2: Spawn daemon (works for local AND remote via executor)
    let pid = executor.spawn_daemon(
        binary_path.to_str().unwrap(),
        &config.args
    ).await?;
    
    n!("spawned", "Daemon spawned with PID: {}", pid);
    
    // Step 3: Poll health endpoint (works over HTTP for both local and remote)
    let health_url = &config.health_url;
    let mut health_config = HealthPollConfig::new(health_url)
        .with_daemon_name(&config.daemon_name);
    
    if let Some(attempts) = config.max_health_attempts {
        health_config = health_config.with_max_attempts(attempts);
    }
    
    if let Some(job_id) = config.job_id.as_deref() {
        health_config = health_config.with_job_id(job_id);
    }
    
    poll_daemon_health(health_config).await?;
    
    // Step 4: Write PID file (only for local)
    // For remote, we skip this since PID file is on remote machine
    if executor.type_id() == TypeId::of::<LocalExecutor>() {
        write_pid_file(&config.daemon_name, pid)?;
    }
    
    Ok(pid)
}
```

---

### Step 5: Add Execution Context to HttpDaemonConfig

```rust
// bin/99_shared_crates/daemon-lifecycle/src/types/start.rs

use crate::executor::{ProcessExecutor, LocalExecutor, SshExecutor};

pub struct HttpDaemonConfig {
    pub daemon_name: String,
    pub health_url: String,
    pub args: Vec<String>,
    pub max_health_attempts: Option<usize>,
    pub job_id: Option<String>,
    
    // NEW: Execution context
    executor: Box<dyn ProcessExecutor>,
}

impl HttpDaemonConfig {
    pub fn new(daemon_name: &str, health_url: &str) -> Self {
        Self {
            daemon_name: daemon_name.to_string(),
            health_url: health_url.to_string(),
            args: vec![],
            max_health_attempts: None,
            job_id: None,
            executor: Box::new(LocalExecutor),  // Default: local
        }
    }
    
    // Builder methods (existing)
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }
    
    pub fn with_job_id(mut self, job_id: &str) -> Self {
        self.job_id = Some(job_id.to_string());
        self
    }
    
    // NEW: Set SSH executor
    pub fn with_ssh(mut self, hostname: &str, user: &str, port: u16) -> Self {
        self.executor = Box::new(SshExecutor::new(
            hostname.to_string(),
            user.to_string(),
            port,
        ));
        self
    }
    
    // Internal: Get executor
    pub(crate) fn executor(&self) -> &dyn ProcessExecutor {
        &*self.executor
    }
}
```

---

### Step 6: Update start_daemon to Use Config's Executor

```rust
// bin/99_shared_crates/daemon-lifecycle/src/start.rs (UPDATED)

pub async fn start_daemon(config: HttpDaemonConfig) -> Result<u32> {
    let executor = config.executor();
    
    // Step 1: Find binary
    let binary_path = executor.find_binary(&config.daemon_name).await?;
    
    n!("spawn", "Spawning daemon: {} with args: {:?}", binary_path.display(), config.args);
    
    // Step 2: Spawn daemon
    let pid = executor.spawn_daemon(
        binary_path.to_str().unwrap(),
        &config.args
    ).await?;
    
    n!("spawned", "Daemon spawned with PID: {}", pid);
    
    // Step 3: Poll health (works over HTTP for both)
    let mut health_config = HealthPollConfig::new(&config.health_url)
        .with_daemon_name(&config.daemon_name);
    
    if let Some(attempts) = config.max_health_attempts {
        health_config = health_config.with_max_attempts(attempts);
    }
    
    if let Some(job_id) = config.job_id.as_deref() {
        health_config = health_config.with_job_id(job_id);
    }
    
    poll_daemon_health(health_config).await?;
    
    // Step 4: Write PID file (local only)
    // TODO: Detect if executor is LocalExecutor
    // For now, skip PID file for simplicity
    
    Ok(pid)
}
```

---

### Step 7: Handler Usage (Clean API)

```rust
// bin/00_rbee_keeper/src/handlers/hive.rs

HiveAction::Start { port } => {
    let port = port.unwrap_or(7835);
    
    // Build config
    let mut config = HttpDaemonConfig::new("rbee-hive", &format!("http://localhost:{}", port))
        .with_args(vec![
            "--port".to_string(),
            port.to_string(),
            "--queen-url".to_string(),
            queen_url.to_string(),
        ]);
    
    // Inject SSH if remote
    if alias != "localhost" {
        let hive = load_hive_config(&alias)?;
        
        // Update health URL to use remote hostname
        let remote_health_url = format!("http://{}:{}", hive.hostname, port);
        config = HttpDaemonConfig::new("rbee-hive", &remote_health_url)
            .with_args(config.args)
            .with_ssh(&hive.hostname, &hive.ssh_user, hive.ssh_port);
    }
    
    // SAME function call - executor injected in config
    daemon_lifecycle::start_daemon(config).await?;
    Ok(())
}
```

---

## Benefits of This Approach

### ✅ Zero Duplication
- ONE `start_daemon()` function
- Works for both local and remote
- Executor trait handles the differences

### ✅ Clean API
```rust
// Local (default)
let config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835");
start_daemon(config).await?;

// Remote (just add .with_ssh())
let config = HttpDaemonConfig::new("rbee-hive", "http://remote:7835")
    .with_ssh("remote", "user", 22);
start_daemon(config).await?;
```

### ✅ Testable
```rust
// Mock executor for tests
struct MockExecutor {
    responses: Vec<(String, String, i32)>,
}

#[async_trait]
impl ProcessExecutor for MockExecutor {
    async fn spawn_daemon(&self, _cmd: &str, _args: &[String]) -> Result<u32> {
        Ok(12345)  // Mock PID
    }
    // ... other methods return mocked data
}

// Test without real processes or SSH
let config = HttpDaemonConfig::new("test", "http://localhost:8080")
    .with_executor(Box::new(MockExecutor::new()));
start_daemon(config).await?;
```

### ✅ Extensible
Want to add Docker support later?
```rust
pub struct DockerExecutor {
    container_id: String,
}

#[async_trait]
impl ProcessExecutor for DockerExecutor {
    async fn spawn_daemon(&self, cmd: &str, args: &[String]) -> Result<u32> {
        // docker exec ...
    }
}

// Usage
let config = HttpDaemonConfig::new("rbee-hive", url)
    .with_docker("container-123");
```

---

## Implementation Timeline

### Week 1: Executor Trait
- Create `executor.rs` with trait definition
- Implement `LocalExecutor` (extract from existing code)
- Add tests

### Week 2: SSH Executor
- Implement `SshExecutor`
- Test with real SSH connection
- Handle edge cases

### Week 3: Integration
- Add `executor` field to `HttpDaemonConfig`
- Add `.with_ssh()` builder method
- Update `start_daemon()` to use executor
- Update `stop_daemon()` to use executor

### Week 4: Other Operations
- Update `install_daemon()` to use executor
- Update `uninstall_daemon()` to use executor
- Update handlers to inject SSH when needed
- Documentation

**Total: 4 weeks**

---

## File Changes Required

### New Files
```
bin/99_shared_crates/daemon-lifecycle/src/
├── executor.rs              # NEW: Trait definition
├── executor/
│   ├── mod.rs              # NEW: Re-exports
│   ├── local.rs            # NEW: LocalExecutor
│   ├── ssh.rs              # NEW: SshExecutor
│   └── mock.rs             # NEW: MockExecutor (for tests)
```

### Modified Files
```
bin/99_shared_crates/daemon-lifecycle/src/
├── start.rs                # Use executor from config
├── stop.rs                 # Use executor from config
├── install.rs              # Use executor from config
├── types/start.rs          # Add executor field to HttpDaemonConfig
└── lib.rs                  # Export executor module
```

### Handler Changes
```
bin/00_rbee_keeper/src/handlers/
└── hive.rs                 # Add .with_ssh() when alias != "localhost"
```

---

## Migration Path

### Phase 1: Add Executor (Backward Compatible)
```rust
// Existing code still works (uses LocalExecutor by default)
let config = HttpDaemonConfig::new("rbee-hive", url);
start_daemon(config).await?;  // ✅ Works unchanged
```

### Phase 2: Add SSH Support
```rust
// New code can use SSH
let config = HttpDaemonConfig::new("rbee-hive", url)
    .with_ssh("remote", "user", 22);
start_daemon(config).await?;  // ✅ Now works remotely
```

### Phase 3: Update Handlers
```rust
// Handlers detect alias and inject SSH
if alias != "localhost" {
    config = config.with_ssh(...);
}
```

---

## Comparison to Previous Approaches

| Aspect | Separate remote/ Module | Executor Injection |
|--------|------------------------|-------------------|
| Code duplication | None (different paths) | None (same path) |
| API changes | New functions | Builder method |
| Testing | Mock SSH calls | Mock executor |
| Extensibility | Add more modules | Add more executors |
| Complexity | Medium | Low |
| Lines of code | ~500 | ~400 |

---

## Key Design Decisions

### Why Trait Instead of Enum?
```rust
// ❌ Enum approach - less flexible
enum Executor {
    Local,
    Ssh(SshConfig),
}

// ✅ Trait approach - extensible
trait ProcessExecutor {
    async fn spawn_daemon(...) -> Result<u32>;
}
```

**Reason:** Trait allows adding Docker, Kubernetes, etc. without modifying daemon-lifecycle

### Why Box<dyn ProcessExecutor>?
```rust
pub struct HttpDaemonConfig {
    executor: Box<dyn ProcessExecutor>,  // Dynamic dispatch
}
```

**Reason:** Config needs to own the executor, trait objects require Box

### Why Not Generic?
```rust
// ❌ Generic approach
pub struct HttpDaemonConfig<E: ProcessExecutor> {
    executor: E,
}
```

**Reason:** Makes API more complex, every function becomes generic

---

## Next Steps

1. Review this design
2. Confirm approach
3. Implement `executor.rs` trait
4. Implement `LocalExecutor` (extract existing logic)
5. Implement `SshExecutor`
6. Add `.with_ssh()` to `HttpDaemonConfig`
7. Update `start_daemon()` to use executor
8. Test with real remote hive

**Estimated delivery: 4 weeks for production-ready implementation**
