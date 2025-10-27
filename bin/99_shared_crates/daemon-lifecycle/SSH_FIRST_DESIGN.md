# SSH-First Design - If We Started Over

**Date:** Oct 27, 2025  
**Question:** What if daemon-lifecycle was designed with SSH from the ground up?  
**Key Insight:** Local execution is just SSH to localhost

---

## Fundamental Shift in Thinking

### Current Design (Local-First)
```
Local execution = default (tokio::Command)
Remote execution = special case (wrap with SSH)
```

### SSH-First Design
```
ALL execution goes through unified interface
Local = SSH to localhost (or optimized local path)
Remote = SSH to remote host
```

---

## Core Principle: Everything is Remote

```rust
// bin/99_shared_crates/daemon-lifecycle/src/lib.rs

/// Core concept: Every daemon runs on a "host"
/// Localhost is just a special case of a host
pub struct DaemonHost {
    pub hostname: String,
    pub user: String,
    pub port: u16,
}

impl DaemonHost {
    /// Create localhost host (optimized path)
    pub fn localhost() -> Self {
        Self {
            hostname: "localhost".to_string(),
            user: whoami::username(),
            port: 22,
        }
    }
    
    /// Create remote host
    pub fn remote(hostname: String, user: String, port: u16) -> Self {
        Self { hostname, user, port }
    }
    
    /// Is this the local machine?
    pub fn is_local(&self) -> bool {
        self.hostname == "localhost" || self.hostname == "127.0.0.1"
    }
}

/// Unified daemon configuration
pub struct DaemonConfig {
    pub name: String,
    pub host: DaemonHost,        // Every daemon has a host
    pub health_url: String,
    pub args: Vec<String>,
}

impl DaemonConfig {
    /// Create config for local daemon
    pub fn local(name: &str, port: u16) -> Self {
        Self {
            name: name.to_string(),
            host: DaemonHost::localhost(),
            health_url: format!("http://localhost:{}/health", port),
            args: vec![],
        }
    }
    
    /// Create config for remote daemon
    pub fn remote(name: &str, host: DaemonHost, port: u16) -> Self {
        Self {
            name: name.to_string(),
            health_url: format!("http://{}:{}/health", host.hostname, port),
            host,
            args: vec![],
        }
    }
}
```

---

## Unified Execution Engine

```rust
// bin/99_shared_crates/daemon-lifecycle/src/executor.rs

/// Single execution engine for all operations
pub struct Executor {
    host: DaemonHost,
}

impl Executor {
    pub fn new(host: DaemonHost) -> Self {
        Self { host }
    }
    
    /// Execute command on host (local or remote)
    pub async fn exec(&self, command: &str) -> Result<String> {
        if self.host.is_local() {
            // Optimized local path (no SSH overhead)
            self.exec_local(command).await
        } else {
            // Remote path (SSH)
            self.exec_ssh(command).await
        }
    }
    
    /// Local execution (optimized)
    async fn exec_local(&self, command: &str) -> Result<String> {
        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await?;
        
        if !output.status.success() {
            anyhow::bail!("Command failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
    
    /// Remote execution (SSH)
    async fn exec_ssh(&self, command: &str) -> Result<String> {
        let output = tokio::process::Command::new("ssh")
            .arg("-p").arg(self.host.port.to_string())
            .arg(format!("{}@{}", self.host.user, self.host.hostname))
            .arg(command)
            .output()
            .await?;
        
        if !output.status.success() {
            anyhow::bail!("SSH command failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}
```

---

## Start Daemon - Unified Implementation

```rust
// bin/99_shared_crates/daemon-lifecycle/src/start.rs

/// Start daemon on any host (local or remote)
/// NO special cases, NO if/else for local vs remote
pub async fn start_daemon(config: DaemonConfig) -> Result<u32> {
    let executor = Executor::new(config.host.clone());
    
    // Step 1: Find binary on host
    let binary_path = find_binary(&executor, &config.name).await?;
    
    // Step 2: Start daemon on host
    let pid = spawn_daemon(&executor, &binary_path, &config.args).await?;
    
    // Step 3: Poll health (HTTP works for both local and remote)
    poll_health(&config.health_url).await?;
    
    Ok(pid)
}

/// Find binary on host (works for local and remote)
async fn find_binary(executor: &Executor, name: &str) -> Result<String> {
    // Single script that works everywhere
    let script = format!(
        r#"
        # Try installed location
        if [ -f ~/.local/bin/{name} ]; then
            echo ~/.local/bin/{name}
            exit 0
        fi
        
        # Try development builds
        if [ -f target/debug/{name} ]; then
            echo target/debug/{name}
            exit 0
        fi
        
        if [ -f target/release/{name} ]; then
            echo target/release/{name}
            exit 0
        fi
        
        echo "Binary not found: {name}" >&2
        exit 1
        "#,
        name = name
    );
    
    let output = executor.exec(&script).await?;
    Ok(output.trim().to_string())
}

/// Spawn daemon on host (works for local and remote)
async fn spawn_daemon(executor: &Executor, binary: &str, args: &[String]) -> Result<u32> {
    let args_str = args.join(" ");
    
    // Universal daemon spawn command
    let script = format!(
        r#"
        # Start daemon in background
        nohup {} {} > /dev/null 2>&1 &
        
        # Capture PID
        echo $!
        "#,
        binary, args_str
    );
    
    let output = executor.exec(&script).await?;
    let pid = output.trim().parse::<u32>()?;
    
    Ok(pid)
}

/// Poll health endpoint (already works for both)
async fn poll_health(health_url: &str) -> Result<()> {
    // HTTP request works for localhost and remote
    for attempt in 1..=10 {
        let client = reqwest::Client::new();
        match client.get(health_url).send().await {
            Ok(response) if response.status().is_success() => {
                return Ok(());
            }
            _ => {
                tokio::time::sleep(Duration::from_millis(200 * attempt)).await;
            }
        }
    }
    
    anyhow::bail!("Daemon failed to become healthy")
}
```

---

## Stop Daemon - Unified Implementation

```rust
// bin/99_shared_crates/daemon-lifecycle/src/stop.rs

/// Stop daemon on any host
pub async fn stop_daemon(config: DaemonConfig) -> Result<()> {
    // Try graceful shutdown via HTTP first (works for both)
    if try_graceful_shutdown(&config.health_url).await.is_ok() {
        return Ok(());
    }
    
    // Fallback: Force kill via executor
    let executor = Executor::new(config.host);
    force_kill(&executor, &config.name).await
}

async fn try_graceful_shutdown(health_url: &str) -> Result<()> {
    let shutdown_url = health_url.replace("/health", "/v1/shutdown");
    
    let client = reqwest::Client::new();
    client.post(&shutdown_url)
        .timeout(Duration::from_secs(5))
        .send()
        .await?;
    
    Ok(())
}

async fn force_kill(executor: &Executor, daemon_name: &str) -> Result<()> {
    let script = format!("pkill -f {}", daemon_name);
    executor.exec(&script).await?;
    Ok(())
}
```

---

## Install Daemon - Unified Implementation

```rust
// bin/99_shared_crates/daemon-lifecycle/src/install.rs

/// Install daemon on any host
pub async fn install_daemon(
    daemon_name: &str,
    host: DaemonHost,
    local_binary: Option<PathBuf>,
) -> Result<()> {
    // Step 1: Build or find binary locally
    let binary_path = match local_binary {
        Some(path) => path,
        None => build_daemon(daemon_name).await?,
    };
    
    // Step 2: Copy to host
    copy_to_host(&binary_path, &host, daemon_name).await?;
    
    // Step 3: Make executable on host
    let executor = Executor::new(host);
    let script = format!("chmod +x ~/.local/bin/{}", daemon_name);
    executor.exec(&script).await?;
    
    Ok(())
}

async fn copy_to_host(local_path: &Path, host: &DaemonHost, daemon_name: &str) -> Result<()> {
    if host.is_local() {
        // Local: Just copy file
        let dest = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("No home dir"))?
            .join(".local/bin")
            .join(daemon_name);
        
        tokio::fs::create_dir_all(dest.parent().unwrap()).await?;
        tokio::fs::copy(local_path, dest).await?;
    } else {
        // Remote: Use SCP
        let dest = format!("{}@{}:~/.local/bin/{}", host.user, host.hostname, daemon_name);
        
        let status = tokio::process::Command::new("scp")
            .arg("-P").arg(host.port.to_string())
            .arg(local_path)
            .arg(&dest)
            .status()
            .await?;
        
        if !status.success() {
            anyhow::bail!("SCP failed");
        }
    }
    
    Ok(())
}
```

---

## Handler Usage - Simpler API

```rust
// bin/00_rbee_keeper/src/handlers/hive.rs

pub async fn handle_hive(action: HiveAction, queen_url: &str) -> Result<()> {
    match action {
        HiveAction::Start { port } => {
            let port = port.unwrap_or(7835);
            
            // Create config based on alias
            let config = if alias == "localhost" {
                DaemonConfig::local("rbee-hive", port)
            } else {
                let hive = load_hive_config(&alias)?;
                let host = DaemonHost::remote(hive.hostname, hive.ssh_user, hive.ssh_port);
                DaemonConfig::remote("rbee-hive", host, port)
            };
            
            // Add args
            let config = config.with_args(vec![
                "--port".to_string(),
                port.to_string(),
                "--queen-url".to_string(),
                queen_url.to_string(),
            ]);
            
            // SAME function - no special cases
            daemon_lifecycle::start_daemon(config).await?;
            Ok(())
        }
        
        HiveAction::Stop { port } => {
            let port = port.unwrap_or(7835);
            
            let config = if alias == "localhost" {
                DaemonConfig::local("rbee-hive", port)
            } else {
                let hive = load_hive_config(&alias)?;
                let host = DaemonHost::remote(hive.hostname, hive.ssh_user, hive.ssh_port);
                DaemonConfig::remote("rbee-hive", host, port)
            };
            
            daemon_lifecycle::stop_daemon(config).await
        }
        
        // ... other actions
    }
}
```

---

## Key Differences from Current Design

### 1. No Executor Trait
```rust
// ❌ Current approach: Trait abstraction
trait ProcessExecutor {
    async fn spawn_daemon(...);
}

// ✅ SSH-first: Single Executor with optimization
struct Executor {
    host: DaemonHost,
}

impl Executor {
    async fn exec(&self, cmd: &str) -> Result<String> {
        if self.host.is_local() {
            self.exec_local(cmd).await  // Fast path
        } else {
            self.exec_ssh(cmd).await    // SSH path
        }
    }
}
```

**Why:** Simpler, no trait objects, clear optimization boundary

---

### 2. Host is Core Concept
```rust
// ❌ Current: Host is optional/implicit
let config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835");

// ✅ SSH-first: Host is explicit
let config = DaemonConfig::local("rbee-hive", 7835);
// or
let config = DaemonConfig::remote("rbee-hive", host, 7835);
```

**Why:** Makes it clear where the daemon runs

---

### 3. Shell Scripts Instead of Rust Logic
```rust
// ❌ Current: Rust logic with tokio::Command
let binary_path = find_in_target("rbee-hive")?;
let mut cmd = Command::new(&binary_path);
cmd.args(&args).stdout(Stdio::null());
let child = cmd.spawn()?;
let pid = child.id();

// ✅ SSH-first: Shell script executed via executor
let script = r#"
    nohup rbee-hive --port 7835 > /dev/null 2>&1 &
    echo $!
"#;
let output = executor.exec(script).await?;
let pid = output.trim().parse::<u32>()?;
```

**Why:** Same script works locally and remotely

---

### 4. HTTP for Everything Possible
```rust
// ✅ SSH-first philosophy:
// - Use SSH only for: start, stop, install, find binary
// - Use HTTP for: health checks, status, capabilities
// - Never use SSH when HTTP works

// Health check (works for both)
reqwest::get("http://localhost:7835/health").await?;
reqwest::get("http://remote:7835/health").await?;

// Graceful shutdown (works for both)
reqwest::post("http://localhost:7835/v1/shutdown").await?;
reqwest::post("http://remote:7835/v1/shutdown").await?;
```

**Why:** HTTP is faster, more reliable, easier to debug

---

### 5. No PID Files
```rust
// ❌ Current: PID files for tracking
write_pid_file("rbee-hive", 12345)?;
let pid = read_pid_file("rbee-hive")?;

// ✅ SSH-first: Query process directly
let script = "pgrep -f rbee-hive";
let pid = executor.exec(script).await?.trim().parse::<u32>()?;
```

**Why:** PID files don't work well across SSH, querying is more reliable

---

### 6. Unified Error Handling
```rust
// ✅ SSH-first: All operations return same error type
pub enum DaemonError {
    ExecutionFailed { command: String, stderr: String },
    NotFound { daemon: String, host: String },
    HealthCheckFailed { url: String },
    Timeout { operation: String },
}

// Every operation uses executor, so errors are consistent
```

**Why:** No special cases for "SSH failed" vs "spawn failed"

---

## File Structure Comparison

### Current Structure (Local-First)
```
daemon-lifecycle/src/
├── start.rs           # Local implementation
├── stop.rs            # Local implementation
├── install.rs         # Local implementation
├── executor/          # NEW: Abstraction layer
│   ├── local.rs       # Wraps existing logic
│   └── ssh.rs         # NEW: Remote logic
```

### SSH-First Structure
```
daemon-lifecycle/src/
├── lib.rs             # DaemonHost, DaemonConfig
├── executor.rs        # Single Executor (local + SSH)
├── start.rs           # Uses executor (works for both)
├── stop.rs            # Uses executor (works for both)
├── install.rs         # Uses executor (works for both)
└── scripts/           # Shell scripts used by executor
    ├── find_binary.sh
    ├── spawn_daemon.sh
    └── kill_daemon.sh
```

**Difference:** No separate local/remote paths, just one implementation

---

## Benefits of SSH-First Design

### ✅ Simpler Codebase
- No trait abstraction
- No separate local/remote modules
- One code path for everything

### ✅ Easier to Understand
```rust
// Clear: This daemon runs on this host
let config = DaemonConfig::remote("rbee-hive", host, 7835);
start_daemon(config).await?;
```

### ✅ Better Testing
```rust
// Test with localhost (no mocks needed)
let config = DaemonConfig::local("test-daemon", 9999);
start_daemon(config).await?;

// Same code works for remote
let config = DaemonConfig::remote("test-daemon", remote_host, 9999);
start_daemon(config).await?;
```

### ✅ Consistent Behavior
- Local and remote use same code paths
- Bugs affect both equally (easier to catch)
- Performance characteristics are similar

### ✅ Shell Scripts are Portable
```bash
# This script works locally and remotely
nohup rbee-hive --port 7835 > /dev/null 2>&1 &
echo $!
```

---

## Downsides of SSH-First Design

### ⚠️ Local Performance
- Shell script overhead for local operations
- SSH to localhost has latency (even if optimized out)
- More process spawning

**Mitigation:** Optimize local path in `executor.exec()`

### ⚠️ Debugging Harder
- Shell scripts hide Rust stack traces
- Errors are string-based, not typed
- Harder to step through in debugger

**Mitigation:** Good logging, clear error messages

### ⚠️ Platform Dependencies
- Requires `sh` on all platforms
- Requires `ssh`/`scp` for remote
- Windows compatibility harder

**Mitigation:** Use PowerShell on Windows, or require WSL

---

## Migration from Current Design

### Phase 1: Add DaemonHost
```rust
// Add to existing code
pub struct DaemonHost { /* ... */ }

// Existing code still works
let config = HttpDaemonConfig::new("rbee-hive", url);
```

### Phase 2: Add Executor
```rust
// New unified executor
pub struct Executor { /* ... */ }

// Existing functions use it internally
pub async fn start_daemon(config: HttpDaemonConfig) -> Result<u32> {
    let executor = Executor::new(DaemonHost::localhost());
    // ... use executor
}
```

### Phase 3: Expose Host in API
```rust
// New API
pub async fn start_daemon_on_host(
    daemon_name: &str,
    host: DaemonHost,
    args: Vec<String>,
) -> Result<u32> {
    // ...
}

// Old API delegates to new
pub async fn start_daemon(config: HttpDaemonConfig) -> Result<u32> {
    start_daemon_on_host(&config.daemon_name, DaemonHost::localhost(), config.args).await
}
```

---

## Recommendation

### For New Project: SSH-First
If starting from scratch, SSH-first is cleaner:
- Simpler architecture
- Fewer abstractions
- Unified code paths

### For Existing Project: Executor Injection
For daemon-lifecycle as it exists today:
- Less disruptive
- Backward compatible
- Gradual migration possible

---

## Key Insight

**The fundamental difference:**

**Current (Local-First):**
```
Local is the default, remote is special
→ Optimize for local, add SSH as wrapper
→ Two code paths (local + remote)
```

**SSH-First:**
```
Everything is remote, localhost is optimization
→ Write for remote, optimize local path
→ One code path (executor with fast path)
```

---

## Code Size Comparison

| Approach | Lines of Code | Complexity |
|----------|---------------|------------|
| Current (Local-First) | ~800 LOC | Medium |
| Executor Injection | ~1200 LOC | Medium-High |
| SSH-First | ~600 LOC | Low-Medium |

**SSH-First is smaller** because:
- No trait abstraction
- No separate modules
- Shell scripts handle complexity

---

## Final Thought

**If I were designing daemon-lifecycle today with SSH as a requirement:**

I would use **SSH-First** design because:
1. Simpler (~600 LOC vs ~1200 LOC)
2. One code path, not two
3. Shell scripts are more portable than Rust process spawning
4. HTTP for monitoring (fast, works everywhere)
5. SSH only for control plane (start/stop/install)

**But for migrating existing code:**

I would use **Executor Injection** because:
1. Less disruptive
2. Backward compatible
3. Can migrate gradually
4. Existing tests still work

---

**Next Steps:**

1. Choose approach based on project constraints
2. If SSH-first: Start fresh implementation
3. If executor injection: Follow previous design doc
4. Prototype both and compare

**Estimated effort:**
- SSH-First: 3 weeks (smaller codebase)
- Executor Injection: 4 weeks (more code, but safer)
