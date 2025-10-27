# Remote Execution Architecture - Zero Duplication Pattern

**Date:** Oct 27, 2025  
**Problem:** SSH wrapper that wraps every command duplicates all daemon-lifecycle logic  
**Requirement:** SSH injected at function level, not line-by-line

---

## Anti-Pattern (What We DON'T Want)

```rust
// ❌ BAD: Wraps every command individually
pub async fn start_daemon_remote(config: HttpDaemonConfig, ssh: SshConfig) -> Result<u32> {
    // Duplicate ALL the logic from start_daemon()
    // But wrap each Command::new() with ssh_exec()
    let binary_path = ssh_exec(&ssh, "which rbee-hive").await?;  // Line 1 wrapped
    ssh_exec(&ssh, &format!("{} --port {}", binary_path, port)).await?;  // Line 2 wrapped
    // ... 50 more lines of duplicated logic with ssh_exec() ...
}
```

**Problem:** This duplicates 100+ LOC from `start_daemon()` just to add SSH.

---

## Solution 1: Remote Execution Agent (RECOMMENDED)

**Pattern:** Deploy a thin agent on remote machines that exposes daemon-lifecycle as HTTP API

### Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│  rbee-keeper    │    HTTP/HTTPS      │  Remote Machine │
│  (localhost)    │───────────────────>│                 │
│                 │                    │  ┌───────────┐  │
│  calls:         │                    │  │  rbee-    │  │
│  start_daemon() │                    │  │  lifecycle│  │
│                 │                    │  │  -agent   │  │
└─────────────────┘                    │  └─────┬─────┘  │
                                       │        │        │
                                       │        v        │
                                       │  daemon-        │
                                       │  lifecycle      │
                                       │  (local exec)   │
                                       └─────────────────┘
```

### Implementation

#### Step 1: Create Agent Binary (100 LOC)

```rust
// bin/99_shared_crates/daemon-lifecycle-agent/src/main.rs
use axum::{Router, Json, extract::State};
use daemon_lifecycle::{start_daemon, stop_daemon, HttpDaemonConfig};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct StartRequest {
    daemon_name: String,
    health_url: String,
    args: Vec<String>,
}

#[derive(Serialize)]
struct StartResponse {
    pid: u32,
}

async fn handle_start(Json(req): Json<StartRequest>) -> Result<Json<StartResponse>, StatusCode> {
    let config = HttpDaemonConfig::new(&req.daemon_name, &req.health_url)
        .with_args(req.args);
    
    // ZERO DUPLICATION: Just calls existing daemon-lifecycle function
    let pid = daemon_lifecycle::start_daemon(config).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(StartResponse { pid }))
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/v1/lifecycle/start", post(handle_start))
        .route("/v1/lifecycle/stop", post(handle_stop))
        .route("/v1/lifecycle/install", post(handle_install))
        .route("/health", get(|| async { "OK" }));
    
    axum::Server::bind(&"0.0.0.0:7999".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

**Agent Size:** ~100 LOC (thin HTTP wrapper around daemon-lifecycle)

#### Step 2: Remote Execution Client (50 LOC)

```rust
// bin/99_shared_crates/daemon-lifecycle/src/remote/agent_client.rs
use reqwest::Client;

pub struct RemoteExecutionClient {
    agent_url: String,
    client: Client,
}

impl RemoteExecutionClient {
    pub fn new(hostname: &str, port: u16) -> Self {
        Self {
            agent_url: format!("http://{}:{}", hostname, port),
            client: Client::new(),
        }
    }
    
    pub async fn start_daemon(&self, config: HttpDaemonConfig) -> Result<u32> {
        let response = self.client
            .post(&format!("{}/v1/lifecycle/start", self.agent_url))
            .json(&config)
            .send()
            .await?;
        
        let result: StartResponse = response.json().await?;
        Ok(result.pid)
    }
}
```

#### Step 3: Handler Integration (10 LOC change)

```rust
// bin/00_rbee_keeper/src/handlers/hive.rs
HiveAction::Start { port } => {
    let port = port.unwrap_or(7835);
    let config = HttpDaemonConfig::new("rbee-hive", &format!("http://localhost:{}", port))
        .with_args(vec![/*...*/]);
    
    if alias == "localhost" {
        // Local execution (existing code)
        daemon_lifecycle::start_daemon(config).await?;
    } else {
        // Remote execution via agent
        let hive = load_hive_config(&alias)?;
        let agent = RemoteExecutionClient::new(&hive.hostname, 7999);
        agent.start_daemon(config).await?;
    }
    Ok(())
}
```

### Deployment

```bash
# One-time setup per remote machine
# Build agent
cargo build --release --bin daemon-lifecycle-agent

# Deploy agent (via SCP or manual copy)
scp target/release/daemon-lifecycle-agent user@remote:/usr/local/bin/

# Start agent as systemd service
ssh user@remote "systemctl enable --now daemon-lifecycle-agent"
```

### Benefits

✅ **Zero duplication** - Agent just calls daemon-lifecycle  
✅ **Function-level injection** - HTTP call wraps entire function  
✅ **No line-by-line wrapping** - Agent executes locally on remote  
✅ **Testable** - Mock HTTP client  
✅ **Secure** - Add mTLS for production  
✅ **Extensible** - Add new endpoints easily  

### Cons

⚠️ Requires agent deployment (one-time)  
⚠️ 1 week implementation  

---

## Solution 2: Smart Command Serialization

**Pattern:** Serialize the ENTIRE operation as data, execute remotely, deserialize results

### Architecture

```rust
// bin/99_shared_crates/daemon-lifecycle/src/remote/serialized_execution.rs

#[derive(Serialize, Deserialize)]
enum LifecycleOperation {
    Start(HttpDaemonConfig),
    Stop(HttpDaemonConfig),
    Install { daemon_name: String },
}

#[derive(Serialize, Deserialize)]
enum LifecycleResult {
    StartSuccess(u32),
    StopSuccess,
    InstallSuccess,
    Error(String),
}

pub struct RemoteExecutor {
    ssh_config: SshConfig,
}

impl RemoteExecutor {
    /// Execute ENTIRE lifecycle operation remotely
    /// NO line-by-line wrapping - sends the whole operation
    pub async fn execute(&self, operation: LifecycleOperation) -> Result<LifecycleResult> {
        // Step 1: Serialize operation to JSON
        let operation_json = serde_json::to_string(&operation)?;
        
        // Step 2: ONE SSH command that executes the operation
        let remote_cmd = format!(
            "rbee-lifecycle-exec '{}'",  // Single remote binary
            operation_json
        );
        
        let output = self.ssh_exec(&remote_cmd).await?;
        
        // Step 3: Deserialize result
        let result: LifecycleResult = serde_json::from_str(&output)?;
        Ok(result)
    }
    
    async fn ssh_exec(&self, cmd: &str) -> Result<String> {
        // ONE ssh call, not per-line
        let output = tokio::process::Command::new("ssh")
            .args([&format!("{}@{}", self.ssh_config.user, self.ssh_config.hostname)])
            .arg(cmd)
            .output()
            .await?;
        
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}
```

### Remote Binary

```rust
// bin/99_shared_crates/daemon-lifecycle-exec/src/main.rs
// Small binary deployed to remote machines

use daemon_lifecycle::*;

#[tokio::main]
async fn main() {
    // Read operation from stdin or argv
    let operation: LifecycleOperation = serde_json::from_str(&std::env::args().nth(1).unwrap()).unwrap();
    
    let result = match operation {
        LifecycleOperation::Start(config) => {
            match start_daemon(config).await {
                Ok(pid) => LifecycleResult::StartSuccess(pid),
                Err(e) => LifecycleResult::Error(e.to_string()),
            }
        }
        LifecycleOperation::Stop(config) => {
            match stop_daemon(config).await {
                Ok(_) => LifecycleResult::StopSuccess,
                Err(e) => LifecycleResult::Error(e.to_string()),
            }
        }
        // ... other operations
    };
    
    // Print result as JSON
    println!("{}", serde_json::to_string(&result).unwrap());
}
```

### Usage

```rust
// Handler
HiveAction::Start { port } => {
    let config = HttpDaemonConfig::new("rbee-hive", &format!("http://localhost:{}", port));
    
    if alias == "localhost" {
        daemon_lifecycle::start_daemon(config).await?;
    } else {
        let hive = load_hive_config(&alias)?;
        let executor = RemoteExecutor::new(hive.ssh_config);
        
        // ONE function call, wraps ENTIRE operation
        let result = executor.execute(LifecycleOperation::Start(config)).await?;
        
        match result {
            LifecycleResult::StartSuccess(pid) => Ok(pid),
            LifecycleResult::Error(e) => Err(anyhow::anyhow!(e)),
            _ => unreachable!(),
        }
    }
}
```

### Benefits

✅ **Zero duplication** - Remote binary reuses daemon-lifecycle  
✅ **Function-level wrapping** - Entire operation sent as one SSH call  
✅ **Type-safe** - Serialized operations with serde  
✅ **Minimal SSH usage** - One ssh call per operation  

### Cons

⚠️ Requires deploying `daemon-lifecycle-exec` binary to remote machines  
⚠️ Serialization overhead  

---

## Solution 3: ExecutionContext Trait (Minimal Changes)

**Pattern:** Add optional execution context to existing functions, default to local

### Implementation

```rust
// bin/99_shared_crates/daemon-lifecycle/src/remote/context.rs

pub enum ExecutionContext {
    Local,
    Remote {
        hostname: String,
        user: String,
        port: u16,
    },
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::Local
    }
}

// Extend HttpDaemonConfig (backward compatible)
pub struct HttpDaemonConfig {
    pub daemon_name: String,
    pub health_url: String,
    pub args: Vec<String>,
    pub execution: ExecutionContext,  // NEW: defaults to Local
    // ... existing fields
}

impl HttpDaemonConfig {
    pub fn new(daemon_name: &str, health_url: &str) -> Self {
        Self {
            daemon_name: daemon_name.to_string(),
            health_url: health_url.to_string(),
            args: vec![],
            execution: ExecutionContext::Local,  // Default: local
        }
    }
    
    // NEW: Builder method for remote execution
    pub fn remote(mut self, hostname: &str, user: &str, port: u16) -> Self {
        self.execution = ExecutionContext::Remote {
            hostname: hostname.to_string(),
            user: user.to_string(),
            port,
        };
        self
    }
}
```

### Modify start_daemon() ONCE

```rust
// bin/99_shared_crates/daemon-lifecycle/src/start.rs

pub async fn start_daemon(config: HttpDaemonConfig) -> Result<u32> {
    match config.execution {
        ExecutionContext::Local => {
            // EXISTING CODE - unchanged
            start_daemon_local(config).await
        }
        ExecutionContext::Remote { hostname, user, port } => {
            // NEW PATH - delegates to remote execution
            start_daemon_remote(hostname, user, port, config).await
        }
    }
}

// Extract existing logic into helper (refactor, not rewrite)
async fn start_daemon_local(config: HttpDaemonConfig) -> Result<u32> {
    // All existing start_daemon() code moved here
    // ZERO changes to the logic
    /* ... current implementation ... */
}

// Remote execution helper (uses Solution 1 or 2)
async fn start_daemon_remote(
    hostname: String,
    user: String, 
    port: u16,
    config: HttpDaemonConfig
) -> Result<u32> {
    // Either:
    // A) Call agent (Solution 1)
    let agent = RemoteExecutionClient::new(&hostname, 7999);
    agent.start_daemon(config).await
    
    // B) Serialize and execute (Solution 2)
    // let executor = RemoteExecutor::new(hostname, user, port);
    // executor.execute(LifecycleOperation::Start(config)).await
}
```

### Handler Usage

```rust
// bin/00_rbee_keeper/src/handlers/hive.rs

HiveAction::Start { port } => {
    let mut config = HttpDaemonConfig::new("rbee-hive", &format!("http://localhost:{}", port))
        .with_args(vec![/*...*/]);
    
    // Inject remote execution if not localhost
    if alias != "localhost" {
        let hive = load_hive_config(&alias)?;
        config = config.remote(&hive.hostname, &hive.ssh_user, hive.ssh_port);
    }
    
    // SAME CALL - daemon-lifecycle handles local vs remote internally
    daemon_lifecycle::start_daemon(config).await?;
    Ok(())
}
```

### Benefits

✅ **Minimal API changes** - Add `.remote()` builder, rest is internal  
✅ **Backward compatible** - Existing code works unchanged  
✅ **Function-level injection** - Context injected once, not per-line  
✅ **Clean separation** - Local vs remote paths clearly separated  

### Cons

⚠️ Still requires ONE of the remote execution mechanisms (agent or serialization)  
⚠️ Adds complexity to daemon-lifecycle API  

---

## Solution 4: Proxy Binary Pattern

**Pattern:** Smallest possible wrapper - just forward commands to remote rbee binary

### Architecture

```rust
// Remote machine already has full rbee installation
// We just need to execute the SAME command remotely

pub struct RemoteCommandProxy {
    ssh_config: SshConfig,
}

impl RemoteCommandProxy {
    /// Execute the SAME rbee command remotely
    /// Example: "rbee hive start --port 7835" becomes
    ///          ssh remote "rbee hive start --port 7835"
    pub async fn execute_remote_command(&self, local_args: &[String]) -> Result<()> {
        // Build remote command - just prepend "rbee" to args
        let remote_cmd = format!("rbee {}", local_args.join(" "));
        
        // ONE ssh call
        tokio::process::Command::new("ssh")
            .args([&format!("{}@{}", self.ssh_config.user, self.ssh_config.hostname)])
            .arg(&remote_cmd)
            .status()
            .await?;
        
        Ok(())
    }
}
```

### Handler

```rust
HiveAction::Start { port } => {
    if alias == "localhost" {
        // Execute locally (existing code)
        daemon_lifecycle::start_daemon(config).await?;
    } else {
        // Forward ENTIRE command to remote
        let hive = load_hive_config(&alias)?;
        let proxy = RemoteCommandProxy::new(hive.ssh_config);
        
        // Just re-execute the same CLI command remotely
        proxy.execute_remote_command(&["hive", "start", "--port", &port.to_string()]).await?;
    }
    Ok(())
}
```

### Benefits

✅ **Simplest possible** - Just forward commands via SSH  
✅ **Zero duplication** - Remote machine runs local code  
✅ **No new binaries** - Uses existing rbee installation  

### Cons

⚠️ Requires rbee pre-installed on remote machine  
⚠️ Less control over remote execution  
⚠️ Harder to capture structured output  

---

## Comparison Matrix

| Solution | Duplication | Lines Changed | Remote Binary Needed | Complexity | Time |
|----------|-------------|---------------|---------------------|------------|------|
| **1. Agent** | None | ~150 LOC | Yes (agent) | Medium | 1 week |
| **2. Serialization** | None | ~200 LOC | Yes (exec) | Medium | 1 week |
| **3. Context Trait** | None | ~100 LOC | Depends (uses 1 or 2) | Low | 3 days |
| **4. Proxy** | None | ~50 LOC | No (uses rbee) | Very Low | 2 days |

---

## Recommendation: Solution 3 + Solution 1

**Phase 1:** Implement Solution 3 (ExecutionContext)
- Add `.remote()` builder to HttpDaemonConfig
- Refactor `start_daemon()` into `start_daemon_local()` + routing logic
- Keep API backward compatible

**Phase 2:** Implement Solution 1 (Agent)
- Create `daemon-lifecycle-agent` binary
- Deploy as systemd service on remote machines
- Wire up remote execution in `start_daemon_remote()`

### Why This Combination?

1. **Clean API** - Solution 3 gives us a clean injection point
2. **Zero duplication** - Agent reuses daemon-lifecycle directly
3. **Function-level wrapping** - HTTP call wraps entire function
4. **Testable** - Mock HTTP client easily
5. **Production-ready** - Agent can add auth, logging, monitoring

### Implementation Order

```rust
// Week 1: Add execution context (Solution 3)
let config = HttpDaemonConfig::new("rbee-hive", url)
    .remote("192.168.1.100", "vince", 22);  // NEW

daemon_lifecycle::start_daemon(config).await?;  // Same call

// Week 2: Implement agent (Solution 1)
// Deploy daemon-lifecycle-agent to remote machines
// Wire up remote path in start_daemon()

// Result: Clean, testable, zero duplication
```

---

## Key Insight

The solution is NOT to wrap commands. The solution is to:

1. **Serialize the intent** (start/stop/install with config)
2. **Execute intent remotely** (via agent or serialization)
3. **Remote executor calls daemon-lifecycle locally** (on remote machine)

This way:
- ✅ No line-by-line wrapping
- ✅ No duplication
- ✅ Function-level injection
- ✅ Same code runs locally and remotely

---

**Next Steps:**

1. Choose between Agent (1) or Serialization (2) for remote execution
2. Implement ExecutionContext trait (Solution 3) first
3. Wire up chosen remote execution mechanism
4. Test with real remote hive

**Estimated Total:** 1-2 weeks for complete, correct implementation
