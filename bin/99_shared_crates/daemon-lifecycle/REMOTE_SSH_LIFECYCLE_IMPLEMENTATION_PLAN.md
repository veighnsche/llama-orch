# Remote SSH Lifecycle Implementation Plan

**Date:** Oct 27, 2025  
**Purpose:** Enable remote hive lifecycle management through SSH wrapper  
**Context:** `hive.rs` handlers accept `alias` parameter but currently hardcode localhost

---

## Current Problem

```rust
// bin/00_rbee_keeper/src/handlers/hive.rs - Current Implementation
HiveAction::Start { port } => {
    // ❌ PROBLEM: Hardcoded localhost, ignores alias parameter
    let config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835")
        .with_args(args);
    daemon_lifecycle::start_daemon(config).await?;
}
```

**User Request Pattern:**
```bash
# Works (localhost)
rbee hive start

# Should work but doesn't (remote via SSH)
rbee hive start --host workstation  # alias != "localhost"
```

---

## Required Pattern

```rust
pub async fn handle_hive_start(alias: String, port: u16) -> Result<()> {
    if alias == "localhost" {
        // Direct local execution (current implementation)
        daemon_lifecycle::start_daemon(config).await
    } else {
        // Load hive from ~/.config/rbee/hives.conf
        // SSH wrapper around daemon lifecycle
        ssh_lifecycle::start_remote_daemon(alias, config).await
    }
}
```

---

## Hive Configuration Format

**File:** `~/.config/rbee/hives.conf` (TOML format)

```toml
[[hive]]
alias = "workstation"
hostname = "192.168.1.100"
ssh_port = 22
ssh_user = "vince"
hive_port = 7835
auto_start = false

[[hive]]
alias = "gpu-server"
hostname = "gpu.example.com"
ssh_port = 2222
ssh_user = "rbee"
hive_port = 8081
auto_start = true
```

---

## Top 5 Implementation Approaches

### ⭐ Option 1: Shell SSH Wrapper (RECOMMENDED - Quick Win)

**Effort:** 1-2 days  
**Reliability:** ⭐⭐⭐⭐ (proven pattern)

#### Architecture
```rust
// bin/99_shared_crates/daemon-lifecycle/src/remote/shell_ssh.rs
pub struct ShellSshExecutor {
    hostname: String,
    user: String,
    port: u16,
}

impl ShellSshExecutor {
    pub async fn exec(&self, command: &str) -> Result<String> {
        let ssh_cmd = format!("ssh -p {} {}@{} '{}'", 
            self.port, self.user, self.hostname, command);
        
        let output = tokio::process::Command::new("ssh")
            .args(["-p", &self.port.to_string()])
            .args([&format!("{}@{}", self.user, self.hostname)])
            .arg(command)
            .output()
            .await?;
        
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}
```

#### Integration Pattern
```rust
// bin/99_shared_crates/daemon-lifecycle/src/remote/mod.rs
pub async fn start_remote_daemon(
    hive_config: HiveConfig,
    daemon_config: HttpDaemonConfig,
) -> Result<u32> {
    let executor = ShellSshExecutor::new(
        hive_config.hostname,
        hive_config.ssh_user,
        hive_config.ssh_port,
    );
    
    // Step 1: Check if binary exists remotely
    let binary_path = executor.exec("which rbee-hive").await?;
    
    // Step 2: Start daemon remotely
    let start_cmd = format!("{} --port {} &", binary_path.trim(), hive_config.hive_port);
    executor.exec(&start_cmd).await?;
    
    // Step 3: Poll health over HTTP (same as local)
    let health_url = format!("http://{}:{}", hive_config.hostname, hive_config.hive_port);
    poll_daemon_health(HealthPollConfig::new(&health_url)).await?;
    
    // Step 4: Get PID remotely
    let pid_cmd = "pgrep -f rbee-hive";
    let pid_output = executor.exec(pid_cmd).await?;
    let pid = pid_output.trim().parse::<u32>()?;
    
    Ok(pid)
}
```

#### Handler Integration
```rust
// bin/00_rbee_keeper/src/handlers/hive.rs
HiveAction::Start { port } => {
    let port = port.unwrap_or(7835);
    
    if alias == "localhost" {
        // Local execution (existing code)
        let config = HttpDaemonConfig::new("rbee-hive", &format!("http://localhost:{}", port))
            .with_args(vec![/* ... */]);
        daemon_lifecycle::start_daemon(config).await?;
    } else {
        // Load hive config
        let hive_config = load_hive_config(&alias)?;
        
        // Remote execution via SSH
        let daemon_config = HttpDaemonConfig::new("rbee-hive", &format!("http://{}:{}", hive_config.hostname, port))
            .with_args(vec![/* ... */]);
        
        daemon_lifecycle::start_remote_daemon(hive_config, daemon_config).await?;
    }
    Ok(())
}
```

**Pros:**
- ✅ 1-2 days implementation
- ✅ Reuses system SSH config, keys, agent
- ✅ No new Rust dependencies
- ✅ Works immediately on Unix systems
- ✅ Minimal code (~200 LOC total)

**Cons:**
- ❌ Platform-dependent (requires ssh binary)
- ❌ Error handling via stderr parsing
- ❌ No connection pooling (reconnects each operation)

---

### ⭐ Option 2: russh Library (Best Long-Term)

**Effort:** 4-6 hours  
**Reliability:** ⭐⭐⭐⭐⭐ (pure Rust, testable)

#### Dependencies
```toml
# bin/99_shared_crates/daemon-lifecycle/Cargo.toml
[dependencies]
russh = "0.44"
russh-keys = "0.44"
async-trait = "0.1"
```

#### Implementation
```rust
// bin/99_shared_crates/daemon-lifecycle/src/remote/russh_executor.rs
use russh::*;
use russh_keys::*;

pub struct RusshExecutor {
    session: Option<client::Handle<RusshHandler>>,
    hostname: String,
    user: String,
    port: u16,
}

impl RusshExecutor {
    pub async fn connect(hostname: String, user: String, port: u16) -> Result<Self> {
        let config = client::Config::default();
        let session = client::connect(
            Arc::new(config),
            (hostname.as_str(), port),
            RusshHandler,
        ).await?;
        
        // Authenticate with SSH agent
        session.authenticate_publickey_auto(&user, None).await?;
        
        Ok(Self { session: Some(session), hostname, user, port })
    }
    
    pub async fn exec(&mut self, command: &str) -> Result<(String, String, i32)> {
        let session = self.session.as_mut().ok_or_else(|| anyhow::anyhow!("Not connected"))?;
        let mut channel = session.channel_open_session().await?;
        
        channel.exec(true, command).await?;
        
        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut exit_code = 0;
        
        loop {
            match channel.wait().await? {
                ChannelMsg::Data { ref data } => {
                    stdout.push_str(&String::from_utf8_lossy(data));
                }
                ChannelMsg::ExtendedData { ref data, ext } if ext == 1 => {
                    stderr.push_str(&String::from_utf8_lossy(data));
                }
                ChannelMsg::ExitStatus { exit_status } => {
                    exit_code = exit_status;
                }
                ChannelMsg::Eof => break,
                _ => {}
            }
        }
        
        Ok((stdout, stderr, exit_code))
    }
}

struct RusshHandler;

#[async_trait::async_trait]
impl client::Handler for RusshHandler {
    type Error = russh::Error;
    
    async fn check_server_key(&mut self, _key: &key::PublicKey) -> Result<bool, Self::Error> {
        // Accept all for now (like ssh -o StrictHostKeyChecking=no)
        Ok(true)
    }
}
```

**Pros:**
- ✅ Cross-platform (Windows, Linux, macOS)
- ✅ Structured errors (no stderr parsing)
- ✅ Connection pooling possible
- ✅ Pure Rust, easy testing with mocks
- ✅ No external dependencies

**Cons:**
- ❌ 4-6 hours implementation
- ❌ Host key verification complexity
- ❌ Need to handle SSH agent properly

---

### ⭐ Option 3: Trait-Based Pluggable Backend

**Effort:** 2-3 days  
**Reliability:** ⭐⭐⭐⭐⭐ (flexible, testable)

#### Abstraction
```rust
// bin/99_shared_crates/daemon-lifecycle/src/remote/executor.rs
#[async_trait]
pub trait RemoteExecutor: Send + Sync {
    async fn exec(&self, command: &str) -> Result<(String, String, i32)>;
    async fn copy_file(&self, local: &str, remote: &str) -> Result<()>;
    async fn get_pid(&self, process_name: &str) -> Result<Option<u32>>;
}

// Shell implementation
pub struct ShellSshExecutor { /* ... */ }

#[async_trait]
impl RemoteExecutor for ShellSshExecutor {
    async fn exec(&self, command: &str) -> Result<(String, String, i32)> {
        // Use tokio::process::Command
    }
    
    async fn copy_file(&self, local: &str, remote: &str) -> Result<()> {
        // Use scp command
    }
}

// Russh implementation
pub struct RusshExecutor { /* ... */ }

#[async_trait]
impl RemoteExecutor for RusshExecutor {
    async fn exec(&self, command: &str) -> Result<(String, String, i32)> {
        // Use russh
    }
}

// Mock for testing
pub struct MockExecutor {
    responses: Vec<(String, String, i32)>,
}

#[async_trait]
impl RemoteExecutor for MockExecutor {
    async fn exec(&self, command: &str) -> Result<(String, String, i32)> {
        Ok(self.responses[0].clone())
    }
}
```

#### Usage
```rust
pub async fn start_remote_daemon<E: RemoteExecutor>(
    executor: &E,
    daemon_config: HttpDaemonConfig,
) -> Result<u32> {
    // Same logic, works with any executor
    let (stdout, stderr, exit_code) = executor.exec("which rbee-hive").await?;
    // ... rest of logic
}

// In handler
let executor = ShellSshExecutor::new(/* ... */);
start_remote_daemon(&executor, config).await?;

// In tests
let mock = MockExecutor::new(vec![("rbee-hive running", "", 0)]);
start_remote_daemon(&mock, config).await?;
```

**Pros:**
- ✅ Easy testing with mocks
- ✅ Can swap backends without code changes
- ✅ Gradual migration (start with shell, migrate to russh)
- ✅ Best for long-term maintenance

**Cons:**
- ❌ Abstraction overhead
- ❌ More code to maintain (~400 LOC)
- ❌ Longer implementation time

---

### Option 4: HTTP Agent Architecture (Future-Proof)

**Effort:** 2-3 weeks  
**Reliability:** ⭐⭐⭐⭐⭐ (modern, scalable)

**Pattern:** Deploy rbee-hive as HTTP service, no SSH needed

```rust
// Remote hive runs as daemon with HTTP API
// Queen communicates via HTTP instead of SSH

// Start remote hive
POST http://remote-hive:7835/v1/lifecycle/start
{
    "daemon": "rbee-hive",
    "args": ["--port", "7835"]
}

// No SSH needed!
```

**Pros:**
- ✅ Firewall-friendly (HTTP/HTTPS)
- ✅ Modern architecture
- ✅ Easy testing
- ✅ TLS support
- ✅ No SSH complexity

**Cons:**
- ❌ Requires pre-installed hive daemon
- ❌ Authentication/authorization needed
- ❌ 2-3 weeks implementation
- ❌ More infrastructure

---

### Option 5: Minimal Scope (Read-Only Operations)

**Effort:** 1 day  
**Reliability:** ⭐⭐⭐ (limited but safe)

**Implement SSH only for:**
- ✅ `HiveStatus` - Check remote health
- ✅ `HiveGet` - Get remote info
- ❌ No start/stop/install/uninstall

```rust
// Only wrap status check with SSH
HiveAction::Status { alias } => {
    if alias == "localhost" {
        check_daemon_health("http://localhost:7835").await
    } else {
        let hive = load_hive_config(&alias)?;
        check_remote_daemon_health(&hive).await
    }
}

async fn check_remote_daemon_health(hive: &HiveConfig) -> Result<bool> {
    // Just HTTP health check over network
    let health_url = format!("http://{}:{}/health", hive.hostname, hive.hive_port);
    let client = reqwest::Client::new();
    let response = client.get(&health_url).send().await?;
    Ok(response.status().is_success())
}
```

**Pros:**
- ✅ 1 day implementation
- ✅ Low risk (read-only)
- ✅ No daemon control complexity
- ✅ Just HTTP checks

**Cons:**
- ❌ Incomplete (no lifecycle management)
- ❌ Users still need to start hives manually

---

## Recommended Implementation Plan

### Phase 1: Quick Win (Week 1)
**Implement Option 1: Shell SSH Wrapper**

1. **Day 1:** Core SSH executor (~4 hours)
   - `ShellSshExecutor::exec(command)` 
   - `ShellSshExecutor::get_pid(process_name)`
   - Basic error handling

2. **Day 2:** Remote lifecycle operations (~6 hours)
   - `start_remote_daemon()`
   - `stop_remote_daemon()`
   - `check_remote_daemon_status()`

3. **Day 3:** Handler integration (~4 hours)
   - Update `hive.rs` handlers with if/else pattern
   - Load hive config from `~/.config/rbee/hives.conf`
   - Add hive config loading utility

4. **Day 4:** Testing (~4 hours)
   - Test with real SSH setup
   - Error scenarios (connection failures, etc.)
   - Documentation

**Deliverable:** Working remote hive lifecycle via SSH

---

### Phase 2: Production Hardening (Week 2)
**Migrate to Option 2: russh Library**

1. Add russh dependencies
2. Implement `RusshExecutor`
3. Swap executor in remote operations
4. Add connection pooling
5. Better error handling

**Deliverable:** Production-ready pure Rust SSH

---

### Phase 3: Future Enhancement (Month 2)
**Evaluate Option 4: HTTP Agent**

- Deploy rbee-hive as systemd service
- Add authentication layer
- Gradual migration from SSH to HTTP

**Deliverable:** Modern agent-based architecture

---

## File Structure

```
bin/99_shared_crates/daemon-lifecycle/
├── src/
│   ├── lib.rs                  # Export remote module
│   ├── remote/
│   │   ├── mod.rs              # Re-exports
│   │   ├── shell_ssh.rs        # Shell SSH executor (Phase 1)
│   │   ├── russh_executor.rs   # Pure Rust SSH (Phase 2)
│   │   ├── executor.rs         # Trait abstraction (Phase 2)
│   │   └── lifecycle.rs        # start_remote_daemon(), etc.
│   └── ... (existing modules)
```

```
bin/00_rbee_keeper/src/
├── handlers/
│   ├── hive.rs                 # Update with if alias == "localhost"
│   └── ...
├── hive_config.rs              # NEW: Load ~/.config/rbee/hives.conf
```

---

## Example Usage Flow

```bash
# 1. Configure remote hive
cat >> ~/.config/rbee/hives.conf << EOF
[[hive]]
alias = "workstation"
hostname = "192.168.1.100"
ssh_port = 22
ssh_user = "vince"
hive_port = 7835
EOF

# 2. Start remote hive (uses SSH automatically)
rbee hive start --host workstation

# Behind the scenes:
# - Loads hive config from ~/.config/rbee/hives.conf
# - Detects alias != "localhost"
# - Executes: ssh vince@192.168.1.100 'rbee-hive --port 7835 &'
# - Polls health: http://192.168.1.100:7835/health
# - Reports success

# 3. Check status
rbee hive status --host workstation

# 4. Stop remote hive
rbee hive stop --host workstation
```

---

## Testing Strategy

### Unit Tests
```rust
#[tokio::test]
async fn test_shell_ssh_exec() {
    let executor = ShellSshExecutor::new("localhost", "testuser", 22);
    let (stdout, stderr, code) = executor.exec("echo hello").await.unwrap();
    assert_eq!(stdout.trim(), "hello");
    assert_eq!(code, 0);
}
```

### Integration Tests
```rust
#[tokio::test]
async fn test_start_remote_hive() {
    let hive_config = HiveConfig {
        alias: "test-hive".to_string(),
        hostname: "localhost".to_string(),
        ssh_port: 22,
        ssh_user: "testuser".to_string(),
        hive_port: 8888,
    };
    
    let config = HttpDaemonConfig::new("rbee-hive", "http://localhost:8888");
    let pid = start_remote_daemon(hive_config, config).await.unwrap();
    assert!(pid > 0);
}
```

### Mock Tests
```rust
#[tokio::test]
async fn test_with_mock_executor() {
    let mock = MockExecutor::new(vec![
        ("/usr/local/bin/rbee-hive", "", 0),  // which rbee-hive
        ("", "", 0),                           // start command
        ("12345", "", 0),                      // pgrep
    ]);
    
    let result = start_remote_daemon(&mock, config).await;
    assert!(result.is_ok());
}
```

---

## Security Considerations

### SSH Key Authentication
- ✅ Use SSH agent (no password prompts)
- ✅ Support `~/.ssh/config` settings
- ✅ Honor `IdentityFile` configuration

### Host Key Verification
```rust
// Phase 1 (Shell SSH): Automatic via system SSH
// Phase 2 (russh): Implement known_hosts checking

async fn verify_host_key(hostname: &str, key: &PublicKey) -> Result<bool> {
    // Read ~/.ssh/known_hosts
    // Compare fingerprints
    // Prompt user if unknown
}
```

### Credential Management
- ❌ Never store passwords in hives.conf
- ✅ Always use SSH keys
- ✅ Support SSH agent forwarding

---

## Migration from Current State

### Current Code
```rust
// BEFORE: All operations ignore alias
HiveAction::Start { port } => {
    let config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835");
    daemon_lifecycle::start_daemon(config).await?;
}
```

### After Phase 1
```rust
// AFTER: Detects alias and routes accordingly
HiveAction::Start { port } => {
    if alias == "localhost" {
        // Local (existing code)
        let config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835");
        daemon_lifecycle::start_daemon(config).await?;
    } else {
        // Remote (new code)
        let hive = load_hive_config(&alias)?;
        let config = HttpDaemonConfig::new("rbee-hive", &format!("http://{}:{}", hive.hostname, port));
        daemon_lifecycle::start_remote_daemon(hive, config).await?;
    }
}
```

---

## Decision Matrix

| Criteria | Option 1 (Shell) | Option 2 (russh) | Option 3 (Trait) | Option 4 (HTTP) | Option 5 (Read-Only) |
|----------|-----------------|-----------------|-----------------|-----------------|---------------------|
| Time to Ship | ⭐⭐⭐⭐⭐ (2 days) | ⭐⭐⭐⭐ (1 week) | ⭐⭐⭐ (2 weeks) | ⭐ (3 weeks) | ⭐⭐⭐⭐⭐ (1 day) |
| Reliability | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Cross-Platform | ⭐⭐ (Unix) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Testability | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Maintenance | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Security | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Conclusion

**RECOMMENDATION: Start with Option 1 (Shell SSH Wrapper)**

**Rationale:**
1. ✅ **Fast delivery** - 2 days to working remote operations
2. ✅ **Proven pattern** - Already worked before TEAM-284 removal
3. ✅ **Low risk** - Minimal code changes
4. ✅ **Incremental** - Can migrate to russh later without breaking changes

**Timeline:**
- **Week 1:** Implement Option 1 (Shell SSH)
- **Week 2:** Harden and test
- **Month 2:** Evaluate migration to Option 2 (russh) or Option 4 (HTTP Agent)

**Next Steps:**
1. Review and approve this plan
2. Create implementation tickets
3. Start with `ShellSshExecutor` implementation
4. Update `hive.rs` handlers with alias detection
5. Test with real remote hive

---

**Document Version:** 1.0  
**Last Updated:** Oct 27, 2025  
**Author:** TEAM-330
