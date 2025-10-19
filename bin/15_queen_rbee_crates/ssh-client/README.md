# queen-rbee-remote

**Status:** 🚧 MIGRATION TARGET (TEAM-135)  
**Purpose:** Remote hive management via SSH (outbound SSH connections)  
**Renamed from:** `ssh-client` → `remote` (more accurate)  
**Source:** `bin/old.queen-rbee/src/ssh.rs` (76 LOC) + `preflight/` (138 LOC)

---

## 🎯 CORE RESPONSIBILITY

**Manage outbound SSH connections to remote rbee-hive instances**

This crate handles queen-rbee's role as an **SSH client** (initiating connections TO remote machines):

```
queen-rbee (SSH client) --SSH--> remote machine (SSH server) --spawns--> rbee-hive daemon
```

**This crate does:**
- ✅ Test SSH connectivity to remote machines
- ✅ Execute commands on remote machines (spawn rbee-hive, check status, shutdown)
- ✅ Preflight validation (check if rbee-hive binary exists, check dependencies)
- ✅ Handle SSH authentication (keys, passwords)

**This crate does NOT:**
- ❌ Receive SSH connections (queen-rbee is NOT an SSH server)
- ❌ Provide SSH access to queen-rbee (no inbound SSH)
- ❌ Handle HTTP communication (that's `http-server` crate)

---

## 📛 NAMING CLARIFICATION

### Why "remote" instead of "ssh-client"?

**Your concern is valid:** "ssh-client" is confusing because:
- ❌ Sounds like it receives SSH connections (but queen-rbee is the SSH **initiator**)
- ❌ Doesn't convey the purpose (managing remote hives)
- ❌ Could be confused with SSH server functionality

**Better name: `remote`**
- ✅ Clear purpose: remote hive management
- ✅ Matches the domain: "remote hives" vs "local hives"
- ✅ Avoids SSH terminology confusion
- ✅ Could include other remote protocols in future (if needed)

**Alternative names considered:**
- `remote-hive` - Too specific (also handles preflight, not just hives)
- `ssh-outbound` - Still SSH-focused
- `network-hive` - Confusing (network != remote)
- `remote` - ✅ **BEST: Clear, concise, purpose-driven**

---

## 🔌 SSH DIRECTION CLARIFICATION

**queen-rbee is an SSH CLIENT (outbound connections):**

```
┌─────────────────────────────────────────────────────┐
│ queen-rbee (THIS MACHINE)                           │
│  - SSH CLIENT (initiates connections)               │
│  - Connects TO remote machines                      │
│  - Spawns remote rbee-hive daemons                  │
└──────────────────┬──────────────────────────────────┘
                   │ SSH (outbound)
                   ▼
┌─────────────────────────────────────────────────────┐
│ Remote Machine (SSH SERVER)                         │
│  - Receives SSH connections                         │
│  - Executes commands from queen-rbee                │
│  - Runs rbee-hive daemon                            │
└─────────────────────────────────────────────────────┘
```

**queen-rbee does NOT:**
- ❌ Accept inbound SSH connections
- ❌ Run an SSH server
- ❌ Provide SSH access to itself

**Access to queen-rbee is via:**
- ✅ HTTP API (for rbee-keeper CLI)
- ✅ Local process (if running on same machine)

---

## 🌊 SSE STREAMING ARCHITECTURE

### Inference Request Flow with SSE

**Your observation about SSE is correct - it's more than just tokens:**

```rust
// 1. rbee-keeper submits task
POST http://localhost:8080/v2/tasks
{
  "node": "gpu-0",
  "model": "llama-7b",
  "prompt": "Hello"
}

// 2. queen-rbee responds with SSE stream URL
HTTP 200 OK
{
  "task_id": "task-123",
  "stream_url": "/v2/tasks/task-123/stream"  // SSE endpoint
}

// 3. rbee-keeper connects to SSE stream
GET http://localhost:8080/v2/tasks/task-123/stream
Accept: text/event-stream

// 4. queen-rbee streams BOTH tokens AND narration
event: narration
data: {"level": "info", "message": "🔍 Looking up hive for node gpu-0"}

event: narration
data: {"level": "info", "message": "🚀 Starting rbee-hive on remote machine"}

event: narration
data: {"level": "info", "message": "⏳ Waiting for worker to load model"}

event: token
data: {"token": "Hello", "token_id": 123}

event: token
data: {"token": " world", "token_id": 456}

event: narration
data: {"level": "info", "message": "✅ Inference complete"}

event: done
data: {"tokens_generated": 20, "duration_ms": 1234}
```

**SSE Stream Contents:**
- ✅ **Tokens** - Actual inference output (from llm-worker)
- ✅ **Narration** - Progress updates, status messages (from queen-rbee orchestration)
- ✅ **Errors** - Failure messages if something goes wrong
- ✅ **Metrics** - Performance data, timing information

**Why this matters:**
- User sees progress in real-time (not just waiting)
- Debugging is easier (see where it fails)
- Better UX (know what's happening)

---

## 🏗️ CRATE STRUCTURE

### Files to Migrate

**From `old.queen-rbee/src/`:**
- `ssh.rs` (76 LOC) - SSH connection and command execution
- `preflight/rbee_hive.rs` (76 LOC) - Hive health checks
- `preflight/ssh.rs` (60 LOC) - SSH validation (stub)
- `preflight/mod.rs` (2 LOC)

**Total:** ~214 LOC

### Public API

```rust
/// SSH connection management
pub struct SshClient {
    host: String,
    port: u16,
    user: String,
    key_path: Option<PathBuf>,
}

impl SshClient {
    pub async fn connect(config: SshConfig) -> Result<Self>;
    
    pub async fn test_connection(&self) -> Result<bool>;
    
    pub async fn execute_command(&self, command: &str) -> Result<CommandOutput>;
    
    pub async fn execute_detached(&self, command: &str) -> Result<()>;
}

pub struct CommandOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

/// Preflight checks before spawning remote hive
pub async fn preflight_check_hive(ssh: &SshClient) -> Result<PreflightResult>;

pub struct PreflightResult {
    pub binary_exists: bool,
    pub dependencies_ok: bool,
    pub port_available: bool,
    pub errors: Vec<String>,
}
```

---

## 🔗 DEPENDENCIES

### External Dependencies

```toml
[dependencies]
tokio = { version = "1", features = ["process"] }
anyhow = "1.0"
tracing = "0.1"
serde = { version = "1.0", features = ["derive"] }
```

**Note:** Uses system `ssh` binary (not `ssh2` crate) because:
- ✅ Respects user's SSH config (~/.ssh/config)
- ✅ Uses user's SSH keys automatically
- ✅ Simpler (no Rust SSH library complexity)
- ⚠️ Requires `ssh` binary installed on system

**Future:** Could switch to `ssh2` crate for pure Rust implementation.

### Internal Dependencies

```toml
[dependencies]
# None - this is a leaf crate
```

---

## 🚨 SECURITY CONSIDERATIONS

### Command Injection Vulnerability (CRITICAL)

**From TEAM-109 audit:**

```rust
// ❌ VULNERABLE (current code)
pub async fn execute_remote_command(host: &str, command: &str) -> Result<()> {
    Command::new("ssh")
        .arg(format!("{}@{}", user, host))
        .arg(command)  // 🔴 INJECTION RISK!
        .spawn()?;
}
```

**Attack vector:**
```rust
// Malicious input
let command = "echo hello; rm -rf /";
execute_remote_command("host", command).await?;
// Executes: ssh user@host "echo hello; rm -rf /"
```

**Fix:**
```rust
// ✅ SAFE (use shell escaping)
pub async fn execute_remote_command(host: &str, command: &str) -> Result<()> {
    Command::new("ssh")
        .arg(format!("{}@{}", user, host))
        .arg("--")  // Prevent option injection
        .arg(command)  // Still needs shell escaping!
        .spawn()?;
}
```

**Better fix:**
```rust
// ✅ SAFER (use ssh2 crate with proper API)
use ssh2::Session;

pub async fn execute_remote_command(host: &str, command: &str) -> Result<()> {
    let session = Session::connect(host)?;
    let mut channel = session.channel_session()?;
    channel.exec(command)?;  // ssh2 handles escaping
}
```

---

## 📊 MIGRATION STATUS

**Source:** `bin/old.queen-rbee/src/ssh.rs` + `preflight/` (214 LOC)  
**Target:** `bin/15_queen_rbee_crates/remote/` (~250 LOC with improvements)

**Changes:**
- ✅ Rename crate from `ssh-client` to `remote`
- ✅ Fix command injection vulnerability
- ✅ Add proper error types
- ✅ Add preflight checks
- ✅ Add tests (mock SSH for testing)

---

## ✅ ACCEPTANCE CRITERIA

### Compilation

```bash
cd bin/15_queen_rbee_crates/remote
cargo check
cargo clippy -- -D warnings
cargo test
```

### Functionality

- [ ] Can test SSH connectivity to remote host
- [ ] Can execute commands on remote host
- [ ] Can spawn detached processes (rbee-hive daemon)
- [ ] Preflight checks work (binary exists, dependencies OK)
- [ ] Handles SSH authentication (keys, passwords)
- [ ] No command injection vulnerabilities

### Security

- [ ] All commands properly escaped
- [ ] No shell injection possible
- [ ] SSH keys handled securely
- [ ] Audit logging for all SSH operations

---

## 📚 REFERENCES

### Planning Documents

- `.plan/.archive-130BC-134/TEAM_132_queen-rbee_INVESTIGATION_REPORT.md`
  - Lines 200-263: HTTP server crate (SSE streaming)
  - Lines 266-299: Orchestrator crate (inference flow)

### Source Code

- `bin/old.queen-rbee/src/ssh.rs` - Original SSH implementation
- `bin/old.queen-rbee/src/preflight/` - Preflight checks

### Security Audits

- TEAM-109 audit (line 1 of ssh.rs): Command injection vulnerability

---

## 🎯 SSE STREAMING IMPLEMENTATION

**This crate does NOT handle SSE streaming** - that's in `http-server` crate.

**But it enables SSE by:**
- Starting remote hives (which spawn workers)
- Checking hive health (so queen knows when to route)
- Executing commands (shutdown, status checks)

**SSE flow:**
```
rbee-keeper → queen-rbee HTTP API → SSE stream (http-server crate)
                ↓
            remote crate (SSH to start hive)
                ↓
            rbee-hive (spawns worker)
                ↓
            llm-worker (generates tokens)
                ↓
            queen-rbee (relays tokens + narration via SSE)
                ↓
            rbee-keeper (displays to user)
```

---

**Migration Status:** 🚧 NOT STARTED  
**Priority:** HIGH (blocking hive lifecycle)  
**Estimated Effort:** 1-2 days  
**Security:** CRITICAL (fix command injection)

**Next Steps:**
1. Rename crate from `ssh-client` to `remote`
2. Migrate `ssh.rs` (76 LOC)
3. Migrate `preflight/` (138 LOC)
4. Fix command injection vulnerability
5. Add tests (mock SSH)
6. Add audit logging for all SSH operations
