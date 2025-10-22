# RUSSH Migration Guide

**Status:** 📋 PLANNED  
**Priority:** Medium  
**Estimated Effort:** 4-6 hours  
**Team:** TBD

---

## Executive Summary

Migrate SSH operations from shell commands (`ssh`/`scp`) to pure Rust using the `russh` library.

**Current State:** Using `tokio::process::Command` to shell out to system `ssh`/`scp`  
**Target State:** Using `russh` library for all SSH operations  
**Reason:** Better error handling, cross-platform support, connection pooling, easier testing

---

## Current Implementation

### Files Using Shell SSH Commands

1. **`hive-lifecycle/src/ssh_helper.rs`**
   - `ssh_exec()` - Executes SSH commands via shell
   - `scp_copy()` - Copies files via shell SCP

2. **`hive-lifecycle/src/start.rs`**
   - Uses `ssh_exec()` to start remote hive daemon

3. **`hive-lifecycle/src/stop.rs`**
   - Uses `ssh_exec()` to stop remote hive daemon

4. **`hive-lifecycle/src/install.rs`**
   - Uses `scp_copy()` to copy binary
   - Uses `ssh_exec()` for chmod and verification

### Current Dependencies

```toml
# None - uses system ssh/scp
```

---

## Migration Plan

### Phase 1: Add Dependencies (30 min)

**File:** `bin/15_queen_rbee_crates/hive-lifecycle/Cargo.toml`

```toml
[dependencies]
# ... existing dependencies ...

# SSH library
russh = "0.44"
russh-keys = "0.44"
russh-sftp = "2.0"  # For file transfers

# Async runtime (already have tokio)
tokio = { version = "1.0", features = ["full"] }
```

**Verification:**
```bash
cargo check -p queen-rbee-hive-lifecycle
```

---

### Phase 2: Create SSH Client Module (2 hours)

**File:** `bin/15_queen_rbee_crates/hive-lifecycle/src/russh_client.rs`

#### 2.1 Basic Connection

```rust
use anyhow::Result;
use russh::*;
use russh_keys::*;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// SSH client for remote hive operations
pub struct RbeeSSHClient {
    session: russh::client::Handle<RbeeSSHHandler>,
}

/// SSH event handler
struct RbeeSSHHandler;

#[async_trait::async_trait]
impl russh::client::Handler for RbeeSSHHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &russh_keys::key::PublicKey,
    ) -> Result<bool, Self::Error> {
        // TODO: Implement proper host key verification
        // For now, accept all (like ssh -o StrictHostKeyChecking=no)
        Ok(true)
    }
}
```

#### 2.2 Connection Management

```rust
impl RbeeSSHClient {
    /// Connect to remote host
    pub async fn connect(
        host: &str,
        port: u16,
        user: &str,
    ) -> Result<Self> {
        let config = russh::client::Config::default();
        let mut session = russh::client::connect(
            Arc::new(config),
            (host, port),
            RbeeSSHHandler,
        )
        .await?;

        // Authenticate using SSH agent
        let auth_result = session
            .authenticate_publickey_auto(user, None)
            .await?;

        if !auth_result {
            return Err(anyhow::anyhow!("SSH authentication failed"));
        }

        Ok(Self { session })
    }

    /// Execute command on remote host
    pub async fn exec(&mut self, command: &str) -> Result<(String, String, i32)> {
        let mut channel = self.session.channel_open_session().await?;
        
        channel.exec(true, command).await?;

        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut code = None;

        loop {
            let msg = channel.wait().await?;
            match msg {
                russh::ChannelMsg::Data { ref data } => {
                    stdout.push_str(&String::from_utf8_lossy(data));
                }
                russh::ChannelMsg::ExtendedData { ref data, ext } => {
                    if ext == 1 {
                        stderr.push_str(&String::from_utf8_lossy(data));
                    }
                }
                russh::ChannelMsg::ExitStatus { exit_status } => {
                    code = Some(exit_status);
                }
                russh::ChannelMsg::Eof => {
                    break;
                }
                _ => {}
            }
        }

        let exit_code = code.unwrap_or(-1);
        Ok((stdout, stderr, exit_code))
    }

    /// Close connection
    pub async fn close(self) -> Result<()> {
        self.session
            .disconnect(russh::Disconnect::ByApplication, "", "")
            .await?;
        Ok(())
    }
}
```

#### 2.3 File Transfer (SFTP)

```rust
use russh_sftp::client::SftpSession;

impl RbeeSSHClient {
    /// Copy file to remote host via SFTP
    pub async fn copy_file(
        &mut self,
        local_path: &str,
        remote_path: &str,
    ) -> Result<()> {
        let channel = self.session.channel_open_session().await?;
        channel.request_subsystem(true, "sftp").await?;
        
        let sftp = SftpSession::new(channel.into_stream()).await?;
        
        // Read local file
        let local_data = tokio::fs::read(local_path).await?;
        
        // Write to remote
        let mut remote_file = sftp
            .create(remote_path)
            .await?;
        
        remote_file.write_all(&local_data).await?;
        remote_file.sync_all().await?;
        
        Ok(())
    }
}
```

---

### Phase 3: Update SSH Helper (1 hour)

**File:** `bin/15_queen_rbee_crates/hive-lifecycle/src/ssh_helper.rs`

#### 3.1 Replace `ssh_exec()`

**Before:**
```rust
pub async fn ssh_exec(
    hive_config: &HiveEntry,
    command: &str,
    job_id: &str,
    action: &'static str,
    description: &str,
) -> Result<String> {
    let output = tokio::process::Command::new("ssh")
        .arg("-p")
        .arg(hive_config.ssh_port.to_string())
        .arg(format!("{}@{}", hive_config.ssh_user, hive_config.hostname))
        .arg(command)
        .output()
        .await?;
    // ...
}
```

**After:**
```rust
use crate::russh_client::RbeeSSHClient;

pub async fn ssh_exec(
    hive_config: &HiveEntry,
    command: &str,
    job_id: &str,
    action: &'static str,
    description: &str,
) -> Result<String> {
    NARRATE
        .action(action)
        .job_id(job_id)
        .context(description)
        .human("🔧 {}")
        .emit();

    // Connect
    let mut client = RbeeSSHClient::connect(
        &hive_config.hostname,
        hive_config.ssh_port,
        &hive_config.ssh_user,
    )
    .await?;

    // Execute
    let (stdout, stderr, exit_code) = client.exec(command).await?;

    // Close
    client.close().await?;

    if exit_code != 0 {
        NARRATE
            .action("ssh_err")
            .job_id(job_id)
            .context(action)
            .context(&stderr)
            .human("❌ {} failed: {}")
            .error_kind("ssh_failed")
            .emit();
        return Err(anyhow::anyhow!("SSH command failed: {}", stderr));
    }

    Ok(stdout)
}
```

#### 3.2 Replace `scp_copy()`

**Before:**
```rust
pub async fn scp_copy(
    hive_config: &HiveEntry,
    local_path: &str,
    remote_path: &str,
    job_id: &str,
) -> Result<()> {
    let output = tokio::process::Command::new("scp")
        .arg("-P")
        .arg(hive_config.ssh_port.to_string())
        .arg(local_path)
        .arg(&scp_target)
        .output()
        .await?;
    // ...
}
```

**After:**
```rust
pub async fn scp_copy(
    hive_config: &HiveEntry,
    local_path: &str,
    remote_path: &str,
    job_id: &str,
) -> Result<()> {
    let scp_target = format!("{}@{}:{}", hive_config.ssh_user, hive_config.hostname, remote_path);

    NARRATE
        .action("hive_scp")
        .job_id(job_id)
        .context(&scp_target)
        .human("📤 Copying to {}...")
        .emit();

    // Connect
    let mut client = RbeeSSHClient::connect(
        &hive_config.hostname,
        hive_config.ssh_port,
        &hive_config.ssh_user,
    )
    .await?;

    // Copy file
    client.copy_file(local_path, remote_path).await?;

    // Close
    client.close().await?;

    NARRATE
        .action("hive_scp")
        .job_id(job_id)
        .human("✅ File copied successfully")
        .emit();

    Ok(())
}
```

---

### Phase 4: Add Connection Pooling (Optional, 1 hour)

**File:** `bin/15_queen_rbee_crates/hive-lifecycle/src/ssh_pool.rs`

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// SSH connection pool for reusing connections
pub struct SSHPool {
    connections: Arc<Mutex<HashMap<String, RbeeSSHClient>>>,
}

impl SSHPool {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get or create connection
    pub async fn get_connection(
        &self,
        host: &str,
        port: u16,
        user: &str,
    ) -> Result<RbeeSSHClient> {
        let key = format!("{}@{}:{}", user, host, port);
        
        let mut pool = self.connections.lock().await;
        
        if let Some(client) = pool.get(&key) {
            // TODO: Check if connection is still alive
            // For now, just return it
            return Ok(client.clone());
        }

        // Create new connection
        let client = RbeeSSHClient::connect(host, port, user).await?;
        pool.insert(key, client.clone());
        
        Ok(client)
    }

    /// Close all connections
    pub async fn close_all(&self) -> Result<()> {
        let mut pool = self.connections.lock().await;
        for (_, client) in pool.drain() {
            client.close().await?;
        }
        Ok(())
    }
}
```

**Usage:**
```rust
// In hive-lifecycle/src/lib.rs
pub static SSH_POOL: once_cell::sync::Lazy<SSHPool> = 
    once_cell::sync::Lazy::new(|| SSHPool::new());

// In ssh_helper.rs
let client = SSH_POOL.get_connection(
    &hive_config.hostname,
    hive_config.ssh_port,
    &hive_config.ssh_user,
).await?;
```

---

### Phase 5: Update Module Exports (15 min)

**File:** `bin/15_queen_rbee_crates/hive-lifecycle/src/lib.rs`

```rust
// Add new module
pub mod russh_client;

// Optional: Add connection pool
pub mod ssh_pool;
```

---

### Phase 6: Testing (1 hour)

#### 6.1 Unit Tests

**File:** `bin/15_queen_rbee_crates/hive-lifecycle/tests/russh_client_tests.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ssh_connection() {
        // Mock SSH server or use real test host
        let client = RbeeSSHClient::connect("localhost", 22, "testuser")
            .await
            .expect("Failed to connect");
        
        client.close().await.expect("Failed to close");
    }

    #[tokio::test]
    async fn test_ssh_exec() {
        let mut client = RbeeSSHClient::connect("localhost", 22, "testuser")
            .await
            .expect("Failed to connect");
        
        let (stdout, stderr, code) = client.exec("echo test").await.expect("Failed to exec");
        
        assert_eq!(code, 0);
        assert_eq!(stdout.trim(), "test");
        assert_eq!(stderr, "");
        
        client.close().await.expect("Failed to close");
    }

    #[tokio::test]
    async fn test_file_copy() {
        let mut client = RbeeSSHClient::connect("localhost", 22, "testuser")
            .await
            .expect("Failed to connect");
        
        // Create temp file
        let temp_file = "/tmp/test_russh.txt";
        tokio::fs::write(temp_file, "test content").await.unwrap();
        
        // Copy to remote
        client.copy_file(temp_file, "/tmp/test_russh_remote.txt")
            .await
            .expect("Failed to copy");
        
        // Verify
        let (stdout, _, _) = client.exec("cat /tmp/test_russh_remote.txt")
            .await
            .expect("Failed to read");
        
        assert_eq!(stdout.trim(), "test content");
        
        // Cleanup
        client.exec("rm /tmp/test_russh_remote.txt").await.ok();
        tokio::fs::remove_file(temp_file).await.ok();
        
        client.close().await.expect("Failed to close");
    }
}
```

#### 6.2 Integration Tests

**Test Plan:**
1. ✅ Import SSH config
2. ✅ Install hive on remote (uses SFTP)
3. ✅ Start hive on remote (uses SSH exec)
4. ✅ Check status (HTTP, no SSH)
5. ✅ Stop hive on remote (uses SSH exec)

**Commands:**
```bash
./rbee hive import-ssh
./rbee hive install -a workstation
./rbee hive start -a workstation
./rbee hive status -a workstation
./rbee hive stop -a workstation
```

---

## Benefits After Migration

### 1. Better Error Handling
```rust
// Before: Parse stderr from shell command
if !output.status.success() {
    let error = String::from_utf8_lossy(&output.stderr);
    // Hope the error message is useful
}

// After: Structured error types
match client.exec(command).await {
    Err(russh::Error::Disconnect) => {
        // Connection lost
    }
    Err(russh::Error::Auth) => {
        // Authentication failed
    }
    // ... specific error handling
}
```

### 2. Connection Pooling
```rust
// Before: New connection for every command
ssh_exec(...).await?;  // Connect, exec, disconnect
ssh_exec(...).await?;  // Connect, exec, disconnect
ssh_exec(...).await?;  // Connect, exec, disconnect

// After: Reuse connections
let client = pool.get_connection(...).await?;
client.exec(...).await?;
client.exec(...).await?;
client.exec(...).await?;
// Connection stays open
```

### 3. Cross-Platform
```rust
// Before: Requires system ssh/scp (not on Windows by default)
// After: Pure Rust, works everywhere
```

### 4. Easier Testing
```rust
// Before: Mock shell commands (hard)
// After: Mock SSH client (easy)

#[cfg(test)]
mod tests {
    struct MockSSHClient;
    
    impl SSHClient for MockSSHClient {
        async fn exec(&mut self, cmd: &str) -> Result<(String, String, i32)> {
            Ok(("mocked output".to_string(), "".to_string(), 0))
        }
    }
}
```

---

## Risks & Mitigation

### Risk 1: SSH Agent Support
**Problem:** `russh` SSH agent support might not work exactly like system SSH  
**Mitigation:** Test thoroughly with different key types (RSA, Ed25519)  
**Fallback:** Keep shell command implementation as backup

### Risk 2: Host Key Verification
**Problem:** Need to implement proper host key checking  
**Mitigation:** 
- Phase 1: Accept all (like `StrictHostKeyChecking=no`)
- Phase 2: Store known hosts in `~/.config/rbee/known_hosts`
- Phase 3: Verify against stored keys

### Risk 3: Performance
**Problem:** Connection overhead might be higher  
**Mitigation:** Use connection pooling (Phase 4)  
**Measurement:** Benchmark before/after

### Risk 4: Compatibility
**Problem:** Some SSH servers might not work with `russh`  
**Mitigation:** Test with OpenSSH, Dropbear, etc.  
**Fallback:** Keep shell commands as option

---

## Rollout Strategy

### Stage 1: Parallel Implementation
- Keep existing shell commands
- Add `russh` implementation alongside
- Feature flag to switch between them

```rust
#[cfg(feature = "russh")]
use crate::russh_client::*;

#[cfg(not(feature = "russh"))]
use crate::shell_ssh::*;
```

### Stage 2: Testing
- Test `russh` implementation thoroughly
- Compare behavior with shell commands
- Fix any issues

### Stage 3: Migration
- Make `russh` the default
- Keep shell commands as fallback
- Monitor for issues

### Stage 4: Cleanup
- Remove shell command implementation
- Remove feature flag
- Update documentation

---

## Acceptance Criteria

- [ ] All SSH operations use `russh` instead of shell commands
- [ ] All existing tests pass
- [ ] New unit tests for `russh_client` module
- [ ] Integration tests with real SSH server
- [ ] Connection pooling implemented (optional)
- [ ] Error handling improved
- [ ] Documentation updated
- [ ] Performance benchmarks show no regression

---

## Estimated Timeline

| Phase | Task | Time | Dependencies |
|-------|------|------|--------------|
| 1 | Add dependencies | 30 min | None |
| 2 | Create SSH client module | 2 hours | Phase 1 |
| 3 | Update SSH helper | 1 hour | Phase 2 |
| 4 | Connection pooling (optional) | 1 hour | Phase 3 |
| 5 | Update exports | 15 min | Phase 3 |
| 6 | Testing | 1 hour | Phase 5 |
| **Total** | **4-6 hours** | | |

---

## References

- **russh documentation:** https://docs.rs/russh/latest/russh/
- **russh examples:** https://github.com/warp-tech/russh/tree/main/russh/examples
- **russh-sftp:** https://docs.rs/russh-sftp/latest/russh_sftp/
- **Current implementation:** `bin/15_queen_rbee_crates/hive-lifecycle/src/ssh_helper.rs`

---

## Questions for Discussion

1. **Do we need connection pooling?** (Probably not for low-volume operations)
2. **How to handle host key verification?** (Accept all? Store known hosts?)
3. **Should we keep shell commands as fallback?** (Safer for initial rollout)
4. **Performance requirements?** (Is current speed acceptable?)

---

## Next Steps

1. Review this guide with the team
2. Assign to a team member
3. Create tracking issue
4. Schedule implementation sprint
5. Plan testing strategy

---

**Document Version:** 1.0  
**Last Updated:** Oct 22, 2025  
**Author:** Cascade AI  
**Reviewers:** TBD
