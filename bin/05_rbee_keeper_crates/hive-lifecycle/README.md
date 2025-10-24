# hive-lifecycle

**TEAM-290:** Remote hive lifecycle management for rbee-keeper

## Purpose

Manages rbee-hive instances remotely via SSH:
- Install/uninstall hives on remote machines
- Start/stop hives remotely
- Health checks
- Uses host SSH config (no custom format)

## Architecture

```
rbee-keeper
  └─> hive-lifecycle (this crate)
       ├─> ssh.rs (SSH client using host config)
       └─> operations.rs (install/uninstall/start/stop)
            └─> daemon-lifecycle (lifecycle patterns)
```

## Design Philosophy

**Combines SSH + daemon-lifecycle:**
- SSH for remote operations
- daemon-lifecycle for lifecycle patterns
- Piggybacks on host ~/.ssh/config
- No custom SSH config format

## Usage

### Install Hive Locally

```rust
use hive_lifecycle::install_hive;

// Install hive locally (auto-detects binary, installs to ~/.local/bin)
install_hive("localhost", None, None).await?;
```

### Install Hive Remotely

```rust
use hive_lifecycle::install_hive;

// Install hive on remote machine via SSH
install_hive("gpu-server", Some("./rbee-hive".to_string()), Some("/usr/local/bin".to_string())).await?;
```

### Start Hive Remotely

```rust
use hive_lifecycle::start_hive;

// Start hive on remote machine
start_hive("gpu-server", "/usr/local/bin", 9000).await?;
```

### Check Hive Status

```rust
use hive_lifecycle::hive_status;

let status = hive_status("gpu-server").await?;
println!("{}", status);
```

## SSH Configuration

### User Setup

1. **Add host to ~/.ssh/config:**
```ssh
Host gpu-server
  HostName 192.168.1.100
  User ubuntu
  IdentityFile ~/.ssh/id_rsa
  Port 22
```

2. **Copy public key to remote:**
```bash
ssh-copy-id gpu-server
```

3. **Use hive-lifecycle:**
```rust
install_hive("gpu-server", "./rbee-hive", "/usr/local/bin").await?;
```

## API

### SSH Client

```rust
use hive_lifecycle::SshClient;

let client = SshClient::connect("gpu-server").await?;

// Execute commands
let output = client.execute("ls -la").await?;

// Upload files
client.upload_file("./local-file", "/remote/path").await?;

// Download files
client.download_file("/remote/path", "./local-file").await?;

// Check file existence
let exists = client.file_exists("/remote/path").await?;
```

### Operations

```rust
use hive_lifecycle::*;

// Install
install_hive("gpu-server", "./rbee-hive", "/usr/local/bin").await?;

// Uninstall
uninstall_hive("gpu-server", "/usr/local/bin").await?;

// Start
start_hive("gpu-server", "/usr/local/bin", 9000).await?;

// Stop
stop_hive("gpu-server").await?;

// Status
let running = is_hive_running("gpu-server").await?;
let status = hive_status("gpu-server").await?;
```

## Dependencies

- **ssh2:** SSH client library
- **daemon-lifecycle:** Lifecycle patterns (from shared crates)
- **observability-narration-core:** Narration for observability

## Location

`bin/05_rbee_keeper_crates/hive-lifecycle/`

Part of rbee-keeper crates (not queen crates).

## Related Crates

- **queen-lifecycle:** Manages queen-rbee lifecycle
- **daemon-lifecycle:** Shared lifecycle patterns
- **rbee-keeper:** CLI that uses this crate
