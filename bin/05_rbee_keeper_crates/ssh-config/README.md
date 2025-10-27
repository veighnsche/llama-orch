# ssh-config

**SSH Configuration Parser and Remote Execution Client**

A shared crate for SSH operations in the rbee ecosystem.

## Purpose

This crate provides two main components for working with SSH:

1. **Config Parser** - Parse `~/.ssh/config` to discover available SSH hosts
2. **SSH Client** - Execute commands and transfer files on remote hosts

## Features

### 1. SSH Config Parser

Parse `~/.ssh/config` to discover available remote hosts (hives) without hardcoding connection details.

**Benefits:**
- ✅ No hardcoded IPs or hostnames
- ✅ Uses existing SSH configuration
- ✅ Supports SSH key authentication
- ✅ Returns structured data for CLI and UI

**Example:**

```rust
use ssh_config::parse_ssh_config;
use std::path::Path;

let ssh_config_path = Path::new("/home/user/.ssh/config");
let targets = parse_ssh_config(&ssh_config_path)?;

for target in targets {
    println!("Host: {}", target.host);
    println!("  Hostname: {}", target.hostname);
    println!("  User: {}", target.user);
    println!("  Port: {}", target.port);
}
```

**SSH Config Format:**

```ssh
Host workstation
    HostName 192.168.1.100
    User vince
    Port 22
    IdentityFile ~/.ssh/id_rsa

Host gpu-server
    HostName 192.168.1.200
    User admin
    Port 2222
```

### 2. SSH Client

Execute commands and transfer files on remote hosts using SSH/SCP.

**Features:**
- ✅ Remote command execution
- ✅ File upload (SCP)
- ✅ File download (SCP)
- ✅ File existence checks
- ✅ Automatic narration for all operations
- ✅ Uses `~/.ssh/config` for connection details

**Example:**

```rust
use ssh_config::SshClient;

// Connect to remote host
let client = SshClient::connect("workstation").await?;

// Execute command
let output = client.execute("uname -a").await?;
println!("Remote OS: {}", output);

// Upload file
client.upload_file("./local-binary", "/usr/local/bin/remote-binary").await?;

// Download file
client.download_file("/var/log/app.log", "./app.log").await?;

// Check if file exists
if client.file_exists("/usr/local/bin/app").await? {
    println!("App is installed");
}
```

## Architecture

### Why This Crate Exists

Originally, SSH client code was duplicated in `hive-lifecycle`. This crate consolidates all SSH-related functionality into a single, reusable component.

**Before (Duplicated):**
```
hive-lifecycle/
  └── src/
      └── ssh.rs (SSH client)

ssh-config/
  └── src/
      └── lib.rs (Config parser only)
```

**After (Shared):**
```
ssh-config/
  └── src/
      ├── lib.rs (Config parser)
      └── client.rs (SSH client) ← SHARED
```

### Design Principles

1. **Piggyback on System SSH** - Uses `ssh` and `scp` commands, not custom SSH library
2. **Respect SSH Config** - Uses `~/.ssh/config` for all connection details
3. **Key-Based Auth Only** - No password prompts (`BatchMode=yes`)
4. **Narration Built-In** - All operations emit structured narration events
5. **Async by Default** - Uses `tokio::process` for non-blocking operations

## Usage in rbee Ecosystem

### hive-lifecycle

Uses `SshClient` for remote hive operations:

```rust
use ssh_config::SshClient;

// Install hive remotely
let client = SshClient::connect("workstation").await?;
client.upload_file("./rbee-hive", "$HOME/.local/bin/rbee-hive").await?;
client.execute("chmod +x $HOME/.local/bin/rbee-hive").await?;
```

### rbee-keeper

Uses both components:

```rust
use ssh_config::{parse_ssh_config, SshClient};

// List available hives
let targets = parse_ssh_config(&ssh_config_path)?;
for target in targets {
    println!("Available hive: {}", target.host);
}

// Connect to specific hive
let client = SshClient::connect("workstation").await?;
```

## API Reference

### Types

#### `SshTarget`

Represents an SSH host from `~/.ssh/config`:

```rust
pub struct SshTarget {
    pub host: String,              // SSH alias (e.g., "workstation")
    pub host_subtitle: Option<String>, // Optional subtitle
    pub hostname: String,          // IP or domain
    pub user: String,              // SSH username
    pub port: u16,                 // SSH port
    pub status: SshTargetStatus,   // Connection status
}
```

#### `SshTargetStatus`

```rust
pub enum SshTargetStatus {
    Online,   // Host is reachable
    Offline,  // Host is unreachable
    Unknown,  // Status not checked
}
```

#### `SshClient`

SSH client for remote operations:

```rust
pub struct SshClient {
    // Internal: SSH host alias
}
```

### Functions

#### Config Parser

```rust
pub fn parse_ssh_config(path: &Path) -> Result<Vec<SshTarget>>
```

Parse SSH config file and return list of hosts.

**Parameters:**
- `path` - Path to SSH config file (usually `~/.ssh/config`)

**Returns:**
- `Ok(Vec<SshTarget>)` - List of SSH hosts
- `Err` - If file cannot be read or parsed

#### SSH Client

```rust
impl SshClient {
    pub async fn connect(host: &str) -> Result<Self>
    pub async fn execute(&self, command: &str) -> Result<String>
    pub async fn upload_file(&self, local_path: &str, remote_path: &str) -> Result<()>
    pub async fn download_file(&self, remote_path: &str, local_path: &str) -> Result<()>
    pub async fn file_exists(&self, remote_path: &str) -> Result<bool>
    pub fn host(&self) -> &str
}
```

## Error Handling

All functions return `anyhow::Result` with detailed error messages:

```rust
// Connection failure
Error: SSH connection to 'workstation' failed:
ssh: Could not resolve hostname workstation: Name or service not known

Make sure:
1. Host is in ~/.ssh/config
2. SSH key is configured
3. Public key is on remote machine (~/.ssh/authorized_keys)

// Command failure
Error: Command failed on 'workstation':
bash: nonexistent-command: command not found

// Upload failure
Error: Local file not found: ./missing-file
```

## Narration

All operations emit narration events for observability:

```
🔌 Connecting to SSH host 'workstation'
✅ Connected to 'workstation'
🔧 Executing on 'workstation': uname -a
✅ Command completed on 'workstation'
📤 Uploading './binary' to '/usr/local/bin/binary' on 'workstation'
✅ Upload complete to 'workstation'
```

## Dependencies

```toml
[dependencies]
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["process"] }
observability-narration-core = { path = "../../../contracts/observability-narration-core" }
```

## Testing

```bash
# Run tests
cargo test -p ssh-config

# Test with actual SSH config
cargo test -p ssh-config -- --ignored
```

## Security Considerations

1. **Key-Based Auth Only** - No password prompts, uses SSH keys
2. **BatchMode** - Prevents interactive prompts that could hang
3. **Timeout** - 5-second connection timeout to prevent hanging
4. **No Credential Storage** - Uses system SSH agent
5. **Respects SSH Config** - Uses existing security settings

## Future Enhancements

- [ ] Connection pooling for multiple operations
- [ ] Progress callbacks for large file transfers
- [ ] Parallel operations on multiple hosts
- [ ] SSH tunnel support
- [ ] Connection status caching
- [ ] Automatic retry with exponential backoff

## History

- **TEAM-294** - Created SSH config parser
- **TEAM-314** - Added SSH client, migrated from hive-lifecycle

## License

GPL-3.0-or-later

## See Also

- `hive-lifecycle` - Uses this crate for remote hive operations
- `rbee-keeper` - Uses this crate for hive discovery
