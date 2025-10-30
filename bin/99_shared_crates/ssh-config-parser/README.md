# ssh-config-parser

**TEAM-365: Created by TEAM-365**

Shared crate for parsing SSH config files (`~/.ssh/config`).

## Purpose

Provides reusable SSH config parsing logic for Queen hive discovery and rbee-keeper SSH target listing.

## Usage

```rust
use ssh_config_parser::{parse_ssh_config, get_default_ssh_config_path, SshTarget};

// Get default SSH config path (~/.ssh/config)
let path = get_default_ssh_config_path();

// Parse SSH config
let targets = parse_ssh_config(&path)?;

for target in targets {
    println!("Host: {}, Hostname: {}, User: {}, Port: {}",
        target.host, target.hostname, target.user, target.port);
}
```

## API

### `SshTarget`

Represents a parsed SSH host entry.

```rust
pub struct SshTarget {
    pub host: String,      // Alias from SSH config (e.g., "workstation")
    pub hostname: String,  // Actual hostname/IP (e.g., "192.168.1.100")
    pub user: String,      // SSH username
    pub port: u16,         // SSH port (default: 22)
}
```

### Functions

- `parse_ssh_config(path: &Path) -> Result<Vec<SshTarget>>` - Parse SSH config file
- `get_default_ssh_config_path() -> PathBuf` - Get default SSH config path (`~/.ssh/config`)

## SSH Config Format

Supports basic SSH config format:

```text
Host workstation
    HostName 192.168.1.100
    User vince
    Port 22

Host server
    HostName example.com
    User admin
```

## Features

- Parses `Host`, `HostName`, `User`, `Port` directives
- Supports multiple aliases per host (e.g., `Host workstation workstation.local`)
- Defaults to current user if `User` not specified
- Defaults to port 22 if `Port` not specified
- Returns empty vec if SSH config doesn't exist

## Used By

- `bin/10_queen_rbee/src/discovery.rs` - Queen hive discovery
- `bin/00_rbee_keeper/src/tauri_commands.rs` - SSH target listing for UI
