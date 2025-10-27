# ssh-contract

SSH-related contracts for the rbee ecosystem.

## Purpose

This crate provides SSH-related types used across rbee components:
- ssh-config (parser)
- rbee-keeper (CLI)
- Tauri UI (frontend)

## Components

### SshTarget

Represents an SSH host from `~/.ssh/config`:

```rust
use ssh_contract::{SshTarget, SshTargetStatus};

let target = SshTarget::new("workstation", "192.168.1.100", "vince", 22)
    .with_status(SshTargetStatus::Online);

println!("Connect to: {}", target.connection_string());
// Output: Connect to: vince@192.168.1.100:22
```

### SshTargetStatus

Connection status:

```rust
use ssh_contract::SshTargetStatus;

let status = SshTargetStatus::Online;
assert!(status.is_online());
```

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
ssh-contract = { path = "../97_contracts/ssh-contract" }
```

## License

GPL-3.0-or-later
